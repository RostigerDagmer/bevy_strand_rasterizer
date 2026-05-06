use crate::{
    allocator::GpuPagingAllocator,
    pipelines::layouts,
    pipelines::task_contract::{BINNING_POOL_CHUNK_SIZE, BINNING_POOL_NUM_HEADS},
    resources::ComputeInvocationDims,
    shader_types::{PushConstants, StrandGeo, StrandMeta},
};
use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, IntoBinding,
            PipelineCache, PushConstantRange, ShaderStages,
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewUniformOffset,
    },
    shader::ShaderDefVal,
};
use bevy_gpu_paging_allocator::BindGroupBuilder;

#[derive(Resource, Default)]
pub struct StrandPrepassResources {
    // inputs
    pub visibility_flags_buffer: Option<Buffer>,
    pub visible_geos_buffer: Option<Buffer>,
    pub geos_prefix_buffer: Option<Buffer>,
    pub prepass_queue: Option<Buffer>,
    pub binning_queue: Option<Buffer>,
    // products
    pub indirect_args: Option<Buffer>,
    // queue-binning allocator buffers
    pub chunk_pool: Option<Buffer>,
    pub free_heads: Option<Buffer>,
    pub frustum_table: Option<Buffer>,
    pub froxel_bucket_heads: Option<Buffer>,
    pub raster_work_queue: Option<Buffer>,
    // capacities
    pub prepass_task_capacity: u32,
    pub binning_task_capacity: u32,
    pub geo_capacity: u32,
    pub frustum_capacity: u32,
    pub froxel_bucket_capacity: u32,
    pub raster_work_capacity: u32,
}

#[derive(Resource)]
pub struct StrandPrepassPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub broad_pipeline: Option<CachedComputePipelineId>,
    pub finalize_pipeline: Option<CachedComputePipelineId>,
    pub fine_pipeline: Option<CachedComputePipelineId>,
    pub finalize_binning_pipeline: Option<CachedComputePipelineId>,
    pub binning_pipeline: Option<CachedComputePipelineId>,
}

impl StrandPrepassPipeline {
    fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn uniform_entry(binding: u32, has_dynamic_offset: bool) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "strand_prepass_bind_group_layout",
            &[
                Self::storage_entry(layouts::prepass::PREPASS_QUEUE, false),
                Self::storage_entry(layouts::prepass::BINNING_QUEUE, false),
                Self::uniform_entry(layouts::prepass::LIGHT_UNIFORM, true),
                Self::storage_entry(layouts::prepass::CLUSTER_INDICES, true),
                Self::storage_entry(layouts::prepass::CLUSTER_OFFSETS_AND_COUNTS, true),
                Self::storage_entry(layouts::prepass::CLUSTERABLE_OBJECTS, true),
                Self::uniform_entry(layouts::prepass::VIEW_UNIFORM, true),
                Self::storage_entry(layouts::prepass::VISIBLE_FLAGS, false),
                Self::storage_entry(layouts::prepass::VISIBLE_GEO, false),
                Self::storage_entry(layouts::prepass::GEO_PREFIX, false),
                Self::storage_entry(layouts::prepass::INDIRECT_BUFFER, false),
                Self::storage_entry(layouts::prepass::FRUSTUM_TABLE, true),
                Self::storage_entry(layouts::prepass::FROXEL_BUCKET_HEADS, false),
                Self::storage_entry(layouts::prepass::CHUNK_POOL, false),
                Self::storage_entry(layouts::prepass::FREE_HEADS, false),
                Self::storage_entry(layouts::prepass::RASTER_WORK_QUEUE, false),
            ],
        )
    }

    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        let descriptor = Self::bind_group_layout_descriptor();
        device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries)
    }
}

fn queue_prepass_pipeline(
    pipeline_cache: &PipelineCache,
    shader: Handle<Shader>,
    _bind_group_layout: BindGroupLayout,
    allocator: &GpuPagingAllocator,
    invocation_dims: &ComputeInvocationDims,
    entry_point: &'static str,
) -> Option<CachedComputePipelineId> {
    if allocator.buffer_bind_group_layout.is_none()
        || allocator.pagetable_bind_group_layout.is_none()
    {
        return None;
    }
    let allocator_layout_entries = allocator.layout_entries();
    let buffer_layout = BindGroupLayoutDescriptor::new(
        "gpu_paging_allocator_buffer_layout",
        &allocator_layout_entries.pools,
    );
    let table_layout = BindGroupLayoutDescriptor::new(
        "gpu_paging_allocator_table_layout",
        &allocator_layout_entries.page_tables,
    );
    let bind_group_layout = StrandPrepassPipeline::bind_group_layout_descriptor();

    let mut shader_defs = vec![layouts::prepass::shader_defs(), allocator.shader_defs()].concat();
    shader_defs.extend([
        ShaderDefVal::UInt(
            "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
            invocation_dims.threads_per_workgroup,
        ),
        ShaderDefVal::UInt(
            "WORKGROUP_SIZE".into(),
            invocation_dims.threads_per_workgroup,
        ),
        ShaderDefVal::UInt(
            "NUMBER_OF_THREADS_PER_SUBGROUP".into(),
            invocation_dims.subgroup_size,
        ),
        ShaderDefVal::UInt(
            "FINE_WORKGROUP_SIZE".into(),
            invocation_dims.threads_per_workgroup,
        ),
        ShaderDefVal::UInt(
            "SIZEOF_METADATA".into(),
            std::mem::size_of::<StrandMeta>() as u32,
        ),
        ShaderDefVal::UInt("SIZEOF_GEO".into(), std::mem::size_of::<StrandGeo>() as u32),
        ShaderDefVal::UInt("POOL_CHUNK_SIZE".into(), BINNING_POOL_CHUNK_SIZE),
        ShaderDefVal::UInt("POOL_NUM_HEADS".into(), BINNING_POOL_NUM_HEADS),
    ]);

    let max_group = allocator
        .buffer_group_idx
        .max(allocator.table_group_idx)
        .max(layouts::prepass::PREPASS_GROUP);
    let mut layout = vec![bind_group_layout.clone(); (max_group + 1) as usize];
    layout[allocator.buffer_group_idx as usize] = buffer_layout;
    layout[allocator.table_group_idx as usize] = table_layout;
    layout[layouts::prepass::PREPASS_GROUP as usize] = bind_group_layout;

    Some(
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_prepass_pipeline".into()),
            layout,
            shader,
            shader_defs,
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: Some(entry_point.into()),
            zero_initialize_workgroup_memory: false,
        }),
    )
}

impl FromWorld for StrandPrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let _invocation_dims = *world.resource::<ComputeInvocationDims>();

        StrandPrepassPipeline {
            bind_group_layout,
            broad_pipeline: None,
            finalize_pipeline: None,
            fine_pipeline: None,
            finalize_binning_pipeline: None,
            binning_pipeline: None,
        }
    }
}

pub fn create_prepass_bind_group(
    device: &RenderDevice,
    pipeline: &StrandPrepassPipeline,
    resources: &StrandPrepassResources,
    view_buffer: &BindingResource,
    light_buffer: &BindingResource,
    view_offsets: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
    cluster_indices: &BindingResource,
    cluster_offsets_and_counts: &BindingResource,
    clusterable_objects: &BindingResource,
) -> Result<(BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;
    let prepass_queue = resources.prepass_queue.as_ref().ok_or(())?;
    let binning_queue = resources.binning_queue.as_ref().ok_or(())?;
    let visibility_flags_buffer = resources.visibility_flags_buffer.as_ref().ok_or(())?;
    let visible_geos_buffer = resources.visible_geos_buffer.as_ref().ok_or(())?;
    let geos_prefix_buffer = resources.geos_prefix_buffer.as_ref().ok_or(())?;
    let dispatch_args = resources.indirect_args.as_ref().ok_or(())?;
    let frustum_table = resources.frustum_table.as_ref().ok_or(())?;
    let froxel_bucket_heads = resources.froxel_bucket_heads.as_ref().ok_or(())?;
    let chunk_pool = resources.chunk_pool.as_ref().ok_or(())?;
    let free_heads = resources.free_heads.as_ref().ok_or(())?;
    let raster_work_queue = resources.raster_work_queue.as_ref().ok_or(())?;

    Ok((
        device.create_bind_group(
            Some("strand_prepass_bind_group"),
            layout,
            &[
                BindGroupEntry {
                    binding: layouts::prepass::PREPASS_QUEUE,
                    resource: prepass_queue.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::BINNING_QUEUE,
                    resource: binning_queue.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::LIGHT_UNIFORM,
                    resource: light_buffer.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::CLUSTER_INDICES,
                    resource: cluster_indices.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::CLUSTER_OFFSETS_AND_COUNTS,
                    resource: cluster_offsets_and_counts.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::CLUSTERABLE_OBJECTS,
                    resource: clusterable_objects.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::VIEW_UNIFORM,
                    resource: view_buffer.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::VISIBLE_FLAGS,
                    resource: visibility_flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::VISIBLE_GEO,
                    resource: visible_geos_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::GEO_PREFIX,
                    resource: geos_prefix_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::INDIRECT_BUFFER,
                    resource: dispatch_args.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FRUSTUM_TABLE,
                    resource: frustum_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FROXEL_BUCKET_HEADS,
                    resource: froxel_bucket_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::CHUNK_POOL,
                    resource: chunk_pool.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FREE_HEADS,
                    resource: free_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::RASTER_WORK_QUEUE,
                    resource: raster_work_queue.as_entire_binding(),
                },
            ],
        ),
        // Dynamic offsets order follows bind-group layout declaration order.
        vec![view_light_uniform_offset.offset, view_offsets.offset],
    ))
}

pub fn run_prepass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandPrepassPipeline,
    allocator: &GpuPagingAllocator,
    settings: &ComputeInvocationDims,
    cull_settings: &crate::resources::StochasticCullSettings,
    bind_group: &BindGroup,
    uniform_offsets: &[u32],
    indirect_args: &Buffer,
) {
    let encoder = render_context.command_encoder();
    let Some(allocator_buffer_bind_group) = &allocator.buffer_bind_group else {
        warn!("allocator buffer bind group is not ready yet.");
        return;
    };
    let Some(allocator_pagetable_bind_group) = &allocator.pagetable_bind_group else {
        warn!("allocator pagetable bind group is not ready yet.");
        return;
    };

    let Some(broad_pipeline_id) = pipeline.broad_pipeline else {
        warn!("Broad prepass pipeline id not ready yet");
        return;
    };
    let Some(finalize_pipeline_id) = pipeline.finalize_pipeline else {
        warn!("Finalize prepass pipeline id not ready yet");
        return;
    };
    let Some(fine_pipeline_id) = pipeline.fine_pipeline else {
        warn!("Fine prepass pipeline id not ready yet");
        return;
    };
    let Some(finalize_binning_pipeline_id) = pipeline.finalize_binning_pipeline else {
        warn!("Finalize binning pipeline id not ready yet");
        return;
    };
    let Some(binning_pipeline_id) = pipeline.binning_pipeline else {
        warn!("Binning queue pipeline id not ready yet");
        return;
    };
    let Some(broad_pipeline) = pipeline_cache.get_compute_pipeline(broad_pipeline_id) else {
        warn!("Broad prepass pipeline not found");
        return;
    };
    let Some(finalize_pipeline) = pipeline_cache.get_compute_pipeline(finalize_pipeline_id) else {
        warn!("Finalize prepass pipeline not found");
        return;
    };
    let Some(fine_pipeline) = pipeline_cache.get_compute_pipeline(fine_pipeline_id) else {
        warn!("Fine prepass pipeline not found");
        return;
    };
    let Some(finalize_binning_pipeline) =
        pipeline_cache.get_compute_pipeline(finalize_binning_pipeline_id)
    else {
        warn!("Finalize binning pipeline not found");
        return;
    };
    let Some(binning_pipeline) = pipeline_cache.get_compute_pipeline(binning_pipeline_id) else {
        warn!("Binning queue pipeline not found");
        return;
    };
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Strand Prepass"),
        ..default()
    });
    let pushconstants = PushConstants {
        stochastic_cull_enabled: u32::from(cull_settings.enabled),
        cull_min_dist: cull_settings.min_dist,
        cull_max_dist: cull_settings.max_dist,
        cull_exponent: cull_settings.exponent,
        ..Default::default()
    };
    pass.set_pipeline(broad_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&pushconstants));
    pass.set_bind_group(allocator.buffer_group_idx, allocator_buffer_bind_group, &[]);
    pass.set_bind_group(
        allocator.table_group_idx,
        allocator_pagetable_bind_group,
        &[],
    );
    pass.set_bind_group(layouts::prepass::PREPASS_GROUP, bind_group, uniform_offsets);
    pass.dispatch_workgroups(
        settings.dispatch_size.0,
        settings.dispatch_size.1,
        settings.dispatch_size.2,
    );
    pass.set_pipeline(finalize_pipeline);
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(fine_pipeline);
    pass.dispatch_workgroups_indirect(indirect_args, 0);
    pass.set_pipeline(finalize_binning_pipeline);
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(binning_pipeline);
    pass.dispatch_workgroups_indirect(indirect_args, 0);
}

pub fn update_strand_prepass_pipeline(
    pipeline_cache: Res<PipelineCache>,
    mut pipeline_res: ResMut<StrandPrepassPipeline>,
    dims: Res<ComputeInvocationDims>,
    mut last_state: Local<Option<(ComputeInvocationDims, u64)>>,
    shader_loader: Res<AssetServer>,
    allocator: Res<GpuPagingAllocator>,
) {
    let current_state = (*dims, allocator.bindgroups_epoch);
    if last_state.as_ref() == Some(&current_state) {
        return;
    }

    if allocator.buffer_bind_group_layout.is_none()
        || allocator.pagetable_bind_group_layout.is_none()
    {
        return;
    }

    *last_state = Some(current_state);

    let broad_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let finalize_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let fine_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let finalize_binning_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let binning_shader = shader_loader.load("shaders/strand_prepass.wgsl");

    let Some(broad_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        broad_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "broad_prepass",
    ) else {
        warn!("Could not queue broad prepass pipeline");
        return;
    };
    let Some(finalize_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        finalize_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "finalize_prepass",
    ) else {
        return;
    };
    let Some(fine_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        fine_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "fine_prepass",
    ) else {
        return;
    };
    let Some(finalize_binning_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        finalize_binning_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "finalize_binning",
    ) else {
        return;
    };
    let Some(binning_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        binning_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "binning_queue_pass",
    ) else {
        return;
    };
    pipeline_res.broad_pipeline = Some(broad_pipeline_id);
    pipeline_res.finalize_pipeline = Some(finalize_pipeline_id);
    pipeline_res.fine_pipeline = Some(fine_pipeline_id);
    pipeline_res.finalize_binning_pipeline = Some(finalize_binning_pipeline_id);
    pipeline_res.binning_pipeline = Some(binning_pipeline_id);
    debug!(
        "Rebuilt strand prepass pipelines: broad={:?} finalize={:?} fine={:?} finalize_binning={:?} binning={:?}",
        broad_pipeline_id,
        finalize_pipeline_id,
        fine_pipeline_id,
        finalize_binning_pipeline_id,
        binning_pipeline_id
    );
}
