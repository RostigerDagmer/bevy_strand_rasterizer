use crate::{
    allocator::GpuPagingAllocator,
    pipelines::layouts,
    resources::ComputeInvocationDims,
    shader_types::{PushConstants, StrandGeo, StrandMeta},
};
use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, Buffer, BufferBindingType, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, IntoBinding, PipelineCache, PushConstantRange, ShaderStages,
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewUniformOffset,
    },
    shader::ShaderDefVal,
};

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
    // capacities
    pub prepass_task_capacity: u32,
    pub binning_task_capacity: u32,
    pub geo_capacity: u32,
}

#[derive(Resource)]
pub struct StrandPrepassPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub broad_pipeline: Option<CachedComputePipelineId>,
    pub finalize_pipeline: Option<CachedComputePipelineId>,
    pub fine_pipeline: Option<CachedComputePipelineId>,
    pub finalize_binning_pipeline: Option<CachedComputePipelineId>,
    pub binning_pipeline: Option<CachedComputePipelineId>,
    pub scan_sums_pipeline: Option<CachedComputePipelineId>,
    pub scan_last_pipeline: Option<CachedComputePipelineId>,
    pub scan_prfx_pipeline: Option<CachedComputePipelineId>,
    pub init_place_pipeline: Option<CachedComputePipelineId>,
    pub place_pipeline: Option<CachedComputePipelineId>,
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

    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        device.create_bind_group_layout(
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
                Self::storage_entry(layouts::prepass::TILE_COUNTS_BUFFER, false),
                Self::uniform_entry(layouts::prepass::FROXEL_CONFIG, false),
                Self::storage_entry(layouts::prepass::TILE_OFFSETS_BUFFER, false),
                Self::storage_entry(layouts::prepass::CURRENT_TILE_WRITE_INDICES, false),
                Self::storage_entry(layouts::prepass::FROXEL_TILE_BUFFER, false),
            ],
        )
    }
}

fn queue_prepass_pipeline(
    pipeline_cache: &PipelineCache,
    shader: Handle<Shader>,
    bind_group_layout: BindGroupLayout,
    allocator: &GpuPagingAllocator,
    invocation_dims: &ComputeInvocationDims,
    entry_point: &'static str,
) -> Option<CachedComputePipelineId> {
    let (Some(buffer_layout), Some(table_layout)) = (
        allocator.buffer_bind_group_layout.clone(),
        allocator.pagetable_bind_group_layout.clone(),
    ) else {
        return None;
    };

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
            scan_sums_pipeline: None,
            scan_last_pipeline: None,
            scan_prfx_pipeline: None,
            init_place_pipeline: None,
            place_pipeline: None,
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
    tile_counts: &BindingResource,
    froxel_config: &BindingResource,
    tile_offsets: &BindingResource,
    current_tile_write_indices: &BindingResource,
    froxel_tile_buffer: &BindingResource,
) -> Result<(BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;
    let prepass_queue = resources.prepass_queue.as_ref().ok_or(())?;
    let binning_queue = resources.binning_queue.as_ref().ok_or(())?;
    let visibility_flags_buffer = resources.visibility_flags_buffer.as_ref().ok_or(())?;
    let visible_geos_buffer = resources.visible_geos_buffer.as_ref().ok_or(())?;
    let geos_prefix_buffer = resources.geos_prefix_buffer.as_ref().ok_or(())?;
    let dispatch_args = resources.indirect_args.as_ref().ok_or(())?;

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
                    binding: layouts::prepass::TILE_COUNTS_BUFFER,
                    resource: tile_counts.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FROXEL_CONFIG,
                    resource: froxel_config.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::TILE_OFFSETS_BUFFER,
                    resource: tile_offsets.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::CURRENT_TILE_WRITE_INDICES,
                    resource: current_tile_write_indices.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FROXEL_TILE_BUFFER,
                    resource: froxel_tile_buffer.clone().into_binding(),
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
    froxel_config: &crate::components::FroxelConfig,
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
    let Some(scan_sums_pipeline_id) = pipeline.scan_sums_pipeline else {
        warn!("Scan sums pipeline id not ready yet");
        return;
    };
    let Some(scan_last_pipeline_id) = pipeline.scan_last_pipeline else {
        warn!("Scan last pipeline id not ready yet");
        return;
    };
    let Some(scan_prfx_pipeline_id) = pipeline.scan_prfx_pipeline else {
        warn!("Scan prefix pipeline id not ready yet");
        return;
    };
    let Some(init_place_pipeline_id) = pipeline.init_place_pipeline else {
        warn!("Init place pipeline id not ready yet");
        return;
    };
    let Some(place_pipeline_id) = pipeline.place_pipeline else {
        warn!("Queue place pipeline id not ready yet");
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
    let Some(scan_sums_pipeline) = pipeline_cache.get_compute_pipeline(scan_sums_pipeline_id)
    else {
        warn!("Scan sums pipeline not found");
        return;
    };
    let Some(scan_last_pipeline) = pipeline_cache.get_compute_pipeline(scan_last_pipeline_id) else {
        warn!("Scan last pipeline not found");
        return;
    };
    let Some(scan_prfx_pipeline) = pipeline_cache.get_compute_pipeline(scan_prfx_pipeline_id)
    else {
        warn!("Scan prefix pipeline not found");
        return;
    };
    let Some(init_place_pipeline) = pipeline_cache.get_compute_pipeline(init_place_pipeline_id)
    else {
        warn!("Init place pipeline not found");
        return;
    };
    let Some(place_pipeline) = pipeline_cache.get_compute_pipeline(place_pipeline_id) else {
        warn!("Queue place pipeline not found");
        return;
    };

    let num_tiles_x =
        (froxel_config.screen_width + froxel_config.froxel_size_x - 1) / froxel_config.froxel_size_x;
    let num_tiles_y =
        (froxel_config.screen_height + froxel_config.froxel_size_y - 1) / froxel_config.froxel_size_y;
    let num_tiles = num_tiles_x * num_tiles_y * froxel_config.depth_slices;
    let scan_threads = settings.threads_per_workgroup.max(1);
    const SCAN_LOAD_BASE_OFFSET: u32 = 8;
    const SCAN_SAVE_BASE_OFFSET: u32 = 12;

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

    // Prefix-sum over tile counts -> tile offsets.
    let mut load_base = 0u32;
    let mut save_base = num_tiles;
    let mut rounds: std::vec::Vec<(u32, u32, u32)> = std::vec::Vec::new();
    while save_base - load_base > scan_threads {
        let number_of_workgroups = (save_base - load_base).div_ceil(scan_threads);
        rounds.push((load_base, save_base, number_of_workgroups));
        load_base = save_base;
        save_base += number_of_workgroups;
    }

    pass.set_pipeline(scan_sums_pipeline);
    for (load_base, save_base, number_of_workgroups) in rounds.iter() {
        pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(load_base));
        pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(save_base));
        pass.dispatch_workgroups(*number_of_workgroups, 1, 1);
    }

    pass.set_pipeline(scan_last_pipeline);
    pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(&load_base));
    pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(&save_base));
    pass.dispatch_workgroups(1, 1, 1);

    pass.set_pipeline(scan_prfx_pipeline);
    for (load_base, save_base, number_of_workgroups) in rounds.iter().rev() {
        pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(save_base));
        pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(load_base));
        pass.dispatch_workgroups(*number_of_workgroups, 1, 1);
    }

    pass.set_pipeline(init_place_pipeline);
    pass.dispatch_workgroups(num_tiles.div_ceil(256), 1, 1);

    pass.set_pipeline(place_pipeline);
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
    let scan_sums_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let scan_last_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let scan_prfx_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let init_place_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let place_shader = shader_loader.load("shaders/strand_prepass.wgsl");

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
    let Some(scan_sums_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        scan_sums_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "scan_sums",
    ) else {
        return;
    };
    let Some(scan_last_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        scan_last_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "scan_last",
    ) else {
        return;
    };
    let Some(scan_prfx_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        scan_prfx_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "scan_prfx",
    ) else {
        return;
    };
    let Some(init_place_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        init_place_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "init_placement_idx",
    ) else {
        return;
    };
    let Some(place_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        place_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "binning_queue_place_pass",
    ) else {
        return;
    };

    pipeline_res.broad_pipeline = Some(broad_pipeline_id);
    pipeline_res.finalize_pipeline = Some(finalize_pipeline_id);
    pipeline_res.fine_pipeline = Some(fine_pipeline_id);
    pipeline_res.finalize_binning_pipeline = Some(finalize_binning_pipeline_id);
    pipeline_res.binning_pipeline = Some(binning_pipeline_id);
    pipeline_res.scan_sums_pipeline = Some(scan_sums_pipeline_id);
    pipeline_res.scan_last_pipeline = Some(scan_last_pipeline_id);
    pipeline_res.scan_prfx_pipeline = Some(scan_prfx_pipeline_id);
    pipeline_res.init_place_pipeline = Some(init_place_pipeline_id);
    pipeline_res.place_pipeline = Some(place_pipeline_id);
    debug!(
        "Rebuilt strand prepass pipelines: broad={:?} finalize={:?} fine={:?} finalize_binning={:?} binning_count={:?} scan_sums={:?} scan_last={:?} scan_prfx={:?} init_place={:?} binning_place={:?}",
        broad_pipeline_id,
        finalize_pipeline_id,
        fine_pipeline_id,
        finalize_binning_pipeline_id,
        binning_pipeline_id,
        scan_sums_pipeline_id,
        scan_last_pipeline_id,
        scan_prfx_pipeline_id,
        init_place_pipeline_id,
        place_pipeline_id
    );
}
