use crate::{
    allocator::GpuPagingAllocator, pipelines::layouts, resources::ComputeInvocationDims,
    shader_types::PushConstants,
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

const SIMULATOR_WORKGROUP_SIZE: u32 = 64;

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
    // pub aabbs: Option<Buffer>,
}

#[derive(Resource)]
pub struct StrandPrepassPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl StrandPrepassPipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        device.create_bind_group_layout(
            "strand_prepass_bind_group_layout",
            &[
                // Light Uniform Buffer
                BindGroupLayoutEntry {
                    binding: layouts::prepass::LIGHT_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cluster Indices
                BindGroupLayoutEntry {
                    binding: layouts::prepass::CLUSTER_INDICES,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cluster Offsets and Counts
                BindGroupLayoutEntry {
                    binding: layouts::prepass::CLUSTER_OFFSETS_AND_COUNTS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Clusterable Objects
                BindGroupLayoutEntry {
                    binding: layouts::prepass::CLUSTERABLE_OBJECTS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // View Uniform Buffer
                BindGroupLayoutEntry {
                    binding: layouts::prepass::VIEW_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                ///////////////////////////////////////////////////////////
                // Indirect Buffer
                BindGroupLayoutEntry {
                    binding: layouts::prepass::INDIRECT_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )
    }
}

impl FromWorld for StrandPrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let invocation_dims = world.resource::<ComputeInvocationDims>();

        let shader_loader = world.resource::<AssetServer>();
        let rasterize_shader = shader_loader.load("shaders/strand_prepass.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();

        let cdefs = [
            vec![
                ShaderDefVal::UInt(
                    "WORKGROUP_SIZE".into(),
                    invocation_dims.threads_per_workgroup,
                ),
                ShaderDefVal::UInt(
                    "NUMBER_OF_THREADS_PER_SUBGROUP".into(),
                    invocation_dims.subgroup_size, // Use detected subgroup size
                ),
            ],
            layouts::binning::shader_defs(),
        ]
        .concat();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_prepass_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: rasterize_shader,
            shader_defs: vec![layouts::prepass::shader_defs(), cdefs].concat(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: Some("broad_phase".into()),
            zero_initialize_workgroup_memory: false,
        });

        debug!(
            "Created strand prepass compute pipelines: prepass={:?}",
            pipeline
        );

        StrandPrepassPipeline {
            bind_group_layout,
            pipeline,
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
            ],
        ),
        vec![view_offsets.offset, view_light_uniform_offset.offset],
    ))
}

pub fn run_prepass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandPrepassPipeline,
    allocator: &GpuPagingAllocator,
    settings: &ComputeInvocationDims,
    bind_group: &BindGroup,
    uniform_offsets: &[u32],
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
    // --- Prepass ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Prepass"),
            ..default()
        });
        let Some(prepass_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
            warn!("Prepass pipeline not found");
            return;
        };
        pass.set_pipeline(prepass_pipeline);
        pass.set_bind_group(
            0,
            bind_group, // Assume correctly populated bind group
            uniform_offsets,
        );
        pass.set_bind_group(
            allocator.buffer_group_idx,
            allocator_buffer_bind_group,
            &[]
        );
        pass.set_bind_group(
            allocator.table_group_idx,
            allocator_pagetable_bind_group,
            &[]
        );

        pass.dispatch_workgroups(settings.dispatch_size.0, settings.dispatch_size.1, settings.dispatch_size.2);

    }
}


fn update_strand_prepass_pipeline(
    pipeline_cache: Res<PipelineCache>,
    mut pipeline_res: ResMut<StrandPrepassPipeline>,
    dims: Res<ComputeInvocationDims>,
    mut last_dims: Local<Option<ComputeInvocationDims>>,
    shader_loader: Res<AssetServer>,
) {
    if last_dims.as_ref() == Some(&*dims) {
        return; // same as before
    }
    *last_dims = Some(dims.clone());

    let bind_group_layout = pipeline_res.bind_group_layout.clone();
    let shader = shader_loader.load("shaders/strand_prepass.wgsl");

    let shader_defs = vec![
        ShaderDefVal::UInt("WORKGROUP_SIZE".into(), dims.threads_per_workgroup),
        ShaderDefVal::UInt("NUMBER_OF_THREADS_PER_SUBGROUP".into(), dims.subgroup_size),
    ];

    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("strand_prepass_pipeline".into()),
        layout: vec![bind_group_layout.clone()],
        shader,
        shader_defs,
        push_constant_ranges: vec![PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<PushConstants>() as u32,
        }],
        entry_point: Some("broad_phase".into()),
        zero_initialize_workgroup_memory: false,
    });

    // Update resource to use the new pipeline handle
    pipeline_res.pipeline = pipeline_id;

    debug!("Rebuilt strand prepass pipeline: {:?}", pipeline_id);
}
