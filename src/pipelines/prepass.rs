use crate::{pipelines::layouts, resources::ComputeInvocationDims, shader_types::PushConstants};
use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, Buffer, BufferBindingType, CachedComputePipelineId, ComputePipelineDescriptor, IntoBinding,
            PipelineCache, PushConstantRange, ShaderStages,
        },
        renderer::RenderDevice,
        view::ViewUniformOffset,
    }, shader::ShaderDefVal,
};

const SIMULATOR_WORKGROUP_SIZE: u32 = 64;

#[derive(Resource, Default)]
pub struct StrandPrepassResources {
    // inputs
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub meta_buffer: Option<Buffer>,
    pub geos_buffer: Option<Buffer>,
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
            "strand_rasterizer_bind_group_layout",
            &[
                BindGroupLayoutEntry {
                    binding: layouts::prepass::VERTEX_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Index Buffer
                BindGroupLayoutEntry {
                    binding: layouts::prepass::INDEX_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Meta buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: layouts::prepass::META_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Geos buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: layouts::prepass::GEO_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            label: Some("strand_rasterize_pipeline".into()),
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
    entity: &Entity,
    device: &RenderDevice,
    pipeline: &StrandPrepassPipeline,
    resources: &StrandPrepassResources,
    view_buffer: &BindingResource,
    light_buffer: &BindingResource,
    view_offsets: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
) -> Result<(BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;
    let vertex_buffer = resources.vertex_buffer.as_ref().ok_or(())?;
    let index_buffer = resources.index_buffer.as_ref().ok_or(())?;
    let meta_buffer = resources.meta_buffer.as_ref().ok_or(())?;
    let geos_buffer = resources.geos_buffer.as_ref().ok_or(())?;
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
                    binding: layouts::prepass::VERTEX_BUFFER,
                    resource: vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::INDEX_BUFFER,
                    resource: index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::META_BUFFER,
                    resource: meta_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::GEO_BUFFER,
                    resource: geos_buffer.as_entire_binding(),
                },
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
