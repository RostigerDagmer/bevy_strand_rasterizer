use bevy::{prelude::*, render::{render_resource::{BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferInitDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, PushConstantRange, ShaderDefVal, ShaderStages, TextureView}, renderer::{RenderContext, RenderDevice}}};
use crate::{pipelines::layouts, shader_types::PushConstants};

const SIMULATOR_WORKGROUP_SIZE: u32 = 64;

#[derive(Resource, Default)]
pub struct StrandPrepassResources {
    // inputs
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub meta_buffer: Option<Buffer>,
    pub geos_buffer: Option<Buffer>,
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
                }
            ]
        )
    }
}

impl FromWorld for StrandPrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let shader_loader = world.resource::<AssetServer>();
        let rasterize_shader = shader_loader.load("shaders/strand_prepass.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_rasterize_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: rasterize_shader,
            shader_defs: layouts::prepass::shader_defs(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "broad_phase".into(),
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


// pub fn create_prepass_bind_group(
//     entity: &Entity,
//     device: &RenderDevice,
//     pipeline: &StrandPrepassPipeline,
//     resources: &StrandPrepassResources,
//     view_buffer: &BindingResource,
//     light_buffer: &BindingResource,
//     view_offsets: &ViewUniformOffset,
//     view_light_uniform_offset: &ViewLightsUniformOffset,
// ) -> Result<(BindGroup, Vec<u32>), ()> {
//     let layout = &pipeline.bind_group_layout;
//     let vertex_buffer = buffers.vertex_buffer.as_ref().ok_or(())?;
//     let index_buffer = buffers.index_buffer.as_ref().ok_or(())?;

//     // TODO: we have to bind all maps created for lights.
//     let light_entities = resources
//         .froxel_config_buffer
//         .keys()
//         .find(|k| *k != entity)
//         .ok_or(())?;
//     let dom_texture = shadow_resources.dom_targets.get(light_entities).ok_or(())?;
//     let dom_sampler = shadow_resources
//         .dom_samplers
//         .get(light_entities)
//         .ok_or(())?;
//     let artifacts = buffers.artifacts.get(entity).ok_or(())?;

//     let tile_offsets_buffer = &artifacts.tile_offsets_buffer;
//     let tile_counts_buffer = &artifacts.tile_counts_buffer;
//     let meta_buffer = buffers.meta_buffer.as_ref().ok_or(())?;
//     let geos_buffer = buffers.geos_buffer.as_ref().ok_or(())?;
//     let material_buffer = shading_resources.materials.as_ref().ok_or(())?;
//     let packed_segments = resources.froxel_buffer.get(entity).ok_or(())?;
//     let output_texture = resources.output_texture.as_ref().ok_or(())?;
//     let output_depth = resources.output_depth.as_ref().ok_or(())?;
//     let froxel_config_buffer = resources.froxel_config_buffer.get(entity).ok_or(())?;
//     let shading_buffer = shading_resources.output_texture.as_ref().ok_or(())?;

//     Ok((
//         device.create_bind_group(
//             Some("strand_rasterizer_bind_group"),
//             layout,
//             &[
//                 BindGroupEntry {
//                     binding: layouts::rasterizer::VERTEX_BUFFER,
//                     resource: vertex_buffer.as_entire_binding(),
//                 },
//                 BindGroupEntry {
//                     binding: layouts::rasterizer::INDEX_BUFFER,
//                     resource: index_buffer.as_entire_binding(),
//                 },
//                 BindGroupEntry {
//                     binding: layouts::rasterizer::META_BUFFER,
//                     resource: meta_buffer.as_entire_binding(),
//                 },
//                 BindGroupEntry {
//                     binding: layouts::rasterizer::GEO_BUFFER,
//                     resource: geos_buffer.as_entire_binding(),
//                 },
//             ]
//         ), vec![view_offsets.offset]))
// }
