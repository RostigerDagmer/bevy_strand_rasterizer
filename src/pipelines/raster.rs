use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer, BufferBindingType, BufferSize, CachedComputePipelineId, CachedRenderPipelineId, ColorTargetState, ColorWrites, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Extent3d, FilterMode, FragmentState, MultisampleState, PipelineCache, PrimitiveState, PushConstantRange, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderDefVal, ShaderStages, ShaderType, StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewUniform,
    },
};

use crate::{components::FroxelConfig, pipelines::layouts, plugin::MAX_TEXTURE_EXTENT, shader_types::PushConstants};

use super::shading::StrandShadingResources;

#[derive(Resource, Default)]
pub struct StrandRasterizerResources {
    pub pipeline: Option<ComputePipeline>,
    pub froxel_buffer: Option<Buffer>,
    pub froxel_config_buffer: Option<Buffer>,
    pub output_texture: Option<TextureView>,
    pub strand_count: Option<u32>,
    pub frustrum_config: Option<FroxelConfig>,
}

#[derive(Resource)]
pub struct StrandRasterizerPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub rasterize_pipeline: CachedComputePipelineId,
}

impl StrandRasterizerPipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        device.create_bind_group_layout(
            "strand_rasterizer_bind_group_layout",
            &[
                // Vertex buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::VERTEX_BUFFER,
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
                    binding: layouts::rasterizer::INDEX_BUFFER,
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
                    binding: layouts::rasterizer::META_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile offsets buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::TILE_OFFSETS_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile counts buffer (for debug; read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::TILE_COUNTS_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Froxel buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::FROXEL_TILE_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output texture (write-only storage texture)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::OUTPUT_TEXTURE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Froxel configuration (uniform buffer)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::FROXEL_CONFIG,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // View Uniform Buffer
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::VIEW_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Shading Buffer
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::SHADING_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        )
    }
}

impl FromWorld for StrandRasterizerPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let shader_loader = world.resource::<AssetServer>();
        let rasterize_shader = shader_loader.load("shaders/strand_rasterizer.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();
        let cdefs = [
            vec![ShaderDefVal::UInt(
                "MAX_TEXTURE_EXTENT".into(),
                MAX_TEXTURE_EXTENT,
            )],
            layouts::rasterizer::shader_defs(),
        ]
        .concat();

        let rasterize_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_rasterize_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: rasterize_shader,
            shader_defs: cdefs,
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "rasterize_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        info!(
            "Created strand raster compute pipelines: rasterize={:?}",
            rasterize_pipeline
        );

        StrandRasterizerPipeline {
            bind_group_layout,
            rasterize_pipeline,
        }
    }
}


// Create the bind group for the strand rasterizer
pub fn create_strand_raster_bind_group(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    vertex_buffer: &Buffer,
    index_buffer: &Buffer,
    tile_offsets_buffer: &Buffer,
    tile_counts_buffer: &Buffer,
    meta_buffer: &Buffer,
    packed_segments: &Buffer,
    output_texture: &TextureView,
    froxel_config_buffer: &Buffer,
    view_buffer: BindingResource,
    shading_buffer: &TextureView,
) -> BindGroup {
    device.create_bind_group(
        Some("strand_rasterizer_bind_group"),
        layout,
        &[
            BindGroupEntry {
                binding: layouts::rasterizer::VERTEX_BUFFER,
                resource: vertex_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::INDEX_BUFFER,
                resource: index_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::META_BUFFER,
                resource: meta_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::TILE_OFFSETS_BUFFER,
                resource: tile_offsets_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::TILE_COUNTS_BUFFER,
                resource: tile_counts_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::FROXEL_TILE_BUFFER,
                resource: packed_segments.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::OUTPUT_TEXTURE,
                resource: BindingResource::TextureView(output_texture),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::FROXEL_CONFIG,
                resource: froxel_config_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::VIEW_UNIFORM,
                resource: view_buffer.clone(),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::SHADING_BUFFER,
                resource: BindingResource::TextureView(shading_buffer),
            },
        ],
    )
}

// Create output texture for the rasterizer
pub fn create_render_target_texture(
    device: &RenderDevice,
    config: &FroxelConfig,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("strand_rasterizer_output"),
        size: Extent3d {
            width: config.screen_width,
            height: config.screen_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}


pub fn run_raster_pass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandRasterizerPipeline,
    froxel_config: &FroxelConfig,
    resources: &StrandRasterizerResources,
    shading_resources: &StrandShadingResources,
    bind_group: &BindGroup,
) {
    let Some(render_target) = &resources.output_texture else {
        warn!("Output texture not found");
        return;
    };
    let Some(packed_buffer) = &resources.froxel_buffer else {
        warn!("Froxel buffer not found");
        return;
    };
    let Some(config_buffer) = &resources.froxel_config_buffer else {
        warn!("Froxel config buffer not found");
        return;
    };

    let encoder = render_context.command_encoder(); // Get CommandEncoder

    // --- Rasterize ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Rasterize"),
            ..default()
        });
        let Some(raster_pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.rasterize_pipeline)
        else {
            warn!("Raster pipeline not found");
            return;
        };
        pass.set_pipeline(raster_pipeline);
        pass.set_bind_group(
            0,
            bind_group, // Assume correctly populated bind group
            &[],
        );
        // Set push constants if needed
        let pushconstants = PushConstants {
            num_elements: resources.strand_count.unwrap_or(0),
            workgroup_offset: shading_resources.max_segments_in_strand.unwrap_or(0), // TODO: maybe its time to make this its own field
            scan_load_base: 0,
            scan_save_base: 0,
        };
        pass.set_push_constants(0, bytemuck::bytes_of(&pushconstants));

        // Dispatch based on number of strands or segments
        let workgroup_size_x = froxel_config.froxel_size_x;
        let workgroup_size_y = froxel_config.froxel_size_y;
        let workgroups_x = froxel_config.screen_width / workgroup_size_x;
        let workgroups_y = froxel_config.screen_height / workgroup_size_y;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // --- Rasterization complete ---
    // output_texture is ready for composition
}