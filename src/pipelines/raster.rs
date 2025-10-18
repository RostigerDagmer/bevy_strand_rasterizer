use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, BlendState, Buffer, BufferBindingType, BufferSize,
            CachedComputePipelineId, CachedRenderPipelineId, ColorTargetState, ColorWrites,
            ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Extent3d,
            FilterMode, FragmentState, MultisampleState, PipelineCache, PrimitiveState,
            PushConstantRange, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            SamplerDescriptor, ShaderDefVal, ShaderStages, ShaderType, StorageTextureAccess,
            Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
            TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniform, ViewUniformOffset},
    },
};

use std::collections::HashMap;

use crate::{
    components::FroxelConfig, pipelines::layouts, plugin::MAX_TEXTURE_EXTENT,
    shader_types::PushConstants,
};

use super::{
    binning::StrandBinningBuffers, shading::StrandShadingResources, shadows::StrandShadowResources,
};

#[derive(Resource, Default)]
pub struct StrandRasterizerResources {
    pub pipeline: Option<ComputePipeline>,
    pub froxel_buffer: HashMap<Entity, Buffer>,
    pub froxel_config_buffer: HashMap<Entity, Buffer>,
    pub output_texture: Option<TextureView>,
    pub output_depth: Option<TextureView>,
    pub strand_count: Option<u32>,
    pub frustrum_config: HashMap<Entity, FroxelConfig>,
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
                // Geos buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::GEO_BUFFER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Material buffer
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::MATERIAL_BUFFER,
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
                // Output depth texture (write-only storage texture)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::OUTPUT_DEPTH,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
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
                // Light Uniform Buffer
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::LIGHT_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
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
                        has_dynamic_offset: true,
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
                // Deep Opacity Texture Array (Opacity layers)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Deep Opacity Texture View
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O_VIEW,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // Deep Opacity Texture Array (Depth)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Deep Opacity Texture View
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D_VIEW,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
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
        let use_spline = false;

        let shader_loader = world.resource::<AssetServer>();
        let rasterize_shader = shader_loader.load("shaders/strand_rasterizer.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();
        let mut cdefs = [
            vec![ShaderDefVal::UInt(
                "MAX_TEXTURE_EXTENT".into(),
                MAX_TEXTURE_EXTENT,
            )],
            layouts::rasterizer::shader_defs(),
        ]
        .concat();
        if use_spline {
            // TODO: fix spline shader (also optimize spline shader)
            cdefs.push("SPLINE".into());
        } else {
            cdefs.push("LINEAR".into());
        }

        let rasterize_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_rasterize_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: rasterize_shader,
            shader_defs: [
                cdefs.as_slice(),
                &[
                    ShaderDefVal::UInt(
                        "DEEP_OPACITY_TEXTURE_O_VIEW".into(),
                        layouts::rasterizer::DEEP_OPACITY_TEXTURE_O_VIEW,
                    ),
                    ShaderDefVal::UInt(
                        "DEEP_OPACITY_TEXTURE_D_VIEW".into(),
                        layouts::rasterizer::DEEP_OPACITY_TEXTURE_D_VIEW,
                    ),
                ],
            ]
            .concat(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "rasterize_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        debug!(
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
    entity: &Entity,
    device: &RenderDevice,
    pipeline: &StrandRasterizerPipeline,
    resources: &StrandRasterizerResources,
    buffers: &StrandBinningBuffers,
    shading_resources: &StrandShadingResources,
    shadow_resources: &StrandShadowResources,
    view_buffer: &BindingResource,
    light_buffer: &BindingResource,
    view_offsets: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
) -> Result<(BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;
    let vertex_buffer = buffers.vertex_buffer.as_ref().ok_or(())?;
    let index_buffer = buffers.index_buffer.as_ref().ok_or(())?;

    // TODO: we have to bind all maps created for lights.
    let light_entities = resources
        .froxel_config_buffer
        .keys()
        .find(|k| *k != entity)
        .ok_or(())?;
    let dom_texture = shadow_resources.dom_targets.get(light_entities).ok_or(())?;
    let dom_sampler = shadow_resources
        .dom_samplers
        .get(light_entities)
        .ok_or(())?;
    let artifacts = buffers.artifacts.get(entity).ok_or(())?;

    let tile_offsets_buffer = &artifacts.tile_offsets_buffer;
    let tile_counts_buffer = &artifacts.tile_counts_buffer;
    let meta_buffer = buffers.meta_buffer.as_ref().ok_or(())?;
    let geos_buffer = buffers.geos_buffer.as_ref().ok_or(())?;
    let material_buffer = shading_resources.materials.as_ref().ok_or(())?;
    let packed_segments = resources.froxel_buffer.get(entity).ok_or(())?;
    let output_texture = resources.output_texture.as_ref().ok_or(())?;
    let output_depth = resources.output_depth.as_ref().ok_or(())?;
    let froxel_config_buffer = resources.froxel_config_buffer.get(entity).ok_or(())?;
    let shading_buffer = shading_resources.output_texture.as_ref().ok_or(())?;

    Ok((
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
                    binding: layouts::rasterizer::GEO_BUFFER,
                    resource: geos_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::MATERIAL_BUFFER,
                    resource: material_buffer.as_entire_binding(),
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
                    binding: layouts::rasterizer::OUTPUT_DEPTH,
                    resource: BindingResource::TextureView(output_depth),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::VIEW_UNIFORM,
                    resource: view_buffer.clone(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::FROXEL_CONFIG,
                    resource: froxel_config_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::LIGHT_UNIFORM,
                    resource: light_buffer.clone(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::SHADING_BUFFER,
                    resource: BindingResource::TextureView(shading_buffer),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O,
                    resource: BindingResource::Sampler(&dom_sampler.0),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O_VIEW,
                    resource: BindingResource::TextureView(&dom_texture.0),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D,
                    resource: BindingResource::Sampler(&dom_sampler.1),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D_VIEW,
                    resource: BindingResource::TextureView(&dom_texture.1),
                },
            ],
        ),
        vec![view_offsets.offset, view_light_uniform_offset.offset],
    ))
}

// Create output texture for the rasterizer
pub fn recreate_render_target_texture(
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

pub fn recreate_render_target_depth_texture(
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
        format: TextureFormat::R32Float,
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
    uniform_offsets: &[u32],
) {
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
            uniform_offsets,
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
