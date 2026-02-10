use bevy::{
    pbr::{ShadowSamplers, ViewLightsUniformOffset},
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, BufferBindingType, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, Extent3d, FilterMode, PipelineCache, PushConstantRange,
            Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType,
            StorageTextureAccess, Texture, TextureAspect, TextureDescriptor, TextureDimension,
            TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
            TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniform, ViewUniformOffset},
    },
    shader::ShaderDefVal,
};
use std::collections::HashMap;

use crate::{
    components::FroxelConfig, pipelines::layouts, plugin::MAX_TEXTURE_EXTENT,
    shader_types::PushConstants,
};

use super::{
    binning::StrandBinningBuffers, raster::StrandRasterizerResources,
    shading::StrandShadingResources,
};

const NUM_DOM_SLICES: u32 = 12;

#[derive(Resource, Default)]
pub struct StrandShadowResources {
    pub dom_targets: HashMap<Entity, (TextureView, TextureView)>,
    pub dom_samplers: HashMap<Entity, (Sampler, Sampler)>,
}

#[derive(Resource)]
pub struct StrandShadowPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shadow_pipeline: CachedComputePipelineId,
    // pub punch_through_pipeline: CachedRenderPipelineId,
}

pub const DOM_FORMAT: TextureFormat = TextureFormat::R32Float;

impl StrandShadowPipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        // We shade in strand space so we only need the vertex, index and meta buffers in terms of geometry.
        // We also need the View and light buffers and an output buffer containing the shading data along line segments.
        device.create_bind_group_layout(
            "strand_shading_bind_group_layout",
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
                // Tile Offsets Buffer
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
                // Tile Counts Buffer
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
                // Froxel Tile Buffer
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
                // View Uniform Buffer
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::VIEW_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                // Froxel Config Buffer
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
                // Cluster Indices
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::CLUSTER_INDICES,
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
                    binding: layouts::rasterizer::CLUSTER_OFFSETS_AND_COUNTS,
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
                    binding: layouts::rasterizer::CLUSTERABLE_OBJECTS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Point Light Depth Texture
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::POINT_LIGHT_DEPTH_TEXTURE_SAMPLER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Directional Light Depth Texture
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::DIRECTIONAL_LIGHT_DEPTH_TEXTURE_SAMPLER,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::POINT_LIGHT_DEPTH_TEXTURE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: None,
                },
                // Directional Light Depth Texture
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::DIRECTIONAL_LIGHT_DEPTH_TEXTURE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output texture (write-only storage texture)
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: DOM_FORMAT,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: DOM_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // NOTE: Additional textures here if necessary for more accurate blending during rasterization
            ],
        )
    }
}

impl FromWorld for StrandShadowPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);
        // let punch_bind_group_layout = Self::create_punch_bind_group_layout(device);

        let shader_loader = world.resource::<AssetServer>();
        let rasterize_shader = shader_loader.load("shaders/strand_rasterizer.wgsl");
        // let punch_shader = shader_loader.load("shaders/shadow_punch.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();
        let cdefs = [
            vec![ShaderDefVal::UInt(
                "MAX_TEXTURE_EXTENT".into(),
                MAX_TEXTURE_EXTENT,
            )],
            layouts::rasterizer::shader_defs(),
        ]
        .concat();

        let shadow_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_shadow_rasterize_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: rasterize_shader,
            shader_defs: [
                cdefs.as_slice(),
                &[
                    "SHADOWS".into(),
                    ShaderDefVal::UInt("NUM_DOM_SLICES".into(), NUM_DOM_SLICES),
                ],
            ]
            .concat(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: Some("rasterize_strands".into()),
            zero_initialize_workgroup_memory: false,
        });

        // let punch_through_pipeline =
        //     pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        //         label: Some("shadow_punch_through_pipeline".into()),
        //         layout: vec![punch_bind_group_layout.clone()],
        //         push_constant_ranges: vec![],
        //         vertex: (),
        //         primitive: PrimitiveState {
        //             topology: TriangleList,
        //             cull_mode: None, // or match Bevy’s shadow pass
        //             ..Default::default()
        //         },
        //         depth_stencil: Some(DepthStencilState {
        //             format: SHADOW_FORMAT, // must match Bevy’s (e.g., Depth32Float)
        //             depth_write_enabled: true,
        //             depth_compare: CompareFunction::Less, // the trick: only replace if DOM is nearer
        //             stencil: Default::default(),
        //             bias: DepthBiasState {
        //                 constant: SAME_AS_SHADOW_PASS, // copy Bevy’s values
        //                 slope_scale: SAME_AS_SHADOW_PASS,
        //                 clamp: 0,
        //             },
        //         }),
        //         multisample: MultisampleState::default(),
        //         fragment: Some(FragmentState {
        //             targets: vec![], // no color
        //             shader: punch_shader,
        //             entry_point: "fs".into(),
        //             shader_defs: layouts::shadows::shader_defs(), // no color targets
        //         }),
        //         zero_initialize_workgroup_memory: false,
        //     });

        debug!(
            "Created strand raster compute pipelines: rasterize={:?}",
            shadow_pipeline
        );

        StrandShadowPipeline {
            bind_group_layout,
            shadow_pipeline,
        }
    }
}

pub fn create_strand_shadow_bind_group(
    entity: &Entity,
    device: &RenderDevice,
    pipeline: &StrandShadowPipeline,
    shadow_resources: &StrandShadowResources,
    shading_resources: &StrandShadingResources,
    raster_resources: &StrandRasterizerResources,
    buffers: &StrandBinningBuffers,
    view_buffer: &BindingResource,
    light_buffer: &BindingResource,
    view_uniform_offset: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
    cluster_indices: &BindingResource,
    cluster_offsets_and_counts: &BindingResource,
    clusterable_objects: &BindingResource,
    shadows: &ShadowSamplers,
    shadow_textures: (&TextureView, &TextureView),
) -> Result<(BindGroup, Vec<u32>), ()> {
    let artifacts = buffers.artifacts.get(entity).ok_or(())?;
    let layout = &pipeline.bind_group_layout;

    let tile_offsets_buffer = &artifacts.tile_offsets_buffer;
    let tile_counts_buffer = &artifacts.tile_counts_buffer;
    let vertex_buffer = buffers.vertex_buffer.as_ref().ok_or(())?;
    let index_buffer = buffers.index_buffer.as_ref().ok_or(())?;
    let meta_buffer = buffers.meta_buffer.as_ref().ok_or(())?;
    let material_buffer = shading_resources.materials.as_ref().ok_or(())?;
    let output_texture = shadow_resources.dom_targets.get(entity).ok_or(())?;
    let packed_segments = raster_resources.froxel_buffer.get(entity).ok_or(())?;
    let froxel_config_buffer = raster_resources
        .froxel_config_buffer
        .get(entity)
        .ok_or(())?;
    let geos_buffer = buffers.geos_buffer.as_ref().ok_or(())?;

    Ok((
        device.create_bind_group(
            Some(&*format!("strand_shadows_{:?}_bind_group", entity)),
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
                    binding: layouts::rasterizer::CLUSTER_INDICES,
                    resource: cluster_indices.clone(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::CLUSTER_OFFSETS_AND_COUNTS,
                    resource: cluster_offsets_and_counts.clone(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::CLUSTERABLE_OBJECTS,
                    resource: clusterable_objects.clone(),
                },
                // Bind the shadow map texture
                BindGroupEntry {
                    binding: layouts::rasterizer::POINT_LIGHT_DEPTH_TEXTURE_SAMPLER,
                    resource: BindingResource::Sampler(&shadows.point_light_linear_sampler),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::DIRECTIONAL_LIGHT_DEPTH_TEXTURE_SAMPLER,
                    resource: BindingResource::Sampler(&shadows.directional_light_linear_sampler),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::POINT_LIGHT_DEPTH_TEXTURE,
                    resource: BindingResource::TextureView(shadow_textures.0),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::DIRECTIONAL_LIGHT_DEPTH_TEXTURE,
                    resource: BindingResource::TextureView(shadow_textures.1),
                },
                // Render target array
                BindGroupEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O,
                    resource: BindingResource::TextureView(&output_texture.0),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D,
                    resource: BindingResource::TextureView(&output_texture.1),
                },
            ],
        ),
        vec![view_uniform_offset.offset, view_light_uniform_offset.offset],
    ))
}

pub fn create_strand_shadow_textures(
    device: &RenderDevice,
    width: u32,
    height: u32,
    slices: u32,
) -> (
    (Texture, TextureView, Sampler),
    (Texture, TextureView, Sampler),
) {
    let depth_texture = device.create_texture(&TextureDescriptor {
        label: Some("strand_shadow_depth_texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1, // we store two slices per layer (rg) and (ba)
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: DOM_FORMAT, // TODO: compact
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST,
        view_formats: &[DOM_FORMAT],
    });

    let depth_texture_view = depth_texture.create_view(&TextureViewDescriptor {
        label: Some("strand_shadow_depth_view"),
        format: Some(DOM_FORMAT),
        dimension: Some(TextureViewDimension::D2),
        aspect: TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
        usage: Some(
            TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
        ),
    });
    let depth_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("strand_shadow_depth_sampler"),
        address_mode_u: bevy::render::render_resource::AddressMode::ClampToEdge,
        address_mode_v: bevy::render::render_resource::AddressMode::ClampToEdge,
        address_mode_w: bevy::render::render_resource::AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Nearest,
        ..default()
    });

    let opacity_texture = device.create_texture(&TextureDescriptor {
        label: Some("strand_shadow_opacity_layers"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: NUM_DOM_SLICES,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: DOM_FORMAT,
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST,
        view_formats: &[DOM_FORMAT],
    });

    let opacity_texture_view = opacity_texture.create_view(&TextureViewDescriptor {
        label: Some("strand_shadow_opacity_view"),
        format: Some(DOM_FORMAT),
        dimension: Some(TextureViewDimension::D3),
        aspect: TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
        usage: Some(
            TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
        ),
    });

    let opacity_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("strand_shadow_layers_sampler"),
        address_mode_u: bevy::render::render_resource::AddressMode::ClampToEdge,
        address_mode_v: bevy::render::render_resource::AddressMode::ClampToEdge,
        address_mode_w: bevy::render::render_resource::AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Nearest,
        ..default()
    });

    (
        (opacity_texture, opacity_texture_view, opacity_sampler),
        (depth_texture, depth_texture_view, depth_sampler),
    )
}

pub fn run_shadow_pass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandShadowPipeline,
    froxel_config: &FroxelConfig,
    resources: &StrandRasterizerResources,
    bind_group: &BindGroup,
    offsets: &[u32],
) {
    let encoder = render_context.command_encoder(); // Get CommandEncoder
    // --- Rasterize ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Rasterize Shadow"),
            ..default()
        });
        let Some(raster_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.shadow_pipeline)
        else {
            warn!("Shadow pipeline not found");
            return;
        };
        pass.set_pipeline(raster_pipeline);
        pass.set_bind_group(
            0, bind_group, // Assume correctly populated bind group
            offsets,
        );
        // Set push constants if needed
        let pushconstants = PushConstants {
            num_elements: resources.strand_count.unwrap_or(0),
            workgroup_offset: 0, // TODO: maybe its time to make this its own field
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
}
