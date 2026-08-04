use bevy::{
    core_pipeline::FullscreenShader,
    prelude::*,
    render::render_resource::{
        BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState,
        BufferBindingType, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
        MultisampleState, PipelineCache, PrimitiveState, RenderPipelineDescriptor, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, TextureFormat,
        TextureSampleType, TextureViewDimension,
    },
};

use crate::pipelines::layouts;
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct TileDebugParams {
    pub inv_norm: f32,
    pub gain: f32,
    pub alpha: f32,
    pub frustum_id: u32,
}

#[derive(Resource)]
pub struct TileDebugPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedRenderPipelineId,
    pub sampler: Sampler,
}

impl FromWorld for TileDebugPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<bevy::render::renderer::RenderDevice>();
        let fullscreen_shader = world.resource::<FullscreenShader>();

        let layout_descriptor = BindGroupLayoutDescriptor::new(
            "tile_debug_layout",
            &[
                BindGroupLayoutEntry {
                    binding: layouts::tile_debug::FRUSTUM_TABLE,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::tile_debug::FROXEL_BUCKET_HEADS,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::tile_debug::CHUNK_POOL,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::tile_debug::COARSE_COUNT_PAGE_TABLE,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::tile_debug::COARSE_COUNT_PAGES,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::tile_debug::PARAMS,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::tile_debug::INPUT_TEXTURE,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::tile_debug::INPUT_SAMPLER,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        );
        let layout = render_device
            .create_bind_group_layout(layout_descriptor.label.as_ref(), &layout_descriptor.entries);
        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let shader = world
            .resource::<AssetServer>()
            .load(crate::plugin::embedded_shader_path(
                "strand_tile_debug.wgsl",
            ));

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("tile_debug_pipeline".into()),
            layout: vec![layout_descriptor],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader,
                shader_defs: layouts::tile_debug::shader_defs(),
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        });

        Self {
            layout,
            pipeline,
            sampler,
        }
    }
}
