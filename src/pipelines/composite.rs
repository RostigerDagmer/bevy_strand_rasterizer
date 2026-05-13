use bevy::{
    core_pipeline::{FullscreenShader, prepass::ViewPrepassTextures},
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
            BindingResource, BindingType, BlendState, BufferBindingType, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, FilterMode, FragmentState, LoadOp, MultisampleState,
            Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            SamplerDescriptor, ShaderStages, ShaderType, StoreOp, TextureFormat, TextureSampleType,
            TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
};

use super::raster::StrandRasterizerResources;

#[derive(Resource)]
pub struct CompositionPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedRenderPipelineId, // Use RenderPipeline for fullscreen quad/triangle
    pub sampler: Sampler,
}

impl FromWorld for CompositionPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let fullscreen_shader = world.resource::<FullscreenShader>();

        let layout_descriptor = BindGroupLayoutDescriptor::new(
            "composition_layout",
            &[
                // Input Scene Texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                // Input Scene Texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true }, // Assuming HDR intermediate
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Strand Rasterizer Output Texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true }, // Or Uint/Sint if format is different, but sampling rgba8unorm as float is fine
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true }, // Or Uint/Sint if format is different, but sampling rgba8unorm as float is fine
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth {},
                        view_dimension: TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
            ],
        );
        let layout = render_device
            .create_bind_group_layout(layout_descriptor.label.as_ref(), &layout_descriptor.entries);

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("composition_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let shader = world
            .resource::<AssetServer>()
            .load(crate::plugin::embedded_shader_path("strand_composite.wgsl"));

        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("composition_pipeline".into()),
            layout: vec![layout_descriptor],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    // IMPORTANT: This format must match the ViewTarget format
                    // Usually HDR first, then tonemapped. Let's assume HDR for now.
                    format: TextureFormat::bevy_default(), // Use bevy's default HDR format
                    blend: Some(BlendState::ALPHA_BLENDING), // Use alpha blending
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(), // Triangle list covering screen
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        CompositionPipeline {
            layout,
            pipeline,
            sampler,
        }
    }
}
