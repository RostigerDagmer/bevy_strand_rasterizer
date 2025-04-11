use bevy::{
    pbr::ViewLightsUniformOffset, prelude::*, render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer, BufferBindingType, BufferSize, CachedComputePipelineId, CachedRenderPipelineId, ColorTargetState, ColorWrites, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Extent3d, FilterMode, FragmentState, MultisampleState, PipelineCache, PrimitiveState, PushConstantRange, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderDefVal, ShaderStages, ShaderType, StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniform, ViewUniformOffset},
    }
};

use crate::{pipelines::layouts, plugin::MAX_TEXTURE_EXTENT, shader_types::PushConstants};

use super::{binning::StrandBinningBuffers, raster::StrandRasterizerResources};

const MAX_SHADING_SUBSAMPLING_FACTOR: u32 = 4; // for shading

#[derive(Resource, Default)]
pub struct StrandShadingResources {
    pub output_texture: Option<TextureView>,
    pub strand_count: Option<u32>,
    pub max_segments_in_strand: Option<u32>,
}


#[derive(Resource)]
pub struct StrandShadingPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shading_pipeline: CachedComputePipelineId,
}

impl StrandShadingPipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        // We shade in strand space so we only need the vertex, index and meta buffers in terms of geometry.
        // We also need the View and light buffers and an output buffer containing the shading data along line segments.
        device.create_bind_group_layout(
            "strand_shading_bind_group_layout",
            &[
                // Vertex buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: layouts::shading::VERTEX_BUFFER,
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
                    binding: layouts::shading::INDEX_BUFFER,
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
                    binding: layouts::shading::META_BUFFER,
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
                    binding: layouts::shading::VIEW_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                // Light Uniform Buffer
                BindGroupLayoutEntry {
                    binding: layouts::shading::LIGHT_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output texture (write-only storage texture)
                BindGroupLayoutEntry {
                    binding: layouts::shading::OUTPUT_TEXTURE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // NOTE: Additional textures here if necessary for more accurate blending during rasterization
            ],
        )
    }
}

impl FromWorld for StrandShadingPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let shader_loader = world.resource::<AssetServer>();
        let shading_shader = shader_loader.load("shaders/strand_shading.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();
        let cdefs = [vec![ShaderDefVal::UInt(
            "MAX_TEXTURE_EXTENT".into(),
            crate::plugin::MAX_TEXTURE_EXTENT,
        )], layouts::shading::shader_defs()].concat();

        let shading_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_shading_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: shading_shader,
            shader_defs: cdefs,
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "shade_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        debug!(
            "Created strand shading compute pipelines: shading={:?}",
            shading_pipeline
        );

        StrandShadingPipeline {
            bind_group_layout,
            shading_pipeline,
        }
    }
}


pub fn create_strand_shading_bind_group(
    device: &RenderDevice,
    pipeline: &StrandShadingPipeline,
    shading_resources: &StrandShadingResources,
    buffers: &StrandBinningBuffers,
    view_buffer: BindingResource,
    light_buffer: BindingResource,
    view_uniform_offset: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
) -> Result<(BindGroup, Vec<u32>), ()> {

    let layout = &pipeline.bind_group_layout;
    let vertex_buffer = buffers.vertex_buffer.as_ref().ok_or(())?;
    let index_buffer = buffers.index_buffer.as_ref().ok_or(())?;
    let meta_buffer = buffers.meta_buffer.as_ref().ok_or(())?;
    let output_texture = shading_resources.output_texture.as_ref().ok_or(())?;
    Ok((
        device.create_bind_group(
            Some("strand_shading_bind_group"),
            layout,
            &[
                BindGroupEntry {
                    binding: layouts::shading::VERTEX_BUFFER,
                    resource: vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::shading::INDEX_BUFFER,
                    resource: index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::shading::META_BUFFER,
                    resource: meta_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::shading::VIEW_UNIFORM,
                    resource: view_buffer.clone(),
                },
                BindGroupEntry {
                    binding: layouts::shading::LIGHT_UNIFORM,
                    resource: light_buffer.clone(),
                },
                BindGroupEntry {
                    binding: layouts::shading::OUTPUT_TEXTURE,
                    resource: BindingResource::TextureView(output_texture),
                },
            ],
        ),
        vec![view_uniform_offset.offset, view_light_uniform_offset.offset],
    ))
}

pub fn create_shading_target_texture(
    device: &RenderDevice,
    strand_count: u32,
    max_strand_segment_count: u32,
) -> (Texture, TextureView) {
    // calculate how many columns we need
    let cols = strand_count.div_ceil(MAX_TEXTURE_EXTENT);

    let texture = device.create_texture(&TextureDescriptor {
        label: Some("strand_rasterizer_output"),
        size: Extent3d {
            width: max_strand_segment_count * MAX_SHADING_SUBSAMPLING_FACTOR * cols,
            height: MAX_TEXTURE_EXTENT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[TextureFormat::Rgba8Unorm],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

pub fn run_shading_pass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandShadingPipeline,
    resources: &StrandShadingResources,
    bind_group: &BindGroup,
    offsets: &[u32],
) {
    let Some(strand_count) = resources.strand_count else {
        warn!("Strand count not set.");
        return;
    };
    let Some(shading_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.shading_pipeline)
    else {
        warn!("Shading pipeline not found");
        return;
    };

    let encoder = render_context.command_encoder(); // Get CommandEncoder
    // --- Shading ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Shading"),
            ..default()
        });
        pass.set_pipeline(shading_pipeline);
        pass.set_bind_group(0, bind_group, offsets);
        // Set push constants if needed
        let pushconstants = PushConstants {
            num_elements: resources.strand_count.unwrap_or(0),
            workgroup_offset: resources.max_segments_in_strand.unwrap_or(0),
            scan_load_base: 0,
            scan_save_base: 0,
        };
        pass.set_push_constants(0, bytemuck::bytes_of(&pushconstants));

        // Dispatch based on number of strands or segments
        // TODO: this limits us to e.g. 65535 strands. With most workgroups staying underutilized.
        pass.dispatch_workgroups(strand_count, 1, 1);
    }

    // --- Shading complete ---
    // output_texture is ready for composition
}