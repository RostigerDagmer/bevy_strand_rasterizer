use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, Buffer, BufferBindingType, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipeline, ComputePipelineDescriptor, Extent3d, PipelineCache, PushConstantRange,
            ShaderStages, StorageTextureAccess, Texture, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewUniformOffset,
    },
    shader::ShaderDefVal,
};

use std::collections::HashMap;

use crate::{
    allocator::GpuPagingAllocator, components::FroxelConfig, pipelines::layouts,
    pipelines::task_contract::BINNING_POOL_CHUNK_SIZE, plugin::MAX_TEXTURE_EXTENT,
    shader_types::PushConstants,
};

use super::{prepass::StrandPrepassResources, shading::StrandShadingResources};

#[derive(Resource, Default)]
pub struct StrandRasterizerResources {
    pub pipeline: Option<ComputePipeline>,
    pub froxel_buffer: HashMap<Entity, Buffer>,
    pub froxel_config_buffer: HashMap<Entity, Buffer>,
    pub output_texture_resource: Option<Texture>,
    pub output_depth_resource: Option<Texture>,
    pub output_texture: Option<TextureView>,
    pub output_depth: Option<TextureView>,
    pub strand_count: Option<u32>,
    pub frustrum_config: HashMap<Entity, FroxelConfig>,
}

#[derive(Resource)]
pub struct StrandRasterizerPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub rasterize_pipeline: Option<CachedComputePipelineId>,
    pub allocator_epoch: u64,
}

impl StrandRasterizerPipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        device.create_bind_group_layout(
            "strand_rasterizer_bind_group_layout",
            &[
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
                // Frustum descriptor table
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::FRUSTUM_TABLE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sparse froxel bucket heads
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::FROXEL_BUCKET_HEADS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sparse chunk pool payload
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::CHUNK_POOL,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Raster work queue items
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::RASTER_WORK_QUEUE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Disabled for correctness-only raster pass:
                // BindGroupLayoutEntry {
                //     binding: layouts::rasterizer::SHADING_BUFFER,
                //     visibility: ShaderStages::COMPUTE,
                //     ty: BindingType::StorageTexture {
                //         access: StorageTextureAccess::ReadOnly,
                //         format: TextureFormat::Rgba8Unorm,
                //         view_dimension: TextureViewDimension::D2,
                //     },
                //     count: None,
                // },
                // BindGroupLayoutEntry {
                //     binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O,
                //     visibility: ShaderStages::COMPUTE,
                //     ty: BindingType::Sampler(SamplerBindingType::Filtering),
                //     count: None,
                // },
                // BindGroupLayoutEntry {
                //     binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O_VIEW,
                //     visibility: ShaderStages::COMPUTE,
                //     ty: BindingType::Texture {
                //         sample_type: TextureSampleType::Float { filterable: true },
                //         view_dimension: TextureViewDimension::D3,
                //         multisampled: false,
                //     },
                //     count: None,
                // },
                // BindGroupLayoutEntry {
                //     binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D,
                //     visibility: ShaderStages::COMPUTE,
                //     ty: BindingType::Sampler(SamplerBindingType::Filtering),
                //     count: None,
                // },
                // BindGroupLayoutEntry {
                //     binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D_VIEW,
                //     visibility: ShaderStages::COMPUTE,
                //     ty: BindingType::Texture {
                //         sample_type: TextureSampleType::Float { filterable: true },
                //         view_dimension: TextureViewDimension::D2,
                //         multisampled: false,
                //     },
                //     count: None,
                // },
            ],
        )
    }
}

impl FromWorld for StrandRasterizerPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        StrandRasterizerPipeline {
            bind_group_layout,
            rasterize_pipeline: None,
            allocator_epoch: u64::MAX,
        }
    }
}

pub fn update_strand_raster_pipeline(
    mut pipeline: ResMut<StrandRasterizerPipeline>,
    allocator: Res<GpuPagingAllocator>,
    shader_loader: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let current_state = allocator.bindgroups_epoch;
    if pipeline.rasterize_pipeline.is_some() && pipeline.allocator_epoch == current_state {
        return;
    }

    let (Some(buffer_layout), Some(table_layout)) = (
        allocator.buffer_bind_group_layout.clone(),
        allocator.pagetable_bind_group_layout.clone(),
    ) else {
        return;
    };

    let mut cdefs = [
        vec![
            ShaderDefVal::UInt("MAX_TEXTURE_EXTENT".into(), MAX_TEXTURE_EXTENT),
            ShaderDefVal::UInt(
                "SIZEOF_METADATA".into(),
                std::mem::size_of::<crate::shader_types::StrandMeta>() as u32,
            ),
            ShaderDefVal::UInt(
                "SIZEOF_MATERIAL".into(),
                std::mem::size_of::<crate::components::StrandMaterial>() as u32,
            ),
            ShaderDefVal::UInt(
                "SIZEOF_GEO".into(),
                std::mem::size_of::<crate::shader_types::StrandGeo>() as u32,
            ),
            ShaderDefVal::UInt("POOL_CHUNK_SIZE".into(), BINNING_POOL_CHUNK_SIZE),
        ],
        layouts::rasterizer::shader_defs(),
        allocator.shader_defs(),
    ]
    .concat();
    cdefs.push("LINEAR".into());

    let max_group = allocator
        .buffer_group_idx
        .max(allocator.table_group_idx)
        .max(layouts::rasterizer::RASTER_GROUP);
    let mut layout = vec![pipeline.bind_group_layout.clone(); (max_group + 1) as usize];
    layout[allocator.buffer_group_idx as usize] = buffer_layout;
    layout[allocator.table_group_idx as usize] = table_layout;
    layout[layouts::rasterizer::RASTER_GROUP as usize] = pipeline.bind_group_layout.clone();

    let rasterize_shader = shader_loader.load("shaders/strand_rasterizer.wgsl");
    let rasterize_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("strand_rasterize_pipeline".into()),
        layout,
        shader: rasterize_shader,
        shader_defs: cdefs,
        push_constant_ranges: vec![PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<PushConstants>() as u32,
        }],
        entry_point: Some("rasterize_strands".into()),
        zero_initialize_workgroup_memory: false,
    });

    pipeline.rasterize_pipeline = Some(rasterize_pipeline);
    pipeline.allocator_epoch = current_state;
    info!("Updated rasterizer pipeline.")
}

// Create the bind group for the strand rasterizer
pub fn create_strand_raster_bind_group(
    entity: &Entity,
    device: &RenderDevice,
    pipeline: &StrandRasterizerPipeline,
    resources: &StrandRasterizerResources,
    prepass_resources: &StrandPrepassResources,
    view_buffer: &BindingResource,
    light_buffer: &BindingResource,
    view_offsets: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
) -> Result<(BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;
    let output_texture = resources.output_texture.as_ref().ok_or(())?;
    let output_depth = resources.output_depth.as_ref().ok_or(())?;
    let froxel_config_buffer = resources.froxel_config_buffer.get(entity).ok_or(())?;
    let frustum_table = prepass_resources.frustum_table.as_ref().ok_or(())?;
    let froxel_bucket_heads = prepass_resources.froxel_bucket_heads.as_ref().ok_or(())?;
    let chunk_pool = prepass_resources.chunk_pool.as_ref().ok_or(())?;
    let raster_work_queue = prepass_resources.raster_work_queue.as_ref().ok_or(())?;

    Ok((
        device.create_bind_group(
            Some("strand_rasterizer_bind_group"),
            layout,
            &[
                BindGroupEntry {
                    binding: layouts::rasterizer::OUTPUT_TEXTURE,
                    resource: BindingResource::TextureView(output_texture),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::OUTPUT_DEPTH,
                    resource: BindingResource::TextureView(output_depth),
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
                    binding: layouts::rasterizer::LIGHT_UNIFORM,
                    resource: light_buffer.clone(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::FRUSTUM_TABLE,
                    resource: frustum_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::FROXEL_BUCKET_HEADS,
                    resource: froxel_bucket_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::CHUNK_POOL,
                    resource: chunk_pool.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::RASTER_WORK_QUEUE,
                    resource: raster_work_queue.as_entire_binding(),
                },
                // Disabled for correctness-only raster pass:
                // BindGroupEntry {
                //     binding: layouts::rasterizer::SHADING_BUFFER,
                //     resource: BindingResource::TextureView(shading_buffer),
                // },
                // BindGroupEntry {
                //     binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O,
                //     resource: BindingResource::Sampler(&dom_sampler.0),
                // },
                // BindGroupEntry {
                //     binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_O_VIEW,
                //     resource: BindingResource::TextureView(&dom_texture.0),
                // },
                // BindGroupEntry {
                //     binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D,
                //     resource: BindingResource::Sampler(&dom_sampler.1),
                // },
                // BindGroupEntry {
                //     binding: layouts::rasterizer::DEEP_OPACITY_TEXTURE_D_VIEW,
                //     resource: BindingResource::TextureView(&dom_texture.1),
                // },
            ],
        ),
        // Dynamic offsets must follow bind-group layout declaration order.
        // Raster layout declares LIGHT_UNIFORM before VIEW_UNIFORM.
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
    allocator: &GpuPagingAllocator,
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
        let Some(raster_pipeline_id) = pipeline.rasterize_pipeline else {
            warn!("Raster pipeline id not ready yet");
            return;
        };
        let Some(raster_pipeline) = pipeline_cache.get_compute_pipeline(raster_pipeline_id) else {
            warn!("Raster pipeline not found");
            return;
        };
        let Some(allocator_buffer_bind_group) = allocator.buffer_bind_group.as_ref() else {
            warn!("allocator buffer bind group is not ready yet.");
            return;
        };
        let Some(allocator_pagetable_bind_group) = allocator.pagetable_bind_group.as_ref() else {
            warn!("allocator pagetable bind group is not ready yet.");
            return;
        };
        pass.set_pipeline(raster_pipeline);
        pass.set_bind_group(allocator.buffer_group_idx, allocator_buffer_bind_group, &[]);
        pass.set_bind_group(
            allocator.table_group_idx,
            allocator_pagetable_bind_group,
            &[],
        );
        pass.set_bind_group(
            layouts::rasterizer::RASTER_GROUP,
            bind_group, // Assume correctly populated bind group
            uniform_offsets,
        );
        // Set push constants if needed
        let pushconstants = PushConstants {
            num_elements: resources.strand_count.unwrap_or(0),
            workgroup_offset: shading_resources.max_segments_in_strand.unwrap_or(0), // TODO: maybe its time to make this its own field
            scan_load_base: 0,
            scan_save_base: 0,
            ..Default::default()
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
