use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipeline,
            ComputePipelineDescriptor, Extent3d, PipelineCache, PushConstantRange, ShaderStages,
            StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewUniformOffset,
    },
    shader::ShaderDefVal,
};
use bevy_gpu_paging_allocator::BindGroupBuilder;

use std::collections::HashMap;

use crate::{
    allocator::GpuPagingAllocator, components::FroxelConfig, pipelines::layouts,
    pipelines::task_contract::BINNING_POOL_CHUNK_SIZE, plugin::MAX_TEXTURE_EXTENT,
    resources::ComputeInvocationDims, shader_types::PushConstants,
};

use super::prepass::StrandPrepassResources;

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
    pub frustum_ids: HashMap<Entity, u32>,
}

#[derive(Resource)]
pub struct StrandRasterizerPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub rasterize_pipeline: Option<CachedComputePipelineId>,
    pub allocator_epoch: u64,
    pub workgroup_size: u32,
}

impl StrandRasterizerPipeline {
    pub fn bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
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
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::FINE_SEG_REFS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::STRAND_INSTANCES,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::COARSE_TILE_WORK_COUNTS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::COARSE_TILE_WORK_OFFSETS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )
    }

    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        let descriptor = Self::bind_group_layout_descriptor();
        device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries)
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
            workgroup_size: 0,
        }
    }
}

pub fn update_strand_raster_pipeline(
    mut pipeline: ResMut<StrandRasterizerPipeline>,
    allocator: Res<GpuPagingAllocator>,
    invocation_dims: Res<ComputeInvocationDims>,
    shader_loader: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let current_state = allocator.bindgroups_epoch;
    let workgroup_size = invocation_dims.threads_per_workgroup.max(1);
    if pipeline.rasterize_pipeline.is_some()
        && pipeline.allocator_epoch == current_state
        && pipeline.workgroup_size == workgroup_size
    {
        return;
    }

    if allocator.buffer_bind_group_layout.is_none()
        || allocator.pagetable_bind_group_layout.is_none()
    {
        return;
    }
    let allocator_layout_entries = allocator.layout_entries();
    let buffer_layout = BindGroupLayoutDescriptor::new(
        "gpu_paging_allocator_buffer_layout",
        &allocator_layout_entries.pools,
    );
    let table_layout = BindGroupLayoutDescriptor::new(
        "gpu_paging_allocator_table_layout",
        &allocator_layout_entries.page_tables,
    );

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
            ShaderDefVal::UInt(
                "COARSE_FINE_TILE_EXTENT".into(),
                crate::plugin::COARSE_FINE_TILE_EXTENT,
            ),
            ShaderDefVal::UInt("WORKGROUP_SIZE".into(), workgroup_size),
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
    let raster_layout = StrandRasterizerPipeline::bind_group_layout_descriptor();
    let mut layout = vec![raster_layout.clone(); (max_group + 1) as usize];
    layout[allocator.buffer_group_idx as usize] = buffer_layout;
    layout[allocator.table_group_idx as usize] = table_layout;
    layout[layouts::rasterizer::RASTER_GROUP as usize] = raster_layout;

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
    pipeline.workgroup_size = workgroup_size;
    info!("Updated rasterizer pipeline.")
}

// Create the bind group for the strand rasterizer
pub fn create_strand_raster_bind_group(
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
    let frustum_table = prepass_resources.frustum_table.as_ref().ok_or(())?;
    let froxel_bucket_heads = prepass_resources.froxel_bucket_heads.as_ref().ok_or(())?;
    let chunk_pool = prepass_resources.chunk_pool.as_ref().ok_or(())?;
    let raster_work_queue = prepass_resources.raster_work_queue.as_ref().ok_or(())?;
    let fine_seg_refs = prepass_resources.fine_seg_refs.as_ref().ok_or(())?;
    let strand_instances = prepass_resources.strand_instances.as_ref().ok_or(())?;
    let coarse_tile_work_counts = prepass_resources
        .coarse_tile_work_counts
        .as_ref()
        .ok_or(())?;
    let coarse_tile_work_offsets = prepass_resources
        .coarse_tile_work_offsets
        .as_ref()
        .ok_or(())?;
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
                BindGroupEntry {
                    binding: layouts::rasterizer::FINE_SEG_REFS,
                    resource: fine_seg_refs.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::STRAND_INSTANCES,
                    resource: strand_instances.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::COARSE_TILE_WORK_COUNTS,
                    resource: coarse_tile_work_counts.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::COARSE_TILE_WORK_OFFSETS,
                    resource: coarse_tile_work_offsets.as_entire_binding(),
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
    allocator: &GpuPagingAllocator,
    _froxel_config: &FroxelConfig,
    frustum_id: u32,
    resources: &StrandRasterizerResources,
    bind_group: &BindGroup,
    uniform_offsets: &[u32],
    dispatch_size: (u32, u32, u32),
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
            workgroup_offset: 0,
            scan_load_base: frustum_id,
            scan_save_base: 0,
            ..Default::default()
        };
        pass.set_push_constants(0, bytemuck::bytes_of(&pushconstants));

        pass.dispatch_workgroups(dispatch_size.0, dispatch_size.1, dispatch_size.2);
    }

    // --- Rasterization complete ---
    // output_texture is ready for composition
}
