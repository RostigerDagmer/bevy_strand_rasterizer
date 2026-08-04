use bevy::{
    core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT,
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType,
            CachedComputePipelineId, CachedRenderPipelineId, CompareFunction,
            ComputePassDescriptor, ComputePipelineDescriptor, DepthBiasState, DepthStencilState,
            FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState,
            RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            ShaderStages, ShaderType, StencilState, StoreOp, TextureSampleType, TextureView,
            TextureViewDimension, VertexState,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniform, ViewUniformOffset},
    },
    shader::ShaderDefVal,
};
use bevy_gpu_paging_allocator::{BindGroupBuilder, GpuPagingAllocator};
use bevy_vsms::allocator::{VirtualSurfacePool, VirtualSurfaceRuntime};
use std::collections::HashMap;

use crate::{
    pipelines::{layouts, prepass::StrandPrepassResources, task_contract::BINNING_POOL_CHUNK_SIZE},
    plugin::MAX_TEXTURE_EXTENT,
    resources::ComputeInvocationDims,
    shader_types::PushConstants,
};

use super::raster::StrandRasterizerResources;

pub const NUM_DOM_SLICES: u32 = 12;

#[derive(Resource, Default)]
pub struct StrandShadowResources {
    pub dom_array_targets: Option<(TextureView, TextureView)>,
    pub light_layer_by_entity: HashMap<Entity, u32>,
    pub light_layer_by_frustum: HashMap<u32, u32>,
    pub light_entity_by_layer: Vec<Entity>,
    pub dom_vsms_proxies: HashMap<Entity, (Entity, Entity)>, // (opacity_proxy, depth_proxy)
    pub stamp_targets: Vec<ShadowStampTarget>,
}

#[derive(Clone, Copy, Debug)]
pub struct ShadowStampTarget {
    pub frustum_id: u32,
    pub light_entity: Entity,
    pub cascade_index: u32,
}

#[derive(Resource)]
pub struct StrandShadowPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub stamp_bind_group_layout: BindGroupLayout,
    pub vsms_table_bind_group_layout: BindGroupLayout,
    pub shadow_pipeline: Option<CachedComputePipelineId>,
    pub stampback_pipeline: Option<CachedRenderPipelineId>,
    pub allocator_epoch: u64,
    pub workgroup_size: u32,
    pub opacity_storage_count: u32,
    pub depth_storage_count: u32,
}

impl StrandShadowPipeline {
    pub fn bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "strand_shadow_bind_group_layout",
            &[
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
                // Fine tile run queue. Shadow rasterization consumes the same
                // ordered runs as camera rasterization, filtered by frustum.
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::RASTER_TILE_RUN_QUEUE,
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
                    binding: layouts::rasterizer::SHADOW_DOM_SURFACE_IDS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            ],
        )
    }

    pub fn vsms_table_bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "strand_shadow_vsms_table_bind_group_layout",
            &[
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::VSMS_VIRTUAL_META_BINDING,
                    visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::VSMS_VIRTUAL_PAGE_TABLE_BINDING,
                    visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
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

    pub fn stamp_bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "strand_shadow_stampback_bind_group_layout",
            &[
                BindGroupLayoutEntry {
                    binding: layouts::shadow_stampback::FRUSTUM_TABLE,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shadow_stampback::SHADOW_DOM_SURFACE_IDS,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shadow_stampback::PARAMS,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(ShadowStampParams::min_size()),
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

impl FromWorld for StrandShadowPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);
        let stamp_descriptor = Self::stamp_bind_group_layout_descriptor();
        let stamp_bind_group_layout = device
            .create_bind_group_layout(stamp_descriptor.label.as_ref(), &stamp_descriptor.entries);
        let vsms_table_descriptor = Self::vsms_table_bind_group_layout_descriptor();
        let vsms_table_bind_group_layout = device.create_bind_group_layout(
            vsms_table_descriptor.label.as_ref(),
            &vsms_table_descriptor.entries,
        );

        StrandShadowPipeline {
            bind_group_layout,
            stamp_bind_group_layout,
            vsms_table_bind_group_layout,
            shadow_pipeline: None,
            stampback_pipeline: None,
            allocator_epoch: u64::MAX,
            workgroup_size: 0,
            opacity_storage_count: 0,
            depth_storage_count: 0,
        }
    }
}

#[derive(Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ShadowStampParams {
    pub frustum_id: u32,
    pub target_width: u32,
    pub target_height: u32,
    pub _pad2: u32,
}

pub fn update_strand_shadow_pipeline(
    mut pipeline: ResMut<StrandShadowPipeline>,
    allocator: Res<GpuPagingAllocator>,
    invocation_dims: Res<ComputeInvocationDims>,
    vsms_runtime: Res<VirtualSurfaceRuntime>,
    shader_loader: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let current_state = allocator.bindgroups_epoch;
    let workgroup_size = invocation_dims.threads_per_workgroup.max(1);
    let Some(opacity_storage_binding) = vsms_runtime
        .pool_storage_bindings
        .get(&bevy_vsms::api::VirtualSurfaceKind::Opacity3D)
    else {
        return;
    };
    let Some(depth_storage_binding) = vsms_runtime
        .pool_storage_bindings
        .get(&bevy_vsms::api::VirtualSurfaceKind::Depth2DArray)
    else {
        return;
    };
    let Some(depth_sample_binding) = vsms_runtime
        .pool_bindings
        .get(&bevy_vsms::api::VirtualSurfaceKind::Depth2DArray)
    else {
        return;
    };
    let opacity_storage_count = opacity_storage_binding.texture_count;
    let depth_storage_count = depth_storage_binding.texture_count;
    if pipeline.shadow_pipeline.is_some()
        && pipeline.allocator_epoch == current_state
        && pipeline.workgroup_size == workgroup_size
        && pipeline.opacity_storage_count == opacity_storage_count
        && pipeline.depth_storage_count == depth_storage_count
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
    let opacity_storage_layout = opacity_storage_binding.layout_descriptor.clone();
    let depth_storage_layout = depth_storage_binding.layout_descriptor.clone();
    let depth_sample_layout = depth_sample_binding.layout_descriptor.clone();

    let cdefs = [
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
            ShaderDefVal::UInt("NUM_DOM_SLICES".into(), NUM_DOM_SLICES),
            ShaderDefVal::UInt("WORKGROUP_SIZE".into(), workgroup_size),
            ShaderDefVal::UInt(
                "COARSE_FINE_TILE_EXTENT".into(),
                crate::plugin::COARSE_FINE_TILE_EXTENT,
            ),
            "SHADOWS".into(),
        ],
        layouts::rasterizer::shader_defs(),
        allocator.shader_defs(),
    ]
    .concat();

    let max_group = allocator
        .buffer_group_idx
        .max(allocator.table_group_idx)
        .max(layouts::rasterizer::RASTER_GROUP)
        .max(layouts::rasterizer::VSMS_OPACITY_WRITE_GROUP)
        .max(layouts::rasterizer::VSMS_DEPTH_WRITE_GROUP)
        .max(layouts::rasterizer::VSMS_OPACITY_TABLE_GROUP)
        .max(layouts::rasterizer::VSMS_DEPTH_TABLE_GROUP);
    let shadow_layout = StrandShadowPipeline::bind_group_layout_descriptor();
    let vsms_table_layout = StrandShadowPipeline::vsms_table_bind_group_layout_descriptor();
    let mut layout = vec![shadow_layout.clone(); (max_group + 1) as usize];
    layout[allocator.buffer_group_idx as usize] = buffer_layout;
    layout[allocator.table_group_idx as usize] = table_layout;
    layout[layouts::rasterizer::RASTER_GROUP as usize] = shadow_layout;
    layout[layouts::rasterizer::VSMS_OPACITY_WRITE_GROUP as usize] = opacity_storage_layout;
    layout[layouts::rasterizer::VSMS_DEPTH_WRITE_GROUP as usize] = depth_storage_layout.clone();
    layout[layouts::rasterizer::VSMS_OPACITY_TABLE_GROUP as usize] = vsms_table_layout.clone();
    layout[layouts::rasterizer::VSMS_DEPTH_TABLE_GROUP as usize] = vsms_table_layout.clone();

    let rasterize_shader = shader_loader.load(crate::plugin::embedded_shader_path(
        "strand_rasterizer.wgsl",
    ));
    let stamp_shader = shader_loader.load(crate::plugin::embedded_shader_path(
        "shadow_dom_stampback.wgsl",
    ));
    let stamp_shader_defs = [
        layouts::shadow_stampback::shader_defs(),
        vec![ShaderDefVal::UInt(
            layouts::rasterizer::VSMS_DEPTH_POOL_TEXTURE_COUNT_DEF.into(),
            depth_storage_count,
        )],
    ]
    .concat();
    pipeline.shadow_pipeline = Some(pipeline_cache.queue_compute_pipeline(
        ComputePipelineDescriptor {
            label: Some("strand_shadow_rasterize_pipeline".into()),
            layout,
            shader: rasterize_shader,
            shader_defs: cdefs,
            immediate_size: std::mem::size_of::<PushConstants>() as u32,
            entry_point: Some("rasterize_strands".into()),
            zero_initialize_workgroup_memory: false,
        },
    ));
    let stamp_layout = StrandShadowPipeline::stamp_bind_group_layout_descriptor();
    pipeline.stampback_pipeline = Some(pipeline_cache.queue_render_pipeline(
        RenderPipelineDescriptor {
            label: Some("strand_shadow_stampback_pipeline".into()),
            layout: vec![stamp_layout, depth_sample_layout, vsms_table_layout.clone()],
            vertex: VertexState {
                shader: stamp_shader.clone(),
                shader_defs: stamp_shader_defs.clone(),
                entry_point: Some("vertex".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: stamp_shader,
                shader_defs: stamp_shader_defs,
                entry_point: Some("fragment".into()),
                targets: vec![],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        },
    ));
    pipeline.allocator_epoch = current_state;
    pipeline.workgroup_size = workgroup_size;
    pipeline.opacity_storage_count = opacity_storage_count;
    pipeline.depth_storage_count = depth_storage_count;
}

pub fn create_strand_shadow_bind_group(
    device: &RenderDevice,
    pipeline: &StrandShadowPipeline,
    prepass_resources: &StrandPrepassResources,
    view_buffer: &BindingResource,
    light_buffer: &BindingResource,
    directional_light_depth_texture_view: &TextureView,
    view_uniform_offset: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
) -> Result<(BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;

    let frustum_table = prepass_resources.frustum_table.as_ref().ok_or(())?;
    let froxel_bucket_heads = prepass_resources.froxel_bucket_heads.as_ref().ok_or(())?;
    let chunk_pool = prepass_resources.chunk_pool.as_ref().ok_or(())?;
    let raster_work_queue = prepass_resources.raster_work_queue.as_ref().ok_or(())?;
    let raster_tile_run_queue = prepass_resources.raster_tile_run_queue.as_ref().ok_or(())?;
    let fine_seg_refs = prepass_resources.fine_seg_refs.as_ref().ok_or(())?;
    let strand_instances = prepass_resources.strand_instances.as_ref().ok_or(())?;
    let shadow_dom_surface_ids = prepass_resources
        .shadow_dom_surface_ids
        .as_ref()
        .ok_or(())?;

    Ok((
        device.create_bind_group(
            Some("strand_shadow_bind_group"),
            layout,
            &[
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
                    binding: layouts::rasterizer::RASTER_TILE_RUN_QUEUE,
                    resource: raster_tile_run_queue.as_entire_binding(),
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
                    binding: layouts::rasterizer::SHADOW_DOM_SURFACE_IDS,
                    resource: shadow_dom_surface_ids.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::DIRECTIONAL_LIGHT_DEPTH_TEXTURE,
                    resource: BindingResource::TextureView(directional_light_depth_texture_view),
                },
            ],
        ),
        vec![view_uniform_offset.offset, view_light_uniform_offset.offset],
    ))
}

pub fn create_vsms_table_bind_group(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    pool: &VirtualSurfacePool,
    label: &'static str,
) -> Option<BindGroup> {
    let meta = pool.virtual_page_table_meta.as_ref()?;
    let table = pool.virtual_page_table.as_ref()?;
    Some(device.create_bind_group(
        Some(label),
        layout,
        &[
            BindGroupEntry {
                binding: layouts::rasterizer::VSMS_VIRTUAL_META_BINDING,
                resource: meta.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::rasterizer::VSMS_VIRTUAL_PAGE_TABLE_BINDING,
                resource: table.as_entire_binding(),
            },
        ],
    ))
}

pub fn create_shadow_stamp_bind_group(
    device: &RenderDevice,
    pipeline: &StrandShadowPipeline,
    prepass_resources: &StrandPrepassResources,
    params: &bevy::render::render_resource::Buffer,
) -> Option<BindGroup> {
    let frustum_table = prepass_resources.frustum_table.as_ref()?;
    let shadow_dom_surface_ids = prepass_resources.shadow_dom_surface_ids.as_ref()?;
    Some(device.create_bind_group(
        Some("strand_shadow_stampback_bind_group"),
        &pipeline.stamp_bind_group_layout,
        &[
            BindGroupEntry {
                binding: layouts::shadow_stampback::FRUSTUM_TABLE,
                resource: frustum_table.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::shadow_stampback::SHADOW_DOM_SURFACE_IDS,
                resource: shadow_dom_surface_ids.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::shadow_stampback::PARAMS,
                resource: params.as_entire_binding(),
            },
        ],
    ))
}

pub fn run_shadow_stampback_pass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandShadowPipeline,
    depth_sample_bind_group: &BindGroup,
    depth_table_bind_group: &BindGroup,
    stamp_bind_group: &BindGroup,
    target_view: &TextureView,
) {
    let Some(pipeline_id) = pipeline.stampback_pipeline else {
        warn!("Shadow stampback pipeline id not ready");
        return;
    };
    let Some(stamp_pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
        warn!("Shadow stampback pipeline not found");
        return;
    };

    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("Strand Shadow Stampback"),
        color_attachments: &[],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: target_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(stamp_pipeline);
    pass.set_bind_group(
        layouts::shadow_stampback::STAMP_GROUP as usize,
        stamp_bind_group,
        &[],
    );
    pass.set_bind_group(
        layouts::shadow_stampback::VSMS_DEPTH_READ_GROUP as usize,
        depth_sample_bind_group,
        &[],
    );
    pass.set_bind_group(
        layouts::shadow_stampback::VSMS_DEPTH_TABLE_GROUP as usize,
        depth_table_bind_group,
        &[],
    );
    pass.draw(0..3, 0..1);
}

pub fn run_shadow_pass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandShadowPipeline,
    allocator: &GpuPagingAllocator,
    opacity_storage_bind_group: &BindGroup,
    depth_storage_bind_group: &BindGroup,
    opacity_table_bind_group: &BindGroup,
    depth_table_bind_group: &BindGroup,
    resources: &StrandRasterizerResources,
    frustum_count: u32,
    raster_tile_run_dispatch_args: &bevy::render::render_resource::Buffer,
    bind_group: &BindGroup,
    offsets: &[u32],
) {
    let Some(pipeline_id) = pipeline.shadow_pipeline else {
        warn!("Shadow pipeline id not ready");
        return;
    };
    let Some(shadow_pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
        warn!("Shadow pipeline not found");
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

    let encoder = render_context.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Strand Rasterize Shadow"),
        ..default()
    });
    pass.set_pipeline(shadow_pipeline);
    pass.set_bind_group(allocator.buffer_group_idx, allocator_buffer_bind_group, &[]);
    pass.set_bind_group(
        allocator.table_group_idx,
        allocator_pagetable_bind_group,
        &[],
    );
    pass.set_bind_group(layouts::rasterizer::RASTER_GROUP, bind_group, offsets);
    pass.set_bind_group(
        layouts::rasterizer::VSMS_OPACITY_WRITE_GROUP,
        opacity_storage_bind_group,
        &[],
    );
    pass.set_bind_group(
        layouts::rasterizer::VSMS_DEPTH_WRITE_GROUP,
        depth_storage_bind_group,
        &[],
    );
    pass.set_bind_group(
        layouts::rasterizer::VSMS_OPACITY_TABLE_GROUP,
        opacity_table_bind_group,
        &[],
    );
    pass.set_bind_group(
        layouts::rasterizer::VSMS_DEPTH_TABLE_GROUP,
        depth_table_bind_group,
        &[],
    );

    let pushconstants = PushConstants {
        num_elements: resources.strand_count.unwrap_or(0),
        frustum_count,
        ..Default::default()
    };
    pass.set_immediates(0, bytemuck::bytes_of(&pushconstants));

    pass.dispatch_workgroups_indirect(raster_tile_run_dispatch_args, 0);
}
