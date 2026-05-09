use crate::pipelines::task_contract::{FinePageMeta, FineSegRef};
use crate::{
    allocator::GpuPagingAllocator,
    pipelines::layouts,
    pipelines::task_contract::{BINNING_POOL_CHUNK_SIZE, BINNING_POOL_NUM_HEADS},
    resources::ComputeInvocationDims,
    shader_types::{PushConstants, StrandGeo, StrandInstance, StrandMeta},
};
use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, IntoBinding,
            PipelineCache, PushConstantRange, ShaderStages,
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewUniformOffset,
    },
    shader::ShaderDefVal,
};
use bevy_gpu_paging_allocator::BindGroupBuilder;
use bevy_vsms::request::VirtualSurfaceRequestBitmapRuntime;

#[derive(Resource, Default)]
pub struct StrandPrepassResources {
    // inputs
    pub visibility_flags_buffer: Option<Buffer>,
    pub visible_geos_buffer: Option<Buffer>,
    pub geos_prefix_buffer: Option<Buffer>,
    pub prepass_queue: Option<Buffer>,
    pub binning_queue: Option<Buffer>,
    pub strand_instances: Option<Buffer>,
    // products
    pub indirect_args: Option<Buffer>,
    // queue-binning allocator buffers
    pub chunk_pool: Option<Buffer>,
    pub free_heads: Option<Buffer>,
    pub frustum_table: Option<Buffer>,
    pub froxel_bucket_heads: Option<Buffer>,
    pub raster_work_queue: Option<Buffer>,
    pub raster_tile_run_queue: Option<Buffer>,
    pub coarse_depth_lut: Option<Buffer>,
    pub coarse_range_queue: Option<Buffer>,
    pub coarse_interval_heads: Option<Buffer>,
    pub coarse_interval_refs: Option<Buffer>,
    pub coarse_range_lookup: Option<Buffer>,
    pub coarse_count_page_table: Option<Buffer>,
    pub coarse_count_pages: Option<Buffer>,
    pub fine_page_meta: Option<Buffer>,
    pub fine_cell_offsets: Option<Buffer>,
    pub fine_cell_work_offsets: Option<Buffer>,
    pub fine_cell_write_cursors: Option<Buffer>,
    pub fine_seg_refs: Option<Buffer>,
    pub coarse_tile_work_counts: Option<Buffer>,
    pub coarse_tile_work_offsets: Option<Buffer>,
    pub shadow_dom_surface_ids: Option<Buffer>,
    pub broad_instance_meta: Option<Buffer>,
    // capacities
    pub prepass_task_capacity: u32,
    pub binning_task_capacity: u32,
    pub instance_capacity: u32,
    pub instance_count: u32,
    pub max_strands_in_instance: u32,
    pub frustum_capacity: u32,
    pub frustum_count: u32,
    pub froxel_bucket_capacity: u32,
    pub raster_work_capacity: u32,
    pub raster_tile_run_capacity: u32,
    pub coarse_depth_tile_capacity: u32,
    pub coarse_range_capacity: u32,
    pub coarse_interval_ref_capacity: u32,
    pub coarse_count_page_capacity: u32,
    pub fine_seg_ref_capacity: u32,
}

#[derive(Resource)]
pub struct StrandPrepassPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub broad_pipeline: Option<CachedComputePipelineId>,
    pub broad_strand_pipeline: Option<CachedComputePipelineId>,
    pub finalize_pipeline: Option<CachedComputePipelineId>,
    pub fine_pipeline: Option<CachedComputePipelineId>,
    pub coarse_interval_pipeline: Option<CachedComputePipelineId>,
    pub build_depth_warp_pipeline: Option<CachedComputePipelineId>,
    pub finalize_binning_pipeline: Option<CachedComputePipelineId>,
    pub mark_coarse_count_pages_pipeline: Option<CachedComputePipelineId>,
    pub allocate_coarse_count_pages_pipeline: Option<CachedComputePipelineId>,
    pub binning_pipeline: Option<CachedComputePipelineId>,
    pub prefix_fine_pages_pipeline: Option<CachedComputePipelineId>,
    pub fill_fine_seg_refs_pipeline: Option<CachedComputePipelineId>,
    pub count_coarse_tile_work_pipeline: Option<CachedComputePipelineId>,
    pub emit_raster_work_pipeline: Option<CachedComputePipelineId>,
}

impl StrandPrepassPipeline {
    fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn uniform_entry(binding: u32, has_dynamic_offset: bool) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "strand_prepass_bind_group_layout",
            &[
                Self::storage_entry(layouts::prepass::PREPASS_QUEUE, false),
                Self::storage_entry(layouts::prepass::BINNING_QUEUE, false),
                Self::uniform_entry(layouts::prepass::LIGHT_UNIFORM, true),
                Self::storage_entry(layouts::prepass::CLUSTER_INDICES, true),
                Self::storage_entry(layouts::prepass::CLUSTER_OFFSETS_AND_COUNTS, true),
                Self::storage_entry(layouts::prepass::CLUSTERABLE_OBJECTS, true),
                Self::uniform_entry(layouts::prepass::VIEW_UNIFORM, true),
                Self::storage_entry(layouts::prepass::VISIBLE_FLAGS, false),
                Self::storage_entry(layouts::prepass::VISIBLE_GEO, false),
                Self::storage_entry(layouts::prepass::GEO_PREFIX, false),
                Self::storage_entry(layouts::prepass::INDIRECT_BUFFER, false),
                Self::storage_entry(layouts::prepass::FRUSTUM_TABLE, true),
                Self::storage_entry(layouts::prepass::FROXEL_BUCKET_HEADS, false),
                Self::storage_entry(layouts::prepass::CHUNK_POOL, false),
                Self::storage_entry(layouts::prepass::FREE_HEADS, false),
                Self::storage_entry(layouts::prepass::RASTER_WORK_QUEUE, false),
                Self::storage_entry(layouts::prepass::RASTER_TILE_RUN_QUEUE, false),
                Self::storage_entry(layouts::prepass::COARSE_DEPTH_LUT, false),
                Self::storage_entry(layouts::prepass::COARSE_RANGE_QUEUE, false),
                Self::storage_entry(layouts::prepass::COARSE_INTERVAL_HEADS, false),
                Self::storage_entry(layouts::prepass::COARSE_INTERVAL_REFS, false),
                Self::storage_entry(layouts::prepass::COARSE_RANGE_LOOKUP, false),
                Self::storage_entry(layouts::prepass::COARSE_COUNT_PAGE_TABLE, false),
                Self::storage_entry(layouts::prepass::COARSE_COUNT_PAGES, false),
                Self::storage_entry(layouts::prepass::STRAND_INSTANCES, true),
                Self::storage_entry(layouts::prepass::FINE_PAGE_META, false),
                Self::storage_entry(layouts::prepass::FINE_CELL_OFFSETS, false),
                Self::storage_entry(layouts::prepass::FINE_CELL_WORK_OFFSETS, false),
                Self::storage_entry(layouts::prepass::FINE_CELL_WRITE_CURSORS, false),
                Self::storage_entry(layouts::prepass::FINE_SEG_REFS, false),
                Self::storage_entry(layouts::prepass::COARSE_TILE_WORK_COUNTS, false),
                Self::storage_entry(layouts::prepass::COARSE_TILE_WORK_OFFSETS, false),
                Self::storage_entry(layouts::prepass::VSMS_REQUEST_META, true),
                Self::storage_entry(layouts::prepass::VSMS_REQUEST_BITS, false),
                Self::storage_entry(layouts::prepass::SHADOW_DOM_SURFACE_IDS, true),
                Self::storage_entry(layouts::prepass::BROAD_INSTANCE_META, false),
            ],
        )
    }

    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        let descriptor = Self::bind_group_layout_descriptor();
        device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries)
    }
}

fn queue_prepass_pipeline(
    pipeline_cache: &PipelineCache,
    shader: Handle<Shader>,
    _bind_group_layout: BindGroupLayout,
    allocator: &GpuPagingAllocator,
    invocation_dims: &ComputeInvocationDims,
    entry_point: &'static str,
) -> Option<CachedComputePipelineId> {
    if allocator.buffer_bind_group_layout.is_none()
        || allocator.pagetable_bind_group_layout.is_none()
    {
        return None;
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
    let bind_group_layout = StrandPrepassPipeline::bind_group_layout_descriptor();

    let mut shader_defs = vec![layouts::prepass::shader_defs(), allocator.shader_defs()].concat();
    shader_defs.extend([
        ShaderDefVal::UInt(
            "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
            invocation_dims.threads_per_workgroup,
        ),
        ShaderDefVal::UInt(
            "WORKGROUP_SIZE".into(),
            invocation_dims.threads_per_workgroup,
        ),
        ShaderDefVal::UInt(
            "NUMBER_OF_THREADS_PER_SUBGROUP".into(),
            invocation_dims.subgroup_size,
        ),
        ShaderDefVal::UInt(
            "FINE_WORKGROUP_SIZE".into(),
            invocation_dims.threads_per_workgroup,
        ),
        ShaderDefVal::UInt(
            "SIZEOF_METADATA".into(),
            std::mem::size_of::<StrandMeta>() as u32,
        ),
        ShaderDefVal::UInt("SIZEOF_GEO".into(), std::mem::size_of::<StrandGeo>() as u32),
        ShaderDefVal::UInt(
            "SIZEOF_INSTANCE".into(),
            std::mem::size_of::<StrandInstance>() as u32,
        ),
        ShaderDefVal::UInt(
            "SIZEOF_FINE_PAGE_META".into(),
            std::mem::size_of::<FinePageMeta>() as u32,
        ),
        ShaderDefVal::UInt(
            "SIZEOF_FINE_SEG_REF".into(),
            std::mem::size_of::<FineSegRef>() as u32,
        ),
        ShaderDefVal::UInt("POOL_CHUNK_SIZE".into(), BINNING_POOL_CHUNK_SIZE),
        ShaderDefVal::UInt("POOL_NUM_HEADS".into(), BINNING_POOL_NUM_HEADS),
        ShaderDefVal::UInt(
            "COARSE_FINE_TILE_EXTENT".into(),
            crate::plugin::COARSE_FINE_TILE_EXTENT,
        ),
        ShaderDefVal::UInt(
            "COARSE_DEPTH_SLICES".into(),
            crate::plugin::COARSE_DEPTH_SLICES,
        ),
        ShaderDefVal::UInt(
            "COARSE_MAX_SLICES_PER_ASSET_INTERVAL".into(),
            crate::plugin::COARSE_MAX_SLICES_PER_ASSET_INTERVAL,
        ),
        ShaderDefVal::UInt("DOM_PAGE_XY".into(), crate::plugin::DOM_PAGE_XY),
        ShaderDefVal::UInt(
            "COARSE_COUNT_PAGE_SIZE".into(),
            crate::plugin::COARSE_COUNT_PAGE_SIZE,
        ),
    ]);

    let max_group = allocator
        .buffer_group_idx
        .max(allocator.table_group_idx)
        .max(layouts::prepass::PREPASS_GROUP);
    let mut layout = vec![bind_group_layout.clone(); (max_group + 1) as usize];
    layout[allocator.buffer_group_idx as usize] = buffer_layout;
    layout[allocator.table_group_idx as usize] = table_layout;
    layout[layouts::prepass::PREPASS_GROUP as usize] = bind_group_layout;

    Some(
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_prepass_pipeline".into()),
            layout,
            shader,
            shader_defs,
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: Some(entry_point.into()),
            zero_initialize_workgroup_memory: false,
        }),
    )
}

impl FromWorld for StrandPrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let _invocation_dims = *world.resource::<ComputeInvocationDims>();

        StrandPrepassPipeline {
            bind_group_layout,
            broad_pipeline: None,
            broad_strand_pipeline: None,
            finalize_pipeline: None,
            fine_pipeline: None,
            coarse_interval_pipeline: None,
            build_depth_warp_pipeline: None,
            finalize_binning_pipeline: None,
            mark_coarse_count_pages_pipeline: None,
            allocate_coarse_count_pages_pipeline: None,
            binning_pipeline: None,
            prefix_fine_pages_pipeline: None,
            fill_fine_seg_refs_pipeline: None,
            count_coarse_tile_work_pipeline: None,
            emit_raster_work_pipeline: None,
        }
    }
}

pub fn create_prepass_bind_group(
    device: &RenderDevice,
    pipeline: &StrandPrepassPipeline,
    resources: &StrandPrepassResources,
    view_buffer: &BindingResource,
    light_buffer: &BindingResource,
    view_offsets: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
    cluster_indices: &BindingResource,
    cluster_offsets_and_counts: &BindingResource,
    clusterable_objects: &BindingResource,
    request_runtime: &VirtualSurfaceRequestBitmapRuntime,
) -> Result<(BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;
    let prepass_queue = resources.prepass_queue.as_ref().ok_or(())?;
    let binning_queue = resources.binning_queue.as_ref().ok_or(())?;
    let visibility_flags_buffer = resources.visibility_flags_buffer.as_ref().ok_or(())?;
    let visible_geos_buffer = resources.visible_geos_buffer.as_ref().ok_or(())?;
    let geos_prefix_buffer = resources.geos_prefix_buffer.as_ref().ok_or(())?;
    let dispatch_args = resources.indirect_args.as_ref().ok_or(())?;
    let frustum_table = resources.frustum_table.as_ref().ok_or(())?;
    let froxel_bucket_heads = resources.froxel_bucket_heads.as_ref().ok_or(())?;
    let chunk_pool = resources.chunk_pool.as_ref().ok_or(())?;
    let free_heads = resources.free_heads.as_ref().ok_or(())?;
    let raster_work_queue = resources.raster_work_queue.as_ref().ok_or(())?;
    let raster_tile_run_queue = resources.raster_tile_run_queue.as_ref().ok_or(())?;
    let coarse_depth_lut = resources.coarse_depth_lut.as_ref().ok_or(())?;
    let coarse_range_queue = resources.coarse_range_queue.as_ref().ok_or(())?;
    let coarse_interval_heads = resources.coarse_interval_heads.as_ref().ok_or(())?;
    let coarse_interval_refs = resources.coarse_interval_refs.as_ref().ok_or(())?;
    let coarse_range_lookup = resources.coarse_range_lookup.as_ref().ok_or(())?;
    let coarse_count_page_table = resources.coarse_count_page_table.as_ref().ok_or(())?;
    let coarse_count_pages = resources.coarse_count_pages.as_ref().ok_or(())?;
    let strand_instances = resources.strand_instances.as_ref().ok_or(())?;
    let fine_page_meta = resources.fine_page_meta.as_ref().ok_or(())?;
    let fine_cell_offsets = resources.fine_cell_offsets.as_ref().ok_or(())?;
    let fine_cell_work_offsets = resources.fine_cell_work_offsets.as_ref().ok_or(())?;
    let fine_cell_write_cursors = resources.fine_cell_write_cursors.as_ref().ok_or(())?;
    let fine_seg_refs = resources.fine_seg_refs.as_ref().ok_or(())?;
    let coarse_tile_work_counts = resources.coarse_tile_work_counts.as_ref().ok_or(())?;
    let coarse_tile_work_offsets = resources.coarse_tile_work_offsets.as_ref().ok_or(())?;
    let shadow_dom_surface_ids = resources.shadow_dom_surface_ids.as_ref().ok_or(())?;
    let broad_instance_meta = resources.broad_instance_meta.as_ref().ok_or(())?;
    let request_meta = request_runtime.metadata_buffer.as_ref().ok_or(())?;
    let request_bits = request_runtime.bits_buffer.as_ref().ok_or(())?;

    Ok((
        device.create_bind_group(
            Some("strand_prepass_bind_group"),
            layout,
            &[
                BindGroupEntry {
                    binding: layouts::prepass::PREPASS_QUEUE,
                    resource: prepass_queue.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::BINNING_QUEUE,
                    resource: binning_queue.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::LIGHT_UNIFORM,
                    resource: light_buffer.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::CLUSTER_INDICES,
                    resource: cluster_indices.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::CLUSTER_OFFSETS_AND_COUNTS,
                    resource: cluster_offsets_and_counts.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::CLUSTERABLE_OBJECTS,
                    resource: clusterable_objects.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::VIEW_UNIFORM,
                    resource: view_buffer.clone().into_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::VISIBLE_FLAGS,
                    resource: visibility_flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::VISIBLE_GEO,
                    resource: visible_geos_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::GEO_PREFIX,
                    resource: geos_prefix_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::INDIRECT_BUFFER,
                    resource: dispatch_args.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FRUSTUM_TABLE,
                    resource: frustum_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FROXEL_BUCKET_HEADS,
                    resource: froxel_bucket_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::CHUNK_POOL,
                    resource: chunk_pool.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FREE_HEADS,
                    resource: free_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::RASTER_WORK_QUEUE,
                    resource: raster_work_queue.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::RASTER_TILE_RUN_QUEUE,
                    resource: raster_tile_run_queue.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::COARSE_DEPTH_LUT,
                    resource: coarse_depth_lut.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::COARSE_RANGE_QUEUE,
                    resource: coarse_range_queue.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::COARSE_INTERVAL_HEADS,
                    resource: coarse_interval_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::COARSE_INTERVAL_REFS,
                    resource: coarse_interval_refs.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::COARSE_RANGE_LOOKUP,
                    resource: coarse_range_lookup.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::COARSE_COUNT_PAGE_TABLE,
                    resource: coarse_count_page_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::COARSE_COUNT_PAGES,
                    resource: coarse_count_pages.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::STRAND_INSTANCES,
                    resource: strand_instances.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FINE_PAGE_META,
                    resource: fine_page_meta.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FINE_CELL_OFFSETS,
                    resource: fine_cell_offsets.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FINE_CELL_WORK_OFFSETS,
                    resource: fine_cell_work_offsets.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FINE_CELL_WRITE_CURSORS,
                    resource: fine_cell_write_cursors.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FINE_SEG_REFS,
                    resource: fine_seg_refs.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::COARSE_TILE_WORK_COUNTS,
                    resource: coarse_tile_work_counts.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::COARSE_TILE_WORK_OFFSETS,
                    resource: coarse_tile_work_offsets.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::VSMS_REQUEST_META,
                    resource: request_meta.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::VSMS_REQUEST_BITS,
                    resource: request_bits.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::SHADOW_DOM_SURFACE_IDS,
                    resource: shadow_dom_surface_ids.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::BROAD_INSTANCE_META,
                    resource: broad_instance_meta.as_entire_binding(),
                },
            ],
        ),
        // Dynamic offsets order follows bind-group layout declaration order.
        vec![view_light_uniform_offset.offset, view_offsets.offset],
    ))
}

pub fn run_prepass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandPrepassPipeline,
    allocator: &GpuPagingAllocator,
    settings: &ComputeInvocationDims,
    cull_settings: &crate::resources::StochasticCullSettings,
    bind_group: &BindGroup,
    uniform_offsets: &[u32],
    indirect_args: &Buffer,
    coarse_depth_lut: &Buffer,
    coarse_count_page_table: &Buffer,
    coarse_count_pages: &Buffer,
    fine_cell_write_cursors: &Buffer,
    fine_seg_refs: &Buffer,
    coarse_tile_work_counts: &Buffer,
    coarse_tile_work_offsets: &Buffer,
    frustum_count: u32,
    instance_count: u32,
    max_strands_in_instance: u32,
    coarse_depth_tile_capacity: u32,
    coarse_range_capacity: u32,
) {
    let encoder = render_context.command_encoder();
    let Some(allocator_buffer_bind_group) = &allocator.buffer_bind_group else {
        warn!("allocator buffer bind group is not ready yet.");
        return;
    };
    let Some(allocator_pagetable_bind_group) = &allocator.pagetable_bind_group else {
        warn!("allocator pagetable bind group is not ready yet.");
        return;
    };

    encoder.clear_buffer(coarse_depth_lut, 0, None);
    encoder.clear_buffer(coarse_count_page_table, 0, None);
    encoder.clear_buffer(coarse_count_pages, 0, None);
    let _ = fine_cell_write_cursors;
    encoder.clear_buffer(fine_seg_refs, 0, Some(4));
    encoder.clear_buffer(coarse_tile_work_counts, 0, None);
    encoder.clear_buffer(coarse_tile_work_offsets, 0, None);

    let Some(broad_pipeline_id) = pipeline.broad_pipeline else {
        warn!("Broad prepass pipeline id not ready yet");
        return;
    };
    let Some(broad_strand_pipeline_id) = pipeline.broad_strand_pipeline else {
        warn!("Broad strand prepass pipeline id not ready yet");
        return;
    };
    let Some(finalize_pipeline_id) = pipeline.finalize_pipeline else {
        warn!("Finalize prepass pipeline id not ready yet");
        return;
    };
    let Some(fine_pipeline_id) = pipeline.fine_pipeline else {
        warn!("Fine prepass pipeline id not ready yet");
        return;
    };
    let Some(coarse_interval_pipeline_id) = pipeline.coarse_interval_pipeline else {
        warn!("Coarse interval pipeline id not ready yet");
        return;
    };
    let Some(build_depth_warp_pipeline_id) = pipeline.build_depth_warp_pipeline else {
        warn!("Depth warp pipeline id not ready yet");
        return;
    };
    let Some(finalize_binning_pipeline_id) = pipeline.finalize_binning_pipeline else {
        warn!("Finalize binning pipeline id not ready yet");
        return;
    };
    let Some(mark_coarse_count_pages_pipeline_id) = pipeline.mark_coarse_count_pages_pipeline
    else {
        warn!("Mark coarse count pages pipeline id not ready yet");
        return;
    };
    let Some(allocate_coarse_count_pages_pipeline_id) =
        pipeline.allocate_coarse_count_pages_pipeline
    else {
        warn!("Allocate coarse count pages pipeline id not ready yet");
        return;
    };
    let Some(binning_pipeline_id) = pipeline.binning_pipeline else {
        warn!("Binning queue pipeline id not ready yet");
        return;
    };
    let Some(prefix_fine_pages_pipeline_id) = pipeline.prefix_fine_pages_pipeline else {
        warn!("Prefix fine pages pipeline id not ready yet");
        return;
    };
    let Some(fill_fine_seg_refs_pipeline_id) = pipeline.fill_fine_seg_refs_pipeline else {
        warn!("Fill fine seg refs pipeline id not ready yet");
        return;
    };
    let Some(count_coarse_tile_work_pipeline_id) = pipeline.count_coarse_tile_work_pipeline else {
        warn!("Count coarse tile work pipeline id not ready yet");
        return;
    };
    let Some(emit_raster_work_pipeline_id) = pipeline.emit_raster_work_pipeline else {
        warn!("Emit raster work pipeline id not ready yet");
        return;
    };
    let Some(broad_pipeline) = pipeline_cache.get_compute_pipeline(broad_pipeline_id) else {
        warn!("Broad prepass pipeline not found");
        return;
    };
    let Some(broad_strand_pipeline) = pipeline_cache.get_compute_pipeline(broad_strand_pipeline_id)
    else {
        warn!("Broad strand prepass pipeline not found");
        return;
    };
    let Some(finalize_pipeline) = pipeline_cache.get_compute_pipeline(finalize_pipeline_id) else {
        warn!("Finalize prepass pipeline not found");
        return;
    };
    let Some(fine_pipeline) = pipeline_cache.get_compute_pipeline(fine_pipeline_id) else {
        warn!("Fine prepass pipeline not found");
        return;
    };
    let Some(coarse_interval_pipeline) =
        pipeline_cache.get_compute_pipeline(coarse_interval_pipeline_id)
    else {
        warn!("Coarse interval pipeline not found");
        return;
    };
    let Some(build_depth_warp_pipeline) =
        pipeline_cache.get_compute_pipeline(build_depth_warp_pipeline_id)
    else {
        warn!("Depth warp pipeline not found");
        return;
    };
    let Some(finalize_binning_pipeline) =
        pipeline_cache.get_compute_pipeline(finalize_binning_pipeline_id)
    else {
        warn!("Finalize binning pipeline not found");
        return;
    };
    let Some(mark_coarse_count_pages_pipeline) =
        pipeline_cache.get_compute_pipeline(mark_coarse_count_pages_pipeline_id)
    else {
        warn!("Mark coarse count pages pipeline not found");
        return;
    };
    let Some(allocate_coarse_count_pages_pipeline) =
        pipeline_cache.get_compute_pipeline(allocate_coarse_count_pages_pipeline_id)
    else {
        warn!("Allocate coarse count pages pipeline not found");
        return;
    };
    let Some(binning_pipeline) = pipeline_cache.get_compute_pipeline(binning_pipeline_id) else {
        warn!("Binning queue pipeline not found");
        return;
    };
    let Some(prefix_fine_pages_pipeline) =
        pipeline_cache.get_compute_pipeline(prefix_fine_pages_pipeline_id)
    else {
        warn!("Prefix fine pages pipeline not found");
        return;
    };
    let Some(fill_fine_seg_refs_pipeline) =
        pipeline_cache.get_compute_pipeline(fill_fine_seg_refs_pipeline_id)
    else {
        warn!("Fill fine seg refs pipeline not found");
        return;
    };
    let Some(count_coarse_tile_work_pipeline) =
        pipeline_cache.get_compute_pipeline(count_coarse_tile_work_pipeline_id)
    else {
        warn!("Count coarse tile work pipeline not found");
        return;
    };
    let Some(emit_raster_work_pipeline) =
        pipeline_cache.get_compute_pipeline(emit_raster_work_pipeline_id)
    else {
        warn!("Emit raster work pipeline not found");
        return;
    };
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Strand Prepass"),
        ..default()
    });
    let pushconstants = PushConstants {
        num_elements: instance_count,
        frustum_count,
        stochastic_cull_enabled: u32::from(cull_settings.enabled),
        cull_min_dist: cull_settings.min_dist,
        cull_max_dist: cull_settings.max_dist,
        cull_exponent: cull_settings.exponent,
        ..Default::default()
    };
    pass.set_pipeline(broad_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&pushconstants));
    pass.set_bind_group(allocator.buffer_group_idx, allocator_buffer_bind_group, &[]);
    pass.set_bind_group(
        allocator.table_group_idx,
        allocator_pagetable_bind_group,
        &[],
    );
    pass.set_bind_group(layouts::prepass::PREPASS_GROUP, bind_group, uniform_offsets);
    pass.dispatch_workgroups(
        instance_count.div_ceil(settings.threads_per_workgroup),
        1,
        1,
    );
    let broad_strand_pushconstants = PushConstants {
        num_elements: max_strands_in_instance.max(1),
        scan_load_base: instance_count,
        ..pushconstants
    };
    pass.set_pipeline(broad_strand_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&broad_strand_pushconstants));
    pass.dispatch_workgroups(
        max_strands_in_instance
            .max(1)
            .div_ceil(settings.threads_per_workgroup),
        instance_count.max(1),
        1,
    );
    pass.set_pipeline(finalize_pipeline);
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(fine_pipeline);
    pass.dispatch_workgroups_indirect(indirect_args, 0);
    let coarse_interval_pushconstants = PushConstants {
        num_elements: coarse_range_capacity,
        ..pushconstants
    };
    pass.set_pipeline(coarse_interval_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&coarse_interval_pushconstants));
    pass.dispatch_workgroups(coarse_range_capacity, 1, 1);
    let depth_warp_pushconstants = PushConstants {
        num_elements: coarse_depth_tile_capacity,
        ..pushconstants
    };
    pass.set_pipeline(build_depth_warp_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&depth_warp_pushconstants));
    pass.dispatch_workgroups(
        coarse_depth_tile_capacity.div_ceil(settings.threads_per_workgroup),
        1,
        1,
    );
    pass.set_pipeline(finalize_binning_pipeline);
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(mark_coarse_count_pages_pipeline);
    pass.dispatch_workgroups_indirect(indirect_args, 0);
    let coarse_count_page_table_capacity =
        coarse_depth_tile_capacity.saturating_mul(crate::plugin::COARSE_DEPTH_SLICES);
    let allocate_count_pages_pushconstants = PushConstants {
        num_elements: coarse_count_page_table_capacity,
        ..pushconstants
    };
    pass.set_pipeline(allocate_coarse_count_pages_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&allocate_count_pages_pushconstants));
    pass.dispatch_workgroups(
        coarse_count_page_table_capacity.div_ceil(settings.threads_per_workgroup),
        1,
        1,
    );
    pass.set_pipeline(binning_pipeline);
    pass.dispatch_workgroups_indirect(indirect_args, 0);
    pass.set_pipeline(prefix_fine_pages_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&allocate_count_pages_pushconstants));
    pass.dispatch_workgroups(
        coarse_count_page_table_capacity.min(65_535),
        coarse_count_page_table_capacity.div_ceil(65_535),
        1,
    );
    pass.set_pipeline(fill_fine_seg_refs_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&pushconstants));
    pass.dispatch_workgroups_indirect(indirect_args, 0);
    let coarse_tile_pushconstants = PushConstants {
        num_elements: coarse_depth_tile_capacity,
        ..pushconstants
    };
    pass.set_pipeline(count_coarse_tile_work_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&coarse_tile_pushconstants));
    let coarse_tile_workgroups_x = coarse_depth_tile_capacity.min(65_535).max(1);
    let coarse_tile_workgroups_y = coarse_depth_tile_capacity
        .div_ceil(coarse_tile_workgroups_x)
        .max(1);
    pass.dispatch_workgroups(coarse_tile_workgroups_x, coarse_tile_workgroups_y, 1);
    pass.set_pipeline(emit_raster_work_pipeline);
    pass.set_push_constants(0, bytemuck::bytes_of(&coarse_tile_pushconstants));
    pass.dispatch_workgroups(coarse_tile_workgroups_x, coarse_tile_workgroups_y, 1);
}

pub fn update_strand_prepass_pipeline(
    pipeline_cache: Res<PipelineCache>,
    mut pipeline_res: ResMut<StrandPrepassPipeline>,
    dims: Res<ComputeInvocationDims>,
    mut last_state: Local<Option<(ComputeInvocationDims, u64)>>,
    shader_loader: Res<AssetServer>,
    allocator: Res<GpuPagingAllocator>,
) {
    let current_state = (*dims, allocator.bindgroups_epoch);
    if last_state.as_ref() == Some(&current_state) {
        return;
    }

    if allocator.buffer_bind_group_layout.is_none()
        || allocator.pagetable_bind_group_layout.is_none()
    {
        return;
    }

    *last_state = Some(current_state);

    let broad_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let broad_strand_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let finalize_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let fine_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let coarse_interval_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let build_depth_warp_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let finalize_binning_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let mark_coarse_count_pages_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let allocate_coarse_count_pages_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let binning_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let prefix_fine_pages_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let fill_fine_seg_refs_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let count_coarse_tile_work_shader = shader_loader.load("shaders/strand_prepass.wgsl");
    let emit_raster_work_shader = shader_loader.load("shaders/strand_prepass.wgsl");

    let Some(broad_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        broad_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "broad_prepass",
    ) else {
        warn!("Could not queue broad prepass pipeline");
        return;
    };
    let Some(broad_strand_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        broad_strand_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "broad_strand_prepass",
    ) else {
        warn!("Could not queue broad strand prepass pipeline");
        return;
    };
    let Some(finalize_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        finalize_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "finalize_prepass",
    ) else {
        return;
    };
    let Some(fine_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        fine_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "fine_prepass",
    ) else {
        return;
    };
    let Some(coarse_interval_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        coarse_interval_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "coarse_interval_pass",
    ) else {
        return;
    };
    let Some(build_depth_warp_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        build_depth_warp_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "build_depth_warp_lut",
    ) else {
        return;
    };
    let Some(finalize_binning_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        finalize_binning_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "finalize_binning",
    ) else {
        return;
    };
    let Some(mark_coarse_count_pages_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        mark_coarse_count_pages_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "mark_coarse_count_pages_pass",
    ) else {
        return;
    };
    let Some(allocate_coarse_count_pages_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        allocate_coarse_count_pages_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "allocate_coarse_count_pages",
    ) else {
        return;
    };
    let Some(binning_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        binning_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "binning_queue_pass",
    ) else {
        return;
    };
    let Some(prefix_fine_pages_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        prefix_fine_pages_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "prefix_fine_pages",
    ) else {
        return;
    };
    let Some(fill_fine_seg_refs_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        fill_fine_seg_refs_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "fill_fine_seg_refs",
    ) else {
        return;
    };
    let Some(count_coarse_tile_work_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        count_coarse_tile_work_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "count_coarse_tile_work",
    ) else {
        return;
    };
    let Some(emit_raster_work_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        emit_raster_work_shader,
        pipeline_res.bind_group_layout.clone(),
        &allocator,
        &dims,
        "emit_raster_work",
    ) else {
        return;
    };
    pipeline_res.broad_pipeline = Some(broad_pipeline_id);
    pipeline_res.broad_strand_pipeline = Some(broad_strand_pipeline_id);
    pipeline_res.finalize_pipeline = Some(finalize_pipeline_id);
    pipeline_res.fine_pipeline = Some(fine_pipeline_id);
    pipeline_res.coarse_interval_pipeline = Some(coarse_interval_pipeline_id);
    pipeline_res.build_depth_warp_pipeline = Some(build_depth_warp_pipeline_id);
    pipeline_res.finalize_binning_pipeline = Some(finalize_binning_pipeline_id);
    pipeline_res.mark_coarse_count_pages_pipeline = Some(mark_coarse_count_pages_pipeline_id);
    pipeline_res.allocate_coarse_count_pages_pipeline =
        Some(allocate_coarse_count_pages_pipeline_id);
    pipeline_res.binning_pipeline = Some(binning_pipeline_id);
    pipeline_res.prefix_fine_pages_pipeline = Some(prefix_fine_pages_pipeline_id);
    pipeline_res.fill_fine_seg_refs_pipeline = Some(fill_fine_seg_refs_pipeline_id);
    pipeline_res.count_coarse_tile_work_pipeline = Some(count_coarse_tile_work_pipeline_id);
    pipeline_res.emit_raster_work_pipeline = Some(emit_raster_work_pipeline_id);
    debug!(
        "Rebuilt strand prepass pipelines: broad={:?} broad_strand={:?} finalize={:?} fine={:?} coarse_interval={:?} depth_warp={:?} finalize_binning={:?} mark_pages={:?} allocate_pages={:?} binning={:?} prefix_pages={:?} fill_refs={:?} count_work={:?} emit_work={:?}",
        broad_pipeline_id,
        broad_strand_pipeline_id,
        finalize_pipeline_id,
        fine_pipeline_id,
        coarse_interval_pipeline_id,
        build_depth_warp_pipeline_id,
        finalize_binning_pipeline_id,
        mark_coarse_count_pages_pipeline_id,
        allocate_coarse_count_pages_pipeline_id,
        binning_pipeline_id,
        prefix_fine_pages_pipeline_id,
        fill_fine_seg_refs_pipeline_id,
        count_coarse_tile_work_pipeline_id,
        emit_raster_work_pipeline_id
    );
}
