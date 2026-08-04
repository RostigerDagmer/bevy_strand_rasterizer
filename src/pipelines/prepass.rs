use crate::pipelines::task_contract::{FinePageMeta, FineSegRef};
use crate::{
    allocator::GpuPagingAllocator,
    pipelines::layouts,
    pipelines::task_contract::{BINNING_POOL_CHUNK_SIZE, BINNING_POOL_NUM_HEADS},
    resources::{ComputeInvocationDims, FineBinningBackend},
    shader_types::{PushConstants, StrandGeo, StrandInstance, StrandMeta},
};
use bevy::{
    core_pipeline::prepass::ViewPrepassTextures,
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, IntoBinding,
            PipelineCache, ShaderStages, TextureSampleType, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewUniformOffset,
    },
    shader::ShaderDefVal,
};
use bevy_gpu_paging_allocator::BindGroupBuilder;
use bevy_vsms::request::VirtualSurfaceRequestBitmapRuntime;

pub const TELEMETRY_HISTOGRAM_BINS: u32 = 32;
pub const TELEMETRY_HISTOGRAM_OFFSET_WORDS: u32 = 8;
pub const TELEMETRY_PAGE_COUNTS_OFFSET_WORDS: u32 =
    TELEMETRY_HISTOGRAM_OFFSET_WORDS + TELEMETRY_HISTOGRAM_BINS;
pub const FRUSTUM_TELEMETRY_STRIDE_WORDS: u32 = 11;

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
    pub prefix_indirect_args: Option<Buffer>,
    pub telemetry: Option<Buffer>,
    pub projected_segments: Option<Buffer>,
    pub page_candidate_counts: Option<Buffer>,
    pub page_candidate_offsets: Option<Buffer>,
    pub page_candidate_cursors: Option<Buffer>,
    pub page_candidates: Option<Buffer>,
    pub virtual_page_candidate_counts: Option<Buffer>,
    pub frustum_instance_keep_probabilities: Option<Buffer>,
    // queue-binning allocator buffers
    pub chunk_pool: Option<Buffer>,
    pub free_heads: Option<Buffer>,
    pub frustum_table: Option<Buffer>,
    pub froxel_bucket_heads: Option<Buffer>,
    pub raster_work_queue: Option<Buffer>,
    pub raster_tile_run_queue: Option<Buffer>,
    pub raster_tile_run_dispatch_args: Option<Buffer>,
    pub coarse_depth_lut: Option<Buffer>,
    pub coarse_range_queue: Option<Buffer>,
    pub coarse_interval_heads: Option<Buffer>,
    pub coarse_interval_refs: Option<Buffer>,
    pub coarse_range_lookup: Option<Buffer>,
    pub coarse_count_page_table: Option<Buffer>,
    pub coarse_count_pages: Option<Buffer>,
    pub fine_page_meta: Option<Buffer>,
    pub fine_cell_offsets: Option<Buffer>,
    pub fine_cell_write_cursors: Option<Buffer>,
    pub fine_seg_refs: Option<Buffer>,
    pub shadow_dom_surface_ids: Option<Buffer>,
    pub broad_instance_meta: Option<Buffer>,
    pub opaque_fine_depth_tiles: Option<Buffer>,
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
    pub page_candidate_capacity: u32,
    pub opaque_fine_depth_tile_capacity: u32,
}

#[derive(Resource)]
pub struct StrandPrepassPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub indirect_args_bind_group_layout: BindGroupLayout,
    pub broad_pipeline: Option<CachedComputePipelineId>,
    pub broad_strand_pipeline: Option<CachedComputePipelineId>,
    pub finalize_pipeline: Option<CachedComputePipelineId>,
    pub fine_pipeline: Option<CachedComputePipelineId>,
    pub coarse_interval_pipeline: Option<CachedComputePipelineId>,
    pub build_depth_warp_pipeline: Option<CachedComputePipelineId>,
    pub finalize_binning_pipeline: Option<CachedComputePipelineId>,
    pub mark_coarse_count_pages_pipeline: Option<CachedComputePipelineId>,
    pub allocate_coarse_count_pages_pipeline: Option<CachedComputePipelineId>,
    pub finalize_prefix_dispatch_pipeline: Option<CachedComputePipelineId>,
    pub segment_scatter: SegmentScatterPipelines,
    pub page_csr: PageCsrPipelines,
    pub finalize_telemetry_pipeline: Option<CachedComputePipelineId>,
    pub emit_raster_work_pipeline: Option<CachedComputePipelineId>,
    pub finalize_raster_dispatch_pipeline: Option<CachedComputePipelineId>,
    pub depth_reduce_bind_group_layout: BindGroupLayout,
    pub depth_reduce_pipeline: Option<CachedComputePipelineId>,
}

#[derive(Default)]
pub struct SegmentScatterPipelines {
    pub count: Option<CachedComputePipelineId>,
    pub prefix: Option<CachedComputePipelineId>,
    pub fill: Option<CachedComputePipelineId>,
}

#[derive(Default)]
pub struct PageCsrPipelines {
    pub prefix_candidates: Option<CachedComputePipelineId>,
    pub scatter_candidates: Option<CachedComputePipelineId>,
    pub build_pages: Option<CachedComputePipelineId>,
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
                Self::storage_entry(layouts::prepass::FRUSTUM_TABLE, true),
                Self::storage_entry(layouts::prepass::FROXEL_BUCKET_HEADS, false),
                Self::storage_entry(layouts::prepass::CHUNK_POOL, false),
                Self::storage_entry(layouts::prepass::FREE_HEADS, false),
                Self::storage_entry(layouts::prepass::RASTER_WORK_QUEUE, false),
                Self::storage_entry(layouts::prepass::RASTER_TILE_RUN_QUEUE, false),
                Self::storage_entry(layouts::prepass::RASTER_TILE_RUN_DISPATCH_ARGS, false),
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
                Self::storage_entry(layouts::prepass::FINE_CELL_WRITE_CURSORS, false),
                Self::storage_entry(layouts::prepass::FINE_SEG_REFS, false),
                Self::storage_entry(layouts::prepass::VSMS_REQUEST_META, true),
                Self::storage_entry(layouts::prepass::VSMS_REQUEST_BITS, false),
                Self::storage_entry(layouts::prepass::SHADOW_DOM_SURFACE_IDS, true),
                Self::storage_entry(layouts::prepass::BROAD_INSTANCE_META, false),
                Self::storage_entry(layouts::prepass::OPAQUE_FINE_DEPTH_TILES, true),
                Self::storage_entry(layouts::prepass::TELEMETRY, false),
                Self::storage_entry(layouts::prepass::PROJECTED_SEGMENTS, false),
                Self::storage_entry(layouts::prepass::PAGE_CANDIDATE_COUNTS, false),
                Self::storage_entry(layouts::prepass::PAGE_CANDIDATE_OFFSETS, false),
                Self::storage_entry(layouts::prepass::PAGE_CANDIDATE_CURSORS, false),
                Self::storage_entry(layouts::prepass::PAGE_CANDIDATES, false),
                Self::storage_entry(layouts::prepass::VIRTUAL_PAGE_CANDIDATE_COUNTS, false),
                Self::storage_entry(layouts::prepass::FRUSTUM_INSTANCE_KEEP_PROBABILITIES, false),
            ],
        )
    }

    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        let descriptor = Self::bind_group_layout_descriptor();
        device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries)
    }

    pub fn indirect_args_bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "strand_prepass_indirect_args_bind_group_layout",
            &[Self::storage_entry(layouts::prepass::INDIRECT_ARGS, false)],
        )
    }

    pub fn create_indirect_args_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        let descriptor = Self::indirect_args_bind_group_layout_descriptor();
        device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries)
    }

    pub fn depth_reduce_bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "strand_depth_reduce_bind_group_layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
                Self::storage_entry(1, true),
                Self::storage_entry(2, false),
            ],
        )
    }

    pub fn create_depth_reduce_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        let descriptor = Self::depth_reduce_bind_group_layout_descriptor();
        device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries)
    }
}

fn queue_prepass_pipeline(
    pipeline_cache: &PipelineCache,
    shader: Handle<Shader>,
    allocator: &GpuPagingAllocator,
    invocation_dims: &ComputeInvocationDims,
    entry_point: &'static str,
    uses_indirect_args_storage: bool,
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
            "DOM_PREFETCH_PAGE_BORDER".into(),
            crate::plugin::DOM_PREFETCH_PAGE_BORDER,
        ),
        ShaderDefVal::UInt(
            "COARSE_COUNT_PAGE_SIZE".into(),
            crate::plugin::COARSE_COUNT_PAGE_SIZE,
        ),
    ]);

    let mut max_group = allocator
        .buffer_group_idx
        .max(allocator.table_group_idx)
        .max(layouts::prepass::PREPASS_GROUP);
    if uses_indirect_args_storage {
        max_group = max_group.max(layouts::prepass::INDIRECT_ARGS_GROUP);
    }
    let mut layout = vec![bind_group_layout.clone(); (max_group + 1) as usize];
    layout[allocator.buffer_group_idx as usize] = buffer_layout;
    layout[allocator.table_group_idx as usize] = table_layout;
    layout[layouts::prepass::PREPASS_GROUP as usize] = bind_group_layout;
    if uses_indirect_args_storage {
        layout[layouts::prepass::INDIRECT_ARGS_GROUP as usize] =
            StrandPrepassPipeline::indirect_args_bind_group_layout_descriptor();
    }

    Some(
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_prepass_pipeline".into()),
            layout,
            shader,
            shader_defs,
            immediate_size: std::mem::size_of::<PushConstants>() as u32,
            entry_point: Some(entry_point.into()),
            zero_initialize_workgroup_memory: false,
        }),
    )
}

impl FromWorld for StrandPrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);
        let indirect_args_bind_group_layout = Self::create_indirect_args_bind_group_layout(device);
        let depth_reduce_bind_group_layout = Self::create_depth_reduce_bind_group_layout(device);

        let _invocation_dims = *world.resource::<ComputeInvocationDims>();

        StrandPrepassPipeline {
            bind_group_layout,
            indirect_args_bind_group_layout,
            broad_pipeline: None,
            broad_strand_pipeline: None,
            finalize_pipeline: None,
            fine_pipeline: None,
            coarse_interval_pipeline: None,
            build_depth_warp_pipeline: None,
            finalize_binning_pipeline: None,
            mark_coarse_count_pages_pipeline: None,
            allocate_coarse_count_pages_pipeline: None,
            finalize_prefix_dispatch_pipeline: None,
            segment_scatter: SegmentScatterPipelines::default(),
            page_csr: PageCsrPipelines::default(),
            finalize_telemetry_pipeline: None,
            emit_raster_work_pipeline: None,
            finalize_raster_dispatch_pipeline: None,
            depth_reduce_bind_group_layout,
            depth_reduce_pipeline: None,
        }
    }
}

pub fn create_prepass_bind_groups(
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
) -> Result<(BindGroup, BindGroup, BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;
    let prepass_queue = resources.prepass_queue.as_ref().ok_or(())?;
    let binning_queue = resources.binning_queue.as_ref().ok_or(())?;
    let visibility_flags_buffer = resources.visibility_flags_buffer.as_ref().ok_or(())?;
    let visible_geos_buffer = resources.visible_geos_buffer.as_ref().ok_or(())?;
    let geos_prefix_buffer = resources.geos_prefix_buffer.as_ref().ok_or(())?;
    let dispatch_args = resources.indirect_args.as_ref().ok_or(())?;
    let prefix_dispatch_args = resources.prefix_indirect_args.as_ref().ok_or(())?;
    let telemetry = resources.telemetry.as_ref().ok_or(())?;
    let projected_segments = resources.projected_segments.as_ref().ok_or(())?;
    let page_candidate_counts = resources.page_candidate_counts.as_ref().ok_or(())?;
    let page_candidate_offsets = resources.page_candidate_offsets.as_ref().ok_or(())?;
    let page_candidate_cursors = resources.page_candidate_cursors.as_ref().ok_or(())?;
    let page_candidates = resources.page_candidates.as_ref().ok_or(())?;
    let virtual_page_candidate_counts =
        resources.virtual_page_candidate_counts.as_ref().ok_or(())?;
    let frustum_instance_keep_probabilities = resources
        .frustum_instance_keep_probabilities
        .as_ref()
        .ok_or(())?;
    let frustum_table = resources.frustum_table.as_ref().ok_or(())?;
    let froxel_bucket_heads = resources.froxel_bucket_heads.as_ref().ok_or(())?;
    let chunk_pool = resources.chunk_pool.as_ref().ok_or(())?;
    let free_heads = resources.free_heads.as_ref().ok_or(())?;
    let raster_work_queue = resources.raster_work_queue.as_ref().ok_or(())?;
    let raster_tile_run_queue = resources.raster_tile_run_queue.as_ref().ok_or(())?;
    let raster_tile_run_dispatch_args =
        resources.raster_tile_run_dispatch_args.as_ref().ok_or(())?;
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
    let fine_cell_write_cursors = resources.fine_cell_write_cursors.as_ref().ok_or(())?;
    let fine_seg_refs = resources.fine_seg_refs.as_ref().ok_or(())?;
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
                    binding: layouts::prepass::RASTER_TILE_RUN_DISPATCH_ARGS,
                    resource: raster_tile_run_dispatch_args.as_entire_binding(),
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
                    binding: layouts::prepass::FINE_CELL_WRITE_CURSORS,
                    resource: fine_cell_write_cursors.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FINE_SEG_REFS,
                    resource: fine_seg_refs.as_entire_binding(),
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
                BindGroupEntry {
                    binding: layouts::prepass::OPAQUE_FINE_DEPTH_TILES,
                    resource: resources
                        .opaque_fine_depth_tiles
                        .as_ref()
                        .ok_or(())?
                        .as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::TELEMETRY,
                    resource: telemetry.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::PROJECTED_SEGMENTS,
                    resource: projected_segments.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::PAGE_CANDIDATE_COUNTS,
                    resource: page_candidate_counts.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::PAGE_CANDIDATE_OFFSETS,
                    resource: page_candidate_offsets.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::PAGE_CANDIDATE_CURSORS,
                    resource: page_candidate_cursors.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::PAGE_CANDIDATES,
                    resource: page_candidates.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::VIRTUAL_PAGE_CANDIDATE_COUNTS,
                    resource: virtual_page_candidate_counts.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::prepass::FRUSTUM_INSTANCE_KEEP_PROBABILITIES,
                    resource: frustum_instance_keep_probabilities.as_entire_binding(),
                },
            ],
        ),
        device.create_bind_group(
            Some("strand_prepass_indirect_args_bind_group"),
            &pipeline.indirect_args_bind_group_layout,
            &[BindGroupEntry {
                binding: layouts::prepass::INDIRECT_ARGS,
                resource: dispatch_args.as_entire_binding(),
            }],
        ),
        device.create_bind_group(
            Some("strand_prefix_indirect_args_bind_group"),
            &pipeline.indirect_args_bind_group_layout,
            &[BindGroupEntry {
                binding: layouts::prepass::INDIRECT_ARGS,
                resource: prefix_dispatch_args.as_entire_binding(),
            }],
        ),
        // Dynamic offsets order follows bind-group layout declaration order.
        vec![view_light_uniform_offset.offset, view_offsets.offset],
    ))
}

pub fn create_depth_reduce_bind_group(
    device: &RenderDevice,
    pipeline: &StrandPrepassPipeline,
    resources: &StrandPrepassResources,
    view_prepass_textures: &ViewPrepassTextures,
) -> Result<BindGroup, ()> {
    let depth_view = view_prepass_textures.depth_view().ok_or(())?;
    let frustum_table = resources.frustum_table.as_ref().ok_or(())?;
    let opaque_fine_depth_tiles = resources.opaque_fine_depth_tiles.as_ref().ok_or(())?;
    Ok(device.create_bind_group(
        Some("strand_depth_reduce_bind_group"),
        &pipeline.depth_reduce_bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(depth_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: frustum_table.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: opaque_fine_depth_tiles.as_entire_binding(),
            },
        ],
    ))
}

pub fn run_depth_reduce(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandPrepassPipeline,
    bind_group: &BindGroup,
    opaque_fine_depth_tiles: &Buffer,
    frustum_id: u32,
    fine_tile_count: u32,
) {
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = render_context.command_encoder();
    encoder.clear_buffer(opaque_fine_depth_tiles, 0, None);
    let Some(pipeline_id) = pipeline.depth_reduce_pipeline else {
        warn!("Depth reduce pipeline id not ready yet");
        return;
    };
    let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
        warn!("Depth reduce pipeline not found");
        return;
    };
    let pushconstants = PushConstants {
        num_elements: fine_tile_count,
        scan_load_base: frustum_id,
        ..Default::default()
    };
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Strand Depth Reduce"),
        ..default()
    });
    pass.set_pipeline(compute_pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&pushconstants));
    let span = diagnostics.time_span(&mut pass, "strand_prepass/depth_reduce");
    pass.dispatch_workgroups(fine_tile_count.max(1).div_ceil(64), 1, 1);
    span.end(&mut pass);
}

pub fn run_prepass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandPrepassPipeline,
    allocator: &GpuPagingAllocator,
    settings: &ComputeInvocationDims,
    cull_settings: &crate::resources::StochasticCullSettings,
    bind_group: &BindGroup,
    indirect_args_bind_group: &BindGroup,
    prefix_indirect_args_bind_group: &BindGroup,
    uniform_offsets: &[u32],
    indirect_args: &Buffer,
    prefix_indirect_args: &Buffer,
    telemetry: &Buffer,
    telemetry_enabled: bool,
    fine_binning_backend: FineBinningBackend,
    coarse_depth_lut: &Buffer,
    coarse_count_page_table: &Buffer,
    coarse_count_pages: &Buffer,
    fine_cell_write_cursors: &Buffer,
    page_candidate_counts: &Buffer,
    page_candidate_cursors: &Buffer,
    virtual_page_candidate_counts: &Buffer,
    fine_seg_refs: &Buffer,
    frustum_count: u32,
    instance_count: u32,
    max_strands_in_instance: u32,
    coarse_depth_tile_capacity: u32,
    coarse_count_page_capacity: u32,
    coarse_range_capacity: u32,
) {
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
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
    if fine_binning_backend == FineBinningBackend::PageCsr {
        encoder.clear_buffer(page_candidate_counts, 0, None);
        encoder.clear_buffer(page_candidate_cursors, 0, None);
        encoder.clear_buffer(virtual_page_candidate_counts, 0, None);
    }
    if telemetry_enabled {
        encoder.clear_buffer(telemetry, 0, None);
    }
    let _ = fine_cell_write_cursors;
    encoder.clear_buffer(fine_seg_refs, 0, Some(4));

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
    let Some(finalize_prefix_dispatch_pipeline_id) = pipeline.finalize_prefix_dispatch_pipeline
    else {
        warn!("Finalize prefix dispatch pipeline id not ready yet");
        return;
    };
    let Some(binning_pipeline_id) = pipeline.segment_scatter.count else {
        warn!("Binning queue pipeline id not ready yet");
        return;
    };
    let Some(prefix_fine_pages_pipeline_id) = pipeline.segment_scatter.prefix else {
        warn!("Prefix fine pages pipeline id not ready yet");
        return;
    };
    let Some(fill_fine_seg_refs_pipeline_id) = pipeline.segment_scatter.fill else {
        warn!("Fill fine seg refs pipeline id not ready yet");
        return;
    };
    let Some(prefix_page_candidates_pipeline_id) = pipeline.page_csr.prefix_candidates else {
        warn!("Prefix page candidates pipeline id not ready yet");
        return;
    };
    let Some(scatter_page_candidates_pipeline_id) = pipeline.page_csr.scatter_candidates else {
        warn!("Scatter page candidates pipeline id not ready yet");
        return;
    };
    let Some(build_fine_pages_csr_pipeline_id) = pipeline.page_csr.build_pages else {
        warn!("Build CSR fine pages pipeline id not ready yet");
        return;
    };
    let Some(finalize_telemetry_pipeline_id) = pipeline.finalize_telemetry_pipeline else {
        warn!("Finalize telemetry pipeline id not ready yet");
        return;
    };
    let Some(emit_raster_work_pipeline_id) = pipeline.emit_raster_work_pipeline else {
        warn!("Emit raster work pipeline id not ready yet");
        return;
    };
    let Some(finalize_raster_dispatch_pipeline_id) = pipeline.finalize_raster_dispatch_pipeline
    else {
        warn!("Finalize raster dispatch pipeline id not ready yet");
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
    let Some(finalize_prefix_dispatch_pipeline) =
        pipeline_cache.get_compute_pipeline(finalize_prefix_dispatch_pipeline_id)
    else {
        warn!("Finalize prefix dispatch pipeline not found");
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
    let Some(prefix_page_candidates_pipeline) =
        pipeline_cache.get_compute_pipeline(prefix_page_candidates_pipeline_id)
    else {
        warn!("Prefix page candidates pipeline not found");
        return;
    };
    let Some(scatter_page_candidates_pipeline) =
        pipeline_cache.get_compute_pipeline(scatter_page_candidates_pipeline_id)
    else {
        warn!("Scatter page candidates pipeline not found");
        return;
    };
    let Some(build_fine_pages_csr_pipeline) =
        pipeline_cache.get_compute_pipeline(build_fine_pages_csr_pipeline_id)
    else {
        warn!("Build CSR fine pages pipeline not found");
        return;
    };
    let Some(finalize_telemetry_pipeline) =
        pipeline_cache.get_compute_pipeline(finalize_telemetry_pipeline_id)
    else {
        warn!("Finalize telemetry pipeline not found");
        return;
    };
    let Some(emit_raster_work_pipeline) =
        pipeline_cache.get_compute_pipeline(emit_raster_work_pipeline_id)
    else {
        warn!("Emit raster work pipeline not found");
        return;
    };
    let Some(finalize_raster_dispatch_pipeline) =
        pipeline_cache.get_compute_pipeline(finalize_raster_dispatch_pipeline_id)
    else {
        warn!("Finalize raster dispatch pipeline not found");
        return;
    };
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Strand Prepass"),
        ..default()
    });
    let pushconstants = PushConstants {
        num_elements: instance_count,
        frustum_count,
        scan_save_base: TELEMETRY_PAGE_COUNTS_OFFSET_WORDS + coarse_count_page_capacity,
        stochastic_cull_enabled: u32::from(cull_settings.enabled),
        target_strands_per_pixel: cull_settings.target_strands_per_pixel,
        min_keep_probability: cull_settings.min_keep_probability,
        shadow_keep_probability: cull_settings.shadow_keep_probability,
        telemetry_enabled: u32::from(telemetry_enabled),
        fine_binning_backend: match fine_binning_backend {
            FineBinningBackend::SegmentScatter => 0,
            FineBinningBackend::PageCsr => 1,
        },
        ..Default::default()
    };
    pass.set_pipeline(broad_pipeline);
    pass.set_immediates(0, bytemuck::bytes_of(&pushconstants));
    pass.set_bind_group(allocator.buffer_group_idx, allocator_buffer_bind_group, &[]);
    pass.set_bind_group(
        allocator.table_group_idx,
        allocator_pagetable_bind_group,
        &[],
    );
    pass.set_bind_group(layouts::prepass::PREPASS_GROUP, bind_group, uniform_offsets);
    let span = diagnostics.time_span(&mut pass, "strand_prepass/broad_instances");
    pass.dispatch_workgroups(
        instance_count.div_ceil(settings.threads_per_workgroup),
        1,
        1,
    );
    span.end(&mut pass);
    let broad_strand_pushconstants = PushConstants {
        num_elements: max_strands_in_instance.max(1),
        scan_load_base: instance_count,
        ..pushconstants
    };
    pass.set_pipeline(broad_strand_pipeline);
    pass.set_immediates(0, bytemuck::bytes_of(&broad_strand_pushconstants));
    let span = diagnostics.time_span(&mut pass, "strand_prepass/broad_strands");
    pass.dispatch_workgroups(
        max_strands_in_instance
            .max(1)
            .div_ceil(settings.threads_per_workgroup),
        instance_count.max(1),
        1,
    );
    span.end(&mut pass);
    let fine_indirect_pushconstants = PushConstants {
        workgroup_offset: crate::plugin::MAX_COMPUTE_WORKGROUPS_PER_DIMENSION
            .saturating_mul(settings.threads_per_workgroup),
        ..pushconstants
    };
    pass.set_pipeline(finalize_pipeline);
    pass.set_bind_group(
        layouts::prepass::INDIRECT_ARGS_GROUP,
        indirect_args_bind_group,
        &[],
    );
    let span = diagnostics.time_span(&mut pass, "strand_prepass/finalize_fine_dispatch");
    pass.dispatch_workgroups(1, 1, 1);
    span.end(&mut pass);
    pass.set_pipeline(fine_pipeline);
    pass.set_immediates(0, bytemuck::bytes_of(&fine_indirect_pushconstants));
    let span = diagnostics.time_span(&mut pass, "strand_prepass/fine");
    pass.dispatch_workgroups_indirect(indirect_args, 0);
    span.end(&mut pass);
    let coarse_interval_pushconstants = PushConstants {
        num_elements: coarse_range_capacity,
        ..pushconstants
    };
    pass.set_pipeline(coarse_interval_pipeline);
    pass.set_immediates(0, bytemuck::bytes_of(&coarse_interval_pushconstants));
    let span = diagnostics.time_span(&mut pass, "strand_prepass/coarse_intervals");
    pass.dispatch_workgroups(coarse_range_capacity, 1, 1);
    span.end(&mut pass);
    let depth_warp_pushconstants = PushConstants {
        num_elements: coarse_depth_tile_capacity,
        ..pushconstants
    };
    pass.set_pipeline(build_depth_warp_pipeline);
    pass.set_immediates(0, bytemuck::bytes_of(&depth_warp_pushconstants));
    let span = diagnostics.time_span(&mut pass, "strand_prepass/depth_warp");
    pass.dispatch_workgroups(
        coarse_depth_tile_capacity.div_ceil(settings.threads_per_workgroup),
        1,
        1,
    );
    span.end(&mut pass);
    pass.set_pipeline(finalize_binning_pipeline);
    pass.set_bind_group(
        layouts::prepass::INDIRECT_ARGS_GROUP,
        indirect_args_bind_group,
        &[],
    );
    let span = diagnostics.time_span(&mut pass, "strand_prepass/finalize_binning_dispatch");
    pass.dispatch_workgroups(1, 1, 1);
    span.end(&mut pass);
    pass.set_pipeline(mark_coarse_count_pages_pipeline);
    pass.set_immediates(0, bytemuck::bytes_of(&fine_indirect_pushconstants));
    let span = diagnostics.time_span(&mut pass, "strand_prepass/mark_count_pages");
    pass.dispatch_workgroups_indirect(indirect_args, 0);
    span.end(&mut pass);
    let coarse_count_page_table_capacity =
        coarse_depth_tile_capacity.saturating_mul(crate::plugin::COARSE_DEPTH_SLICES);
    let allocate_count_pages_pushconstants = PushConstants {
        num_elements: coarse_count_page_table_capacity,
        ..pushconstants
    };
    pass.set_pipeline(allocate_coarse_count_pages_pipeline);
    pass.set_immediates(0, bytemuck::bytes_of(&allocate_count_pages_pushconstants));
    let span = diagnostics.time_span(&mut pass, "strand_prepass/allocate_count_pages");
    pass.dispatch_workgroups(
        coarse_count_page_table_capacity.div_ceil(settings.threads_per_workgroup),
        1,
        1,
    );
    span.end(&mut pass);
    pass.set_pipeline(finalize_prefix_dispatch_pipeline);
    pass.set_bind_group(
        layouts::prepass::INDIRECT_ARGS_GROUP,
        prefix_indirect_args_bind_group,
        &[],
    );
    let span = diagnostics.time_span(&mut pass, "strand_prepass/finalize_prefix_dispatch");
    pass.dispatch_workgroups(1, 1, 1);
    span.end(&mut pass);
    // The next indirect dispatch must not have its argument buffer bound as storage.
    pass.set_bind_group(
        layouts::prepass::INDIRECT_ARGS_GROUP,
        indirect_args_bind_group,
        &[],
    );
    match fine_binning_backend {
        FineBinningBackend::SegmentScatter => {
            pass.set_pipeline(binning_pipeline);
            pass.set_immediates(0, bytemuck::bytes_of(&fine_indirect_pushconstants));
            let span = diagnostics.time_span(&mut pass, "strand_prepass/binning");
            pass.dispatch_workgroups_indirect(indirect_args, 0);
            span.end(&mut pass);
            pass.set_pipeline(prefix_fine_pages_pipeline);
            let span = diagnostics.time_span(&mut pass, "strand_prepass/prefix_fine_pages");
            pass.dispatch_workgroups_indirect(prefix_indirect_args, 0);
            span.end(&mut pass);
            pass.set_pipeline(fill_fine_seg_refs_pipeline);
            pass.set_immediates(0, bytemuck::bytes_of(&fine_indirect_pushconstants));
            let span = diagnostics.time_span(&mut pass, "strand_prepass/fill_segment_refs");
            pass.dispatch_workgroups_indirect(indirect_args, 0);
            span.end(&mut pass);
        }
        FineBinningBackend::PageCsr => {
            pass.set_pipeline(prefix_page_candidates_pipeline);
            let span =
                diagnostics.time_span(&mut pass, "strand_prepass/page_csr/prefix_candidates");
            pass.dispatch_workgroups(1, 1, 1);
            span.end(&mut pass);

            pass.set_pipeline(scatter_page_candidates_pipeline);
            pass.set_immediates(0, bytemuck::bytes_of(&fine_indirect_pushconstants));
            let span =
                diagnostics.time_span(&mut pass, "strand_prepass/page_csr/scatter_candidates");
            pass.dispatch_workgroups_indirect(indirect_args, 0);
            span.end(&mut pass);

            pass.set_pipeline(build_fine_pages_csr_pipeline);
            let span = diagnostics.time_span(&mut pass, "strand_prepass/page_csr/build_fine_pages");
            pass.dispatch_workgroups_indirect(prefix_indirect_args, 0);
            span.end(&mut pass);
        }
    }
    if telemetry_enabled {
        pass.set_pipeline(finalize_telemetry_pipeline);
        pass.set_immediates(0, bytemuck::bytes_of(&allocate_count_pages_pushconstants));
        pass.dispatch_workgroups(
            coarse_count_page_table_capacity.div_ceil(settings.threads_per_workgroup),
            1,
            1,
        );
    }
    let fine_tile_stack_pushconstants = PushConstants {
        num_elements: coarse_depth_tile_capacity
            .saturating_mul(crate::plugin::COARSE_FINE_TILE_EXTENT)
            .saturating_mul(crate::plugin::COARSE_FINE_TILE_EXTENT),
        ..pushconstants
    };
    let fine_tile_stack_count = fine_tile_stack_pushconstants.num_elements.max(1);
    let fine_tile_stack_workgroups_x = fine_tile_stack_count.min(65_535).max(1);
    let fine_tile_stack_workgroups_y = fine_tile_stack_count
        .div_ceil(fine_tile_stack_workgroups_x)
        .max(1);
    pass.set_pipeline(emit_raster_work_pipeline);
    pass.set_immediates(0, bytemuck::bytes_of(&fine_tile_stack_pushconstants));
    let span = diagnostics.time_span(&mut pass, "strand_prepass/emit_raster_work");
    pass.dispatch_workgroups(
        fine_tile_stack_workgroups_x,
        fine_tile_stack_workgroups_y,
        1,
    );
    span.end(&mut pass);
    pass.set_pipeline(finalize_raster_dispatch_pipeline);
    let span = diagnostics.time_span(&mut pass, "strand_prepass/finalize_raster_dispatch");
    pass.dispatch_workgroups(1, 1, 1);
    span.end(&mut pass);
    drop(pass);

    if telemetry_enabled {
        const FIELDS: [(&str, u64); 6] = [
            ("allocated_pages", 0),
            ("binning_tasks", 1),
            ("page_candidates", 2),
            ("fine_cell_hits", 3),
            ("active_candidate_pages", 4),
            ("max_candidates_per_page", 5),
        ];
        for (name, word) in FIELDS {
            diagnostics.record_u32(
                encoder,
                &telemetry.slice(word * 4..word * 4 + 4),
                format!("strand_prepass/telemetry/{name}"),
            );
        }
        for bin in 0..TELEMETRY_HISTOGRAM_BINS {
            let word = u64::from(TELEMETRY_HISTOGRAM_OFFSET_WORDS + bin);
            diagnostics.record_u32(
                encoder,
                &telemetry.slice(word * 4..word * 4 + 4),
                format!("strand_prepass/telemetry/candidate_histogram/{bin}"),
            );
        }
        const FRUSTUM_FIELDS: [(&str, u64); FRUSTUM_TELEMETRY_STRIDE_WORDS as usize] = [
            ("visible_instances", 0),
            ("visible_strands", 1),
            ("retained_strands", 2),
            ("emitted_segments", 3),
            ("allocated_pages", 4),
            ("page_candidates", 5),
            ("fine_refs", 6),
            ("raster_runs", 7),
            ("raster_load", 8),
            ("max_page_candidates", 9),
            ("max_raster_load", 10),
        ];
        let frustum_base =
            u64::from(TELEMETRY_PAGE_COUNTS_OFFSET_WORDS + coarse_count_page_capacity);
        for frustum_id in 0..frustum_count {
            for (name, field) in FRUSTUM_FIELDS {
                let word = frustum_base
                    + u64::from(frustum_id) * u64::from(FRUSTUM_TELEMETRY_STRIDE_WORDS)
                    + field;
                diagnostics.record_u32(
                    encoder,
                    &telemetry.slice(word * 4..word * 4 + 4),
                    format!("strand_prepass/telemetry/frustum/{frustum_id}/{name}"),
                );
            }
        }
    }
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

    let prepass_shader_path = crate::plugin::embedded_shader_path("strand_prepass.wgsl");
    let broad_shader = shader_loader.load(prepass_shader_path.clone());
    let broad_strand_shader = shader_loader.load(prepass_shader_path.clone());
    let finalize_shader = shader_loader.load(prepass_shader_path.clone());
    let fine_shader = shader_loader.load(prepass_shader_path.clone());
    let coarse_interval_shader = shader_loader.load(prepass_shader_path.clone());
    let build_depth_warp_shader = shader_loader.load(prepass_shader_path.clone());
    let finalize_binning_shader = shader_loader.load(prepass_shader_path.clone());
    let mark_coarse_count_pages_shader = shader_loader.load(prepass_shader_path.clone());
    let allocate_coarse_count_pages_shader = shader_loader.load(prepass_shader_path.clone());
    let finalize_prefix_dispatch_shader = shader_loader.load(prepass_shader_path.clone());
    let binning_shader = shader_loader.load(prepass_shader_path.clone());
    let prefix_fine_pages_shader = shader_loader.load(prepass_shader_path.clone());
    let prefix_page_candidates_shader = shader_loader.load(prepass_shader_path.clone());
    let scatter_page_candidates_shader = shader_loader.load(prepass_shader_path.clone());
    let build_fine_pages_csr_shader = shader_loader.load(prepass_shader_path.clone());
    let finalize_telemetry_shader = shader_loader.load(prepass_shader_path.clone());
    let fill_fine_seg_refs_shader = shader_loader.load(prepass_shader_path.clone());
    let emit_raster_work_shader = shader_loader.load(prepass_shader_path.clone());
    let finalize_raster_dispatch_shader = shader_loader.load(prepass_shader_path);
    let depth_reduce_shader = shader_loader.load(crate::plugin::embedded_shader_path(
        "opaque_depth_reduce.wgsl",
    ));

    let Some(broad_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        broad_shader,
        &allocator,
        &dims,
        "broad_prepass",
        false,
    ) else {
        warn!("Could not queue broad prepass pipeline");
        return;
    };
    let Some(broad_strand_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        broad_strand_shader,
        &allocator,
        &dims,
        "broad_strand_prepass",
        false,
    ) else {
        warn!("Could not queue broad strand prepass pipeline");
        return;
    };
    let Some(finalize_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        finalize_shader,
        &allocator,
        &dims,
        "finalize_prepass",
        true,
    ) else {
        return;
    };
    let Some(fine_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        fine_shader,
        &allocator,
        &dims,
        "fine_prepass",
        false,
    ) else {
        return;
    };
    let Some(coarse_interval_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        coarse_interval_shader,
        &allocator,
        &dims,
        "coarse_interval_pass",
        false,
    ) else {
        return;
    };
    let Some(build_depth_warp_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        build_depth_warp_shader,
        &allocator,
        &dims,
        "build_depth_warp_lut",
        false,
    ) else {
        return;
    };
    let Some(finalize_binning_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        finalize_binning_shader,
        &allocator,
        &dims,
        "finalize_binning",
        true,
    ) else {
        return;
    };
    let Some(mark_coarse_count_pages_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        mark_coarse_count_pages_shader,
        &allocator,
        &dims,
        "mark_coarse_count_pages_pass",
        false,
    ) else {
        return;
    };
    let Some(allocate_coarse_count_pages_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        allocate_coarse_count_pages_shader,
        &allocator,
        &dims,
        "allocate_coarse_count_pages",
        false,
    ) else {
        return;
    };
    let Some(finalize_prefix_dispatch_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        finalize_prefix_dispatch_shader,
        &allocator,
        &dims,
        "finalize_prefix_dispatch",
        true,
    ) else {
        return;
    };
    let Some(binning_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        binning_shader,
        &allocator,
        &dims,
        "binning_queue_pass",
        false,
    ) else {
        return;
    };
    let Some(prefix_fine_pages_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        prefix_fine_pages_shader,
        &allocator,
        &dims,
        "prefix_fine_pages",
        false,
    ) else {
        return;
    };
    let Some(prefix_page_candidates_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        prefix_page_candidates_shader,
        &allocator,
        &dims,
        "prefix_page_candidates",
        false,
    ) else {
        return;
    };
    let Some(scatter_page_candidates_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        scatter_page_candidates_shader,
        &allocator,
        &dims,
        "scatter_page_candidates",
        false,
    ) else {
        return;
    };
    let Some(build_fine_pages_csr_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        build_fine_pages_csr_shader,
        &allocator,
        &dims,
        "build_fine_pages_csr",
        false,
    ) else {
        return;
    };
    let Some(finalize_telemetry_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        finalize_telemetry_shader,
        &allocator,
        &dims,
        "finalize_telemetry",
        false,
    ) else {
        return;
    };
    let Some(fill_fine_seg_refs_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        fill_fine_seg_refs_shader,
        &allocator,
        &dims,
        "fill_fine_seg_refs",
        false,
    ) else {
        return;
    };
    let Some(emit_raster_work_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        emit_raster_work_shader,
        &allocator,
        &dims,
        "emit_raster_work",
        false,
    ) else {
        return;
    };
    let Some(finalize_raster_dispatch_pipeline_id) = queue_prepass_pipeline(
        &pipeline_cache,
        finalize_raster_dispatch_shader,
        &allocator,
        &dims,
        "finalize_raster_dispatch",
        false,
    ) else {
        return;
    };
    let depth_reduce_pipeline_id =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_depth_reduce_pipeline".into()),
            layout: vec![StrandPrepassPipeline::depth_reduce_bind_group_layout_descriptor()],
            shader: depth_reduce_shader,
            shader_defs: vec![ShaderDefVal::UInt("WORKGROUP_SIZE".into(), 64)],
            immediate_size: std::mem::size_of::<PushConstants>() as u32,
            entry_point: Some("reduce_opaque_depth".into()),
            zero_initialize_workgroup_memory: false,
        });
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
    pipeline_res.finalize_prefix_dispatch_pipeline = Some(finalize_prefix_dispatch_pipeline_id);
    pipeline_res.segment_scatter.count = Some(binning_pipeline_id);
    pipeline_res.segment_scatter.prefix = Some(prefix_fine_pages_pipeline_id);
    pipeline_res.page_csr.prefix_candidates = Some(prefix_page_candidates_pipeline_id);
    pipeline_res.page_csr.scatter_candidates = Some(scatter_page_candidates_pipeline_id);
    pipeline_res.page_csr.build_pages = Some(build_fine_pages_csr_pipeline_id);
    pipeline_res.finalize_telemetry_pipeline = Some(finalize_telemetry_pipeline_id);
    pipeline_res.segment_scatter.fill = Some(fill_fine_seg_refs_pipeline_id);
    pipeline_res.emit_raster_work_pipeline = Some(emit_raster_work_pipeline_id);
    pipeline_res.finalize_raster_dispatch_pipeline = Some(finalize_raster_dispatch_pipeline_id);
    pipeline_res.depth_reduce_pipeline = Some(depth_reduce_pipeline_id);
    debug!(
        "Rebuilt strand prepass pipelines: broad={:?} broad_strand={:?} finalize={:?} fine={:?} coarse_interval={:?} depth_warp={:?} finalize_binning={:?} mark_pages={:?} allocate_pages={:?} finalize_prefix={:?} binning={:?} prefix_pages={:?} fill_refs={:?} emit_work={:?} finalize_raster_dispatch={:?}",
        broad_pipeline_id,
        broad_strand_pipeline_id,
        finalize_pipeline_id,
        fine_pipeline_id,
        coarse_interval_pipeline_id,
        build_depth_warp_pipeline_id,
        finalize_binning_pipeline_id,
        mark_coarse_count_pages_pipeline_id,
        allocate_coarse_count_pages_pipeline_id,
        finalize_prefix_dispatch_pipeline_id,
        binning_pipeline_id,
        prefix_fine_pages_pipeline_id,
        fill_fine_seg_refs_pipeline_id,
        emit_raster_work_pipeline_id,
        finalize_raster_dispatch_pipeline_id
    );
}
