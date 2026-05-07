use bevy::{
    core_pipeline::core_3d::graph::Core3d,
    math::bounding::Aabb3d,
    pbr::ExtractedDirectionalLight,
    platform::collections::HashMap,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::ExtractComponentPlugin,
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssets,
        render_graph::RenderGraphExt,
        render_resource::Buffer,
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
    },
    transform::TransformSystems,
    window::WindowResized,
};
use bevy_vsms::{
    allocator::VirtualSurfaceRuntime,
    prelude::{VirtualSurfaceConfigBuilder, VirtualSurfaceKind, VirtualSurfaceResidencyMode},
};
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use wgpu::{BufferDescriptor, BufferUsages};

use crate::{
    allocator::*,
    components::*,
    dson::DsonAsset,
    nodes,
    pipelines::{
        composite::*,
        prepass::*,
        raster::*,
        shading::*,
        shadows::*,
        task_contract::{
            BINNING_POOL_CHUNK_SIZE, BINNING_POOL_MAX_CHUNKS, BINNING_POOL_MIN_CHUNKS,
            BINNING_POOL_NUM_HEADS, FinePageMeta, FineSegRef, QUEUE_HEADER_WORDS, RasterWorkItem,
        },
        tile_debug::*,
    },
    resources::*,
    shader_types::*,
};

pub const MAX_TEXTURE_EXTENT: u32 = 8192; // for shading (TODO: get this from device limits)
pub const COARSE_FINE_TILE_EXTENT: u32 = 4;
pub const COARSE_DEPTH_SLICES: u32 = 16;
pub const COARSE_MAX_SLICES_PER_ASSET_INTERVAL: u32 = 2;
pub const COARSE_COUNT_PAGE_SIZE: u32 =
    COARSE_FINE_TILE_EXTENT * COARSE_FINE_TILE_EXTENT * COARSE_DEPTH_SLICES;
const COARSE_COUNT_PAGE_CAPACITY_DIVISOR: u32 = 2;
const COARSE_INTERVAL_REFS_PER_TILE: u32 = 64;

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
enum DomVsmsProxyKind {
    Opacity,
    Depth,
}

use lazy_static::lazy_static;

lazy_static! {
    static ref BIND_MAP: HashMap<SlabKind, u32> = [
        (SlabKind::Vert, 0),
        (SlabKind::Index, 1),
        (SlabKind::StrandMaterial, 5),
        (SlabKind::StrandGeo, 6),
        (SlabKind::StrandMeta, 7),
    ]
    .iter()
    .copied()
    .collect();
}

lazy_static! {
    static ref LABEL_MAP: HashMap<SlabKind, &'static str> = [
        (SlabKind::Vert, "VERTICES"),
        (SlabKind::Index, "INDICES"),
        (SlabKind::StrandMaterial, "STRAND_MATERIALS"),
        (SlabKind::StrandGeo, "STRAND_GEOS"),
        (SlabKind::StrandMeta, "STRAND_METADATA"),
    ]
    .iter()
    .copied()
    .collect();
}

pub struct StrandRasterizerPlugin;

impl Plugin for StrandRasterizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<FroxelConfig>::default(),
            ExtractComponentPlugin::<StrandGeometry>::default(),
            ExtractComponentPlugin::<StrandInstanceTransform>::default(),
            ExtractComponentPlugin::<StrandMaterial>::default(),
            ExtractResourcePlugin::<TileDebugSettings>::default(),
            ExtractResourcePlugin::<StochasticCullSettings>::default(),
            GpuPagingAllocatorPlugin,
        ));
        app.init_resource::<TileDebugSettings>();
        app.init_resource::<StochasticCullSettings>();
        app.init_resource::<StrandAssetResources>();
        app.add_systems(
            Update,
            (
                set_strand_geometry,
                flag_realloc_on_view_change,
                flag_realloc_on_config_change,
                flag_realloc_on_tie_change,
                tie_view_to_froxel_config,
            ),
        );
        app.add_systems(
            PostUpdate,
            sync_strand_instance_transforms.after(TransformSystems::Propagate),
        );
    }
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.insert_resource(GpuPagingAllocatorSettings {
            label_map: LABEL_MAP.clone(),
            bind_map: BIND_MAP.clone(),
            buffer_group_idx: 0,
            table_group_idx: 1,
        });
        render_app.init_resource::<StrandRasterizerResources>();
        render_app.init_resource::<StrandShadowResources>();
        render_app.init_resource::<StrandPrepassResources>();
        render_app.init_resource::<ComputeInvocationDims>();
        render_app.init_resource::<StrandPrepassPipeline>();
        render_app.init_resource::<TileDebugPipeline>();
        render_app.add_systems(
            Render,
            ((
                use_froxel_buffer,
                use_prepass_buffers,
                update_material_buffer,
                // use_strand_geometry.after(prepare_view_uniforms),
            )
                .chain()
                .in_set(RenderSystems::Prepare),),
        );
        render_app.add_systems(
            Render,
            update_strand_prepass_pipeline.after(RenderSystems::PrepareBindGroups),
        );
        render_app
            .add_render_graph_node::<nodes::prepass::WorkPreparationNode>(
                Core3d,
                nodes::prepass::WorkPreparationLabel,
            )
            .add_render_graph_node::<nodes::debug::TileDebugNode>(
                Core3d,
                nodes::debug::TileDebugLabel,
            )
            .add_render_graph_edge(
                Core3d,
                bevy::core_pipeline::core_3d::graph::Node3d::StartMainPass,
                nodes::prepass::WorkPreparationLabel,
            )
            .add_render_graph_edge(
                Core3d,
                nodes::prepass::WorkPreparationLabel,
                nodes::debug::TileDebugLabel,
            )
            // Keep the debug overlay after the standard 3D main pass while the strand raster,
            // shading, composition, and shadow passes are disconnected during the hierarchy rewrite.
            .add_render_graph_edge(
                Core3d,
                bevy::core_pipeline::core_3d::graph::Node3d::EndMainPass,
                nodes::debug::TileDebugLabel,
            );
        // Disabled transition graph:
        // WorkPreparation -> Shading -> Raster -> Composition -> Debug
        // WorkPreparation -> ShadowRaster
        //
        // The current migration target is:
        // WorkPreparation -> Debug -> PostProcessing
        // .add_render_graph_edge(
        //     Core3d,
        //     nodes::debug::TileDebugLabel,
        //     bevy::core_pipeline::core_3d::graph::Node3d::PostProcessing,
        // );
    }
}

fn use_prepass_buffers(
    geometry_query: Query<(Entity, &StrandGeometry, Option<&StrandInstanceTransform>)>,
    light_frusta_query: Query<Entity, With<ExtractedDirectionalLight>>,
    storage_buffers: Res<RenderAssets<GpuVirtualShaderStorageBuffer>>,
    device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    mut prepass_resources: ResMut<StrandPrepassResources>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
    mut shadow_resources: ResMut<StrandShadowResources>,
) {
    let mut total_strands = 0u32;
    let mut total_segment_budget = 0u32;
    let mut instances = Vec::new();
    let mut sorted_geometry: Vec<_> = geometry_query.iter().collect();
    sorted_geometry.sort_by_key(|(entity, _, _)| entity.index());
    for (entity, geom, transform) in sorted_geometry {
        let Some(asset_id) = strand_asset_table_id(entity, geom, &storage_buffers) else {
            continue;
        };
        total_strands = total_strands.saturating_add(geom.strand_count);
        total_segment_budget = total_segment_budget.saturating_add(
            geom.strand_count
                .saturating_mul(geom.max_segments_in_strand),
        );
        let transform = transform.copied().unwrap_or_default();
        instances.push(StrandInstance {
            asset_id,
            material_id: asset_id,
            pad0: 0,
            pad1: 0,
            world_from_local: transform.world_from_local,
            local_from_world: transform.local_from_world,
        });
    }

    if total_strands == 0 {
        return;
    }

    raster_resources.strand_count = Some(total_strands);

    let prepass_capacity = total_strands.next_power_of_two().max(2048);
    let base_binning_capacity = total_segment_budget
        .next_power_of_two()
        .max(prepass_capacity);
    let instance_count = instances.len() as u32;
    let instance_capacity = instance_count.next_power_of_two().max(2048);
    let mut frustum_descs: Vec<GpuFrustumDesc> = Vec::new();
    let mut bucket_base = 0u32;
    let mut coarse_depth_tile_base = 0u32;
    let light_entities: HashSet<Entity> = light_frusta_query.iter().collect();
    raster_resources.frustum_ids.clear();
    shadow_resources.light_layer_by_frustum.clear();
    let mut light_layer: u32 = 0;
    let mut frusta: Vec<_> = raster_resources
        .frustrum_config
        .iter()
        .map(|(entity, cfg)| (*entity, cfg.clone()))
        .collect();
    frusta.sort_by_key(|(entity, _)| entity.index());
    for (frustum_id, (entity, cfg)) in frusta.into_iter().enumerate() {
        raster_resources
            .frustum_ids
            .insert(entity, frustum_id as u32);
        let is_light = light_entities.contains(&entity);
        if is_light {
            shadow_resources
                .light_layer_by_frustum
                .insert(frustum_id as u32, light_layer);
            light_layer = light_layer.saturating_add(1);
        }
        let (fine_tiles_x, fine_tiles_y, bucket_count) = cfg.get_num_tiles();
        let coarse_tiles_x = fine_tiles_x.div_ceil(COARSE_FINE_TILE_EXTENT).max(1);
        let coarse_tiles_y = fine_tiles_y.div_ceil(COARSE_FINE_TILE_EXTENT).max(1);
        let coarse_depth_tile_count = coarse_tiles_x.saturating_mul(coarse_tiles_y);
        frustum_descs.push(GpuFrustumDesc {
            screen_width: cfg.screen_width,
            screen_height: cfg.screen_height,
            froxel_size_x: cfg.froxel_size_x,
            froxel_size_y: cfg.froxel_size_y,
            depth_slices: cfg.depth_slices,
            bucket_base,
            bucket_count,
            kind: u32::from(is_light), // 0 = camera, 1 = light
            coarse_depth_tile_base,
            coarse_depth_tile_count,
            coarse_tiles_x,
            coarse_tiles_y,
        });
        bucket_base = bucket_base.saturating_add(bucket_count);
        coarse_depth_tile_base = coarse_depth_tile_base.saturating_add(coarse_depth_tile_count);
    }
    if frustum_descs.is_empty() {
        frustum_descs.push(GpuFrustumDesc {
            screen_width: 1,
            screen_height: 1,
            froxel_size_x: 1,
            froxel_size_y: 1,
            depth_slices: 1,
            bucket_base: 0,
            bucket_count: 1,
            kind: 0,
            coarse_depth_tile_base: 0,
            coarse_depth_tile_count: 1,
            coarse_tiles_x: 1,
            coarse_tiles_y: 1,
        });
        bucket_base = 1;
        coarse_depth_tile_base = 1;
    }
    let frustum_task_multiplier = (frustum_descs.len() as u32).max(1);
    let binning_capacity = base_binning_capacity
        .saturating_mul(frustum_task_multiplier)
        .next_power_of_two()
        .max(base_binning_capacity);
    let frustum_capacity = (frustum_descs.len() as u32).next_power_of_two().max(1);
    let froxel_bucket_capacity = bucket_base.next_power_of_two().max(1024);
    let coarse_depth_tile_capacity = coarse_depth_tile_base.next_power_of_two().max(1);
    let coarse_count_page_table_capacity = coarse_depth_tile_capacity
        .saturating_mul(COARSE_DEPTH_SLICES)
        .next_power_of_two()
        .max(1);
    let coarse_range_capacity = instance_capacity
        .saturating_mul(frustum_capacity)
        .next_power_of_two()
        .max(1);
    let coarse_interval_ref_capacity = coarse_depth_tile_capacity
        .saturating_mul(COARSE_INTERVAL_REFS_PER_TILE)
        .next_power_of_two()
        .max(coarse_range_capacity);
    let coarse_count_page_capacity = coarse_count_page_table_capacity
        .div_ceil(COARSE_COUNT_PAGE_CAPACITY_DIVISOR)
        .max(1);
    let fine_cell_capacity = coarse_count_page_capacity.saturating_mul(COARSE_COUNT_PAGE_SIZE);
    let fine_seg_ref_capacity = binning_capacity
        .saturating_mul(4)
        .next_power_of_two()
        .max(binning_capacity.max(1024));
    let raster_work_capacity = binning_capacity
        .saturating_mul(2)
        .next_power_of_two()
        .max(binning_capacity.max(1024));

    let needs_realloc = prepass_resources.prepass_queue.is_none()
        || prepass_resources.binning_queue.is_none()
        || prepass_resources.visibility_flags_buffer.is_none()
        || prepass_resources.visible_geos_buffer.is_none()
        || prepass_resources.geos_prefix_buffer.is_none()
        || prepass_resources.indirect_args.is_none()
        || prepass_resources.chunk_pool.is_none()
        || prepass_resources.free_heads.is_none()
        || prepass_resources.frustum_table.is_none()
        || prepass_resources.froxel_bucket_heads.is_none()
        || prepass_resources.raster_work_queue.is_none()
        || prepass_resources.coarse_depth_lut.is_none()
        || prepass_resources.coarse_range_queue.is_none()
        || prepass_resources.coarse_interval_heads.is_none()
        || prepass_resources.coarse_interval_refs.is_none()
        || prepass_resources.coarse_range_lookup.is_none()
        || prepass_resources.coarse_count_page_table.is_none()
        || prepass_resources.coarse_count_pages.is_none()
        || prepass_resources.fine_page_meta.is_none()
        || prepass_resources.fine_cell_offsets.is_none()
        || prepass_resources.fine_cell_write_cursors.is_none()
        || prepass_resources.fine_seg_refs.is_none()
        || prepass_resources.coarse_tile_work_counts.is_none()
        || prepass_resources.coarse_tile_work_offsets.is_none()
        || prepass_resources.strand_instances.is_none()
        || prepass_resources.prepass_task_capacity < prepass_capacity
        || prepass_resources.binning_task_capacity < binning_capacity
        || prepass_resources.instance_capacity < instance_capacity
        || prepass_resources.frustum_capacity < frustum_capacity
        || prepass_resources.froxel_bucket_capacity < froxel_bucket_capacity
        || prepass_resources.raster_work_capacity < raster_work_capacity
        || prepass_resources.coarse_depth_tile_capacity < coarse_depth_tile_capacity
        || prepass_resources.coarse_range_capacity < coarse_range_capacity
        || prepass_resources.coarse_interval_ref_capacity < coarse_interval_ref_capacity
        || prepass_resources.coarse_count_page_capacity < coarse_count_page_capacity
        || prepass_resources.fine_seg_ref_capacity < fine_seg_ref_capacity;

    if needs_realloc {
        let prepass_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (prepass_capacity as u64) * (std::mem::size_of::<FinePrepassTask>() as u64);
        let binning_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (binning_capacity as u64) * (std::mem::size_of::<BinningTask>() as u64);
        let instance_id_bytes = (instance_capacity as u64) * (std::mem::size_of::<u32>() as u64);
        let instance_prefix_bytes =
            ((instance_capacity as u64) + 1) * (std::mem::size_of::<u32>() as u64);
        let strand_instances_bytes =
            (instance_capacity as u64) * (std::mem::size_of::<StrandInstance>() as u64);
        let chunk_count = raster_work_capacity
            .max(BINNING_POOL_MIN_CHUNKS)
            .min(BINNING_POOL_MAX_CHUNKS);
        let chunk_stride_bytes =
            (2u64 + BINNING_POOL_CHUNK_SIZE as u64) * std::mem::size_of::<u32>() as u64;
        let chunk_pool_bytes = chunk_count as u64 * chunk_stride_bytes;
        let frustum_table_bytes =
            (frustum_capacity as u64) * (std::mem::size_of::<GpuFrustumDesc>() as u64);
        let froxel_bucket_heads_bytes =
            (froxel_bucket_capacity as u64) * (std::mem::size_of::<u32>() as u64);
        let raster_work_queue_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (raster_work_capacity as u64) * (std::mem::size_of::<RasterWorkItem>() as u64);
        let coarse_depth_lut_bytes = (coarse_depth_tile_capacity as u64)
            * (COARSE_DEPTH_SLICES as u64)
            * (std::mem::size_of::<GpuCoarseDepthLutEntry>() as u64);
        let coarse_range_queue_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (coarse_range_capacity as u64) * (std::mem::size_of::<GpuCoarseAssetRange>() as u64);
        let coarse_interval_heads_bytes =
            (coarse_depth_tile_capacity as u64) * (std::mem::size_of::<u32>() as u64);
        let coarse_interval_refs_bytes = std::mem::size_of::<u32>() as u64
            + (coarse_interval_ref_capacity as u64)
                * (std::mem::size_of::<GpuCoarseIntervalRef>() as u64);
        let coarse_range_lookup_bytes =
            (coarse_range_capacity as u64) * std::mem::size_of::<u32>() as u64;
        let coarse_count_page_table_bytes =
            (coarse_count_page_table_capacity as u64) * std::mem::size_of::<u32>() as u64;
        let coarse_count_pages_bytes = std::mem::size_of::<u32>() as u64
            + (coarse_count_page_capacity as u64)
                * (COARSE_COUNT_PAGE_SIZE as u64)
                * std::mem::size_of::<u32>() as u64;
        let fine_page_meta_bytes =
            (coarse_count_page_capacity as u64) * std::mem::size_of::<FinePageMeta>() as u64;
        let fine_cell_offsets_bytes =
            (fine_cell_capacity as u64) * std::mem::size_of::<u32>() as u64;
        let fine_cell_write_cursors_bytes =
            (fine_cell_capacity as u64) * std::mem::size_of::<u32>() as u64;
        let fine_seg_refs_bytes = std::mem::size_of::<u32>() as u64
            + (fine_seg_ref_capacity as u64) * std::mem::size_of::<FineSegRef>() as u64;
        let coarse_tile_work_counts_bytes =
            (coarse_depth_tile_capacity as u64) * std::mem::size_of::<u32>() as u64;
        let coarse_tile_work_offsets_bytes =
            (coarse_depth_tile_capacity as u64) * std::mem::size_of::<u32>() as u64;

        prepass_resources.prepass_queue = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_queue"),
            size: prepass_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.binning_queue = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_binning_queue"),
            size: binning_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.visibility_flags_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_visible_flags"),
            size: instance_id_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.visible_geos_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_visible_instances"),
            size: instance_id_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.geos_prefix_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_instance_prefix"),
            size: instance_prefix_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.strand_instances = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_instances"),
            size: strand_instances_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.indirect_args = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_indirect_args"),
            size: (3 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.chunk_pool = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_binning_chunk_pool"),
            size: chunk_pool_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.free_heads = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_binning_free_heads"),
            size: (BINNING_POOL_NUM_HEADS as u64) * std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.frustum_table = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_frustum_table"),
            size: frustum_table_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.froxel_bucket_heads = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_froxel_bucket_heads"),
            size: froxel_bucket_heads_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.raster_work_queue = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_raster_work_queue"),
            size: raster_work_queue_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.coarse_depth_lut = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_coarse_depth_lut"),
            size: coarse_depth_lut_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.coarse_range_queue = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_coarse_range_queue"),
            size: coarse_range_queue_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.coarse_interval_heads = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_coarse_interval_heads"),
            size: coarse_interval_heads_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.coarse_interval_refs = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_coarse_interval_refs"),
            size: coarse_interval_refs_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.coarse_range_lookup = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_coarse_range_lookup"),
            size: coarse_range_lookup_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.coarse_count_page_table = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_coarse_count_page_table"),
            size: coarse_count_page_table_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.coarse_count_pages = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_coarse_count_pages"),
            size: coarse_count_pages_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.fine_page_meta = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_fine_page_meta"),
            size: fine_page_meta_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.fine_cell_offsets = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_fine_cell_offsets"),
            size: fine_cell_offsets_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.fine_cell_write_cursors = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_fine_cell_write_cursors"),
            size: fine_cell_write_cursors_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.fine_seg_refs = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_fine_seg_refs"),
            size: fine_seg_refs_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.coarse_tile_work_counts = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_coarse_tile_work_counts"),
            size: coarse_tile_work_counts_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.coarse_tile_work_offsets =
            Some(device.create_buffer(&BufferDescriptor {
                label: Some("strand_coarse_tile_work_offsets"),
                size: coarse_tile_work_offsets_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        prepass_resources.prepass_task_capacity = prepass_capacity;
        prepass_resources.binning_task_capacity = binning_capacity;
        prepass_resources.instance_capacity = instance_capacity;
        prepass_resources.instance_count = instance_count;
        prepass_resources.frustum_capacity = frustum_capacity;
        prepass_resources.frustum_count = frustum_descs.len() as u32;
        prepass_resources.froxel_bucket_capacity = froxel_bucket_capacity;
        prepass_resources.raster_work_capacity = raster_work_capacity;
        prepass_resources.coarse_depth_tile_capacity = coarse_depth_tile_capacity;
        prepass_resources.coarse_range_capacity = coarse_range_capacity;
        prepass_resources.coarse_interval_ref_capacity = coarse_interval_ref_capacity;
        prepass_resources.coarse_count_page_capacity = coarse_count_page_capacity;
        prepass_resources.fine_seg_ref_capacity = fine_seg_ref_capacity;

        info!(
            "Allocated prepass buffers: strands={} segment_budget={} instances={} frusta={} prepass_cap={} binning_cap={} bucket_cap={} coarse_depth_tiles={} coarse_ranges={} coarse_interval_refs={} coarse_count_pages={} fine_seg_refs={} raster_work_cap={}",
            total_strands,
            total_segment_budget,
            instance_count,
            frustum_descs.len(),
            prepass_capacity,
            binning_capacity,
            froxel_bucket_capacity,
            coarse_depth_tile_capacity,
            coarse_range_capacity,
            coarse_interval_ref_capacity,
            coarse_count_page_capacity,
            fine_seg_ref_capacity,
            raster_work_capacity,
        );
    }
    prepass_resources.instance_count = instance_count;
    prepass_resources.frustum_count = frustum_descs.len() as u32;

    let zero_queue_hdr = [0u32, 0u32];
    let zero_dispatch = [0u32, 1u32, 1u32];

    if let Some(queue_buf) = &prepass_resources.prepass_queue {
        render_queue.write_buffer(queue_buf, 0, bytemuck::cast_slice(&zero_queue_hdr));
    }
    if let Some(queue_buf) = &prepass_resources.binning_queue {
        render_queue.write_buffer(queue_buf, 0, bytemuck::cast_slice(&zero_queue_hdr));
    }
    if let Some(queue_buf) = &prepass_resources.raster_work_queue {
        render_queue.write_buffer(queue_buf, 0, bytemuck::cast_slice(&zero_queue_hdr));
    }
    if let Some(seg_refs) = &prepass_resources.fine_seg_refs {
        render_queue.write_buffer(seg_refs, 0, bytemuck::cast_slice(&[0u32]));
    }
    if let Some(queue_buf) = &prepass_resources.coarse_range_queue {
        render_queue.write_buffer(queue_buf, 0, bytemuck::cast_slice(&zero_queue_hdr));
    }
    if let Some(interval_refs) = &prepass_resources.coarse_interval_refs {
        render_queue.write_buffer(interval_refs, 0, bytemuck::cast_slice(&[0u32]));
    }
    if let Some(indirect) = &prepass_resources.indirect_args {
        render_queue.write_buffer(indirect, 0, bytemuck::cast_slice(&zero_dispatch));
    }
    if let Some(free_heads) = &prepass_resources.free_heads {
        let zero_heads = vec![0u32; BINNING_POOL_NUM_HEADS as usize];
        render_queue.write_buffer(free_heads, 0, bytemuck::cast_slice(&zero_heads));
    }
    if let Some(interval_heads) = &prepass_resources.coarse_interval_heads {
        let invalid_heads = vec![u32::MAX; prepass_resources.coarse_depth_tile_capacity as usize];
        render_queue.write_buffer(interval_heads, 0, bytemuck::cast_slice(&invalid_heads));
    }
    if let Some(range_lookup) = &prepass_resources.coarse_range_lookup {
        let invalid_lookup = vec![u32::MAX; prepass_resources.coarse_range_capacity as usize];
        render_queue.write_buffer(range_lookup, 0, bytemuck::cast_slice(&invalid_lookup));
    }
    if let Some(frustum_table) = &prepass_resources.frustum_table {
        render_queue.write_buffer(frustum_table, 0, bytemuck::cast_slice(&frustum_descs));
    }
    if let Some(strand_instances) = &prepass_resources.strand_instances {
        render_queue.write_buffer(strand_instances, 0, bytemuck::cast_slice(&instances));
    }
}

fn strand_asset_table_id(
    entity: Entity,
    geometry: &StrandGeometry,
    storage_buffers: &RenderAssets<GpuVirtualShaderStorageBuffer>,
) -> Option<u32> {
    let ids = [
        ("vertices", &geometry.vertices),
        ("indices", &geometry.indices),
        ("metadata", &geometry.meta),
        ("geos", &geometry.geos),
    ]
    .map(|(label, handle)| {
        let gpu_buffer = storage_buffers.get(handle)?;
        let (allocation, _) = gpu_buffer.allocation.as_ref()?;
        Some((label, allocation.id.0))
    });

    let [
        Some((_, vertex_id)),
        Some((_, index_id)),
        Some((_, meta_id)),
        Some((_, geo_id)),
    ] = ids
    else {
        return None;
    };

    if vertex_id != geo_id || index_id != geo_id || meta_id != geo_id {
        warn!(
            "Skipping strand entity {:?}: allocator table ids differ (vertices={}, indices={}, metadata={}, geos={})",
            entity, vertex_id, index_id, meta_id, geo_id
        );
        return None;
    }

    Some(geo_id)
}

fn sync_strand_instance_transforms(
    mut commands: Commands,
    query: Query<(Entity, &GlobalTransform, Option<&StrandInstanceTransform>), With<StrandAsset>>,
) {
    for (entity, global_transform, current) in &query {
        let next = StrandInstanceTransform::from(global_transform);
        match current {
            Some(current)
                if current.world_from_local == next.world_from_local
                    && current.local_from_world == next.local_from_world => {}
            _ => {
                commands.entity(entity).insert(next);
            }
        }
    }
}

// main world buffer initialization
// packs StrandGeometry
fn set_strand_geometry(
    query: Query<(Entity, &StrandAsset, &StrandMaterial), Without<StrandGeometry>>,
    assets: Res<Assets<DsonAsset>>,
    mut storage_buffers: ResMut<Assets<VirtualShaderStorageBuffer>>,
    mut commands: Commands,
) {
    for (entity, strand_asset, material) in query.iter() {
        let Some(asset) = assets.get(&strand_asset.handle) else {
            continue;
        };

        let Some(geometry_library) = &asset.dson_file.geometry_library else {
            warn!("Geometry library not found for entity: {:?}", entity);
            continue;
        };

        if geometry_library.is_empty() {
            warn!("Geometry library is empty for entity: {:?}", entity);
            continue;
        }

        let geometry = &geometry_library[0];
        // Extract vertices
        // TODO: time this. Could also be done in a compute shader
        let mut aabb = Aabb3d::from_point_cloud(
            Isometry3d::IDENTITY,
            geometry.vertices.values.iter().cloned(),
        ); // todo: pass transform
        aabb.min *= 0.0254; // TODO: pass transform to shaders
        aabb.max *= 0.0254; // TODO: pass transform to shaders
        let vertices: Vec<[f32; 4]> = geometry
            .vertices
            .values
            .clone()
            .iter()
            .map(|v| {
                [
                    v[0] * 0.0254, // TODO: pass transform to shaders
                    v[1] * 0.0254, // TODO: pass transform to shaders
                    v[2] * 0.0254, // TODO: pass transform to shaders
                    1.0,
                ]
            })
            .collect();

        // Extract indices from polyline_list
        let Some(polyline_list) = &geometry.polyline_list else {
            warn!("Polyline list not found for entity: {:?}", entity);
            continue;
        };

        // Flatten the polyline indices
        // For each strand in values, skip first two elements (group_idx, mat_group_idx)
        // and collect the vertex indices
        let packed_strand_info =
            polyline_list
                .values
                .iter()
                .fold((Vec::new(), Vec::new(), 0), |mut acc, strand| {
                    let strand_indices = &strand[2..];
                    acc.0.extend_from_slice(strand_indices);
                    if strand_indices.len() > acc.2 {
                        acc.2 = strand_indices.len();
                    }
                    match acc.1.last().copied() {
                        Some((last_strand_count, last_strand_offset)) => {
                            acc.1.push((
                                strand_indices.len() as u32,
                                last_strand_offset + last_strand_count,
                            ));
                        }
                        None => acc.1.push((strand_indices.len() as u32, 0)),
                    }
                    acc
                });

        let indices = packed_strand_info.0;
        let meta: Vec<StrandMeta> = packed_strand_info
            .1
            .into_iter()
            .map(StrandMeta::from)
            .collect();
        let max_segments_in_strand = packed_strand_info.2 as u32;

        let geos_data = vec![StrandGeo::new(
            polyline_list.values.len() as u32,
            max_segments_in_strand,
            aabb,
        )];

        let vertex_buffer = VirtualShaderStorageBuffer::from((SlabKind::Vert, vertices));
        let index_buffer = VirtualShaderStorageBuffer::from((SlabKind::Index, indices));
        let meta_buffer = VirtualShaderStorageBuffer::from((SlabKind::StrandMeta, meta));
        let geo_buffer = VirtualShaderStorageBuffer::from((SlabKind::StrandGeo, geos_data));
        let material_buffer =
            VirtualShaderStorageBuffer::from((SlabKind::StrandMaterial, vec![material]));

        let vertex_buffer_handle = storage_buffers.add(vertex_buffer);
        let index_buffer_handle = storage_buffers.add(index_buffer);
        let meta_buffer_handle = storage_buffers.add(meta_buffer);
        let geo_buffer_handle = storage_buffers.add(geo_buffer);
        let material_buffer_handle = storage_buffers.add(material_buffer);

        info!("set strand geometry: {:?}", vertex_buffer_handle);

        commands.entity(entity).insert(StrandGeometry {
            vertices: vertex_buffer_handle,
            indices: index_buffer_handle,
            meta: meta_buffer_handle,
            geos: geo_buffer_handle,
            materials: material_buffer_handle,
            strand_count: polyline_list.values.len() as u32,
            max_segments_in_strand,
            aabb,
        });
        commands
            .entity(entity)
            .insert(StrandInstanceTransform::default());
    }
}

// uploads changed material paramters to the buffer
// TODO
pub fn update_material_buffer(
    query: Query<(Entity, &StrandGeometry, &StrandMaterial)>,
    storage_buffers: Res<RenderAssets<GpuVirtualShaderStorageBuffer>>,
    allocator: Res<GpuPagingAllocator>,
    render_queue: Res<RenderQueue>,
) {
    for (entity, geometry, material) in query.iter() {
        let Some(material_buffer) = storage_buffers.get(&geometry.materials) else {
            warn!("Material storage buffer not found for entity: {:?}", entity);
            continue;
        };
        let Some((allocation, kind)) = &material_buffer.allocation else {
            warn!("Material buffer allocation missing");
            continue;
        };
        let Some(buffer) = allocator.slab_buffer(*kind, allocation.slab_index) else {
            warn!("Material slab not found in allocator");
            continue;
        };
        let material_bytes = bytemuck::bytes_of(material);
        render_queue.write_buffer(
            &buffer,        // Get the underlying wgpu::Buffer
            0,              // Offset in the buffer to start writing (0 for the start)
            material_bytes, // The byte slice to write
        );
    }
}

// Create froxel configuration uniform buffer
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuFroxelConfigStd140 {
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuFrustumDesc {
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
    bucket_base: u32,
    bucket_count: u32,
    kind: u32,
    coarse_depth_tile_base: u32,
    coarse_depth_tile_count: u32,
    coarse_tiles_x: u32,
    coarse_tiles_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuCoarseDepthLutEntry {
    z_min_q: u32,
    z_max_q: u32,
    virtual_start_q: u32,
    virtual_count_q: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuCoarseAssetRange {
    geo_id: u32,
    frustum_id: u32,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
    z_min_q: u32,
    z_max_q: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuCoarseIntervalRef {
    next: u32,
    range_id: u32,
    z_min_q: u32,
    z_max_q: u32,
}

pub fn create_froxel_config_buffer(device: &RenderDevice, config: &FroxelConfig) -> Buffer {
    let gpu_cfg = GpuFroxelConfigStd140 {
        screen_width: config.screen_width,
        screen_height: config.screen_height,
        froxel_size_x: config.froxel_size_x,
        froxel_size_y: config.froxel_size_y,
        depth_slices: config.depth_slices,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("strand_froxel_config_buffer"),
        size: std::mem::size_of::<GpuFroxelConfigStd140>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // Initialize with configuration
    let mut mapped = buffer.slice(..).get_mapped_range_mut();
    mapped.copy_from_slice(bytemuck::bytes_of(&gpu_cfg));
    drop(mapped);
    buffer.unmap();
    buffer
}

fn flag_realloc_on_view_change(
    mut commands: Commands,
    cams: Query<Entity, With<TieFroxelsToView>>,
    mut resize_reader: MessageReader<WindowResized>,
    // or listen to WindowResized and map to camera(s)
) {
    for _ in resize_reader.read() {
        info!("window changed");
        for e in &cams {
            commands.entity(e).insert(NeedsRealloc);
        }
    }
}

fn flag_realloc_on_config_change(mut commands: Commands, q: Query<Entity, Changed<FroxelConfig>>) {
    for e in &q {
        commands.entity(e).insert(NeedsRealloc);
    }
}

fn flag_realloc_on_tie_change(mut commands: Commands, q: Query<Entity, Changed<TieFroxelsToView>>) {
    for e in &q {
        commands.entity(e).insert(NeedsRealloc);
    }
}

fn tie_view_to_froxel_config(
    cams: Query<
        (Entity, &TieFroxelsToView, &ExtractedView, &mut FroxelConfig),
        Changed<ExtractedView>,
    >,
) {
    for (entity, tie_mode, extracted_view, mut config) in cams {
        info!("Viewport change: {:?}", extracted_view.viewport);
        let viewport_x = extracted_view.viewport.z;
        let viewport_y = extracted_view.viewport.w;
        match tie_mode {
            TieFroxelsToView::Fixed(v) => {}
            TieFroxelsToView::Native => {
                config.screen_width = viewport_x;
                config.screen_height = viewport_y;
            }
            TieFroxelsToView::Scaled(s) => {
                let w = ((viewport_x as f32) * s) as u32;
                let h = ((viewport_y as f32) * s) as u32;
                config.screen_width = w;
                config.screen_height = h;
            }
        }
    }
}

// render world buffer retrieval
fn use_froxel_buffer(
    query: Query<
        (Entity, &FroxelConfig, Option<&ExtractedView>),
        Or<(Added<FroxelConfig>, With<NeedsRealloc>)>,
    >,
    device: Res<RenderDevice>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
) {
    for (entity, config, _extracted_view) in query.iter() {
        let config_buffer = create_froxel_config_buffer(&device, config);

        raster_resources
            .froxel_config_buffer
            .insert(entity, config_buffer);
        raster_resources
            .frustrum_config
            .insert(entity, config.clone());
        debug!("Updated froxel config + render targets");
    }
}

fn use_deep_opacity_maps(
    query: Query<(Entity, &FroxelConfig), With<ExtractedDirectionalLight>>,
    mut shadow_resources: ResMut<StrandShadowResources>,
) {
    let mut lights: Vec<(Entity, FroxelConfig)> =
        query.iter().map(|(e, cfg)| (e, cfg.clone())).collect();
    lights.sort_by_key(|(e, _)| e.index());
    if lights.is_empty() {
        shadow_resources.light_layer_by_entity.clear();
        shadow_resources.light_entity_by_layer.clear();
        return;
    }

    let max_width = lights
        .iter()
        .map(|(_, cfg)| cfg.screen_width)
        .max()
        .unwrap_or(1);
    let max_height = lights
        .iter()
        .map(|(_, cfg)| cfg.screen_height)
        .max()
        .unwrap_or(1);
    let light_layers = lights.len() as u32;
    let _ = (max_width, max_height, light_layers);
    shadow_resources.light_layer_by_entity.clear();
    shadow_resources.light_entity_by_layer.clear();
    for (layer, (entity, _)) in lights.iter().enumerate() {
        shadow_resources
            .light_layer_by_entity
            .insert(*entity, layer as u32);
        shadow_resources.light_entity_by_layer.push(*entity);
    }
}

fn bind_vsms_dom_targets(
    runtime: Res<VirtualSurfaceRuntime>,
    mut shadow_resources: ResMut<StrandShadowResources>,
) {
    let Some(opacity_binding) = runtime
        .pool_storage_bindings
        .get(&VirtualSurfaceKind::Opacity3D)
    else {
        return;
    };
    let Some(depth_binding) = runtime
        .pool_storage_bindings
        .get(&VirtualSurfaceKind::Depth2DArray)
    else {
        return;
    };
    shadow_resources.dom_array_targets = Some((
        opacity_binding.first_view.clone(),
        depth_binding.first_view.clone(),
    ));
}

fn sync_vsms_dom_configs(
    mut commands: Commands,
    lights: Query<(Entity, &FroxelConfig), With<ExtractedDirectionalLight>>,
    mut shadow_resources: ResMut<StrandShadowResources>,
) {
    let mut light_entities: Vec<(Entity, FroxelConfig)> =
        lights.iter().map(|(e, cfg)| (e, *cfg)).collect();
    light_entities.sort_by_key(|(e, _)| e.index());

    let live: HashSet<Entity> = light_entities.iter().map(|(e, _)| *e).collect();
    let stale: Vec<Entity> = shadow_resources
        .dom_vsms_proxies
        .keys()
        .copied()
        .filter(|e| !live.contains(e))
        .collect();
    for entity in stale {
        if let Some((op_proxy, d_proxy)) = shadow_resources.dom_vsms_proxies.remove(&entity) {
            commands.entity(op_proxy).despawn();
            commands.entity(d_proxy).despawn();
        }
    }

    for (entity, cfg) in light_entities {
        const DOM_VIRTUAL_SCALE: u32 = 2;
        const DOM_PAGE_XY: u32 = 256;

        let virtual_w = cfg
            .screen_width
            .saturating_mul(DOM_VIRTUAL_SCALE)
            .clamp(DOM_PAGE_XY, 8192);
        let virtual_h = cfg
            .screen_height
            .saturating_mul(DOM_VIRTUAL_SCALE)
            .clamp(DOM_PAGE_XY, 8192);
        // TEMP DEBUG: force dense DOM residency (full virtual tile coverage) while fixing
        // shadow shader logic. Revert to smaller budgets + demand-driven residency afterwards.
        let depth_tile_count = virtual_w
            .div_ceil(DOM_PAGE_XY)
            .saturating_mul(virtual_h.div_ceil(DOM_PAGE_XY))
            .max(1);
        let opacity_tile_count = depth_tile_count.max(1);
        let op_cfg = VirtualSurfaceConfigBuilder::new()
            .enabled(true)
            .surface_kind(VirtualSurfaceKind::Opacity3D)
            .page_size(UVec3::new(DOM_PAGE_XY, DOM_PAGE_XY, NUM_DOM_SLICES))
            .virtual_extent(UVec3::new(virtual_w, virtual_h, NUM_DOM_SLICES))
            .mip_levels(1)
            .layer_count(1)
            .max_resident_pages(opacity_tile_count)
            .priority_bias(1.0)
            .residency_mode(VirtualSurfaceResidencyMode::PreallocatePages(
                opacity_tile_count,
            ))
            .build();
        let d_cfg = VirtualSurfaceConfigBuilder::new()
            .enabled(true)
            .surface_kind(VirtualSurfaceKind::Depth2DArray)
            .page_size(UVec3::new(DOM_PAGE_XY, DOM_PAGE_XY, 1))
            .virtual_extent(UVec3::new(virtual_w, virtual_h, 1))
            .mip_levels(1)
            .layer_count(1)
            .max_resident_pages(depth_tile_count)
            .priority_bias(1.0)
            .residency_mode(VirtualSurfaceResidencyMode::PreallocatePages(
                depth_tile_count,
            ))
            .build();

        if let Some(&(op_proxy, d_proxy)) = shadow_resources.dom_vsms_proxies.get(&entity) {
            commands
                .entity(op_proxy)
                .insert((op_cfg, DomVsmsProxyKind::Opacity));
            commands
                .entity(d_proxy)
                .insert((d_cfg, DomVsmsProxyKind::Depth));
        } else {
            let op_proxy = commands.spawn((op_cfg, DomVsmsProxyKind::Opacity)).id();
            let d_proxy = commands.spawn((d_cfg, DomVsmsProxyKind::Depth)).id();
            shadow_resources
                .dom_vsms_proxies
                .insert(entity, (op_proxy, d_proxy));
        }
    }
}
