use bevy::{
    core_pipeline::{
        core_3d::main_opaque_pass_3d,
        schedule::{Core3d, Core3dSystems},
        tonemapping::tonemapping,
    },
    math::bounding::Aabb3d,
    pbr::ExtractedDirectionalLight,
    platform::collections::HashMap,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::ExtractComponentPlugin,
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssets,
        render_resource::Buffer,
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
    },
    shader::load_shader_library,
    transform::TransformSystems,
    window::WindowResized,
};
use bevy_vsms::{
    allocator::VirtualSurfaceRuntime,
    prelude::{
        VirtualSurfaceConfigBuilder, VirtualSurfaceKind, VirtualSurfaceMap,
        VirtualSurfaceResidencyMode,
    },
};
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use wgpu::{BufferDescriptor, BufferUsages};

use crate::{
    allocator::*,
    components::*,
    nodes,
    pipelines::{
        composite::*,
        prepass::*,
        raster::*,
        shading::*,
        shadows::*,
        task_contract::{
            BINNING_POOL_CHUNK_SIZE, BINNING_POOL_MAX_CHUNKS, BINNING_POOL_MIN_CHUNKS,
            BINNING_POOL_NUM_HEADS, FinePageMeta, FineSegRef, QUEUE_HEADER_WORDS, RasterTileRun,
            RasterWorkItem,
        },
        tile_debug::*,
    },
    resources::*,
    shader_types::*,
    strand_cache::{StrandCacheAsset, StrandCacheAssetLoader, StrandCacheStrand},
};

pub const MAX_TEXTURE_EXTENT: u32 = 8192; // for shading (TODO: get this from device limits)
pub const COARSE_FINE_TILE_EXTENT: u32 = 4;
pub const COARSE_DEPTH_SLICES: u32 = 16;
pub const COARSE_MAX_SLICES_PER_ASSET_INTERVAL: u32 = 2;
pub const COARSE_COUNT_PAGE_SIZE: u32 =
    COARSE_FINE_TILE_EXTENT * COARSE_FINE_TILE_EXTENT * COARSE_DEPTH_SLICES;
const COARSE_COUNT_PAGE_CAPACITY_DIVISOR: u32 = 2;
const COARSE_INTERVAL_REFS_PER_TILE: u32 = 64;
const DOM_RESIDENT_PAGE_BUDGET_DIVISOR: u32 = 8;
const DOM_MIN_RESIDENT_PAGES: u32 = 4;
const DOM_MAX_RESIDENT_PAGES: u32 = 256;
pub const DOM_PAGE_XY: u32 = 256;
pub const DOM_PREFETCH_PAGE_BORDER: u32 = 1;
pub(crate) const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65_535;
pub(crate) const SHADER_PREFIX: &str = "embedded://strand_software_rasterizer/shaders/";

pub(crate) fn embedded_shader_path(shader: &str) -> String {
    format!("{SHADER_PREFIX}{shader}")
}

fn load_strand_shader_libraries(app: &mut App) {
    load_shader_library!(app, "shaders/allocator_debug_noop.wgsl");
    load_shader_library!(app, "shaders/common.wgsl");
    load_shader_library!(app, "shaders/opaque_depth_reduce.wgsl");
    load_shader_library!(app, "shaders/pool.wgsl");
    load_shader_library!(app, "shaders/prefix_sum.wgsl");
    load_shader_library!(app, "shaders/queues.wgsl");
    load_shader_library!(app, "shaders/shading_LUTs.wgsl");
    load_shader_library!(app, "shaders/shadow_dom_stampback.wgsl");
    load_shader_library!(app, "shaders/spline.wgsl");
    load_shader_library!(app, "shaders/strand_binning.wgsl");
    load_shader_library!(app, "shaders/strand_binning_queue.wgsl");
    load_shader_library!(app, "shaders/strand_composite.wgsl");
    load_shader_library!(app, "shaders/strand_prepass.wgsl");
    load_shader_library!(app, "shaders/strand_rasterizer.wgsl");
    load_shader_library!(app, "shaders/strand_rasterizer_.wgsl");
    load_shader_library!(app, "shaders/strand_shading.wgsl");
    load_shader_library!(app, "shaders/strand_tile_debug.wgsl");
    load_shader_library!(app, "shaders/task_contract.wgsl");
    load_shader_library!(app, "shaders/types.wgsl");
}

fn cap_storage_capacity(
    requested: u32,
    header_bytes: u64,
    item_bytes: u64,
    max_binding_bytes: u64,
) -> u32 {
    if item_bytes == 0 {
        return requested;
    }
    let max_items = max_binding_bytes.saturating_sub(header_bytes) / item_bytes;
    let max_items = max_items.min(u32::MAX as u64) as u32;
    requested.min(max_items)
}

fn log_storage_capacity_cap(label: &str, requested: u32, capped: u32, max_binding_bytes: u64) {
    if capped < requested {
        warn!(
            "Capping {label} capacity from {requested} to {capped} to fit max storage buffer binding size {max_binding_bytes} bytes"
        );
    }
}

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
enum DomVsmsProxyKind {
    Opacity,
    Depth,
}

use lazy_static::lazy_static;

lazy_static! {
    pub static ref BIND_MAP: HashMap<SlabKind, u32> = [
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

fn dom_resident_page_budget(tile_count: u32) -> u32 {
    tile_count
        .div_ceil(DOM_RESIDENT_PAGE_BUDGET_DIVISOR)
        .clamp(DOM_MIN_RESIDENT_PAGES, DOM_MAX_RESIDENT_PAGES)
        .min(tile_count.max(1))
}

lazy_static! {
    pub static ref LABEL_MAP: HashMap<SlabKind, &'static str> = [
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
        load_strand_shader_libraries(app);
        app.add_plugins((
            ExtractComponentPlugin::<FroxelConfig>::default(),
            ExtractComponentPlugin::<StrandGeometry>::default(),
            ExtractComponentPlugin::<StrandInstanceTransform>::default(),
            ExtractComponentPlugin::<StrandMaterial>::default(),
            ExtractResourcePlugin::<TileDebugSettings>::default(),
            ExtractResourcePlugin::<StochasticCullSettings>::default(),
            ExtractResourcePlugin::<PrepassTelemetrySettings>::default(),
        ));
        app.init_resource::<TileDebugSettings>();
        app.init_resource::<StochasticCullSettings>();
        app.init_resource::<PrepassTelemetrySettings>();
        app.init_resource::<StrandAssetResources>();
        app.init_asset::<StrandCacheAsset>();
        app.init_asset_loader::<StrandCacheAssetLoader>();
        app.add_systems(
            Update,
            (
                set_strand_list_geometry,
                set_cached_strand_geometry,
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
        render_app.init_resource::<StrandRasterizerResources>();
        render_app.init_resource::<StrandShadingResources>();
        render_app.init_resource::<StrandShadowResources>();
        render_app.init_resource::<StrandPrepassResources>();
        render_app.init_resource::<ComputeInvocationDims>();
        render_app.init_resource::<StrandPrepassPipeline>();
        render_app.init_resource::<StrandShadingPipeline>();
        render_app.init_resource::<StrandRasterizerPipeline>();
        render_app.init_resource::<StrandShadowPipeline>();
        render_app.init_resource::<CompositionPipeline>();
        render_app.init_resource::<TileDebugPipeline>();
        render_app.add_systems(
            Render,
            ((
                use_froxel_buffer,
                sync_vsms_dom_configs,
                use_deep_opacity_maps,
                use_prepass_buffers,
                update_material_buffer,
                // use_strand_geometry.after(prepare_view_uniforms),
            )
                .chain()
                .in_set(RenderSystems::Prepare),),
        );
        render_app.add_systems(
            Render,
            (
                update_strand_prepass_pipeline,
                update_strand_shading_pipeline,
                update_strand_raster_pipeline,
                update_strand_shadow_pipeline,
                bind_vsms_dom_targets,
            )
                .chain()
                .after(RenderSystems::PrepareBindGroups),
        );
        render_app.add_systems(
            Core3d,
            (
                (
                    nodes::prepass::work_preparation_pass,
                    nodes::shadows::strand_shadow_rasterizer_pass,
                    nodes::shading::strand_shading_pass,
                    nodes::raster::strand_rasterizer_pass,
                )
                    .chain()
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
                (
                    nodes::composite::composition_pass,
                    nodes::debug::tile_debug_pass,
                )
                    .chain()
                    .before(tonemapping)
                    .in_set(Core3dSystems::PostProcess),
            ),
        );
    }
}

fn use_prepass_buffers(
    geometry_query: Query<(Entity, &StrandGeometry, Option<&StrandInstanceTransform>)>,
    light_frusta_query: Query<(Entity, &ExtractedDirectionalLight)>,
    view_query: Query<(Entity, &ExtractedView), Without<ExtractedDirectionalLight>>,
    storage_buffers: Res<RenderAssets<GpuVirtualShaderStorageBuffer>>,
    device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    mut prepass_resources: ResMut<StrandPrepassResources>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
    mut shading_resources: ResMut<StrandShadingResources>,
    mut shadow_resources: ResMut<StrandShadowResources>,
    surface_map: Res<VirtualSurfaceMap>,
) {
    let mut total_strands = 0u32;
    let mut total_segment_budget = 0u32;
    let mut max_strands_in_instance = 0u32;
    let mut max_segments_in_strand = 0u32;
    let mut max_shading_texels_in_instance = 0u32;
    let mut instances = Vec::new();
    let mut strand_world_centers = Vec::new();
    let mut sorted_geometry: Vec<_> = geometry_query.iter().collect();
    sorted_geometry.sort_by_key(|(entity, _, _)| entity.index());
    for (_entity, geom, transform) in sorted_geometry {
        let Some(table_ids) = strand_asset_table_ids(geom, &storage_buffers) else {
            continue;
        };
        total_strands = total_strands.saturating_add(geom.strand_count);
        total_segment_budget =
            total_segment_budget.saturating_add(geom.index_count.saturating_sub(geom.strand_count));
        max_strands_in_instance = max_strands_in_instance.max(geom.strand_count);
        max_segments_in_strand = max_segments_in_strand.max(geom.max_segments_in_strand);
        max_shading_texels_in_instance = max_shading_texels_in_instance.max(geom.index_count);
        let transform = transform.copied().unwrap_or_default();
        instances.push(StrandInstance {
            vertex_id: table_ids.vertex_id,
            index_id: table_ids.index_id,
            meta_id: table_ids.meta_id,
            geo_id: table_ids.geo_id,
            material_id: table_ids.material_id,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            world_from_local: transform.world_from_local,
            local_from_world: transform.local_from_world,
        });
        let local_center = (geom.aabb.min + geom.aabb.max) * 0.5;
        strand_world_centers.push(
            transform
                .world_from_local
                .transform_point3(local_center.into()),
        );
    }

    if total_strands == 0 {
        return;
    }

    raster_resources.strand_count = Some(total_strands);

    let max_storage_binding_bytes = device.limits().max_storage_buffer_binding_size as u64;
    let queue_header_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64;

    let requested_prepass_capacity = total_strands.next_power_of_two().max(2048);
    let prepass_capacity = cap_storage_capacity(
        requested_prepass_capacity,
        queue_header_bytes,
        std::mem::size_of::<FinePrepassTask>() as u64,
        max_storage_binding_bytes,
    );
    let base_binning_capacity = total_segment_budget
        .next_power_of_two()
        .max(prepass_capacity);
    let instance_count = instances.len() as u32;
    let needs_shading_realloc = shading_resources.output_texture.is_none()
        || shading_resources.max_shading_texels_in_instance != Some(max_shading_texels_in_instance)
        || shading_resources.layer_count != Some(instance_count);
    shading_resources.strand_count = Some(total_strands);
    shading_resources.max_shading_texels_in_instance = Some(max_shading_texels_in_instance);
    shading_resources.layer_count = Some(instance_count);
    let instance_capacity = instance_count.next_power_of_two().max(2048);
    let mut frustum_descs: Vec<GpuFrustumDesc> = Vec::new();
    let mut bucket_base = 0u32;
    let mut coarse_depth_tile_base = 0u32;
    let mut fine_depth_tile_base = 0u32;
    let light_entities: HashSet<Entity> = light_frusta_query
        .iter()
        .map(|(entity, _)| entity)
        .collect();
    let main_view = view_query
        .iter()
        .find(|(entity, _)| {
            raster_resources.frustrum_config.contains_key(entity)
                && !light_entities.contains(entity)
        })
        .map(|(_, view)| view);
    let mut shadow_cascade_by_entity: HashMap<Entity, u32> = HashMap::default();
    let mut shadow_texture_base_by_entity: HashMap<Entity, u32> = HashMap::default();
    let mut next_shadow_texture_layer = 0u32;
    let mut sorted_lights: Vec<_> = light_frusta_query.iter().collect();
    sorted_lights.sort_by_key(|(entity, _)| entity.index());
    for (entity, light) in sorted_lights {
        let cascade_index =
            select_authoritative_shadow_cascade(main_view, &strand_world_centers, light);
        shadow_cascade_by_entity.insert(entity, cascade_index);
        if light.shadow_maps_enabled {
            shadow_texture_base_by_entity.insert(entity, next_shadow_texture_layer);
            let cascade_count = light
                .cascade_shadow_config
                .bounds
                .len()
                .min(bevy::pbr::MAX_CASCADES_PER_LIGHT)
                .max(1) as u32;
            next_shadow_texture_layer = next_shadow_texture_layer.saturating_add(cascade_count);
        }
    }
    raster_resources.frustum_ids.clear();
    shadow_resources.light_layer_by_frustum.clear();
    shadow_resources.stamp_targets.clear();
    let mut shadow_dom_surface_ids: Vec<[u32; 2]> = Vec::new();
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
        let cascade_index = shadow_cascade_by_entity
            .get(&entity)
            .copied()
            .unwrap_or_default();
        if is_light {
            shadow_resources
                .light_layer_by_frustum
                .insert(frustum_id as u32, light_layer);
            if shadow_texture_base_by_entity.contains_key(&entity) {
                shadow_resources.stamp_targets.push(ShadowStampTarget {
                    frustum_id: frustum_id as u32,
                    light_entity: entity,
                    cascade_index,
                });
            }
            light_layer = light_layer.saturating_add(1);
        }
        let dom_surface_ids = shadow_resources
            .dom_vsms_proxies
            .get(&entity)
            .and_then(|(opacity_proxy, depth_proxy)| {
                Some([
                    *surface_map.entity_to_surface.get(opacity_proxy)?,
                    *surface_map.entity_to_surface.get(depth_proxy)?,
                ])
            })
            .unwrap_or([u32::MAX, u32::MAX]);
        shadow_dom_surface_ids.push(dom_surface_ids);
        let (fine_tiles_x, fine_tiles_y, bucket_count) = cfg.get_num_tiles();
        let fine_depth_tile_count = fine_tiles_x.saturating_mul(fine_tiles_y);
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
            cascade_index,
            fine_depth_tile_base,
            _pad1: 0,
            _pad2: 0,
        });
        bucket_base = bucket_base.saturating_add(bucket_count);
        coarse_depth_tile_base = coarse_depth_tile_base.saturating_add(coarse_depth_tile_count);
        fine_depth_tile_base = fine_depth_tile_base.saturating_add(fine_depth_tile_count);
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
            cascade_index: 0,
            fine_depth_tile_base: 0,
            _pad1: 0,
            _pad2: 0,
        });
        bucket_base = 1;
        coarse_depth_tile_base = 1;
        fine_depth_tile_base = 1;
    }
    let frustum_task_multiplier = (frustum_descs.len() as u32).max(1);
    let requested_binning_capacity = base_binning_capacity
        .saturating_mul(frustum_task_multiplier)
        .next_power_of_two()
        .max(base_binning_capacity);
    let binning_capacity = cap_storage_capacity(
        requested_binning_capacity,
        queue_header_bytes,
        std::mem::size_of::<BinningTask>() as u64,
        max_storage_binding_bytes,
    );
    let frustum_capacity = (frustum_descs.len() as u32).next_power_of_two().max(1);
    let froxel_bucket_capacity = bucket_base.next_power_of_two().max(1024);
    let requested_coarse_depth_tile_capacity = coarse_depth_tile_base.next_power_of_two().max(1);
    let coarse_depth_tile_capacity = cap_storage_capacity(
        requested_coarse_depth_tile_capacity,
        0,
        (COARSE_DEPTH_SLICES as u64) * (std::mem::size_of::<GpuCoarseDepthLutEntry>() as u64),
        max_storage_binding_bytes,
    );
    let coarse_count_page_table_capacity = coarse_depth_tile_capacity
        .saturating_mul(COARSE_DEPTH_SLICES)
        .next_power_of_two()
        .max(1);
    let requested_coarse_range_capacity = instance_capacity
        .saturating_mul(frustum_capacity)
        .next_power_of_two()
        .max(1);
    let coarse_range_capacity = cap_storage_capacity(
        requested_coarse_range_capacity,
        queue_header_bytes,
        std::mem::size_of::<GpuCoarseAssetRange>() as u64,
        max_storage_binding_bytes,
    );
    let requested_coarse_interval_ref_capacity = coarse_depth_tile_capacity
        .saturating_mul(COARSE_INTERVAL_REFS_PER_TILE)
        .next_power_of_two()
        .max(coarse_range_capacity);
    let coarse_interval_ref_capacity = cap_storage_capacity(
        requested_coarse_interval_ref_capacity,
        std::mem::size_of::<u32>() as u64,
        std::mem::size_of::<GpuCoarseIntervalRef>() as u64,
        max_storage_binding_bytes,
    );
    let requested_coarse_count_page_capacity = coarse_count_page_table_capacity
        .div_ceil(COARSE_COUNT_PAGE_CAPACITY_DIVISOR)
        .max(1);
    let coarse_count_page_capacity = cap_storage_capacity(
        requested_coarse_count_page_capacity,
        std::mem::size_of::<u32>() as u64,
        (COARSE_COUNT_PAGE_SIZE as u64) * std::mem::size_of::<u32>() as u64,
        max_storage_binding_bytes,
    );
    let fine_cell_capacity = coarse_count_page_capacity.saturating_mul(COARSE_COUNT_PAGE_SIZE);
    let requested_fine_seg_ref_capacity = binning_capacity
        .saturating_mul(4)
        .next_power_of_two()
        .max(binning_capacity.max(1024));
    let fine_seg_ref_capacity = cap_storage_capacity(
        requested_fine_seg_ref_capacity,
        std::mem::size_of::<u32>() as u64,
        std::mem::size_of::<FineSegRef>() as u64,
        max_storage_binding_bytes,
    );
    let requested_raster_work_capacity = binning_capacity
        .saturating_mul(2)
        .next_power_of_two()
        .max(binning_capacity.max(1024));
    let raster_work_capacity = cap_storage_capacity(
        requested_raster_work_capacity,
        queue_header_bytes,
        std::mem::size_of::<RasterWorkItem>() as u64,
        max_storage_binding_bytes,
    );
    let requested_raster_tile_run_capacity = coarse_depth_tile_capacity
        .saturating_mul(COARSE_FINE_TILE_EXTENT)
        .saturating_mul(COARSE_FINE_TILE_EXTENT)
        .next_power_of_two()
        .max(1024);
    let raster_tile_run_capacity = cap_storage_capacity(
        requested_raster_tile_run_capacity,
        queue_header_bytes,
        std::mem::size_of::<RasterTileRun>() as u64,
        max_storage_binding_bytes,
    );
    let opaque_fine_depth_tile_capacity = fine_depth_tile_base.next_power_of_two().max(1);

    let needs_realloc = prepass_resources.prepass_queue.is_none()
        || prepass_resources.binning_queue.is_none()
        || prepass_resources.visibility_flags_buffer.is_none()
        || prepass_resources.visible_geos_buffer.is_none()
        || prepass_resources.geos_prefix_buffer.is_none()
        || prepass_resources.indirect_args.is_none()
        || prepass_resources.prefix_indirect_args.is_none()
        || prepass_resources.telemetry.is_none()
        || prepass_resources.projected_segments.is_none()
        || prepass_resources.chunk_pool.is_none()
        || prepass_resources.free_heads.is_none()
        || prepass_resources.frustum_table.is_none()
        || prepass_resources.froxel_bucket_heads.is_none()
        || prepass_resources.raster_work_queue.is_none()
        || prepass_resources.raster_tile_run_queue.is_none()
        || prepass_resources.raster_tile_run_dispatch_args.is_none()
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
        || prepass_resources.shadow_dom_surface_ids.is_none()
        || prepass_resources.broad_instance_meta.is_none()
        || prepass_resources.opaque_fine_depth_tiles.is_none()
        || prepass_resources.strand_instances.is_none()
        || prepass_resources.prepass_task_capacity < prepass_capacity
        || prepass_resources.binning_task_capacity < binning_capacity
        || prepass_resources.instance_capacity < instance_capacity
        || prepass_resources.frustum_capacity < frustum_capacity
        || prepass_resources.froxel_bucket_capacity < froxel_bucket_capacity
        || prepass_resources.raster_work_capacity < raster_work_capacity
        || prepass_resources.raster_tile_run_capacity < raster_tile_run_capacity
        || prepass_resources.coarse_depth_tile_capacity < coarse_depth_tile_capacity
        || prepass_resources.coarse_range_capacity < coarse_range_capacity
        || prepass_resources.coarse_interval_ref_capacity < coarse_interval_ref_capacity
        || prepass_resources.coarse_count_page_capacity < coarse_count_page_capacity
        || prepass_resources.fine_seg_ref_capacity < fine_seg_ref_capacity
        || prepass_resources.opaque_fine_depth_tile_capacity < opaque_fine_depth_tile_capacity;

    if needs_realloc {
        log_storage_capacity_cap(
            "prepass queue",
            requested_prepass_capacity,
            prepass_capacity,
            max_storage_binding_bytes,
        );
        log_storage_capacity_cap(
            "binning queue",
            requested_binning_capacity,
            binning_capacity,
            max_storage_binding_bytes,
        );
        log_storage_capacity_cap(
            "coarse depth LUT",
            requested_coarse_depth_tile_capacity,
            coarse_depth_tile_capacity,
            max_storage_binding_bytes,
        );
        log_storage_capacity_cap(
            "coarse range queue",
            requested_coarse_range_capacity,
            coarse_range_capacity,
            max_storage_binding_bytes,
        );
        log_storage_capacity_cap(
            "coarse interval refs",
            requested_coarse_interval_ref_capacity,
            coarse_interval_ref_capacity,
            max_storage_binding_bytes,
        );
        log_storage_capacity_cap(
            "coarse count pages",
            requested_coarse_count_page_capacity,
            coarse_count_page_capacity,
            max_storage_binding_bytes,
        );
        log_storage_capacity_cap(
            "fine seg refs",
            requested_fine_seg_ref_capacity,
            fine_seg_ref_capacity,
            max_storage_binding_bytes,
        );
        log_storage_capacity_cap(
            "raster work queue",
            requested_raster_work_capacity,
            raster_work_capacity,
            max_storage_binding_bytes,
        );
        log_storage_capacity_cap(
            "raster tile run queue",
            requested_raster_tile_run_capacity,
            raster_tile_run_capacity,
            max_storage_binding_bytes,
        );

        let prepass_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (prepass_capacity as u64) * (std::mem::size_of::<FinePrepassTask>() as u64);
        let binning_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (binning_capacity as u64) * (std::mem::size_of::<BinningTask>() as u64);
        let projected_segments_bytes = (binning_capacity as u64)
            * std::mem::size_of::<crate::pipelines::task_contract::ProjectedSegment>() as u64;
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
        let shadow_dom_surface_ids_bytes =
            (frustum_capacity as u64) * (std::mem::size_of::<[u32; 2]>() as u64);
        let broad_instance_meta_bytes =
            (instance_capacity as u64) * (4 * std::mem::size_of::<u32>() as u64);
        let froxel_bucket_heads_bytes =
            (froxel_bucket_capacity as u64) * (std::mem::size_of::<u32>() as u64);
        let raster_work_queue_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (raster_work_capacity as u64) * (std::mem::size_of::<RasterWorkItem>() as u64);
        let raster_tile_run_queue_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (raster_tile_run_capacity as u64) * (std::mem::size_of::<RasterTileRun>() as u64);
        let raster_tile_run_dispatch_args_bytes = (3 * std::mem::size_of::<u32>()) as u64;
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
        let opaque_fine_depth_tiles_bytes =
            (opaque_fine_depth_tile_capacity as u64) * (2 * std::mem::size_of::<u32>() as u64);
        let telemetry_bytes = ((crate::pipelines::prepass::TELEMETRY_PAGE_COUNTS_OFFSET_WORDS
            + coarse_count_page_capacity) as u64)
            * std::mem::size_of::<u32>() as u64;

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
        prepass_resources.projected_segments = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_projected_segments"),
            size: projected_segments_bytes,
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
        prepass_resources.prefix_indirect_args = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prefix_indirect_args"),
            size: (3 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.telemetry = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_telemetry"),
            size: telemetry_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
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
        prepass_resources.shadow_dom_surface_ids = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_shadow_dom_surface_ids"),
            size: shadow_dom_surface_ids_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.broad_instance_meta = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_broad_instance_meta"),
            size: broad_instance_meta_bytes,
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
        prepass_resources.raster_tile_run_queue = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_raster_tile_run_queue"),
            size: raster_tile_run_queue_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.raster_tile_run_dispatch_args =
            Some(device.create_buffer(&BufferDescriptor {
                label: Some("strand_raster_tile_run_dispatch_args"),
                size: raster_tile_run_dispatch_args_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
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
        prepass_resources.opaque_fine_depth_tiles = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_opaque_fine_depth_tiles"),
            size: opaque_fine_depth_tiles_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        prepass_resources.prepass_task_capacity = prepass_capacity;
        prepass_resources.binning_task_capacity = binning_capacity;
        prepass_resources.instance_capacity = instance_capacity;
        prepass_resources.instance_count = instance_count;
        prepass_resources.max_strands_in_instance = max_strands_in_instance;
        prepass_resources.frustum_capacity = frustum_capacity;
        prepass_resources.frustum_count = frustum_descs.len() as u32;
        prepass_resources.froxel_bucket_capacity = froxel_bucket_capacity;
        prepass_resources.raster_work_capacity = raster_work_capacity;
        prepass_resources.raster_tile_run_capacity = raster_tile_run_capacity;
        prepass_resources.coarse_depth_tile_capacity = coarse_depth_tile_capacity;
        prepass_resources.coarse_range_capacity = coarse_range_capacity;
        prepass_resources.coarse_interval_ref_capacity = coarse_interval_ref_capacity;
        prepass_resources.coarse_count_page_capacity = coarse_count_page_capacity;
        prepass_resources.fine_seg_ref_capacity = fine_seg_ref_capacity;
        prepass_resources.opaque_fine_depth_tile_capacity = opaque_fine_depth_tile_capacity;

        info!(
            "Allocated prepass buffers: strands={} segment_budget={} instances={} frusta={} prepass_cap={} binning_cap={} bucket_cap={} coarse_depth_tiles={} coarse_ranges={} coarse_interval_refs={} coarse_count_pages={} fine_seg_refs={} raster_work_cap={} raster_tile_run_cap={}",
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
            raster_tile_run_capacity,
        );
    }
    if needs_shading_realloc {
        let (texture, view) =
            create_shading_target_texture(&device, instance_count, max_shading_texels_in_instance);
        shading_resources.output_texture_resource = Some(texture);
        shading_resources.output_texture = Some(view);
        for i in 0..2 {
            let (history_texture, history_view) = create_shadow_history_texture(
                &device,
                instance_count,
                max_shading_texels_in_instance,
            );
            shading_resources.shadow_history_texture_resources[i] = Some(history_texture);
            shading_resources.shadow_history_textures[i] = Some(history_view);
        }
        shading_resources
            .shadow_history_index
            .store(0, std::sync::atomic::Ordering::Relaxed);
        shading_resources
            .shadow_history_needs_clear
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    prepass_resources.instance_count = instance_count;
    prepass_resources.max_strands_in_instance = max_strands_in_instance;
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
    if let Some(queue_buf) = &prepass_resources.raster_tile_run_queue {
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
    if let Some(indirect) = &prepass_resources.prefix_indirect_args {
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
    if let Some(surface_ids) = &prepass_resources.shadow_dom_surface_ids {
        render_queue.write_buffer(
            surface_ids,
            0,
            bytemuck::cast_slice(shadow_dom_surface_ids.as_slice()),
        );
    }
    if let Some(strand_instances) = &prepass_resources.strand_instances {
        render_queue.write_buffer(strand_instances, 0, bytemuck::cast_slice(&instances));
    }
}

fn select_authoritative_shadow_cascade(
    main_view: Option<&ExtractedView>,
    strand_world_centers: &[Vec3],
    light: &ExtractedDirectionalLight,
) -> u32 {
    let Some(view) = main_view else {
        return 0;
    };
    if strand_world_centers.is_empty() || light.cascade_shadow_config.bounds.is_empty() {
        return 0;
    }

    let view_from_world = view.world_from_view.to_matrix().inverse();
    let max_depth = strand_world_centers
        .iter()
        .map(|center| -view_from_world.transform_point3(*center).z)
        .filter(|depth| depth.is_finite() && *depth >= 0.0)
        .fold(0.0_f32, f32::max);

    light
        .cascade_shadow_config
        .bounds
        .iter()
        .position(|bound| max_depth < *bound)
        .unwrap_or_else(|| light.cascade_shadow_config.bounds.len().saturating_sub(1))
        .try_into()
        .unwrap_or(0)
}

struct StrandTableIds {
    vertex_id: u32,
    index_id: u32,
    meta_id: u32,
    geo_id: u32,
    material_id: u32,
}

fn buffer_table_id(
    handle: &Handle<VirtualShaderStorageBuffer>,
    storage_buffers: &RenderAssets<GpuVirtualShaderStorageBuffer>,
) -> Option<u32> {
    storage_buffers
        .get(handle)?
        .allocation
        .as_ref()
        .map(|(allocation, _)| allocation.id.0)
}

fn strand_asset_table_ids(
    geometry: &StrandGeometry,
    storage_buffers: &RenderAssets<GpuVirtualShaderStorageBuffer>,
) -> Option<StrandTableIds> {
    let vertex_id = buffer_table_id(&geometry.vertices, storage_buffers)?;
    let index_id = buffer_table_id(&geometry.indices, storage_buffers)?;
    let meta_id = buffer_table_id(&geometry.meta, storage_buffers)?;
    let geo_id = buffer_table_id(&geometry.geos, storage_buffers)?;

    let material_id = storage_buffers
        .get(&geometry.materials)?
        .allocation
        .as_ref()?
        .0
        .id
        .0;

    Some(StrandTableIds {
        vertex_id,
        index_id,
        meta_id,
        geo_id,
        material_id,
    })
}

fn sync_strand_instance_transforms(
    mut commands: Commands,
    query: Query<
        (Entity, &GlobalTransform, Option<&StrandInstanceTransform>),
        With<StrandGeometry>,
    >,
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

struct PackedStrandBuffers {
    vertices: Vec<[f32; 4]>,
    indices: Vec<u32>,
    meta: Vec<StrandMeta>,
    aabb: Aabb3d,
    strand_count: u32,
    index_count: u32,
    max_segments_in_strand: u32,
}

fn pack_strands(
    vertices: impl IntoIterator<Item = Vec3>,
    strands: impl IntoIterator<Item = Vec<u32>>,
) -> Option<PackedStrandBuffers> {
    let vertices: Vec<[f32; 4]> = vertices.into_iter().map(|v| [v.x, v.y, v.z, 1.0]).collect();
    if vertices.is_empty() {
        return None;
    }

    let mut indices = Vec::new();
    let mut meta = Vec::new();
    let mut max_segments_in_strand = 0u32;

    for strand in strands {
        if strand.len() < 2 {
            continue;
        }
        let count = strand.len() as u32;
        let offset = indices.len() as u32;
        max_segments_in_strand = max_segments_in_strand.max(count);
        indices.extend(strand);
        meta.push(StrandMeta::from((count, offset)));
    }

    if meta.is_empty() {
        return None;
    }

    let points = vertices.iter().map(|v| Vec3::new(v[0], v[1], v[2]));
    let strand_count = meta.len() as u32;
    let index_count = indices.len() as u32;
    Some(PackedStrandBuffers {
        aabb: Aabb3d::from_point_cloud(Isometry3d::IDENTITY, points),
        vertices,
        indices,
        meta,
        strand_count,
        index_count,
        max_segments_in_strand,
    })
}

fn pack_indexed_strands(
    vertices: impl IntoIterator<Item = Vec3>,
    strands: &[StrandCacheStrand],
    indices: &[u32],
) -> Option<PackedStrandBuffers> {
    let vertices: Vec<[f32; 4]> = vertices.into_iter().map(|v| [v.x, v.y, v.z, 1.0]).collect();
    if vertices.is_empty() || strands.is_empty() {
        return None;
    }

    let mut meta = Vec::with_capacity(strands.len());
    let mut max_segments_in_strand = 0u32;
    for strand in strands {
        if strand.count < 2 {
            continue;
        }
        max_segments_in_strand = max_segments_in_strand.max(strand.count);
        meta.push(StrandMeta::from((strand.count, strand.offset)));
    }

    if meta.is_empty() {
        return None;
    }

    let points = vertices.iter().map(|v| Vec3::new(v[0], v[1], v[2]));
    let strand_count = meta.len() as u32;
    let index_count = indices.len() as u32;
    Some(PackedStrandBuffers {
        aabb: Aabb3d::from_point_cloud(Isometry3d::IDENTITY, points),
        vertices,
        indices: indices.to_vec(),
        meta,
        strand_count,
        index_count,
        max_segments_in_strand,
    })
}

fn insert_packed_strand_geometry(
    entity: Entity,
    material: &StrandMaterial,
    packed: PackedStrandBuffers,
    storage_buffers: &mut Assets<VirtualShaderStorageBuffer>,
    commands: &mut Commands,
) {
    let geos_data = vec![StrandGeo::new(
        packed.strand_count,
        packed.max_segments_in_strand,
        packed.aabb,
    )];

    let vertex_buffer = VirtualShaderStorageBuffer::from((SlabKind::Vert, packed.vertices));
    let index_buffer = VirtualShaderStorageBuffer::from((SlabKind::Index, packed.indices));
    let meta_buffer = VirtualShaderStorageBuffer::from((SlabKind::StrandMeta, packed.meta));
    let geo_buffer = VirtualShaderStorageBuffer::from((SlabKind::StrandGeo, geos_data));
    let material_buffer =
        VirtualShaderStorageBuffer::from((SlabKind::StrandMaterial, vec![material]));

    let vertex_buffer_handle = storage_buffers.add(vertex_buffer);
    let index_buffer_handle = storage_buffers.add(index_buffer);
    let meta_buffer_handle = storage_buffers.add(meta_buffer);
    let geo_buffer_handle = storage_buffers.add(geo_buffer);
    let material_buffer_handle = storage_buffers.add(material_buffer);

    info!(
        "set strand geometry: {:?}, strands={}, max_vertices_per_strand={}",
        vertex_buffer_handle, packed.strand_count, packed.max_segments_in_strand
    );

    commands.entity(entity).insert(StrandGeometry {
        vertices: vertex_buffer_handle,
        indices: index_buffer_handle,
        meta: meta_buffer_handle,
        geos: geo_buffer_handle,
        materials: material_buffer_handle,
        strand_count: packed.strand_count,
        index_count: packed.index_count,
        max_segments_in_strand: packed.max_segments_in_strand,
        aabb: packed.aabb,
    });
    commands
        .entity(entity)
        .insert(StrandInstanceTransform::default());
}

// main world buffer initialization
fn set_strand_list_geometry(
    query: Query<(Entity, &StrandList, &StrandMaterial), Without<StrandGeometry>>,
    mut storage_buffers: ResMut<Assets<VirtualShaderStorageBuffer>>,
    mut commands: Commands,
) {
    for (entity, strand_list, material) in query.iter() {
        let Some(packed) = pack_strands(
            strand_list.vertices.iter().copied(),
            strand_list.strands.clone(),
        ) else {
            warn!("No valid strands found for entity: {:?}", entity);
            continue;
        };

        insert_packed_strand_geometry(
            entity,
            material,
            packed,
            &mut storage_buffers,
            &mut commands,
        );
        commands.entity(entity).remove::<StrandList>();
    }
}

fn set_cached_strand_geometry(
    query: Query<(Entity, &StrandCache, &StrandMaterial), Without<StrandGeometry>>,
    assets: Res<Assets<StrandCacheAsset>>,
    mut storage_buffers: ResMut<Assets<VirtualShaderStorageBuffer>>,
    mut commands: Commands,
) {
    for (entity, strand_cache, material) in query.iter() {
        let Some(asset) = assets.get(&strand_cache.handle) else {
            continue;
        };

        let scale = asset.cache.scale;
        let vertices = asset
            .cache
            .vertices
            .iter()
            .map(|v| Vec3::new(v[0], v[1], v[2]) * scale);

        let Some(packed) =
            pack_indexed_strands(vertices, &asset.cache.strands, &asset.cache.indices)
        else {
            warn!("No valid cached strands found for entity: {:?}", entity);
            continue;
        };

        insert_packed_strand_geometry(
            entity,
            material,
            packed,
            &mut storage_buffers,
            &mut commands,
        );
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
        render_queue.write_buffer(&buffer, allocation.range.start, material_bytes);
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
    cascade_index: u32,
    fine_depth_tile_base: u32,
    _pad1: u32,
    _pad2: u32,
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
    mut commands: Commands,
    mut cams: Query<(Entity, &TieFroxelsToView, &Camera, &mut FroxelConfig)>,
) {
    for (entity, tie_mode, camera, mut config) in &mut cams {
        let Some(viewport) = camera.physical_viewport_size() else {
            continue;
        };
        let desired_size = froxel_size_from_viewport(viewport, *tie_mode);
        if config.screen_width == desired_size.x && config.screen_height == desired_size.y {
            continue;
        }
        config.screen_width = desired_size.x;
        config.screen_height = desired_size.y;
        commands.entity(entity).insert(NeedsRealloc);
    }
}

fn froxel_size_from_viewport(viewport: UVec2, tie_mode: TieFroxelsToView) -> UVec2 {
    match tie_mode {
        TieFroxelsToView::Native => viewport,
        TieFroxelsToView::Scaled {
            numerator,
            denominator,
        } => {
            let denominator = denominator.max(1);
            UVec2::new(
                viewport
                    .x
                    .saturating_mul(numerator)
                    .div_ceil(denominator)
                    .max(1),
                viewport
                    .y
                    .saturating_mul(numerator)
                    .div_ceil(denominator)
                    .max(1),
            )
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
    for (entity, config, extracted_view) in query.iter() {
        let config_buffer = create_froxel_config_buffer(&device, config);

        raster_resources
            .froxel_config_buffer
            .insert(entity, config_buffer);
        raster_resources
            .frustrum_config
            .insert(entity, config.clone());
        if extracted_view.is_some() {
            let (target_texture, target_view) = recreate_render_target_texture(&device, config);
            let (depth_texture, depth_view) = recreate_render_target_depth_texture(&device, config);
            raster_resources.output_texture_resource = Some(target_texture);
            raster_resources.output_texture = Some(target_view);
            raster_resources.output_depth_resource = Some(depth_texture);
            raster_resources.output_depth = Some(depth_view);
        }
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
        let virtual_w = cfg
            .screen_width
            .saturating_mul(DOM_VIRTUAL_SCALE)
            .clamp(DOM_PAGE_XY, 8192);
        let virtual_h = cfg
            .screen_height
            .saturating_mul(DOM_VIRTUAL_SCALE)
            .clamp(DOM_PAGE_XY, 8192);
        let depth_tile_count = virtual_w
            .div_ceil(DOM_PAGE_XY)
            .saturating_mul(virtual_h.div_ceil(DOM_PAGE_XY))
            .max(1);
        let opacity_tile_count = depth_tile_count.max(1);
        let depth_resident_pages = dom_resident_page_budget(depth_tile_count);
        let opacity_resident_pages = dom_resident_page_budget(opacity_tile_count);
        let op_cfg = VirtualSurfaceConfigBuilder::new()
            .enabled(true)
            .surface_kind(VirtualSurfaceKind::Opacity3D)
            .page_size(UVec3::new(DOM_PAGE_XY, DOM_PAGE_XY, NUM_DOM_SLICES))
            .virtual_extent(UVec3::new(virtual_w, virtual_h, NUM_DOM_SLICES))
            .mip_levels(1)
            .layer_count(1)
            .max_resident_pages(opacity_resident_pages)
            .priority_bias(1.0)
            .residency_mode(VirtualSurfaceResidencyMode::DemandDriven)
            .build();
        let d_cfg = VirtualSurfaceConfigBuilder::new()
            .enabled(true)
            .surface_kind(VirtualSurfaceKind::Depth2DArray)
            .page_size(UVec3::new(DOM_PAGE_XY, DOM_PAGE_XY, 1))
            .virtual_extent(UVec3::new(virtual_w, virtual_h, 1))
            .mip_levels(1)
            .layer_count(1)
            .max_resident_pages(depth_resident_pages)
            .priority_bias(1.0)
            .residency_mode(VirtualSurfaceResidencyMode::DemandDriven)
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
