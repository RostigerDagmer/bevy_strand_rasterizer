use bevy::{
    core_pipeline::core_3d::graph::Core3d,
    math::bounding::Aabb3d,
    pbr::ExtractedDirectionalLight,
    platform::collections::HashMap,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems, extract_component::ExtractComponentPlugin,
        extract_resource::ExtractResourcePlugin, render_graph::RenderGraphExt,
        render_resource::Buffer, renderer::RenderDevice, view::ExtractedView,
    },
    window::WindowResized,
};
use bytemuck::{Pod, Zeroable};
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
            BINNING_POOL_CHUNK_SIZE, BINNING_POOL_MIN_CHUNKS, BINNING_POOL_NUM_HEADS,
            QUEUE_HEADER_WORDS, RasterWorkItem,
        },
        tile_debug::*,
    },
    resources::*,
    shader_types::*,
};

pub const MAX_TEXTURE_EXTENT: u32 = 8192; // for shading (TODO: get this from device limits)

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
        render_app.init_resource::<StrandRasterizerPipeline>();
        render_app.init_resource::<StrandShadingPipeline>();
        render_app.init_resource::<StrandShadingResources>();
        // render_app.init_resource::<StrandShadowPipeline>();
        render_app.init_resource::<StrandShadowResources>();
        render_app.init_resource::<StrandPrepassResources>();
        render_app.init_resource::<ComputeInvocationDims>();
        render_app.init_resource::<StrandPrepassPipeline>();
        render_app.init_resource::<CompositionPipeline>();
        render_app.init_resource::<TileDebugPipeline>();
        render_app.add_systems(
            Render,
            ((
                use_froxel_buffer,
                use_deep_opacity_maps,
                use_prepass_buffers,
                // update_material_buffer,
                // use_strand_geometry.after(prepare_view_uniforms),
            )
                .chain()
                .in_set(RenderSystems::Prepare),),
        );
        render_app.add_systems(
            Render,
            update_strand_prepass_pipeline.after(RenderSystems::PrepareBindGroups),
        );
        render_app.add_systems(
            Render,
            update_strand_raster_pipeline.after(RenderSystems::PrepareBindGroups),
        );
        render_app.add_systems(
            Render,
            update_strand_shading_pipeline.after(RenderSystems::PrepareBindGroups),
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
            .add_render_graph_node::<nodes::raster::StrandRasterizerNode>(
                Core3d,
                nodes::raster::StrandRasterizerLabel,
            )
            // .add_render_graph_node::<StrandShadingNode>(Core3d, StrandShadingLabel)
            // .add_render_graph_node::<StrandShadowRasterizerNode>(
            //     Core3d,
            //     StrandShadowRasterizerLabel,
            // )
            .add_render_graph_node::<nodes::composite::CompositionNode>(
                Core3d,
                nodes::composite::CompositionLabel,
            )
            // connect nodes
            // .add_render_graph_edge(
            //     Core3d,
            //     bevy::core_pipeline::core_3d::graph::Node3d::EndPrepasses,
            //     StrandShadowRasterizerLabel,
            // )
            // .add_render_graph_edge(Core3d, StrandShadowRasterizerLabel, StrandShadingLabel)
            // .add_render_graph_edge(Core3d, StrandShadingLabel, StrandRasterizerLabel) // run after ALL shadow maps are present
            // .add_render_graph_edge(
            //     Core3d,
            //     StrandRasterizerLabel,
            //     CompositionLabel, // Run after strand rasterization
            // )
            .add_render_graph_edge(
                Core3d,
                bevy::core_pipeline::core_3d::graph::Node3d::StartMainPass, // Run before composition
                nodes::prepass::WorkPreparationLabel,
            )
            .add_render_graph_edge(
                Core3d,
                nodes::prepass::WorkPreparationLabel,
                nodes::raster::StrandRasterizerLabel,
            )
            .add_render_graph_edge(
                Core3d,
                nodes::raster::StrandRasterizerLabel,
                nodes::composite::CompositionLabel,
            )
            .add_render_graph_edge(
                Core3d,
                nodes::composite::CompositionLabel,
                nodes::debug::TileDebugLabel,
            )
            .add_render_graph_edge(
                Core3d,
                nodes::debug::TileDebugLabel, // Run after composition
                bevy::core_pipeline::core_3d::graph::Node3d::PostProcessing, // Before standard post-processing
            );
    }
}

fn use_prepass_buffers(
    geometry_query: Query<&StrandGeometry>,
    device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    mut prepass_resources: ResMut<StrandPrepassResources>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
) {
    let mut total_strands = 0u32;
    let mut total_segment_budget = 0u32;
    let mut geo_count = 0u32;
    for geom in &geometry_query {
        total_strands = total_strands.saturating_add(geom.strand_count);
        total_segment_budget = total_segment_budget.saturating_add(
            geom.strand_count
                .saturating_mul(geom.max_segments_in_strand),
        );
        geo_count = geo_count.saturating_add(1);
    }

    if total_strands == 0 {
        return;
    }

    raster_resources.strand_count = Some(total_strands);

    let prepass_capacity = total_strands.next_power_of_two().max(2048);
    let binning_capacity = total_segment_budget
        .next_power_of_two()
        .max(prepass_capacity);
    let geo_capacity = geo_count.next_power_of_two().max(2048);
    let mut frustum_descs: Vec<GpuFrustumDesc> = Vec::new();
    let mut bucket_base = 0u32;
    raster_resources.frustum_ids.clear();
    let mut frusta: Vec<_> = raster_resources
        .frustrum_config
        .iter()
        .map(|(entity, cfg)| (*entity, cfg.clone()))
        .collect();
    frusta.sort_by_key(|(entity, _)| entity.index());
    for (frustum_id, (entity, cfg)) in frusta.into_iter().enumerate() {
        raster_resources.frustum_ids.insert(entity, frustum_id as u32);
        let (_, _, bucket_count) = cfg.get_num_tiles();
        frustum_descs.push(GpuFrustumDesc {
            screen_width: cfg.screen_width,
            screen_height: cfg.screen_height,
            froxel_size_x: cfg.froxel_size_x,
            froxel_size_y: cfg.froxel_size_y,
            depth_slices: cfg.depth_slices,
            bucket_base,
            bucket_count,
            kind: 0, // 0 = camera
        });
        bucket_base = bucket_base.saturating_add(bucket_count);
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
        });
        bucket_base = 1;
    }
    let frustum_capacity = (frustum_descs.len() as u32).next_power_of_two().max(1);
    let froxel_bucket_capacity = bucket_base.next_power_of_two().max(1024);
    let raster_work_capacity = binning_capacity
        .saturating_mul(8)
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
        || prepass_resources.prepass_task_capacity < prepass_capacity
        || prepass_resources.binning_task_capacity < binning_capacity
        || prepass_resources.geo_capacity < geo_capacity
        || prepass_resources.frustum_capacity < frustum_capacity
        || prepass_resources.froxel_bucket_capacity < froxel_bucket_capacity
        || prepass_resources.raster_work_capacity < raster_work_capacity;

    if needs_realloc {
        let prepass_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (prepass_capacity as u64) * (std::mem::size_of::<FinePrepassTask>() as u64);
        let binning_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (binning_capacity as u64) * (std::mem::size_of::<BinningTask>() as u64);
        let geo_bytes = (geo_capacity as u64) * (std::mem::size_of::<u32>() as u64);
        let geo_prefix_bytes = ((geo_capacity as u64) + 1) * (std::mem::size_of::<u32>() as u64);
        let chunk_count = raster_work_capacity.max(BINNING_POOL_MIN_CHUNKS);
        let chunk_stride_bytes =
            (2u64 + BINNING_POOL_CHUNK_SIZE as u64) * std::mem::size_of::<u32>() as u64;
        let chunk_pool_bytes = chunk_count as u64 * chunk_stride_bytes;
        let frustum_table_bytes =
            (frustum_capacity as u64) * (std::mem::size_of::<GpuFrustumDesc>() as u64);
        let froxel_bucket_heads_bytes =
            (froxel_bucket_capacity as u64) * (std::mem::size_of::<u32>() as u64);
        let raster_work_queue_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (raster_work_capacity as u64) * (std::mem::size_of::<RasterWorkItem>() as u64);

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
            size: geo_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.visible_geos_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_visible_geos"),
            size: geo_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.geos_prefix_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_geo_prefix"),
            size: geo_prefix_bytes,
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

        prepass_resources.prepass_task_capacity = prepass_capacity;
        prepass_resources.binning_task_capacity = binning_capacity;
        prepass_resources.geo_capacity = geo_capacity;
        prepass_resources.frustum_capacity = frustum_capacity;
        prepass_resources.froxel_bucket_capacity = froxel_bucket_capacity;
        prepass_resources.raster_work_capacity = raster_work_capacity;

        info!(
            "Allocated prepass buffers: strands={} segment_budget={} geos={} frusta={} prepass_cap={} binning_cap={} bucket_cap={} raster_work_cap={}",
            total_strands,
            total_segment_budget,
            geo_count,
            frustum_descs.len(),
            prepass_capacity,
            binning_capacity,
            froxel_bucket_capacity,
            raster_work_capacity,
        );
    }

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
    if let Some(indirect) = &prepass_resources.indirect_args {
        render_queue.write_buffer(indirect, 0, bytemuck::cast_slice(&zero_dispatch));
    }
    if let Some(free_heads) = &prepass_resources.free_heads {
        let zero_heads = vec![0u32; BINNING_POOL_NUM_HEADS as usize];
        render_queue.write_buffer(free_heads, 0, bytemuck::cast_slice(&zero_heads));
    }
    if let Some(bucket_heads) = &prepass_resources.froxel_bucket_heads {
        let invalid_heads = vec![u32::MAX; prepass_resources.froxel_bucket_capacity as usize];
        render_queue.write_buffer(bucket_heads, 0, bytemuck::cast_slice(&invalid_heads));
    }
    if let Some(frustum_table) = &prepass_resources.frustum_table {
        render_queue.write_buffer(frustum_table, 0, bytemuck::cast_slice(&frustum_descs));
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
    }
}

// uploads changed material paramters to the buffer
// TODO
// pub fn update_material_buffer(
//     query: Query<(Entity, &StrandGeometry, &StrandMaterial)>,
//     storage_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
//     render_queue: Res<RenderQueue>,
// ) {
//     for (entity, geometry, material) in query.iter() {
//         let Some(material_buffer) = storage_buffers.get(&geometry.materials) else {
//             warn!("Material storage buffer not found for entity: {:?}", entity);
//             continue;
//         };
//         let material_bytes = bytemuck::bytes_of(material);
//         render_queue.write_buffer(
//             &material_buffer.buffer, // Get the underlying wgpu::Buffer
//             0,                       // Offset in the buffer to start writing (0 for the start)
//             material_bytes,          // The byte slice to write
//         );
//     }
// }

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
    cams: Query<(Entity, Option<&TieFroxelsToView>)>,
    mut resize_reader: MessageReader<WindowResized>,
    // or listen to WindowResized and map to camera(s)
) {
    for _ in resize_reader.read() {
        info!("window changed");
        for (e, _) in &cams {
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
    query: Query<(Entity, &FroxelConfig), Or<(Added<FroxelConfig>, With<NeedsRealloc>)>>,
    device: Res<RenderDevice>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
) {
    for (entity, config) in query.iter() {
        let config_buffer = create_froxel_config_buffer(&device, config);

        let (target_texture, target_view) = recreate_render_target_texture(&device, config);
        let (depth_texture, depth_view) = recreate_render_target_depth_texture(&device, config);

        info!("Recreated render target: {:?}", config);

        raster_resources.output_texture_resource = Some(target_texture);
        raster_resources.output_depth_resource = Some(depth_texture);
        raster_resources.output_texture = Some(target_view);
        raster_resources.output_depth = Some(depth_view);
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
    device: Res<RenderDevice>,
    mut shadow_resources: ResMut<StrandShadowResources>,
) {
    for (entity, config) in query.iter() {
        if shadow_resources.dom_targets.contains_key(&entity) {
            continue;
        }
        let (
            (opacity_texture, opacity_view, opacity_sampler),
            (depth_texture, depth_view, depth_sampler),
        ) = create_strand_shadow_textures(
            &device,
            config.screen_width,
            config.screen_height,
            config.depth_slices,
        );
        shadow_resources
            .dom_targets
            .insert(entity, (opacity_view, depth_view));
        shadow_resources
            .dom_samplers
            .insert(entity, (opacity_sampler, depth_sampler));
        info!("Added deep opacity maps to resource");
    }
}
