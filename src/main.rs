use std::hash::Hash;

use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphApp, RenderGraphContext, RenderLabel,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::storage::GpuShaderStorageBuffer;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::render::view::ViewUniform;
use bevy::render::view::ViewUniforms;
use bevy::render::{Render, RenderApp, RenderSet, render_resource::*};
use bevy::utils::HashMap;
use bevy_radix_sort::{self, GetSubgroupSizePlugin, RadixSortPlugin, dispatch_workgroup_ext};
use bytemuck::{Pod, Zeroable};
mod dson;
use dson::*;
mod components;
use components::*;
mod resources;
use resources::*;
mod shader_types;
use shader_types::*;

const MAX_NUMBER_OF_STRANDS: u32 = 1024 * 1024; // for physics
const SCAN_NUMBER_OF_THREADS_PER_WORKGROUP: u32 = 256; // Or whatever the scan shader uses
/// The row size of the `keys` processed by each workgroup.
pub const NUMBER_OF_ROWS_PER_WORKGROUP: u32 = 16;
pub const NUMBER_OF_THREADS_PER_WORKGROUP: u32 = 256;

pub struct StrandRasterizerPlugin;

impl Plugin for StrandRasterizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            // ExtractComponentPlugin::<Strands>::default(),
            ExtractComponentPlugin::<FroxelConfig>::default(),
            ExtractComponentPlugin::<StrandGeometry>::default(),
        ));
        app.add_plugins(GetSubgroupSizePlugin);
        app.init_resource::<StrandAssetResources>();
        app.add_systems(Update, set_strand_geometry);
    }
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<StrandRasterizerResources>();
        render_app.init_resource::<StrandComputePipeline>();
        render_app.add_systems(
            Render,
            ((use_froxel_buffer, use_strand_geometry)
                .chain()
                .in_set(RenderSet::Prepare),),
        );
        render_app
            // Bevy's renderer uses a render graph which is a collection of nodes in a directed acyclic graph.
            // It currently runs on each view/camera and executes each node in the specified order.
            // It will make sure that any node that needs a dependency from another node
            // only runs when that dependency is done.
            //
            // Each node can execute arbitrary work, but it generally runs at least one render pass.
            // A node only has access to the render world, so if you need data from the main world
            // you need to extract it manually or with the plugin like above.
            // Add a [`Node`] to the [`RenderGraph`]
            // The Node needs to impl FromWorld (dealt with by derive(Default))
            // .add_render_graph_node::<SpatialHashingNode>(Core3d, SpatialHashingLabel)
            // .add_render_graph_node::<SimpleGpuSortNode>(Core3d, SimpleGpuSortNodeLabel)
            .add_render_graph_node::<StrandRasterizerNode>(Core3d, StrandRasterizerLabel)
            // .add_render_graph_edge(
            //     Core3d,
            //     SpatialHashingLabel,
            //     bevy::core_pipeline::core_3d::graph::Node3d::EndPrepasses,
            // )
            // .add_render_graph_edge(Core3d, SimpleGpuSortNodeLabel, SpatialHashingLabel)
            // .add_render_graph_edge(Core3d, StrandRasterizerLabel, SimpleGpuSortNodeLabel)
            .add_render_graph_edge(
                Core3d,
                bevy::core_pipeline::core_3d::graph::Node3d::PostProcessing,
                StrandRasterizerLabel,
            );
    }
}

// Create the bind group for the strand rasterizer
pub fn create_strand_bind_group(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    vertex_buffer: &Buffer,
    index_buffer: &Buffer,
    meta_buffer: &Buffer,
    froxel_buffer: &Buffer,
    output_texture: &TextureView,
    froxel_config_buffer: &Buffer,
    view_buffer: &DynamicUniformBuffer<ViewUniform>,
) -> BindGroup {
    device.create_bind_group(
        Some("strand_rasterizer_bind_group"),
        layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: vertex_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: index_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: meta_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: froxel_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(output_texture),
            },
            BindGroupEntry {
                binding: 5,
                resource: froxel_config_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: view_buffer.binding().unwrap(),
            },
        ],
    )
}


pub fn create_strand_binning_bind_group(
    device: &RenderDevice,
    bind_group_layout: &BindGroupLayout,
    froxel_buffer: &Buffer,
    tile_counts_buffer: &Buffer,
    tile_offsets_buffer: &Buffer,
    current_tile_write_indices_buffer: &Buffer,
    packed_segments_buffer: &Buffer,
) -> (
    BindGroup, // count_bind_group
    BindGroup, // scan_bind_group
    BindGroup, // init_placement_idx_bind_group
    BindGroup, // place_bind_group
) {
    let count_bind_group = device.create_bind_group(
        Some("strand_count_bind_group"),
        bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: froxel_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: tile_counts_buffer.as_entire_binding(),
            },
        ],
    );

    let scan_bind_group = device.create_bind_group(
        Some("strand_scan_bind_group"),
        bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: tile_counts_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: tile_offsets_buffer.as_entire_binding(),
            },
        ],
    );

    let init_placement_idx_bind_group = device.create_bind_group(
        Some("strand_init_placement_idx_bind_group"),
        bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: tile_offsets_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: current_tile_write_indices_buffer.as_entire_binding(),
            },
        ],
    );

    let place_bind_group = device.create_bind_group(
        Some("strand_place_bind_group"),
        bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: tile_offsets_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: current_tile_write_indices_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: packed_segments_buffer.as_entire_binding(),
            },
        ],
    );

    (count_bind_group, scan_bind_group, init_placement_idx_bind_group, place_bind_group)
}

// Create froxel buffer based on screen dimensions and froxel size
fn create_froxel_buffer(device: &RenderDevice, config: &FroxelConfig) -> (Buffer, (u32, u32, u32)) {
    // Calculate froxel grid dimensions
    let froxels_x = (config.screen_width + config.froxel_size_x - 1) / config.froxel_size_x;
    let froxels_y = (config.screen_height + config.froxel_size_y - 1) / config.froxel_size_y;
    let froxels_z = config.depth_slices;

    let froxel_count = froxels_x * froxels_y * froxels_z;

    // Define maximum strands per froxel
    const MAX_STRANDS_PER_FROXEL: u32 = 256;

    // Calculate buffer size:
    // - 4 bytes for the strand count (atomic<u32>)
    // - 4 bytes per strand index * MAX_STRANDS_PER_FROXEL
    let strand_indices_size = MAX_STRANDS_PER_FROXEL * 4;
    let froxel_size = 4 + strand_indices_size;
    let buffer_size = froxel_count * froxel_size;

    // Create the buffer
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("strand_froxel_buffer"),
        size: buffer_size as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // Initialize buffer with zeros (important for atomic counters)
    let mut mapped = buffer.slice(..).get_mapped_range_mut();
    for byte in mapped.iter_mut() {
        *byte = 0;
    }
    drop(mapped);
    buffer.unmap();

    info!(
        "Created froxel buffer with dimensions {}x{}x{} ({} froxels, {} bytes)",
        froxels_x, froxels_y, froxels_z, froxel_count, buffer_size
    );

    (buffer, (froxels_x, froxels_y, froxels_z))
}

// Create froxel configuration uniform buffer
pub fn create_froxel_config_buffer(device: &RenderDevice, config: &FroxelConfig) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("strand_froxel_config_buffer"),
        size: std::mem::size_of::<FroxelConfig>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // Initialize with configuration
    let mut mapped = buffer.slice(..).get_mapped_range_mut();
    mapped.copy_from_slice(bytemuck::bytes_of(config));
    drop(mapped);
    buffer.unmap();
    buffer
}

// Create output texture for the rasterizer
pub fn create_render_target_texture(
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

fn run_binning_pass(
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    bind_groups: &StrandBinningBindGroup, // Assume correctly populated bind groups
    pipelines: &StrandBinningPipeline,
    render_context: &mut RenderContext,
    froxel_config: &FroxelConfig,
    num_strands_or_segments: u32,
    // Query config, number of strands/segments, tile count
) {
    if [
        bind_groups.tile_counts_buffer,
        bind_groups.tile_offsets_buffer,
        bind_groups.current_tile_write_indices_buffer,
        bind_groups.packed_segments_buffer,
    ]
    .iter()
    .any(|buffer| buffer.is_none())
    {
        warn!("Binning buffers not ready");
        return;
    }

    let encoder = render_context.command_encoder(); // Get CommandEncoder

    let tile_size_x = froxel_config.froxel_size_x;
    let tile_size_y = froxel_config.froxel_size_y;
    let depth_slices = froxel_config.depth_slices;
    let screen_width = froxel_config.screen_width;

    let num_tiles_x = (screen_width + tile_size_x - 1) / tile_size_x;
    let num_tiles_y = (froxel_config.screen_height + tile_size_y - 1) / tile_size_y;
    let num_tiles = num_tiles_x * num_tiles_y * depth_slices;

    let max_compute_workgroups_per_dimension =
        render_device.limits().max_compute_workgroups_per_dimension;

    // --- Clear count buffer (important!) ---
    // Use encoder.clear_buffer(...) or a small compute shader pass
    encoder.clear_buffer(&bind_groups.tile_counts_buffer.unwrap(), 0, None); // Clear whole buffer

    // --- Pass 1: Count ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Count"),
            ..default()
        });
        let count_pipeline = pipeline_cache
            .get_compute_pipeline(pipelines.count_pipeline)
            .unwrap();
        pass.set_pipeline(count_pipeline);
        pass.set_bind_group(0, &bind_groups.count_bind_group.unwrap(), &[]);
        // Set push constants if needed (e.g., num_strands_or_segments)
        // pass.set_push_constants(...);

        // Dispatch based on number of strands or segments
        let workgroup_size_x = 64; // Match shader
        let num_workgroups_x = (num_strands_or_segments + workgroup_size_x - 1) / workgroup_size_x;
        pass.dispatch_workgroups(num_workgroups_x, 1, 1); // Adjust dispatch logic as needed
    }

    // --- Pass 2: Scan (Adapted from reference) ---
    let total_count_buffer_offset = num_tiles * 4; // Offset to the last u32 element storing the total
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Scan"),
            ..default()
        });
        pass.set_bind_group(0, &bind_groups.scan_bind_group.unwrap(), &[]); // Scan bind group

        let number_of_elements_to_scan = num_tiles; // We are scanning the tile counts

        // Reuse scan pipeline handles from StrandBinningPipeline
        let scan_sums_pipeline = pipeline_cache
            .get_compute_pipeline(pipelines.scan_sums_pipeline)
            .unwrap();
        let scan_last_pipeline = pipeline_cache
            .get_compute_pipeline(pipelines.scan_last_pipeline)
            .unwrap();
        let scan_prfx_pipeline = pipeline_cache
            .get_compute_pipeline(pipelines.scan_prfx_pipeline)
            .unwrap();

        // Calculate scan hierarchy (copy/adapt logic from reference `run` function)
        let mut load_base = 0u32;
        // Where hierarchical sums are stored. Might be within tile_offsets_buffer itself after num_tiles+1 elements
        // Or could be a separate temp buffer included in the scan_bind_group.
        // Assuming reuse of tile_offsets_buffer for simplicity here, needs careful size calculation.
        let mut save_base = num_tiles; // Start writing sums after the main counts
        let mut rounds = vec![];
        while save_base - load_base > SCAN_NUMBER_OF_THREADS_PER_WORKGROUP {
            let number_of_workgroups =
                (save_base - load_base).div_ceil(SCAN_NUMBER_OF_THREADS_PER_WORKGROUP);
            rounds.push((load_base, save_base, number_of_workgroups));
            load_base = save_base;
            save_base += number_of_workgroups;
        }

        // Scan Sums (Hierarchical reduction)
        pass.set_pipeline(scan_sums_pipeline);
        for (load_base, save_base, number_of_workgroups) in rounds.iter() {
            // Set push constants for scan load/save base (offsets in tile_offsets_buffer)
            pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(load_base));
            pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(save_base));
            dispatch_workgroup_ext(
                &mut pass,
                *number_of_workgroups,
                max_compute_workgroups_per_dimension,
                0, // Handle large dispatches if needed
            );
        }

        // Scan Last (Scan the final reduced block)
        pass.set_pipeline(scan_last_pipeline);
        pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(&load_base));
        pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(&save_base));
        // Set push constant for total count offset if separate buffer not used
        pass.set_push_constants(0, bytemuck::bytes_of(&total_count_buffer_offset));
        pass.dispatch_workgroups(1, 1, 1);

        // Scan Prefix (Propagate scan results back down)
        pass.set_pipeline(scan_prfx_pipeline);
        for (load_base, save_base, number_of_workgroups) in rounds.iter().rev() {
            pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(load_base));
            pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(save_base));
            dispatch_workgroup_ext(
                &mut pass,
                *number_of_workgroups,
                max_compute_workgroups_per_dimension,
                0,
            );
        }
    }

    // --- (Optional) Read back total count for buffer resizing ---
    // let total_segment_refs = read_buffer_value_u32(..., &bind_groups.tile_offsets_buffer, total_count_buffer_offset);
    // resize_packed_segments_buffer_if_needed(..., bind_groups.packed_segments_buffer, total_segment_refs);
    // Update bind groups if buffer was recreated

    // --- Pass 2.5: Initialize Placement Indices ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init Placement Indices"),
            ..default()
        });
        let init_pipeline = pipeline_cache
            .get_compute_pipeline(pipelines.init_placement_idx_pipeline)
            .unwrap();
        pass.set_pipeline(init_pipeline);
        pass.set_bind_group(0, &bind_groups.init_placement_idx_bind_group.unwrap(), &[]);
        // Dispatch one thread per tile
        let workgroup_size = 256; // Example
        let num_workgroups = (num_tiles + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // --- Pass 3: Place ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Place"),
            ..default()
        });
        let place_pipeline = pipeline_cache
            .get_compute_pipeline(pipelines.place_pipeline)
            .unwrap();
        pass.set_pipeline(place_pipeline);
        pass.set_bind_group(0, &bind_groups.place_bind_group.unwrap(), &[]);
        // Set push constants if needed
        // pass.set_push_constants(...);

        // Dispatch based on number of strands or segments (same as Pass 1)
        let workgroup_size_x = 64; // Match shader
        let num_workgroups_x = (num_strands_or_segments + workgroup_size_x - 1) / workgroup_size_x;
        pass.dispatch_workgroups(num_workgroups_x, 1, 1);
    }

    // --- Binning complete ---
    // packed_segments_buffer and tile_offsets_buffer are ready for the rasterizer
}

#[derive(Debug, Clone, Default)]
pub struct StrandRasterizerNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct StrandRasterizerLabel;

impl Node for StrandRasterizerNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Check if we have resources
        if !world.contains_resource::<StrandRasterizerResources>() {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let strand_raster_pipeline = world.resource::<StrandComputePipeline>();
        let strand_binning_pipeline = world.resource::<StrandBinningPipeline>();
        let strand_binning_bind_groups = world.resource::<StrandBinningBindGroup>();
        let resources = world.resource::<StrandRasterizerResources>();

        // Get the dimensions to calculate dispatch size
        let Some(frustrum) = resources.frustrum_config else {
            warn!("No frustum size defined.");
            return Ok(());
        };

        let Some(strand_count) = resources.strand_count else {
            warn!("No strand count set.");
            return Ok(());
        };

        run_binning_pass(
            render_device,
            pipeline_cache,
            strand_binning_bind_groups,
            strand_binning_pipeline,
            render_context,
            &frustrum,
            strand_count,
        );

        // For now, we'll skip the rasterization pass
        // In a future implementation, we would:
        // 1. Wait for the binning pass to complete
        // 2. Execute the rasterization pass
        // 3. Integrate with the PBR pipeline

        Ok(())
    }
}

fn prepare_binning_buffers(
    render_device: &RenderDevice,
    froxel_config: &FroxelConfig,
    // Query for number of strands, calculate num_tiles etc.
) -> (Buffer, Buffer, Buffer, Buffer) {
    // Create/resize buffers here based on num_strands, screen_res, tile_size
    // tile_counts_buffer: size = num_tiles * 4
    // tile_offsets_buffer: size = (num_tiles + 1) * 4 (potentially larger if scan needs more temp space)
    // current_tile_write_indices_buffer: size = num_tiles * 4
    // packed_segments_buffer: Initially small or from a pool. Will be resized after scan.
    // Calculate the number of tiles based on screen resolution and tile size
    let tile_size_x = froxel_config.froxel_size_x;
    let tile_size_y = froxel_config.froxel_size_y;
    let depth_slices = froxel_config.depth_slices;
    let screen_width = froxel_config.screen_width;

    let num_tiles_x = (screen_width + tile_size_x - 1) / tile_size_x;
    let num_tiles_y = (froxel_config.screen_height + tile_size_y - 1) / tile_size_y;
    let num_tiles = num_tiles_x * num_tiles_y * depth_slices;

    // Create the tile counts buffer
    let tile_counts_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_tile_counts_buffer"),
        size: num_tiles as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true, // zero initialization happens in ComputePipelineDescriptor
    });

    // Create the tile offsets buffer
    let tile_offsets_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_tile_offsets_buffer"),
        size: (num_tiles + 1) as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true, // zero initialization happens in ComputePipelineDescriptor
    });

    // Create the current tile write indices buffer
    let current_tile_write_indices_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_current_tile_write_indices_buffer"),
        size: num_tiles as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true, // zero initialization happens in ComputePipelineDescriptor
    });

    // Create the packed segments buffer
    let packed_segments_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_packed_segments_buffer"),
        size: 1024 * 1024 * 4, // Initial size, will be resized after scan
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true, // zero initialization happens in ComputePipelineDescriptor
    });

    (
        tile_counts_buffer,
        tile_offsets_buffer,
        current_tile_write_indices_buffer,
        packed_segments_buffer,
    )
}

// main world buffer initialization
// packs StrandGeometry
fn set_strand_geometry(
    query: Query<(Entity, &StrandAsset), Without<StrandGeometry>>,
    assets: Res<Assets<DsonAsset>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut commands: Commands,
) {
    for (entity, strand_asset) in query.iter() {
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
        let vertices: Vec<[f32; 3]> = geometry.vertices.values.clone();
        let vertex_buffer = ShaderStorageBuffer::from(vertices);
        // info!("Vertex buffer: {:?}", vertex_buffer);
        let vertex_buffer_handle = storage_buffers.add(vertex_buffer);

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
                .fold((Vec::new(), Vec::new()), |mut acc, strand| {
                    let strand_indices = &strand[2..];
                    acc.0.extend_from_slice(strand_indices);
                    let last_strand_offset = acc.1.last().copied().unwrap_or((0, 0)).1;
                    acc.1.push((
                        strand_indices.len() as u32,
                        last_strand_offset + strand_indices.len() as u32,
                    ));
                    acc
                });

        let strand_indices = packed_strand_info.0;
        let strand_meta: Vec<StrandMeta> = packed_strand_info
            .1
            .into_iter()
            .map(StrandMeta::from)
            .collect();

        let index_buffer = ShaderStorageBuffer::from(strand_indices);
        let meta_buffer = ShaderStorageBuffer::from(strand_meta);
        // info!("Index buffer: {:?}", index_buffer);
        let index_buffer_handle = storage_buffers.add(index_buffer);
        let meta_buffer_handle = storage_buffers.add(meta_buffer);

        commands.entity(entity).insert(StrandGeometry {
            vertices: vertex_buffer_handle,
            indices: index_buffer_handle,
            meta: meta_buffer_handle,
            strand_count: polyline_list.values.len() as u32,
        });
    }
}

// render world buffe retrieval
fn use_froxel_buffer(
    query: Query<(Entity, &FroxelConfig), Added<FroxelConfig>>,
    device: Res<RenderDevice>,
    pipeline: Res<StrandBinningPipeline>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
    mut binning_resources: ResMut<StrandBinningBindGroup>,
) {
    for (entity, config) in query.iter() {
        let (froxel_buffer, size) = create_froxel_buffer(&device, config);
        let config_buffer = create_froxel_config_buffer(&device, config);
        let (
            tile_counts_buffer,
            tile_offsets_buffer,
            current_tile_write_indices_buffer,
            packed_segments_buffer,
        ) = prepare_binning_buffers(&device, config);
        let (count_bind_group, scan_bind_group, init_placement_idx_bind_group, place_bind_group) =
            create_strand_binning_bind_group(
                &device,
                &pipeline.bind_group_layout,
                &froxel_buffer,
                &tile_counts_buffer,
                &tile_offsets_buffer,
                &current_tile_write_indices_buffer,
                &packed_segments_buffer,
            );
        let (texture, view) = create_render_target_texture(&device, config);
        raster_resources.output_texture = Some(view);
        // modify the resource
        raster_resources.froxel_buffer = Some(froxel_buffer);
        raster_resources.froxel_config_buffer = Some(config_buffer);
        raster_resources.frustrum_config = Some(config.clone());

        binning_resources.tile_counts_buffer = Some(tile_counts_buffer);
        binning_resources.tile_offsets_buffer = Some(tile_offsets_buffer);
        binning_resources.current_tile_write_indices_buffer =
            Some(current_tile_write_indices_buffer);
        binning_resources.packed_segments_buffer = Some(packed_segments_buffer);

        binning_resources.count_bind_group = Some(count_bind_group);
        binning_resources.scan_bind_group = Some(scan_bind_group);
        binning_resources.init_placement_idx_bind_group = Some(init_placement_idx_bind_group);
        binning_resources.place_bind_group = Some(place_bind_group);

        info!("Added froxel buffers to resource");
    }
}

// render world buffer retrieval
fn use_strand_geometry(
    query: Query<(Entity, &StrandGeometry), (Added<StrandGeometry>)>,
    storage_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    device: Res<RenderDevice>,
    pipeline: Res<StrandComputePipeline>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
    view_uniforms: Res<ViewUniforms>,
) {
    // This is an example of how to retrieve the shader storage buffer created in the main world above
    // and use it in the render world.
    for (entity, geometry) in query.iter() {
        if raster_resources.bind_group.is_some() {
            continue;
        }
        info!("Using strand geometry for entity: {:?}", entity);
        let Some(index_storage_buffer) = storage_buffers.get(&geometry.indices) else {
            warn!("Index storage buffer not found for entity: {:?}", entity);
            continue;
        };
        info!("[{:?}] Index storage buffer created.", entity);
        let Some(vertex_storage_buffer) = storage_buffers.get(&geometry.vertices) else {
            warn!("Vertex storage buffer not found for entity: {:?}", entity);
            continue;
        };

        let Some(meta_storage_buffer) = storage_buffers.get(&geometry.meta) else {
            warn!("Meta storage buffer not found for entity: {:?}", entity);
            continue;
        };
        info!(
            "[{:?}] Vertex storage buffer created: {:?}",
            entity, vertex_storage_buffer.buffer
        );
        let Some(froxel_buffer) = raster_resources.froxel_buffer.as_ref() else {
            warn!("Froxel buffer not found");
            continue;
        };
        let Some(output_texture) = raster_resources.output_texture.as_ref() else {
            warn!("Output texture not found");
            continue;
        };
        let Some(froxel_config_buffer) = raster_resources.froxel_config_buffer.as_ref() else {
            warn!("Froxel config buffer not found");
            continue;
        };

        raster_resources.bind_group = Some(create_strand_bind_group(
            &device,
            &pipeline.bind_group_layout,
            &vertex_storage_buffer.buffer,
            &index_storage_buffer.buffer,
            &meta_storage_buffer.buffer,
            &froxel_buffer,
            &output_texture,
            &froxel_config_buffer,
            &view_uniforms.uniforms,
        ));
        raster_resources.strand_count = Some(geometry.strand_count);

        info!("Created bind group for strand rasterizer");
    }
}

#[derive(Component)]
struct StrandAsset {
    handle: Handle<DsonAsset>,
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle: Handle<DsonAsset> = asset_server.load("dForce Pixie Cut_708408.dsf".to_string());
    commands.spawn((StrandAsset { handle }));

    commands.spawn((
        Camera3d::default(),
        FroxelConfig::default(),
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
    ));
}

fn debug_print_geo(query: Query<&StrandAsset>, assets: Res<Assets<DsonAsset>>) {
    for st_asset in query.iter() {
        let Some(asset) = assets.get(&st_asset.handle) else {
            continue;
        };
        info!(
            "geo: {:?}",
            asset.dson_file.geometry_library.as_ref().unwrap()[0].polyline_list
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(StrandRasterizerPlugin)
        .init_asset::<DsonAsset>()
        .init_asset_loader::<DsonAssetLoader>()
        .add_systems(Startup, setup)
        // .add_systems(Update, debug_print_geo)
        .run();
}
