use std::hash::Hash;

use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::gizmos::config;
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
use bevy::render::view::{ViewTarget, ViewUniform, ViewUniformOffset};
use bevy::render::view::{ViewUniforms, prepare_view_uniforms};
use bevy::render::{Render, RenderApp, RenderSet, render_resource::*};
use bevy::utils::HashMap;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
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

/// The row size of the `keys` processed by each workgroup.
pub const NUMBER_OF_ROWS_PER_WORKGROUP: u32 = 16;
pub const NUMBER_OF_THREADS_PER_WORKGROUP: u32 = 256;

/// The number of keys processed by this workgroup.
const SCAN_NUMBER_OF_KEYS_OFFSET: u32 = 4;
/// The scan step reads from the `number_segs` buffer starting at this index.
const SCAN_LOAD_BASE_OFFSET: u32 = 8;
/// The scan step writes to the `number_segs` buffer starting at this index.
const SCAN_SAVE_BASE_OFFSET: u32 = 12;

pub struct StrandRasterizerPlugin;

impl Plugin for StrandRasterizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            // ExtractComponentPlugin::<Strands>::default(),
            ExtractComponentPlugin::<FroxelConfig>::default(),
            ExtractComponentPlugin::<StrandGeometry>::default(),
        ));
        app.init_resource::<StrandAssetResources>();
        app.add_systems(Update, set_strand_geometry);
    }
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        // render_app.add_plugins(GetSubgroupSizePlugin);
        render_app.init_resource::<StrandRasterizerResources>();
        render_app.init_resource::<StrandRasterizerPipeline>();
        render_app.init_resource::<StrandBinningPipeline>();
        render_app.init_resource::<StrandBinningBuffers>();
        render_app.init_resource::<CompositionPipeline>();
        render_app.add_systems(
            Render,
            ((
                use_froxel_buffer,
                use_strand_geometry.after(prepare_view_uniforms),
            )
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
            .add_render_graph_node::<CompositionNode>(Core3d, CompositionLabel)
            // .add_render_graph_edge(
            //     Core3d,
            //     SpatialHashingLabel,
            //     bevy::core_pipeline::core_3d::graph::Node3d::EndPrepasses,
            // )
            // .add_render_graph_edge(Core3d, SimpleGpuSortNodeLabel, SpatialHashingLabel)
            // .add_render_graph_edge(Core3d, StrandRasterizerLabel, SimpleGpuSortNodeLabel)
            .add_render_graph_edge(
                Core3d,
                bevy::core_pipeline::core_3d::graph::Node3d::EndMainPass,
                StrandRasterizerLabel,
            )
            .add_render_graph_edge(
                Core3d,
                StrandRasterizerLabel, // Run after strand rasterization
                CompositionLabel,
            )
            .add_render_graph_edge(
                Core3d,
                CompositionLabel, // Run after composition
                bevy::core_pipeline::core_3d::graph::Node3d::PostProcessing, // Before standard post-processing
            );
    }
}

// Create the bind group for the strand rasterizer
pub fn create_strand_bind_group(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    vertex_buffer: &Buffer,
    index_buffer: &Buffer,
    tile_offsets_buffer: &Buffer,
    tile_counts_buffer: &Buffer,
    meta_buffer: &Buffer,
    packed_segments: &Buffer,
    output_texture: &TextureView,
    froxel_config_buffer: &Buffer,
    view_buffer: BindingResource,
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
                resource: tile_offsets_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: tile_counts_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: packed_segments.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: BindingResource::TextureView(output_texture),
            },
            BindGroupEntry {
                binding: 7,
                resource: froxel_config_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: view_buffer.clone(),
            },
        ],
    )
}

pub fn create_strand_binning_bind_group(
    device: &RenderDevice,
    pipeline: &StrandBinningPipeline,
    strand_points_buffer: &Buffer,
    index_buffer: &Buffer,
    strand_metadata_buffer: &Buffer,
    view_uniforms: BindingResource,
    raster_resources: &StrandRasterizerResources,
    binning_resources: &StrandBinningBuffers,
) -> StrandBinningBindGroup {
    // Count Bind Group
    let count_bind_group = device.create_bind_group(
        Some("strand_count_bind_group"),
        &pipeline.count_layout, // Use count_layout
        &[
            BindGroupEntry {
                binding: 0,
                resource: strand_points_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: index_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: strand_metadata_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: binning_resources
                    .tile_counts_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: raster_resources
                    .froxel_config_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: view_uniforms.clone(),
            },
        ],
    );

    // Scan Bind Group
    let scan_bind_group = device.create_bind_group(
        Some("strand_scan_bind_group"),
        &pipeline.scan_layout, // Use scan_layout
        &[
            BindGroupEntry {
                binding: 0,
                resource: binning_resources
                    .tile_counts_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: binning_resources
                    .tile_offsets_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            // Implicitly handles total count via buffer structure
        ],
    );

    // Init Placement Index Bind Group
    let init_placement_idx_bind_group = device.create_bind_group(
        Some("strand_init_placement_idx_bind_group"),
        &pipeline.init_place_layout, // Use init_place_layout
        &[
            BindGroupEntry {
                binding: 0,
                resource: binning_resources
                    .tile_offsets_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: binning_resources
                    .current_tile_write_indices_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
        ],
    );

    // Place Bind Group
    let place_bind_group = device.create_bind_group(
        Some("strand_place_bind_group"),
        &pipeline.place_layout, // Use place_layout
        &[
            BindGroupEntry {
                binding: 0,
                resource: strand_points_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: index_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: strand_metadata_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: binning_resources
                    .current_tile_write_indices_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: binning_resources
                    .packed_segments_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: raster_resources
                    .froxel_config_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: view_uniforms.clone(),
            },
        ],
    );
    return StrandBinningBindGroup {
        count_bind_group,
        scan_bind_group,
        init_placement_idx_bind_group,
        place_bind_group,
    };
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
    buffers: &StrandBinningBuffers,       // Assume correctly populated buffers
    pipelines: &StrandBinningPipeline,
    render_context: &mut RenderContext,
    froxel_config: &FroxelConfig,
    num_strands_or_segments: u32,
    // Query config, number of strands/segments, tile count
) {
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
    let threads_per_workgroup = NUMBER_OF_THREADS_PER_WORKGROUP;

    // --- Clear count buffer (important!) ---
    // Use encoder.clear_buffer(...) or a small compute shader pass
    encoder.clear_buffer(buffers.tile_counts_buffer.as_ref().unwrap(), 0, None); // Clear whole buffer

    // --- Pass 1: Count ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Count"),
            ..default()
        });
        let Some(count_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.count_pipeline)
        else {
            warn!("Count pipeline not found");
            return;
        };
        pass.set_pipeline(count_pipeline);
        pass.set_bind_group(0, &bind_groups.count_bind_group, &[]);
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
        pass.set_bind_group(0, &bind_groups.scan_bind_group, &[]); // Scan bind group

        // Reuse scan pipeline handles from StrandBinningPipeline
        let Some(scan_sums_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.scan_sums_pipeline)
        else {
            warn!("Scan sums pipeline not found");
            return;
        };
        let Some(scan_last_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.scan_last_pipeline)
        else {
            warn!("Scan last pipeline not found");
            return;
        };
        let Some(scan_prfx_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.scan_prfx_pipeline)
        else {
            warn!("Scan prefix pipeline not found");
            return;
        };

        // Calculate scan hierarchy (copy/adapt logic from reference `run` function)
        let mut load_base = 0u32;
        // Where hierarchical sums are stored. Might be within tile_offsets_buffer itself after num_tiles+1 elements
        // Or could be a separate temp buffer included in the scan_bind_group.
        // Assuming reuse of tile_offsets_buffer for simplicity here, needs careful size calculation.
        let mut save_base = num_tiles; // Start writing sums after the main counts
        let mut rounds = vec![];
        while save_base - load_base > NUMBER_OF_THREADS_PER_WORKGROUP {
            let number_of_workgroups =
                (save_base - load_base).div_ceil(NUMBER_OF_THREADS_PER_WORKGROUP);
            rounds.push((load_base, save_base, number_of_workgroups));
            load_base = save_base;
            save_base += number_of_workgroups;
        }
        info!("Scan rounds: {:?}", rounds);

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
        // info!("Scan last: load_base: {}, save_base: {}", load_base, save_base);
        pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(&load_base));
        pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(&save_base));
        // Set push constant for total count offset if separate buffer not used
        pass.set_push_constants(0, bytemuck::bytes_of(&total_count_buffer_offset));
        pass.dispatch_workgroups(1, 1, 1);

        // Scan Prefix (Propagate scan results back down)
        pass.set_pipeline(scan_prfx_pipeline);
        for (load_base, save_base, number_of_workgroups) in rounds.iter().rev() {
            // info!("Scan prefix: load_base: {}, save_base: {}", save_base, load_base);
            pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(save_base));
            pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(load_base));
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
        let Some(init_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.init_placement_idx_pipeline)
        else {
            warn!("Init placement indices pipeline not found");
            return;
        };
        pass.set_pipeline(init_pipeline);
        pass.set_bind_group(0, &bind_groups.init_placement_idx_bind_group, &[]);
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
        let Some(place_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.place_pipeline)
        else {
            warn!("Place pipeline not found");
            return;
        };
        pass.set_pipeline(place_pipeline);
        pass.set_bind_group(0, &bind_groups.place_bind_group, &[]);
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

fn run_raster_pass(
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandRasterizerPipeline,
    render_context: &mut RenderContext,
    froxel_config: &FroxelConfig,
    resources: &StrandRasterizerResources,
    bind_group: &BindGroup,
) {
    let Some(render_target) = &resources.output_texture else {
        warn!("Output texture not found");
        return;
    };
    let Some(packed_buffer) = &resources.froxel_buffer else {
        warn!("Froxel buffer not found");
        return;
    };
    let Some(config_buffer) = &resources.froxel_config_buffer else {
        warn!("Froxel config buffer not found");
        return;
    };

    let encoder = render_context.command_encoder(); // Get CommandEncoder

    // --- Rasterize ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Rasterize"),
            ..default()
        });
        let Some(raster_pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.rasterize_pipeline)
        else {
            warn!("Raster pipeline not found");
            return;
        };
        pass.set_pipeline(raster_pipeline);
        pass.set_bind_group(
            0,
            bind_group, // Assume correctly populated bind group
            &[],
        );
        // Set push constants if needed
        let pushconstants = PushConstants {
            num_elements: resources.strand_count.unwrap_or(0),
            workgroup_offset: 0,
            scan_load_base: 0,
            scan_save_base: 0,
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

#[derive(Debug, Clone, Default)]
pub struct StrandRasterizerNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct StrandRasterizerLabel;

impl Node for StrandRasterizerNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Check if we have resources
        if !world.contains_resource::<StrandRasterizerResources>() {
            return Ok(());
        }
        let view_entity = graph.view_entity(); // Get the entity this node instance is running for

        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let binning_pipeline = world.resource::<StrandBinningPipeline>();
        let raster_pipeline = world.resource::<StrandRasterizerPipeline>();
        let binning_buffers = world.resource::<StrandBinningBuffers>(); // Get the buffers
        let raster_resources = world.resource::<StrandRasterizerResources>();
        let view_uniforms = world.resource::<ViewUniforms>(); // Get current view uniforms

        let Some(view_uniform_offset) = world.get::<ViewUniformOffset>(view_entity) else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without ViewUniformOffset",
                view_entity
            );
            return Ok(());
        };

        // --- Check Prerequisites ---
        let Some(view_binding) = view_uniforms.uniforms.buffer() else {
            // This can happen early on, or if the buffer is empty
            warn!("ViewUniforms binding not available.");
            return Ok(());
        };

        let view_binding = BindingResource::Buffer(BufferBinding {
            buffer: view_binding,
            offset: view_uniform_offset.offset as u64,
            size: Some(ViewUniform::min_size()),
        });

        let Some(vertex_buffer) = binning_buffers.vertex_buffer.as_ref() else {
            warn!("Vertex buffer handle not found in resources.");
            return Ok(());
        };
        let Some(index_buffer) = binning_buffers.index_buffer.as_ref() else {
            warn!("Index buffer handle not found in resources.");
            return Ok(());
        };
        let Some(meta_buffer) = binning_buffers.meta_buffer.as_ref() else {
            warn!("Meta buffer handle not found in resources.");
            return Ok(());
        };
        let Some(tile_counts_buffer) = binning_buffers.tile_counts_buffer.as_ref() else {
            /* ... */
            return Ok(());
        };
        let Some(tile_offsets_buffer) = binning_buffers.tile_offsets_buffer.as_ref() else {
            /* ... */
            return Ok(());
        };
        let Some(current_tile_write_indices_buffer) =
            binning_buffers.current_tile_write_indices_buffer.as_ref()
        else {
            /* ... */
            return Ok(());
        };
        let Some(packed_segments_buffer) = binning_buffers.packed_segments_buffer.as_ref() else {
            /* ... */
            return Ok(());
        };
        let Some(froxel_config_buffer) = raster_resources.froxel_config_buffer.as_ref() else {
            /* ... */
            return Ok(());
        };
        let Some(frustrum) = raster_resources.frustrum_config else {
            /* ... */
            return Ok(());
        };
        let Some(strand_count) = raster_resources.strand_count else {
            /* ... */
            return Ok(());
        };

        // Get the dimensions to calculate dispatch size
        let Some(frustrum) = raster_resources.frustrum_config else {
            warn!("No frustum size defined.");
            return Ok(());
        };

        let Some(strand_count) = raster_resources.strand_count else {
            warn!("No strand count set.");
            return Ok(());
        };

        let strand_binning_bind_groups = &create_strand_binning_bind_group(
            render_device,
            binning_pipeline,
            vertex_buffer,
            index_buffer,
            meta_buffer,
            view_binding.clone(),
            raster_resources,
            binning_buffers,
        );

        run_binning_pass(
            render_device,
            pipeline_cache,
            strand_binning_bind_groups,
            binning_buffers,
            binning_pipeline,
            render_context,
            &frustrum,
            strand_count,
        );

        let raster_bind_group = create_strand_bind_group(
            render_device,
            &raster_pipeline.bind_group_layout,
            vertex_buffer,
            index_buffer,
            tile_offsets_buffer,
            tile_counts_buffer,
            meta_buffer,
            packed_segments_buffer,
            raster_resources.output_texture.as_ref().unwrap(),
            froxel_config_buffer,
            view_binding,
        );

        // For now, we'll skip the rasterization pass
        // In a future implementation, we would:
        // 1. Wait for the binning pass to complete

        run_raster_pass(
            render_device,
            pipeline_cache,
            raster_pipeline,
            render_context,
            &frustrum,
            raster_resources,
            &raster_bind_group,
        );

        // 3. Integrate with the PBR pipeline

        Ok(())
    }
}

#[derive(Default)]
pub struct CompositionNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct CompositionLabel;

impl Node for CompositionNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();
        let Some(view_target) = world.get::<ViewTarget>(view_entity) else {
            // This can happen if the view doesn't have a ViewTarget
            // (e.g., shadow map views, reflection probes)
            debug!("View entity {:?} does not have a ViewTarget", view_entity);
            return Ok(());
        };
        let Some(composition_pipeline) = world.get_resource::<CompositionPipeline>() else {
            warn!("CompositionPipeline not found");
            return Ok(());
        };
        let Some(strand_raster_resources) = world.get_resource::<StrandRasterizerResources>()
        else {
            warn!("StrandRasterizerResources not found");
            return Ok(());
        };
        let Some(strand_output_texture) = strand_raster_resources.output_texture.as_ref() else {
            warn!("Strand output texture not ready");
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(composition_pipeline.pipeline)
        else {
            warn!("Composition render pipeline not ready");
            return Ok(());
        };

        // Get the input texture (result of main pass)
        // In 0.13+, use get_color_attachment() which handles intermediate textures
        let input_texture = view_target.main_texture_view();

        let bind_group = render_context.render_device().create_bind_group(
            "composition_bind_group",
            &composition_pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(input_texture), // Main scene texture
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&composition_pipeline.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(strand_output_texture), // Strand texture
                },
            ],
        );

        // Use the ViewTarget's post_process_write to get the correct target
        let post_process = view_target.post_process_write();
        let destination_texture = post_process.destination; // TextureView to write to

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("composition_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: destination_texture, // Write to the destination
                resolve_target: None,
                ops: Operations {
                    // Load the existing contents (result of main pass, potentially clear if first post-proc)
                    // If this is the *first* post-processing pass Bevy runs, it might
                    // have cleared it already. If it runs *after* other passes, load.
                    // Load is usually safer.
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Draw a fullscreen triangle

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
        mapped_at_creation: true,
    });

    // initializes with zeros
    let tile_counts = vec![0u32; num_tiles as usize];
    let mut mapped = tile_counts_buffer.slice(..).get_mapped_range_mut();
    mapped.copy_from_slice(bytemuck::cast_slice(&tile_counts));
    drop(mapped);
    tile_counts_buffer.unmap();

    // Create the tile offsets buffer
    let tile_offsets_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_tile_offsets_buffer"),
        size: (num_tiles * 2 + 1) as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // initializes with zeros
    let tile_offsets = vec![0u32; (num_tiles * 2 + 1) as usize];
    let mut mapped = tile_offsets_buffer.slice(..).get_mapped_range_mut();
    mapped.copy_from_slice(bytemuck::cast_slice(&tile_offsets));
    drop(mapped);
    tile_offsets_buffer.unmap();

    // Create the current tile write indices buffer
    let current_tile_write_indices_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_current_tile_write_indices_buffer"),
        size: num_tiles as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // initializes with zeros
    let current_tile_write_indices = vec![0u32; num_tiles as usize];
    let mut mapped = current_tile_write_indices_buffer
        .slice(..)
        .get_mapped_range_mut();
    mapped.copy_from_slice(bytemuck::cast_slice(&current_tile_write_indices));
    drop(mapped);
    current_tile_write_indices_buffer.unmap();

    // Create the packed segments buffer
    let packed_segments_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_packed_segments_buffer"),
        size: 1024 * 1024 * 8 * 4, // Initial size, will be resized after scan
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
    mut binning_resources: ResMut<StrandBinningBuffers>,
) {
    for (entity, config) in query.iter() {
        let config_buffer = create_froxel_config_buffer(&device, config);
        let (
            tile_counts_buffer,
            tile_offsets_buffer,
            current_tile_write_indices_buffer,
            packed_segments_buffer,
        ) = prepare_binning_buffers(&device, config);

        let (texture, view) = create_render_target_texture(&device, config);
        raster_resources.output_texture = Some(view);
        // modify the resource
        raster_resources.froxel_buffer = Some(packed_segments_buffer.clone());
        raster_resources.froxel_config_buffer = Some(config_buffer);
        raster_resources.frustrum_config = Some(config.clone());

        binning_resources.tile_counts_buffer = Some(tile_counts_buffer);
        binning_resources.tile_offsets_buffer = Some(tile_offsets_buffer);
        binning_resources.current_tile_write_indices_buffer =
            Some(current_tile_write_indices_buffer);
        binning_resources.packed_segments_buffer = Some(packed_segments_buffer);

        info!("Added froxel buffers to resource");
    }
}

// render world buffer retrieval
fn use_strand_geometry(
    query: Query<(Entity, &StrandGeometry)>,
    storage_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    device: Res<RenderDevice>,
    raster_pipeline: Res<StrandRasterizerPipeline>,
    binning_pipeline: Res<StrandBinningPipeline>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
    mut binning_resources: ResMut<StrandBinningBuffers>,
    view_uniforms: Res<ViewUniforms>,
) {
    // This is an example of how to retrieve the shader storage buffer created in the main world above
    // and use it in the render world.
    for (entity, geometry) in query.iter() {
        // if raster_resources.bind_group.is_some() {
        //     continue;
        // }

        // --- Raster resources ---
        info!("Using strand geometry for entity: {:?}", entity);
        let Some(index_storage_buffer) = storage_buffers.get(&geometry.indices) else {
            warn!("Index storage buffer not found for entity: {:?}", entity);
            continue;
        };
        info!("[{:?}] Index storage buffer found.", entity);
        let Some(vertex_storage_buffer) = storage_buffers.get(&geometry.vertices) else {
            warn!("Vertex storage buffer not found for entity: {:?}", entity);
            continue;
        };

        let Some(meta_storage_buffer) = storage_buffers.get(&geometry.meta) else {
            warn!("Meta storage buffer not found for entity: {:?}", entity);
            continue;
        };
        info!(
            "[{:?}] Vertex storage buffer found: {:?}",
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

        let Some(tile_counts_buffer) = binning_resources.tile_counts_buffer.as_ref() else {
            warn!("Tile counts buffer not found");
            continue;
        };

        binning_resources.vertex_buffer = Some(vertex_storage_buffer.buffer.clone());
        binning_resources.meta_buffer = Some(meta_storage_buffer.buffer.clone());
        binning_resources.index_buffer = Some(index_storage_buffer.buffer.clone());
        raster_resources.strand_count = Some(geometry.strand_count);

        info!("Created bind group for strand rasterizer");
    }
}

#[derive(Component)]
struct StrandAsset {
    handle: Handle<DsonAsset>,
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let handle: Handle<DsonAsset> = asset_server.load("dForce Pixie Cut_708408.dsf".to_string());
    commands.spawn((StrandAsset { handle }));

    commands.spawn((
        // Camera3d::default(),
        FroxelConfig::default(),
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
        PanOrbitCamera::default(),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.05, 0.05, 0.05))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
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
        .add_plugins(PanOrbitCameraPlugin)
        .init_asset::<DsonAsset>()
        .init_asset_loader::<DsonAssetLoader>()
        .add_systems(Startup, setup)
        // .add_systems(Update, debug_print_geo)
        .run();
}
