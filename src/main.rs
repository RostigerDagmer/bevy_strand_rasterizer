use std::hash::Hash;

use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::ecs as bevy_ecs;
use bevy::ecs::label::DynHash;
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
use bevy::render::{render_resource::*, Render, RenderApp, RenderSet};
use bevy::utils::HashMap;
use bytemuck::{Pod, Zeroable};
use bevy_radix_sort::{self, GetSubgroupSizePlugin, RadixSortPlugin};
mod dson;
use dson::*;
mod spatial_hashing;
use spatial_hashing::*;
mod sorting;
use sorting::*;


#[derive(Component, ExtractComponent, Debug, Clone)]
pub struct StrandGeometry {
    pub vertices: Handle<ShaderStorageBuffer>,
    pub indices: Handle<ShaderStorageBuffer>,
    pub strand_count: u32,
}

// #[derive(Component, Clone, ExtractComponent)]
// pub struct Strands; // marker component for strand assets

const MAX_NUMBER_OF_STRANDS: u32 = 1024 * 1024;

pub struct StrandRasterizerPlugin;

impl Plugin for StrandRasterizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            // ExtractComponentPlugin::<Strands>::default(),
            ExtractComponentPlugin::<FroxelConfig>::default(),
            ExtractComponentPlugin::<StrandGeometry>::default(),
        ));
        app.add_plugins(GetSubgroupSizePlugin)
        .add_plugins(RadixSortPlugin {
            settings: MAX_NUMBER_OF_STRANDS.into(),
        })
        .add_plugins(SpatialHashingPlugin);
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
            (
                render_strand_rasterizer,
                (set_froxel_buffer, use_strand_geometry).chain().in_set(RenderSet::Prepare),
            ),
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
            .add_render_graph_node::<StrandRasterizerNode>(
                // Specify the label of the graph, in this case we want the graph for 3d
                Core3d,
                // It also needs the label of the node
                StrandRasterizerLabel,
            )
            .add_render_graph_edge(
                Core3d,
                StrandRasterizerLabel,
                bevy::core_pipeline::core_3d::graph::Node3d::MainOpaquePass,
            );
    }
}

#[derive(Resource)]
struct StrandComputePipeline {
    // stub
    bind_group_layout: BindGroupLayout,
    binning_pipeline: CachedComputePipelineId,
    rasterize_pipeline: CachedComputePipelineId,
}

#[derive(Copy, Clone, Pod, Zeroable, Debug)]
#[repr(C)]
struct PushConstants {
    strand_count: u32, // stub in case we need push constants
}

#[derive(Component, ExtractComponent, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct FroxelConfig {
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
    aabb_min_x: u32,
    aabb_min_y: u32,
    aabb_min_z: f32,
    aabb_max_x: u32,
    aabb_max_y: u32,
    aabb_max_z: f32,
    // _padding: [u32; 4],
}
impl Default for FroxelConfig {
    fn default() -> Self {
        Self {
            screen_width: 1920,
            screen_height: 1080,
            froxel_size_x: 8,
            froxel_size_y: 8,
            depth_slices: 16,
            aabb_min_x: 0,
            aabb_min_y: 0,
            aabb_min_z: 0.0,
            aabb_max_x: 1920 / 8,
            aabb_max_y: 1080 / 8,
            aabb_max_z: 1.0,
            // _padding: [0; 4],
        }
    }
}

// Update the bind group layout to include froxel buffer and config
pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
    device.create_bind_group_layout(
        "strand_rasterizer_bind_group_layout",
        &[
            // Vertex buffer (read-only storage buffer)
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Index buffer (read-only storage buffer)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Froxel buffer (read-write storage buffer)
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output texture (write-only storage texture)
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            // Froxel configuration (uniform buffer)
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // View-Projection matrix (uniform buffer)
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    )
}

// Create the bind group for the strand rasterizer
pub fn create_strand_bind_group(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    vertex_buffer: &Buffer,
    index_buffer: &Buffer,
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
                resource: froxel_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(output_texture),
            },
            BindGroupEntry {
                binding: 4,
                resource: froxel_config_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: view_buffer.binding().unwrap(),
            },
        ],
    )
}

// Create froxel buffer based on screen dimensions and froxel size
pub fn create_froxel_buffer(
    device: &RenderDevice,
    config: &FroxelConfig,
) -> (Buffer, (u32, u32, u32)) {
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
pub fn create_output_texture(
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

impl FromWorld for StrandComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = create_bind_group_layout(device);

        let shader_loader = world.resource::<AssetServer>();
        let binning_shader = shader_loader.load("shaders/strand_binning.wgsl");
        let rasterize_shader = shader_loader.load("shaders/strand_rasterizer.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();

        let binning_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: binning_shader,
            shader_defs: vec![],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "bin_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        let rasterize_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_rasterize_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: rasterize_shader,
            shader_defs: vec![],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "rasterize_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        info!(
            "Created strand compute pipelines: binning={:?}, rasterize={:?}",
            binning_pipeline, rasterize_pipeline
        );

        StrandComputePipeline {
            bind_group_layout,
            binning_pipeline,
            rasterize_pipeline,
        }
    }
}

#[derive(Resource, Default)]
pub struct StrandRasterizerResources {
    pub pipeline: Option<ComputePipeline>,
    pub bind_group: Option<BindGroup>,
    pub froxel_buffer: Option<Buffer>,
    pub froxel_config_buffer: Option<Buffer>,
    pub output_texture: Option<TextureView>,
    pub froxel_dimensions: Option<(u32, u32, u32)>, // (width, height, depth)
    pub strand_count: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct StrandAssetInstance {
    push_constants: PushConstants,
    bind_group: BindGroup,
}

#[derive(Resource, Default)]
pub struct StrandAssetResources {
    instances: HashMap<Entity, StrandAssetInstance>,
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
        let strand_pipeline = world.resource::<StrandComputePipeline>();
        let resources = world.resource::<StrandRasterizerResources>();

        // Get the binning pipeline
        let Some(binning_pipeline) =
            pipeline_cache.get_compute_pipeline(strand_pipeline.binning_pipeline)
        else {
            warn!("Strand binning pipeline not ready");
            return Ok(());
        };

        // Get the dimensions to calculate dispatch size
        let Some((froxels_x, froxels_y, froxels_z)) = resources.froxel_dimensions else {
            warn!("No frustum size defined.");
            return Ok(());
        };

        let Some(bind_group) = &resources.bind_group else {
            warn!("No bindgroup set.");
            return Ok(());
        };

        let Some(strand_count) = resources.strand_count else {
            warn!("No strand count set.");
            return Ok(());
        };

        // Execute the binning pass
        {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_pipeline(binning_pipeline);
            pass.set_push_constants(0, bytemuck::bytes_of(&PushConstants {
                strand_count,
            }));
            pass.set_bind_group(0, bind_group, &[]);

            // TODO: Calculate the number of strands to process
            // For now, use a fixed number for testing

            // Round up to workgroup size (64)
            let workgroups_x = (strand_count + 63) / 64;
            info!("Dispatching binning pipeline");
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        // For now, we'll skip the rasterization pass
        // In a future implementation, we would:
        // 1. Wait for the binning pass to complete
        // 2. Execute the rasterization pass
        // 3. Integrate with the PBR pipeline

        Ok(())
    }
}

fn render_strand_rasterizer(world: &mut World) {
    // TODO: Update buffers and bind group with current frame data
}

// main world buffer initialization
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
        let mut strand_indices = Vec::new();

        for strand in &polyline_list.values {
            if strand.len() < 3 {
                // Need at least one vertex index
                continue;
            }

            // Skip first two values (group_idx, mat_group_idx)
            let vertex_indices = &strand[2..];
            strand_indices.extend_from_slice(vertex_indices);

            // Add a separator (u32::MAX) to mark end of strand
            strand_indices.push(u32::MAX);
        }

        let index_buffer = ShaderStorageBuffer::from(strand_indices);
        // info!("Index buffer: {:?}", index_buffer);
        let index_buffer_handle = storage_buffers.add(index_buffer);

        commands.entity(entity).insert(StrandGeometry {
            vertices: vertex_buffer_handle,
            indices: index_buffer_handle,
            strand_count: polyline_list.values.len() as u32,
        });
    }
}

fn set_froxel_buffer(
    query: Query<(Entity, &FroxelConfig), Added<FroxelConfig>>,
    device: Res<RenderDevice>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
) {
    for (entity, config) in query.iter() {
        let (froxel_buffer, size) = create_froxel_buffer(&device, config);
        let config_buffer = create_froxel_config_buffer(&device, config);
        let (texture, view) = create_output_texture(&device, config);
        raster_resources.output_texture = Some(view);
        // modify the resource
        raster_resources.froxel_buffer = Some(froxel_buffer);
        raster_resources.froxel_config_buffer = Some(config_buffer);
        raster_resources.froxel_dimensions = Some(size);
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
        info!("[{:?}] Vertex storage buffer created: {:?}", entity, vertex_storage_buffer.buffer);
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
