use std::hash::Hash;

use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::ecs::label::DynHash;
use bevy::ecs as bevy_ecs;
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
use bevy::render::{render_resource::*, Render, RenderApp};
use bevy::utils::HashMap;
use bytemuck::{Pod, Zeroable};
mod dson;
use dson::*;

#[derive(Component, Debug, Clone)]
pub struct StrandGeometry {
    pub vertices: Handle<ShaderStorageBuffer>,
    pub indices: Handle<ShaderStorageBuffer>,
}

#[derive(Component, Clone, ExtractComponent)]
pub struct Strands; // marker component for strand assets

pub struct StrandRasterizerPlugin;

impl Plugin for StrandRasterizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((ExtractComponentPlugin::<Strands>::default(),));
        app.init_resource::<StrandAssetResources>();
        app.add_systems(Update, set_strand_geometry)
            .add_systems(Render, render_strand_rasterizer);
        setup_render_graph(app);
    }
}

#[derive(Resource)]
struct StrandComputePipeline {
    // stub
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

#[derive(Copy, Clone, Pod, Zeroable, Debug)]
#[repr(C)]
struct PushConstants {
    stub: u32, // stub in case we need push constants
}

#[derive(Component, Clone)]
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
}

pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
    device.create_bind_group_layout(
        "bind_group_layout_label",
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
        ],
    )
}

impl FromWorld for StrandComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let layout = create_bind_group_layout(device);

        let shader = world.load_asset("shaders/strand_rasterizer.wgsl");

        let pipeline_cache = world.get_resource::<PipelineCache>().unwrap();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("morph_compute_pipeline".into()),
            layout: vec![layout.clone()],
            shader,
            shader_defs: vec![],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });
        info!("Created compute pipeline {:?}", pipeline);

        StrandComputePipeline { layout, pipeline }
    }
}

#[derive(Resource)]
pub struct StrandRasterizerResources {
    pub pipeline: ComputePipeline,
    pub bind_group: BindGroup,
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

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StrandRasterizerLabel;

impl RenderLabel for StrandRasterizerLabel {
    fn dyn_clone(&self) -> Box<dyn RenderLabel> {
        Box::new(self.clone())
    }

    fn as_dyn_eq(&self) -> &dyn bevy_ecs::label::DynEq {
        self
    }

    fn dyn_hash(&self, state: &mut dyn ::core::hash::Hasher) {
        std::any::TypeId::of::<Self>().dyn_hash(state);
    }
}

impl Node for StrandRasterizerNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let resources = world.resource::<StrandComputePipeline>();
        let asset_instances = world.resource::<StrandAssetResources>();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(resources.pipeline) else {
            warn!("Pipeline not ready");
            return Ok(());
        };
        for (_, instance) in asset_instances.instances.iter() {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &instance.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // TODO: Calculate proper workgroup sizes
        }

        Ok(())
    }
}

fn setup_render_graph(app: &mut App) {
    // let mut render_graph = app.world().resource_mut::<RenderGraph>();
    // render_graph.add_node(StrandRasterizerLabel, StrandRasterizerNode);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };

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
        );
    // TODO: Add edges to connect to other nodes (e.g., before main pass)
}

fn render_strand_rasterizer(world: &mut World) {
    // TODO: Update buffers and bind group with current frame data
}

// main world buffer initialization
pub fn set_strand_geometry(
    query: Query<(Entity, &StrandAsset), Without<StrandGeometry>>, 
    assets: Res<Assets<DsonAsset>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut commands: Commands
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
        info!("Vertex buffer: {:?}", vertex_buffer);
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
            if strand.len() < 3 {  // Need at least one vertex index
                continue;
            }
            
            // Skip first two values (group_idx, mat_group_idx)
            let vertex_indices = &strand[2..];
            strand_indices.extend_from_slice(vertex_indices);
            
            // Add a separator (u32::MAX) to mark end of strand
            strand_indices.push(u32::MAX);
        }
        
        let index_buffer = ShaderStorageBuffer::from(strand_indices);
        info!("Index buffer: {:?}", index_buffer);
        let index_buffer_handle = storage_buffers.add(index_buffer);
        
        commands.entity(entity).insert(StrandGeometry {
            vertices: vertex_buffer_handle,
            indices: index_buffer_handle
        });
    }
}

// pub fn calculate_strand_aabb(
//     query: Query<(Entity, &StrandGeometry, &GlobalTransform)>,
//     storage_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
//     view: Res<ViewUniforms>,
//     mut strand_rasterizer_resources: ResMut<StrandRasterizerResources>,
// ) {
//     for (entity, geometry, transform) in query.iter() {
//         let Some(vertex_buffer) = storage_buffers.get(&geometry.vertices) else {
//             continue;
//         };
        
//         // Read vertex data to calculate AABB
//         // This is a simplified approach - in practice, we'd do this on the GPU
//         let vertices: &[Vec3] = vertex_buffer.buffer.mapped_slice().unwrap();
        
//         // Initialize with extreme values
//         let mut min_pos = Vec3::splat(f32::MAX);
//         let mut max_pos = Vec3::splat(f32::MIN);
        
//         // Find world-space AABB
//         for &vertex_pos in vertices {
//             // Transform vertex to world space
//             let world_pos = transform.transform_point(vertex_pos);
            
//             // Update min/max
//             min_pos = min_pos.min(world_pos);
//             max_pos = max_pos.max(world_pos);
//         }
        
//         // Transform AABB corners to screen space
//         let view_proj = view.uniforms.view_proj;
//         let mut screen_min = Vec2::splat(f32::MAX);
//         let mut screen_max = Vec2::splat(f32::MIN);
//         let mut depth_min = 1.0;
//         let mut depth_max = 0.0;
        
//         // Check all 8 corners of the AABB
//         for i in 0..8 {
//             let x = if i & 1 != 0 { max_pos.x } else { min_pos.x };
//             let y = if i & 2 != 0 { max_pos.y } else { min_pos.y };
//             let z = if i & 4 != 0 { max_pos.z } else { min_pos.z };
            
//             let world_pos = Vec3::new(x, y, z);
//             let clip_pos = view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
            
//             // Perspective divide
//             let ndc = clip_pos.xyz() / clip_pos.w;
            
//             // Convert to screen space
//             let screen_x = (ndc.x * 0.5 + 0.5) * view.viewport.z;
//             let screen_y = (ndc.y * 0.5 + 0.5) * view.viewport.w;
//             let depth = ndc.z * 0.5 + 0.5; // Convert to [0, 1] range
            
//             screen_min = screen_min.min(Vec2::new(screen_x, screen_y));
//             screen_max = screen_max.max(Vec2::new(screen_x, screen_y));
//             depth_min = depth_min.min(depth);
//             depth_max = depth_max.max(depth);
//         }
        
//         // Convert to froxel coordinates
//         let froxel_size_x = 8; // Should match shader constants
//         let froxel_size_y = 8;
//         let depth_slices = 16;
        
//         let froxel_min_x = (screen_min.x / froxel_size_x as f32).floor() as u32;
//         let froxel_min_y = (screen_min.y / froxel_size_y as f32).floor() as u32;
//         let froxel_min_z = depth_min;
        
//         let froxel_max_x = (screen_max.x / froxel_size_x as f32).ceil() as u32;
//         let froxel_max_y = (screen_max.y / froxel_size_y as f32).ceil() as u32;
//         let froxel_max_z = depth_max;
        
//         // Store AABB for this entity
//         // (Assuming we have a map from entity to froxel config in resources)
//         strand_rasterizer_resources.entity_froxel_config.insert(entity, FroxelConfig {
//             screen_width: view.viewport.z as u32,
//             screen_height: view.viewport.w as u32,
//             froxel_size_x,
//             froxel_size_y,
//             depth_slices,
//             aabb_min_x: froxel_min_x,
//             aabb_min_y: froxel_min_y,
//             aabb_min_z: froxel_min_z,
//             aabb_max_x: froxel_max_x,
//             aabb_max_y: froxel_max_y,
//             aabb_max_z: froxel_max_z,
//         });
//     }
// }

// render world buffer retrieval
pub fn use_strand_geometry(
    query: Query<(Entity, &StrandGeometry), (With<Strands>, Added<StrandGeometry>)>,
    storage_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // This is an example of how to retrieve the shader storage buffer created in the main world above
    // and use it in the render world.
    for (entity, geometry) in query.iter() {
        let Some(index_storage_buffer) = storage_buffers.get(&geometry.indices) else {
            warn!("Index storage buffer not found for entity: {:?}", entity);
            continue;
        };
    }
}

#[derive(Component)]
struct StrandAsset {
    handle: Handle<DsonAsset>,
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle: Handle<DsonAsset> = asset_server.load("dForce Pixie Cut_708408.dsf".to_string());
    commands.spawn((StrandAsset { handle }));
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
