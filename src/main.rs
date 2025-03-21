use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphApp, RenderGraphContext, RenderLabel
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::{render_resource::*, Render, RenderApp};
use bevy::ecs as bevy_ecs;
use bevy::utils::HashMap;
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone)]
pub struct StrandGeometry {
    pub vertices: Vec<Vec3>,    // 3D positions of vertices
    pub strands: Vec<Vec<u32>>, // Each Vec<u32> is a strand (vertex indices)
}

#[derive(Component, Clone, ExtractComponent)]
pub struct Strands; // marker component for strand assets

pub struct StrandRasterizerPlugin;

impl Plugin for StrandRasterizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<Strands>::default(),
        ));
        app.init_resource::<StrandAssetResources>();
        app.add_systems(Update, setup_strand_rasterizer_resources)
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
    stub: u32,           // stub in case we need push constants
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
    bind_group: BindGroup
}

#[derive(Resource, Default)]
pub struct StrandAssetResources {
    instances: HashMap<Entity, StrandAssetInstance>
}

#[derive(Debug, Clone, Default)]
pub struct StrandRasterizerNode;

#[derive(Debug, Clone)]
pub struct StrandRasterizerLabel;
impl RenderLabel for StrandRasterizerLabel {
    #[doc = r" Clones this `"]
    #[doc = stringify!(RenderLabel)]
    #[doc = r"`."]
    fn dyn_clone(&self) -> bevy_ecs::label::Box<dyn RenderLabel> {
        todo!()
    }

    #[doc = r" Casts this value to a form where it can be compared with other type-erased values."]
    fn as_dyn_eq(&self) -> &dyn bevy_ecs::label::DynEq {
        todo!()
    }

    #[doc = r" Feeds this value into the given [`Hasher`]."]
    fn dyn_hash(&self, state: &mut dyn ::core::hash::Hasher) {
        todo!()
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
            return Ok(())
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

fn setup_strand_rasterizer_resources() {
    // stub.
    // Can be used for setting up staging buffers or GpuShaderStorageBuffers with data from the main world (so it can be transported to the render world)
}

fn render_strand_rasterizer(world: &mut World) {
    // TODO: Update buffers and bind group with current frame data
}

pub fn set_strand_geometry(world: &mut World, geometry: StrandGeometry) {
    let device = world.resource::<RenderDevice>();
    // TODO: Create vertex and index buffers
    // TODO: Update froxel buffer and output texture
    // TODO: Recreate bind group with new buffer views
}

fn main() {
    println!("Hello, world!");
}
