use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
            BindingType, Buffer, BufferBindingType, BufferInitDescriptor, BufferUsages,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            PipelineCache, ShaderStages, TextureView,
        },
        renderer::{RenderContext, RenderDevice},
    },
    shader::ShaderDefVal,
};

const SIMULATOR_WORKGROUP_SIZE: u32 = 64;

#[derive(Resource, Default)]
pub struct StrandSimulatorResources {
    pub force_grids: Option<TextureView>,
    pub indirect_args: Option<Buffer>,
}

#[derive(Resource)]
pub struct StrandSimulatorPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl StrandSimulatorPipeline {
    pub fn bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "strand_simulation_bind_group_layout",
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
                // TODO
            ],
        )
    }

    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        let dispatch_args = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("strand_dispatch_args"),
            contents: bytemuck::cast_slice(&[1u32, 1u32, 1u32]),
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let descriptor = Self::bind_group_layout_descriptor();
        device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries)
    }
}

impl FromWorld for StrandSimulatorPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let shader_loader = world.resource::<AssetServer>();
        let shading_shader = shader_loader.load("shaders/strand_simulation.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();
        let cdefs = [vec![
            ShaderDefVal::UInt(
                "MAX_TEXTURE_EXTENT".into(),
                crate::plugin::MAX_TEXTURE_EXTENT,
            ),
            ShaderDefVal::UInt("WORKGROUP_SIZE".into(), SIMULATOR_WORKGROUP_SIZE),
        ]]
        .concat();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_simulation_pipeline".into()),
            layout: vec![Self::bind_group_layout_descriptor()],
            shader: shading_shader,
            shader_defs: cdefs,
            immediate_size: 0,
            entry_point: Some("xpbd_solve".into()),
            zero_initialize_workgroup_memory: false,
        });

        debug!(
            "Created strand simulation compute pipelines: sim={:?}",
            pipeline
        );

        StrandSimulatorPipeline {
            bind_group_layout,
            pipeline,
        }
    }
}

pub fn run_sim_pass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandSimulatorPipeline,
    resources: &StrandSimulatorResources,
    bind_group: &BindGroup,
    offsets: &[u32],
) {
    let Some(shading_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
        warn!("Simulator pipeline not found");
        return;
    };

    let encoder = render_context.command_encoder();
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Simulation"),
            ..default()
        });
        pass.set_pipeline(shading_pipeline);
        pass.set_bind_group(0, bind_group, offsets);

        // pass.dispatch_workgroups_indirect(1, 1, 1);
    }
}
