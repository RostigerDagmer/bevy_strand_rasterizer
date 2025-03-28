use bevy::{
    ecs::{system::Resource, world::World},
    log::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{self, RenderLabel},
        render_resource::{Buffer, BufferAddress, PipelineCache},
        renderer::{RenderContext, RenderDevice},
        storage::GpuShaderStorageBuffer,
    },
};
use bevy_radix_sort::{
    EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE, LoadState,
    RadixSortBindGroup, RadixSortPipeline,
};

#[derive(Resource, Default, Clone, Copy)]
pub struct SortCommand {
    pub requested: bool,
    pub length: usize,
}

#[derive(Resource)]
pub struct SimpleGpuSortResource {
    // cpu-buffer -> gpu-staging-buffer -> gpu-destination-buffer
    pub i_keys_buf: Buffer,
    pub i_vals_buf: Buffer,
    // gpu-source-buffer -> gpu-staging-buffer -> cpu-buffer
    pub o_keys_buf: Buffer,
    pub o_vals_buf: Buffer,
    // the length of the keys and vals
    pub length: usize,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, RenderLabel)]
pub struct SimpleGpuSortNodeLabel;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum SimpleGpuSortState {
    #[default]
    OnLoad,
    Loaded,
}

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct SimpleGpuSortNode {
    state: SimpleGpuSortState,
}

impl render_graph::Node for SimpleGpuSortNode {
    fn update(&mut self, world: &mut World) {
        if matches!(self.state, SimpleGpuSortState::OnLoad) {
            let radix_sort_load_state = bevy_radix_sort::check_load_state(world);

            if let LoadState::Failed(err) = &radix_sort_load_state {
                panic!("{}", err);
            }

            if matches!(radix_sort_load_state, LoadState::Loaded) {
                self.state = SimpleGpuSortState::Loaded;
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if matches!(self.state, SimpleGpuSortState::OnLoad)
            || !world.resource::<SortCommand>().requested
        {
            return Ok(());
        }

        let max_compute_workgroups_per_dimension = {
            let render_device = world.resource::<RenderDevice>();
            render_device.limits().max_compute_workgroups_per_dimension
        };

        let pipeline_cache: &PipelineCache = world.resource::<PipelineCache>();
        let radix_sort_pipeline = world.resource::<RadixSortPipeline>();
        let radix_sort_bind_group = world.resource::<RadixSortBindGroup>();

        let simple_gpu_sort_resource = world.resource::<SimpleGpuSortResource>();

        let storage_buffers = world.resource::<RenderAssets<GpuShaderStorageBuffer>>();
        let eve_global_keys_buf = storage_buffers
            .get(EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let eve_global_vals_buf = storage_buffers
            .get(EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();

        let encoder = render_context.command_encoder();

        let size = (simple_gpu_sort_resource.length * std::mem::size_of::<u32>()) as BufferAddress;
        
        // NOTE: Probably not strictly necessary for our use case. TODO: Investigate performance impact.
        encoder.copy_buffer_to_buffer(
            &simple_gpu_sort_resource.i_keys_buf,
            0,
            &eve_global_keys_buf.buffer,
            0,
            size,
        );

        encoder.copy_buffer_to_buffer(
            &simple_gpu_sort_resource.i_vals_buf,
            0,
            &eve_global_vals_buf.buffer,
            0,
            size,
        );

        info!("before radix_sort: copy key/val from staging buffer to gpu storage buffer");

        bevy_radix_sort::run(
            encoder,
            pipeline_cache,
            radix_sort_pipeline,
            radix_sort_bind_group,
            max_compute_workgroups_per_dimension,
            simple_gpu_sort_resource.length as u32,
            0..4,
            false,
            true,
        );

        // NOTE: Probably not strictly necessary for our use case. TODO: Investigate performance impact.
        encoder.copy_buffer_to_buffer(
            &eve_global_keys_buf.buffer,
            0,
            &simple_gpu_sort_resource.o_keys_buf,
            0,
            size,
        );

        encoder.copy_buffer_to_buffer(
            &eve_global_vals_buf.buffer,
            0,
            &simple_gpu_sort_resource.o_vals_buf,
            0,
            size,
        );

        info!("after radix_sort: copy key/val from gpu storage buffer to staging buffer");

        Ok(())
    }
}
