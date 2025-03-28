use bevy::{
    asset::load_internal_asset,
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedComputePipelineId, CachedPipelineState, CommandEncoder, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, PushConstantRange, ShaderDefVal,
            ShaderStages,
            binding_types::{storage_buffer, storage_buffer_read_only},
        },
        renderer::RenderDevice,
        storage::GpuShaderStorageBuffer,
    },
};
use bevy_radix_sort::{LoadState, dispatch_workgroup_ext};
use bevy_radix_sort::{
    EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE,
    ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE,
};

pub const SPATIAL_HASHING_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(125521088680071949847982117826760245819);

pub const NUMBER_OF_THREADS_PER_WORKGROUP: u32 = 256;

pub struct SpatialHashingPlugin;

impl Plugin for SpatialHashingPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SPATIAL_HASHING_SHADER_HANDLE,
            "../assets/shaders/spatial_hashing.wgsl",
            Shader::from_wgsl
        );

        app.sub_app_mut(RenderApp).add_systems(
            Render,
            SpatialHashingBindGroup::initialize
                .in_set(RenderSet::PrepareBindGroups)
                .run_if(not(resource_exists::<SpatialHashingBindGroup>)),
        );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<SpatialHashingPipeline>();
    }
}

#[derive(Resource, Debug, Clone)]
pub struct SpatialHashingPipeline {
    zeroing_pipeline: CachedComputePipelineId,
    hashing_pipeline: CachedComputePipelineId,
    bind_group_layout: BindGroupLayout,
}

impl SpatialHashingPipeline {
    pub fn new(render_device: &RenderDevice, pipeline_cache: &PipelineCache) -> Self {
        let bind_group_layout = render_device.create_bind_group_layout(
            "spatial hashing bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // A buffer stores the `collision grid` index
                    storage_buffer_read_only::<u32>(false),
                    // A buffer writes the start index of each `collision grid`
                    storage_buffer::<u32>(false),
                    // A buffer writes the end index of each `collision grid`
                    storage_buffer::<u32>(false),
                ),
            ),
        );

        let cdefs = vec![ShaderDefVal::UInt(
            "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
            NUMBER_OF_THREADS_PER_WORKGROUP,
        )];

        let zeroing_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("spatial hashing: zeroing pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: SPATIAL_HASHING_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["ZEROING_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        let hashing_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("spatial hashing: hashing pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: SPATIAL_HASHING_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["HASHING_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            zeroing_pipeline,
            hashing_pipeline,
            bind_group_layout,
        }
    }

    pub fn zeroing_pipeline(&self) -> CachedComputePipelineId {
        self.zeroing_pipeline
    }

    pub fn hashing_pipeline(&self) -> CachedComputePipelineId {
        self.hashing_pipeline
    }

    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}

impl FromWorld for SpatialHashingPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        Self::new(render_device, pipeline_cache)
    }
}

#[derive(Resource, Debug, Clone)]
pub struct SpatialHashingBindGroup {
    eve_bind_group: BindGroup,
    odd_bind_group: BindGroup,
}

impl SpatialHashingBindGroup {
    pub fn initialize(
        mut commands: Commands,
        spatial_hashing_pipeline: Res<SpatialHashingPipeline>,
        render_device: Res<RenderDevice>,
        sbufs: Res<RenderAssets<GpuShaderStorageBuffer>>,
    ) {
        let bind_group_layout = spatial_hashing_pipeline.bind_group_layout();

        let eve_global_keys_buf = sbufs
            .get(EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let eve_global_vals_buf = sbufs
            .get(EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_global_keys_buf = sbufs
            .get(ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_global_vals_buf = sbufs
            .get(ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();

        let eve_bind_group = render_device.create_bind_group(
            "spatial hashing bindgroup eve",
            bind_group_layout,
            &BindGroupEntries::sequential((
                eve_global_keys_buf.buffer.as_entire_binding(),
                odd_global_keys_buf.buffer.as_entire_binding(),
                odd_global_vals_buf.buffer.as_entire_binding(),
            )),
        );

        let odd_bind_group = render_device.create_bind_group(
            "spatial hashing bindgroup odd",
            bind_group_layout,
            &BindGroupEntries::sequential((
                odd_global_keys_buf.buffer.as_entire_binding(),
                eve_global_keys_buf.buffer.as_entire_binding(),
                eve_global_vals_buf.buffer.as_entire_binding(),
            )),
        );

        let spatial_hashing_bind_group = Self {
            eve_bind_group,
            odd_bind_group,
        };

        commands.insert_resource(spatial_hashing_bind_group);
    }
}

pub fn run(
    encoder: &mut CommandEncoder,
    pipeline_cache: &PipelineCache,
    spatial_hashing_pipeline: &SpatialHashingPipeline,
    spatial_hashing_bind_group: &SpatialHashingBindGroup,
    max_compute_workgroups_per_dimension: u32,
    number_of_keys: u32,
    // Read from `eve_global_keys_buf`?
    read_from_even: bool,
) {
    let zeroing_pipeline = pipeline_cache
        .get_compute_pipeline(spatial_hashing_pipeline.zeroing_pipeline())
        .unwrap();

    let hashing_pipeline = pipeline_cache
        .get_compute_pipeline(spatial_hashing_pipeline.hashing_pipeline())
        .unwrap();

    let number_of_workgroups = number_of_keys.div_ceil(NUMBER_OF_THREADS_PER_WORKGROUP);

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("spatial_hashing compute pass"),
            ..default()
        });

        pass.set_pipeline(zeroing_pipeline);

        if read_from_even {
            pass.set_bind_group(0, &spatial_hashing_bind_group.eve_bind_group, &[]);
        } else {
            pass.set_bind_group(0, &spatial_hashing_bind_group.odd_bind_group, &[]);
        }

        pass.set_push_constants(NUMBER_OF_KEYS_OFFSET, bytemuck::bytes_of(&number_of_keys));

        dispatch_workgroup_ext(
            &mut pass,
            number_of_workgroups,
            max_compute_workgroups_per_dimension,
            WORKGROUP_OFFSET_OFFSET,
        );

        pass.set_pipeline(hashing_pipeline);

        dispatch_workgroup_ext(
            &mut pass,
            number_of_workgroups,
            max_compute_workgroups_per_dimension,
            WORKGROUP_OFFSET_OFFSET,
        );
    }
}

const WORKGROUP_OFFSET_OFFSET: u32 = 0;
const NUMBER_OF_KEYS_OFFSET: u32 = 4;

const PUSH_CONSTANT_RANGES: PushConstantRange = PushConstantRange {
    stages: ShaderStages::COMPUTE,
    range: 0..8,
};

pub(super) fn check_load_state(world: &World) -> LoadState {
    let pipeline_cache = world.resource::<PipelineCache>();
    let spatial_hashing_pipeline = world.resource::<SpatialHashingPipeline>();

    let (zeroing_pipeline_state, hashing_pipeline_state) = (
        pipeline_cache.get_compute_pipeline_state(spatial_hashing_pipeline.zeroing_pipeline()),
        pipeline_cache.get_compute_pipeline_state(spatial_hashing_pipeline.hashing_pipeline()),
    );

    if let CachedPipelineState::Err(err) = zeroing_pipeline_state {
        return LoadState::Failed(format!("Failed to load spatial_zeroing_pipeline: {}", err));
    }

    if let CachedPipelineState::Err(err) = hashing_pipeline_state {
        return LoadState::Failed(format!("Failed to load spatial_hashing_pipeline: {}", err));
    }

    if matches!(zeroing_pipeline_state, CachedPipelineState::Ok(_))
        && matches!(hashing_pipeline_state, CachedPipelineState::Ok(_))
    {
        return LoadState::Loaded;
    }

    LoadState::OnLoad
}