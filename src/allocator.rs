use std::{
    fmt::{self, Display, Formatter},
    ops::Range,
};

use bevy::{
    app::Plugin,
    asset::{Asset, AssetApp, AssetId, RenderAssetUsages},
    ecs::{
        resource::Resource,
        system::{SystemParamItem, lifetimeless::SResMut},
        world::FromWorld,
    },
    platform::collections::HashMap,
    prelude::{Deref, DerefMut},
    reflect::{Reflect, prelude::ReflectDefault},
    render::{
        render_asset::RenderAsset,
        render_resource::{
            BindGroupEntry, BindGroupLayoutEntry, BindingType, Buffer, BufferAddress,
            BufferBindingType, BufferDescriptor, BufferUsages, ShaderStages, ShaderType,
            encase::{self, private::WriteInto},
        },
        renderer::{RenderDevice, RenderQueue},
    },
};

use range_alloc;

// Host-side row (std430 same layout in WGSL)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct HandleRow {
    slab: u32,
    offset: u32, // in bytes
    size: u32,   // in bytes (optional)
    aux: u32,    // e.g., base index, vertex format tag, etc.
}

#[derive(Default, Debug)]
struct HandleTable {
    buffer: Option<Buffer>,
    rows: Vec<HandleRow>, // shadow copy; upload when dirty
}

/// A single device buffer slab.
struct Slab {
    pub buffer: Buffer,
    free: range_alloc::RangeAllocator<BufferAddress>,
    pub capacity_bytes: u64,
    pub usage: BufferUsages, // STORAGE | COPY_DST | COPY_SRC (usually)
}

/// Holds information about all slabs scheduled to be allocated or reallocated.
#[derive(Default, Deref, DerefMut)]
struct SlabsToReallocate(HashMap<SlabId, SlabToReallocate>);

/// Holds information about a slab that's scheduled to be allocated or
/// reallocated.
#[derive(Default)]
struct SlabToReallocate {
    /// The capacity of the slab before we decided to grow it.
    old_slot_capacity: u32,
}

impl Display for SlabId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug)]
/// An allocation within a slab.
pub struct SlabAllocation {
    slab_index: usize,
    device_ptr: DevicePtr,
    range: Range<BufferAddress>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlabKind {
    Vert,
    Index,
    Meshlet,
    MeshletCull,
    Bvh,
    Material,
    Geo,
    StrandMeta,
}

const DEFAULT_SLAB_SIZE: u64 = 2048;

#[derive(Resource)]
pub struct GpuPagingAllocator {
    pub pools: HashMap<SlabKind, SlabPool>,
    pub device: RenderDevice,
    pub label_map: HashMap<SlabKind, &'static str>,
    pub bind_map: HashMap<SlabKind, u32>,
    // One bind group that contains a binding_array per kind (see §3)
    // pub bind_group: BindGroup,
    // pub layout: BindGroupLayout,
    // GPU page/handle table (see §2)
    pub handle_table: HandleTable,
}

impl FromWorld for GpuPagingAllocator {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.resource::<RenderDevice>();
        Self {
            pools: HashMap::default(),
            device: device.clone(),
            label_map: HashMap::default(),
            bind_map: HashMap::default(),
            handle_table: HandleTable::default(),
        }
    }
}

pub type AllocKey = (SlabId, SlabKind);

impl GpuPagingAllocator {
    pub fn allocate<T: Into<Vec<u8>>>(&mut self, kind: SlabKind, data: T) -> AllocKey {
        let data: Vec<u8> = data.into();
        let size = data.len() as u64;
        let align = 4 as u64;
        let allocation = self
            .pools
            .entry(kind)
            .or_insert_with(|| {
                SlabPool::new(self.device.clone(), DEFAULT_SLAB_SIZE, BufferUsages::all())
            })
            .allocate(size, align);
        (SlabId(allocation.slab_index as u32), kind)
    }

    // pub fn get_slab<'a>(self, key: AllocKey) -> Option<&'a Slab> {

    // }
}

pub trait BindGroupBuilder {
    fn layout_entries(&self) -> Vec<BindGroupLayoutEntry>;
    fn entries(&self) -> Vec<BindGroupEntry>;
}

impl BindGroupBuilder for GpuPagingAllocator {
    fn layout_entries(&self) -> Vec<BindGroupLayoutEntry> {
        self.pools
            .iter()
            .filter_map(|(kind, pool)| {
                let count = pool.slabs.len();
                (count > 0).then(|| BindGroupLayoutEntry {
                    binding: *self.bind_map.get(kind).unwrap_or_else(|| {
                        panic!("GpuPagingAllocator missing binding index for {:?}", kind)
                    }),
                    visibility: ShaderStages::COMPUTE
                        | ShaderStages::VERTEX
                        | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: Some(
                        (count as u32)
                            .try_into()
                            .expect("bind_array_len to be NonZero"),
                    ),
                })
            })
            .collect()
    }
    fn entries(&self) -> Vec<BindGroupEntry> {
        self.pools
            .iter()
            .flat_map(|(kind, pool)| {
                let binding_index = *self.bind_map.get(kind).unwrap();
                pool.slabs
                    .iter()
                    .enumerate()
                    .map(move |(i, slab)| BindGroupEntry {
                        binding: binding_index,
                        resource: slab.buffer.as_entire_binding(),
                    })
            })
            .collect()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SlabId(u32);

#[derive(Copy, Clone, Debug)]
pub struct DevicePtr {
    pub slab: u32,   // index into binding_array
    pub offset: u32, // byte offset in slab (<= 4 GB if packed in u32)
    pub size: u32,   // optional; useful for bounds checks
}

pub struct SlabPool {
    slabs: Vec<Slab>,
    pointer_table: HashMap<SlabId, SlabAllocation>,
    device: RenderDevice,
    default_slab_size: u64,
    usages: BufferUsages, // Optional: LRU of slabs if you want eviction later
}

impl SlabPool {
    pub fn new(device: RenderDevice, default_slab_size: u64, usages: BufferUsages) -> Self {
        Self {
            slabs: Vec::default(),
            pointer_table: HashMap::default(),
            device,
            default_slab_size,
            usages,
        }
    }

    // pub fn get<'a>(self, slab_id: SlabId) -> Option<&'a Slab> {
    //     self.slabs.get(slab_id.0 as usize)
    // }

    pub fn allocate(&mut self, size: u64, align: u64) -> SlabAllocation {
        let (slab_index, range) = self.ensure_slab(size);
        // let slab = &mut self.slabs[slab_index];

        let device_ptr = DevicePtr {
            slab: slab_index as u32,
            offset: range.start as u32,
            size: (range.end - range.start) as u32,
        };
        SlabAllocation {
            slab_index,
            device_ptr,
            range,
        }
    }

    pub fn free(&mut self, ptr: DevicePtr) {}
    pub fn write(&mut self, ticket: Range<BufferAddress>, bytes: &[u8], queue: RenderQueue) {}
    /// Ensure there's at least one slab with `min_size` bytes of free capacity.
    /// Returns the index of the slab to allocate from.
    pub fn ensure_slab(&mut self, min_size: u64) -> (usize, Range<u64>) {
        // 1. Try to find an existing slab with enough free space.
        for (i, slab) in self.slabs.iter_mut().enumerate() {
            if let Ok(range) = slab.free.allocate_range(min_size) {
                return (i, range);
            }
        }

        // 2. None found → create a new slab buffer.
        let slab_capacity = self.default_slab_size.max(min_size);

        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("slab buffer"),
            size: slab_capacity,
            usage: self.usages,
            mapped_at_creation: false,
        });

        let mut allocator = range_alloc::RangeAllocator::new(0..slab_capacity);
        let range = allocator.allocate_range(min_size).expect("This must fit");
        let new_slab = Slab {
            buffer,
            free: allocator,
            capacity_bytes: slab_capacity,
            usage: self.usages,
        };

        self.slabs.push(new_slab);
        ((self.slabs.len() - 1), range)
    }
}

// Main world component

#[derive(Asset, Reflect, Debug, Clone)]
#[reflect(opaque)]
#[reflect(Default, Debug, Clone)]
pub struct VirtualShaderStorageBuffer {
    /// Optional data used to initialize the buffer.
    pub data: Option<Vec<u8>>,
    /// kind of data
    pub kind: SlabKind,
    /// The asset usage of the storage buffer.
    pub asset_usage: RenderAssetUsages,
}

impl Default for VirtualShaderStorageBuffer {
    fn default() -> Self {
        Self {
            data: None,
            kind: SlabKind::Vert,
            asset_usage: RenderAssetUsages::default(),
        }
    }
}

impl VirtualShaderStorageBuffer {
    /// Creates a new storage buffer with the given data and asset usage.
    pub fn new(data: &[u8], kind: SlabKind, asset_usage: RenderAssetUsages) -> Self {
        let mut storage = VirtualShaderStorageBuffer {
            data: Some(data.to_vec()),
            kind: kind,
            ..Default::default()
        };
        storage.asset_usage = asset_usage;
        storage
    }
}

impl<T: ShaderType + WriteInto> From<(SlabKind, T)> for VirtualShaderStorageBuffer {
    fn from(value: (SlabKind, T)) -> Self {
        let (kind, value) = value;
        let size = value.size().get() as usize;
        let mut wrapper = encase::StorageBuffer::<Vec<u8>>::new(Vec::with_capacity(size));
        wrapper.write(&value).unwrap();
        Self::new(wrapper.as_ref(), kind, RenderAssetUsages::default())
    }
}

/// A storage buffer that is prepared as a [`RenderAsset`] and uploaded to the GPU.
pub struct GpuVirtualShaderStorageBuffer {
    pub allocation: Option<AllocKey>,
}

impl RenderAsset for GpuVirtualShaderStorageBuffer {
    type SourceAsset = VirtualShaderStorageBuffer;
    type Param = SResMut<GpuPagingAllocator>;

    fn asset_usage(source_asset: &Self::SourceAsset) -> RenderAssetUsages {
        source_asset.asset_usage
    }

    fn prepare_asset(
        source_asset: Self::SourceAsset,
        _: AssetId<Self::SourceAsset>,
        allocator: &mut SystemParamItem<Self::Param>,
        _: Option<&Self>,
    ) -> std::result::Result<
        GpuVirtualShaderStorageBuffer,
        bevy::render::render_asset::PrepareAssetError<VirtualShaderStorageBuffer>,
    > {
        match source_asset.data {
            Some(data) => {
                let allocation = allocator.allocate(source_asset.kind, data);
                Ok(GpuVirtualShaderStorageBuffer {
                    allocation: Some(allocation),
                })
            }
            None => Ok(GpuVirtualShaderStorageBuffer { allocation: None }),
        }
    }
    fn unload_asset(
        _source_asset: AssetId<Self::SourceAsset>,
        _param: &mut SystemParamItem<Self::Param>,
    ) {
        todo!()
    }
}

pub struct GpuPagingAllocatorPlugin;

impl Plugin for GpuPagingAllocatorPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.init_asset::<VirtualShaderStorageBuffer>();
    }

    fn finish(&self, app: &mut bevy::app::App) {
        let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
            return;
        };
        render_app.init_resource::<GpuPagingAllocator>();
    }
}
