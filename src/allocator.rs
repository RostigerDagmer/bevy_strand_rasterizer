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
    prelude::{Deref, DerefMut, *},
    reflect::{Reflect, prelude::ReflectDefault},
    render::{
        Render, RenderSystems,
        render_asset::{RenderAsset, RenderAssetPlugin},
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType, Buffer,
            BufferAddress, BufferBindingType, BufferDescriptor, BufferUsages, ShaderStages,
            ShaderType,
            encase::{self, private::WriteInto},
        },
        renderer::{RenderDevice, RenderQueue},
    },
};

use bytemuck::{Pod, Zeroable};
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
    StrandMaterial,
    StrandGeo,
    StrandMeta,
}

const DEFAULT_SLAB_SIZE: u64 = 2048;

// because phf does not support Enum variants as key types.
use lazy_static::lazy_static;
use wgpu::{CommandEncoder, CommandEncoderDescriptor};

lazy_static! {
    static ref DEFAULT_BIND_MAP: HashMap<SlabKind, u32> = [
        (SlabKind::Vert, 0),
        (SlabKind::Index, 1),
        (SlabKind::Meshlet, 2),
        (SlabKind::MeshletCull, 3),
        (SlabKind::Bvh, 4),
        (SlabKind::StrandMaterial, 5),
        (SlabKind::StrandGeo, 6),
        (SlabKind::StrandMeta, 7),
    ]
    .iter()
    .copied()
    .collect();
}

lazy_static! {
    static ref DEFAULT_LABEL_MAP: HashMap<SlabKind, &'static str> = [
        (SlabKind::Vert, "VERTICES"),
        (SlabKind::Index, "INDICES"),
        (SlabKind::Meshlet, "MESHLETS"),
        (SlabKind::MeshletCull, "MESHLET_CULL"),
        (SlabKind::Bvh, "BVH"),
        (SlabKind::StrandMaterial, "STRAND_MATERIALS"),
        (SlabKind::StrandGeo, "STRAND_GEOS"),
        (SlabKind::StrandMeta, "STRAND_METADATA"),
    ]
    .iter()
    .copied()
    .collect();
}

#[derive(Resource)]
pub struct GpuPagingAllocatorSettings {
    pub label_map: HashMap<SlabKind, &'static str>,
    pub bind_map: HashMap<SlabKind, u32>,
    pub buffer_group_idx: u32,
    pub table_group_idx: u32,
}

impl Default for GpuPagingAllocatorSettings {
    fn default() -> Self {
        Self {
            label_map: DEFAULT_LABEL_MAP.clone(),
            bind_map: DEFAULT_BIND_MAP.clone(),
            buffer_group_idx: 5,
            table_group_idx: 6,
        }
    }
}

#[derive(Resource)]
pub struct GpuPagingAllocator {
    pub device: RenderDevice,
    pub queue: RenderQueue, // <- TODO: once wgpu supports multi-queue
    pub pools: HashMap<SlabKind, SlabPool>,
    pub label_map: HashMap<SlabKind, &'static str>,
    pub bind_map: HashMap<SlabKind, u32>,
    // GPU page/handle table
    pub handle_table: HandleTable,
    pub buffer_bind_group: Option<BindGroup>,
    pub buffer_group_idx: u32,
    pub pagetable_bind_group: Option<BindGroup>,
    pub table_group_idx: u32,
}

impl FromWorld for GpuPagingAllocator {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.resource::<RenderDevice>();
        let queue = world.resource::<RenderQueue>();
        let settings = world.resource::<GpuPagingAllocatorSettings>();

        Self {
            device: device.clone(),
            queue: queue.clone(),
            pools: HashMap::default(),
            label_map: settings.label_map.clone(),
            bind_map: settings.bind_map.clone(),
            handle_table: HandleTable::default(),
            buffer_bind_group: None,
            buffer_group_idx: settings.buffer_group_idx,
            pagetable_bind_group: None,
            table_group_idx: settings.table_group_idx,
        }
    }
}

pub type AllocKey = (SlabId, SlabKind);

impl GpuPagingAllocator {
    pub fn allocate<T: Into<Vec<u8>>>(&mut self, kind: SlabKind, data: T) -> AllocKey {
        info!("Allocating data on pool[{:?}]", kind);
        let data: Vec<u8> = data.into();
        let size = data.len() as u64;
        let align = 64 as u64;
        let pool = self.pools.entry(kind).or_insert_with(|| {
            SlabPool::new(self.device.clone(), DEFAULT_SLAB_SIZE, BufferUsages::all() ^ BufferUsages::MAP_READ ^ BufferUsages::INDIRECT ^ BufferUsages::UNIFORM)
        });
        let allocation = pool.allocate(size, align);
        pool.write(&allocation, &data, &self.queue); // queue upload
        (SlabId(allocation.slab_index as u32), kind)
    }

    pub fn write_tables(&mut self) {
        for (kind, pool) in self.pools.iter_mut() {
            info!("Writing page tables for pool[{:?}]", kind);
            pool.write_table(&self.queue);
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SlabId(u32);

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
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
    usages: BufferUsages,
    page_table: Option<Buffer>,
}

impl SlabPool {
    pub fn new(device: RenderDevice, default_slab_size: u64, usages: BufferUsages) -> Self {
        Self {
            slabs: Vec::default(),
            pointer_table: HashMap::default(),
            device,
            default_slab_size,
            usages,
            page_table: None,
        }
    }

    pub fn allocate(&mut self, size: u64, align: u64) -> SlabAllocation {
        let (slab_index, range) = self.ensure_slab(size);
        let device_ptr = DevicePtr {
            slab: slab_index as u32,
            offset: range.start as u32,
            size: (range.end - range.start) as u32,
        };
        let alloc = SlabAllocation {
            slab_index,
            device_ptr,
            range,
        };
        self.ensure_page_table();
        self.pointer_table
            .insert(SlabId(slab_index as u32), alloc.clone());
        alloc
    }

    pub fn free(&mut self, slab_id: SlabId, queue: &RenderQueue) {
        if let Some(allocation) = self.pointer_table.remove(&slab_id) {
            if let Some(buffer) = &self.page_table {
                let invalid = DevicePtr {
                    slab: u32::MAX,
                    offset: 0,
                    size: 0,
                };
                let offset_bytes = allocation.range.start * std::mem::size_of::<DevicePtr>() as u64;
                queue.write_buffer(buffer, offset_bytes, bytemuck::bytes_of(&invalid));
            } else {
                warn!(
                    "Free didn't find the buffer for slab {:?} with allocation {:?} in pointer table",
                    slab_id, allocation
                );
            }
            if let Some(slab) = self.slabs.get_mut(slab_id.0 as usize) {
                slab.free.free_range(allocation.range);
            } else {
                warn!(
                    "Free didn't find the slab {:?} in the slab list to free the allocation {:?}",
                    slab_id, allocation
                );
            }
        }
    }

    pub fn write(&mut self, ticket: &SlabAllocation, bytes: &[u8], queue: &RenderQueue) {
        if let Some(slab) = self.slabs.get(ticket.slab_index) {
            queue.write_buffer(&slab.buffer, ticket.range.start, bytes);
        }
    }

    pub fn write_table(&mut self, queue: &RenderQueue) {
        self.ensure_page_table();
        if let Some(buffer) = &self.page_table {
            let mut table: Vec<DevicePtr> = self
                .pointer_table
                .iter()
                .map(|(_, alloc)| alloc.device_ptr)
                .collect();
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&table));
        }
    }

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

    fn ensure_page_table(&mut self) {
        let needed_bytes =
            (self.pointer_table.len() as u64) * std::mem::size_of::<DevicePtr>() as u64;
        let current_bytes = self.page_table.as_ref().map(|b| b.size()).unwrap_or(0);

        if needed_bytes > current_bytes {
            let new_size = needed_bytes.next_power_of_two().max(2 ^ 22); // start with ~4 MiB
            let buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("SlabPool_PageTable"),
                size: new_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.page_table = Some(buffer);
        }
    }
}

pub trait BindGroupBuilder {
    fn layout_entries(&self) -> PagingBindGroupLayout;
    fn entries(&self) -> PagingBindGroupEntries;
}

pub struct PagingBindGroupLayout {
    pub pools: Vec<BindGroupLayoutEntry>,
    pub page_tables: Vec<BindGroupLayoutEntry>,
}

pub struct PagingBindGroupEntries<'a> {
    pub pools: Vec<BindGroupEntry<'a>>,
    pub page_tables: Vec<BindGroupEntry<'a>>,
}

impl BindGroupBuilder for GpuPagingAllocator {
    fn layout_entries(&self) -> PagingBindGroupLayout {
        let (pools, page_tables) = self
            .pools
            .iter()
            .filter_map(|(kind, pool)| {
                let count = pool.slabs.len();
                (count > 0).then(|| {
                    (
                        // Pool bind_array
                        BindGroupLayoutEntry {
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
                        },
                        // Pagetable
                        BindGroupLayoutEntry {
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
                            count: None,
                        },
                    )
                })
            })
            .unzip();
        PagingBindGroupLayout { pools, page_tables }
    }
    fn entries(&self) -> PagingBindGroupEntries<'_> {
        let pools = self
            .pools
            .iter()
            .filter_map(|(kind, pool)| {
                let count = pool.slabs.len();
                (count > 0).then(|| (kind, pool))
            })
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
            .collect();

        let page_tables = self
            .pools
            .iter()
            .filter_map(|(kind, pool)| {
                let count = pool.slabs.len();
                let binding_index = *self.bind_map.get(kind).unwrap();
                let Some(page_table_buffer) = &pool.page_table else {
                    return None;
                };
                (count > 0).then(move || BindGroupEntry {
                    binding: binding_index,
                    resource: page_table_buffer.as_entire_binding(),
                })
            })
            .collect();
        PagingBindGroupEntries { pools, page_tables }
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
        info!("GpuVirtualShaderStorageBuffer::prepare on {:?}", source_asset.kind);
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

fn update_bindgroups(mut allocator: ResMut<GpuPagingAllocator>, device: Res<RenderDevice>) {
    // TODO: change detection (probably not here)

    let buffer_layout_entries = allocator.layout_entries();
    let buffer_entries = allocator.entries();

    // Binding Arrays
    let buffer_layout = device.create_bind_group_layout(
        "gpu_paging_allocator_buffer_layout",
        &buffer_layout_entries.pools,
    );
    let buffer_bind_group = device.create_bind_group(
        "gpu_paging_allocator_buffer_group",
        &buffer_layout,
        &buffer_entries.pools,
    );

    // Page table
    let table_layout = device.create_bind_group_layout(
        "gpu_paging_allocator_table_layout",
        &buffer_layout_entries.page_tables,
    );
    let table_bind_group = device.create_bind_group(
        "gpu_paging_allocator_buffer_group",
        &table_layout,
        &buffer_entries.page_tables,
    );

    allocator.buffer_bind_group = Some(buffer_bind_group);
    allocator.pagetable_bind_group = Some(table_bind_group);
}

fn upload_buffers(mut allocator: ResMut<GpuPagingAllocator>) {
    allocator.write_tables();
}

pub struct GpuPagingAllocatorPlugin;

impl Plugin for GpuPagingAllocatorPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.init_asset::<VirtualShaderStorageBuffer>();
        app.add_plugins(RenderAssetPlugin::<GpuVirtualShaderStorageBuffer>::default());
    }

    fn finish(&self, app: &mut bevy::app::App) {
        let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
            return;
        };
        render_app.init_resource::<GpuPagingAllocator>();
        render_app
            .add_systems(
                Render,
                (update_bindgroups).in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                (upload_buffers).in_set(RenderSystems::PrepareResourcesFlush),
            );
    }
}
