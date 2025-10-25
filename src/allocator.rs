use std::{
    fmt::{self, Display, Formatter},
    ops::{Range, RangeBounds},
};

use bevy::{
    ecs::resource::Resource,
    platform::collections::HashMap,
    prelude::{Deref, DerefMut},
    render::{
        render_resource::{
            AsBindGroup, BindGroup, BindGroupLayout, Buffer, BufferAddress, BufferDescriptor,
            BufferUsages, ShaderType, encase::private::WriteInto,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use offset_allocator::Allocation;
use range_alloc;

// Host-side row (std430 same layout in WGSL)
#[repr(C)]
#[derive(Clone, Copy)]
struct HandleRow {
    slab: u32,
    offset: u32, // in bytes
    size: u32,   // in bytes (optional)
    aux: u32,    // e.g., base index, vertex format tag, etc.
}

struct HandleTable {
    buffer: Buffer,
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
    // One bind group that contains a binding_array per kind (see §3)
    pub bind_group: BindGroup,
    pub layout: BindGroupLayout,
    // GPU page/handle table (see §2)
    pub handle_table: HandleTable,
}

pub type AllocKey = (SlabId, SlabKind);

impl GpuPagingAllocator {
    pub fn allocate<T: ShaderType + WriteInto>(&mut self, kind: SlabKind, data: T) -> AllocKey {
        let size = data.size().get();
        let align = 4 as u64;
        let allocation = self.pools.entry(kind).or_insert_with(|| { SlabPool::new(self.device.clone(), DEFAULT_SLAB_SIZE, BufferUsages::all()) }).allocate(size, align);
        (SlabId(allocation.slab_index as u32), kind)
    }

    // pub fn get_slab<'a>(self, key: AllocKey) -> Option<&'a Slab> {

    // }
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
