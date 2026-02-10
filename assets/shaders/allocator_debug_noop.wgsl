#import "shaders/types.wgsl"::{
    DevicePtr
} 

struct RawWords {
    words: array<u32>,
}

@group(#{BIND_ARRAYS}) @binding(#{VERTICES}) var<storage, read_write> vertices: binding_array<RawWords>;
@group(#{BIND_ARRAYS}) @binding(#{INDICES}) var<storage, read_write> indices: binding_array<RawWords>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_MATERIALS}) var<storage, read_write> materials: binding_array<RawWords>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_GEOS}) var<storage, read_write> geos: binding_array<RawWords>;
@group(#{BIND_ARRAYS}) @binding(#{STRAND_METADATA}) var<storage, read_write> metas: binding_array<RawWords>;

@group(#{PAGE_TABLES}) @binding(#{VERTICES}) var<storage, read_write> t_vertices: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{INDICES}) var<storage, read_write> t_indices: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_MATERIALS}) var<storage, read_write> t_materials: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_GEOS}) var<storage, read_write> t_geos: array<DevicePtr>;
@group(#{PAGE_TABLES}) @binding(#{STRAND_METADATA}) var<storage, read_write> t_metas: array<DevicePtr>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    var sink = 0u;
    sink ^= vertices[0].words[0] ^ t_vertices[0].slab;
    sink ^= indices[0].words[0] ^ t_indices[0].slab;
    sink ^= materials[0].words[0] ^ t_materials[0].slab;
    sink ^= geos[0].words[0] ^ t_geos[0].slab;
    sink ^= metas[0].words[0] ^ t_metas[0].slab;
    // t_vertices[0] = vec4<u32>(0u, 0u, 0u, 0u);
}