#import bevy_vsms::virtual_surface_types::{
    VirtualPageTableEntry,
    VirtualPageTableMetaRow,
    vsms_virtual_page_table_address,
}

const VSMS_DEPTH_POOL_TEXTURE_COUNT: u32 = #{VSMS_DEPTH_POOL_TEXTURE_COUNT};

struct FrustumDesc {
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
    bucket_base: u32,
    bucket_count: u32,
    kind: u32,
    coarse_depth_tile_base: u32,
    coarse_depth_tile_count: u32,
    coarse_tiles_x: u32,
    coarse_tiles_y: u32,
    cascade_index: u32,
    fine_depth_tile_base: u32,
    pad1: u32,
    pad2: u32,
}

struct StampParams {
    frustum_id: u32,
    target_width: u32,
    target_height: u32,
    _pad2: u32,
}

@group(#{STAMP_GROUP}) @binding(#{FRUSTUM_TABLE}) var<storage, read> frustum_table: array<FrustumDesc>;
@group(#{STAMP_GROUP}) @binding(#{SHADOW_DOM_SURFACE_IDS}) var<storage, read> shadow_dom_surface_ids: array<vec2<u32>>;
@group(#{STAMP_GROUP}) @binding(#{PARAMS}) var<uniform> params: StampParams;

@group(#{VSMS_DEPTH_READ_GROUP}) @binding(#{VSMS_POOL_TEXTURE_BINDING}) var shadow_depth_maps: binding_array<texture_2d_array<f32>>;
@group(#{VSMS_DEPTH_READ_GROUP}) @binding(#{VSMS_POOL_SAMPLER_BINDING}) var shadow_depth_sampler: sampler;

@group(#{VSMS_DEPTH_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_META_BINDING}) var<storage, read> depth_virtual_meta: array<VirtualPageTableMetaRow>;
@group(#{VSMS_DEPTH_TABLE_GROUP}) @binding(#{VSMS_VIRTUAL_PAGE_TABLE_BINDING}) var<storage, read> depth_virtual_pages: array<VirtualPageTableEntry>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var out: VertexOut;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

fn sample_dom_depth(px: vec2<u32>, row: VirtualPageTableMetaRow) -> f32 {
    let page_size = vec2<u32>(max(row.page_size_x, 1u), max(row.page_size_y, 1u));
    let page_tile = vec3<u32>(px.x / page_size.x, px.y / page_size.y, 0u);
    let addr = vsms_virtual_page_table_address(row, page_tile, 0u, 0u);
    if addr.valid == 0u || addr.entry_index >= arrayLength(&depth_virtual_pages) {
        return 0.0;
    }
    let entry = depth_virtual_pages[addr.entry_index];
    if entry.valid == 0u || entry.physical_index >= VSMS_DEPTH_POOL_TEXTURE_COUNT {
        return 0.0;
    }

    let local_px = vec2<i32>(i32(px.x % page_size.x), i32(px.y % page_size.y));
    return textureLoad(shadow_depth_maps[i32(entry.physical_index)], local_px, 0, 0).x;
}

@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @builtin(frag_depth) f32 {
    let frustum_id = params.frustum_id;
    if frustum_id >= arrayLength(&frustum_table) || frustum_id >= arrayLength(&shadow_dom_surface_ids) {
        discard;
    }
    let frustum = frustum_table[frustum_id];
    if frustum.kind != 1u {
        discard;
    }
    if params.target_width == 0u || params.target_height == 0u {
        discard;
    }
    let target_px = vec2<u32>(u32(position.x), u32(position.y));
    if target_px.x >= params.target_width || target_px.y >= params.target_height {
        discard;
    }
    if frustum.screen_width == 0u || frustum.screen_height == 0u {
        discard;
    }

    let depth_surface_id = shadow_dom_surface_ids[frustum_id].y;
    if depth_surface_id == 0xFFFFFFFFu || depth_surface_id >= arrayLength(&depth_virtual_meta) {
        discard;
    }
    let row = depth_virtual_meta[depth_surface_id];
    if row.entry_count == 0u {
        discard;
    }

    let dom_min = vec2<u32>(
        min((target_px.x * frustum.screen_width) / params.target_width, frustum.screen_width - 1u),
        min((target_px.y * frustum.screen_height) / params.target_height, frustum.screen_height - 1u),
    );
    let dom_max = vec2<u32>(
        min(((target_px.x + 1u) * frustum.screen_width + params.target_width - 1u) / params.target_width, frustum.screen_width),
        min(((target_px.y + 1u) * frustum.screen_height + params.target_height - 1u) / params.target_height, frustum.screen_height),
    );

    var z = 0.0;
    for (var y = dom_min.y; y < dom_max.y; y = y + 1u) {
        for (var x = dom_min.x; x < dom_max.x; x = x + 1u) {
            z = max(z, sample_dom_depth(vec2<u32>(x, y), row));
        }
    }
    if z <= 0.0 {
        discard;
    }
    return z;
}
