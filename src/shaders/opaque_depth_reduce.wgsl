#import "embedded://strand_software_rasterizer/shaders/types.wgsl"::{ PushConstants }

const WORKGROUP_SIZE: u32 = #WORKGROUP_SIZE;
const DEPTH_QUANT_MAX: u32 = 16777215u;

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

var<push_constant> pc: PushConstants;

@group(0) @binding(0) var opaque_depth: texture_depth_multisampled_2d;
@group(0) @binding(1) var<storage, read> frustum_table: array<FrustumDesc>;
@group(0) @binding(2) var<storage, read_write> opaque_fine_depth_tiles: array<vec2<u32>>;

fn quantize_depth01(z: f32) -> u32 {
    return u32(round(clamp(z, 0.0, 1.0) * f32(DEPTH_QUANT_MAX)));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn reduce_opaque_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_idx = gid.x;
    if tile_idx >= pc.num_elements || pc.scan_load_base >= arrayLength(&frustum_table) {
        return;
    }

    let frustum = frustum_table[pc.scan_load_base];
    if frustum.kind != 0u {
        return;
    }

    let fine_tiles_x = (frustum.screen_width + frustum.froxel_size_x - 1u) / frustum.froxel_size_x;
    let fine_tiles_y = (frustum.screen_height + frustum.froxel_size_y - 1u) / frustum.froxel_size_y;
    if tile_idx >= fine_tiles_x * fine_tiles_y {
        return;
    }

    let fine_x = tile_idx % fine_tiles_x;
    let fine_y = tile_idx / fine_tiles_x;
    let pixel_min = vec2<u32>(fine_x * frustum.froxel_size_x, fine_y * frustum.froxel_size_y);
    let pixel_max = vec2<u32>(
        min(pixel_min.x + frustum.froxel_size_x, frustum.screen_width),
        min(pixel_min.y + frustum.froxel_size_y, frustum.screen_height),
    );

    var covered = 0u;
    var total = 0u;
    var farthest = 1.0;
    for (var y = pixel_min.y; y < pixel_max.y; y = y + 1u) {
        for (var x = pixel_min.x; x < pixel_max.x; x = x + 1u) {
            total = total + 1u;
            var depth = textureLoad(opaque_depth, vec2<i32>(i32(x), i32(y)), 0);
            depth = min(depth, textureLoad(opaque_depth, vec2<i32>(i32(x), i32(y)), 1));
            depth = min(depth, textureLoad(opaque_depth, vec2<i32>(i32(x), i32(y)), 2));
            depth = min(depth, textureLoad(opaque_depth, vec2<i32>(i32(x), i32(y)), 3));
            if depth > 0.0 {
                covered = covered + 1u;
                farthest = min(farthest, depth);
            }
        }
    }

    let write_idx = frustum.fine_depth_tile_base + tile_idx;
    if write_idx >= arrayLength(&opaque_fine_depth_tiles) {
        return;
    }
    opaque_fine_depth_tiles[write_idx] = vec2<u32>(
        select(0u, 1u, covered == total && total > 0u),
        quantize_depth01(farthest),
    );
}
