#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct TileDebugParams {
    inv_norm: f32,
    gain: f32,
    alpha: f32,
    frustum_id: u32,
}

struct FrustumDesc {
    screen_width: u32,
    screen_height: u32,
    froxel_size_x: u32,
    froxel_size_y: u32,
    depth_slices: u32,
    bucket_base: u32,
    bucket_count: u32,
    kind: u32,
}

const POOL_CHUNK_SIZE: u32 = #POOL_CHUNK_SIZE;
const CHUNK_WORD_STRIDE: u32 = 2u + POOL_CHUNK_SIZE;

@group(0) @binding(#{FRUSTUM_TABLE}) var<storage, read> frustum_table: array<FrustumDesc>;
@group(0) @binding(#{FROXEL_BUCKET_HEADS}) var<storage, read> froxel_bucket_heads: array<atomic<u32>>;
@group(0) @binding(#{CHUNK_POOL}) var<storage, read> chunk_pool_words: array<u32>;
@group(0) @binding(#{PARAMS}) var<uniform> params: TileDebugParams;

fn heatmap_precise(value: f32) -> vec3<f32> {
    let v = clamp(value, 0.0, 1.0);

    if v < 0.25 {
        let t = v / 0.25;
        return vec3<f32>(0.0, t, 1.0);
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        return vec3<f32>(0.0, 1.0, 1.0 - t);
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        return vec3<f32>(t, 1.0, 0.0);
    } else {
        let t = (v - 0.75) / 0.25;
        return vec3<f32>(1.0, 1.0 - t, 0.0);
    }
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    if params.frustum_id >= arrayLength(&frustum_table) {
        return vec4<f32>(0.0);
    }
    let config = frustum_table[params.frustum_id];
    if config.screen_width == 0u || config.screen_height == 0u {
        return vec4<f32>(0.0);
    }

    let sx = min(u32(clamp(in.uv.x, 0.0, 0.999999) * f32(config.screen_width)), config.screen_width - 1u);
    let sy = min(u32(clamp(in.uv.y, 0.0, 0.999999) * f32(config.screen_height)), config.screen_height - 1u);

    let tiles_x = (config.screen_width + config.froxel_size_x - 1u) / config.froxel_size_x;
    let tiles_y = (config.screen_height + config.froxel_size_y - 1u) / config.froxel_size_y;

    let tile_coord_x = min(sx / config.froxel_size_x, tiles_x - 1u);
    let tile_coord_y = min(sy / config.froxel_size_y, tiles_y - 1u);

    var frag_count: u32 = 0u;
    for (var dz: u32 = 0u; dz < config.depth_slices; dz = dz + 1u) {
        let local_idx = (dz * tiles_y + tile_coord_y) * tiles_x + tile_coord_x;
        if local_idx >= config.bucket_count {
            continue;
        }
        let bucket_idx = config.bucket_base + local_idx;
        if bucket_idx >= arrayLength(&froxel_bucket_heads) {
            continue;
        }
        var chunk_idx = atomicLoad(&froxel_bucket_heads[bucket_idx]);
        var guard = 0u;
        loop {
            if chunk_idx == 0xFFFFFFFFu {
                break;
            }
            guard = guard + 1u;
            if guard > 4096u {
                break;
            }
            let base = chunk_idx * CHUNK_WORD_STRIDE;
            if base + 1u >= arrayLength(&chunk_pool_words) {
                break;
            }
            frag_count = frag_count + min(chunk_pool_words[base + 1u], POOL_CHUNK_SIZE);
            chunk_idx = chunk_pool_words[base];
        }
    }

    if frag_count == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let norm = clamp(f32(frag_count) * params.inv_norm * params.gain, 0.0, 1.0);
    let debug_color = heatmap_precise(norm);
    return vec4<f32>(debug_color, params.alpha);
}
