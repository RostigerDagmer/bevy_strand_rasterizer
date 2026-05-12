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
    coarse_depth_tile_base: u32,
    coarse_depth_tile_count: u32,
    coarse_tiles_x: u32,
    coarse_tiles_y: u32,
    cascade_index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

const POOL_CHUNK_SIZE: u32 = #POOL_CHUNK_SIZE;
const CHUNK_WORD_STRIDE: u32 = 2u + POOL_CHUNK_SIZE;
const COARSE_FINE_TILE_EXTENT: u32 = #COARSE_FINE_TILE_EXTENT;
const COARSE_DEPTH_SLICES: u32 = #COARSE_DEPTH_SLICES;
const COARSE_COUNT_PAGE_SIZE: u32 = #COARSE_COUNT_PAGE_SIZE;
const MARKED_COUNT_PAGE: u32 = 0xFFFFFFFEu;

struct CoarseCountPages {
    tail: u32,
    counts: array<u32>,
}

@group(0) @binding(#{FRUSTUM_TABLE}) var<storage, read> frustum_table: array<FrustumDesc>;
@group(0) @binding(#{FROXEL_BUCKET_HEADS}) var<storage, read> froxel_bucket_heads: array<u32>;
@group(0) @binding(#{CHUNK_POOL}) var<storage, read> chunk_pool_words: array<u32>;
@group(0) @binding(#{COARSE_COUNT_PAGE_TABLE}) var<storage, read> coarse_count_page_table: array<u32>;
@group(0) @binding(#{COARSE_COUNT_PAGES}) var<storage, read> coarse_count_pages: CoarseCountPages;
@group(0) @binding(#{PARAMS}) var<uniform> params: TileDebugParams;
@group(0) @binding(#{INPUT_TEXTURE}) var input_texture: texture_2d<f32>;
@group(0) @binding(#{INPUT_SAMPLER}) var input_sampler: sampler;

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
    let scene_color = textureSample(input_texture, input_sampler, in.uv);
    if params.frustum_id >= arrayLength(&frustum_table) {
        return scene_color;
    }
    let config = frustum_table[params.frustum_id];
    if config.screen_width == 0u || config.screen_height == 0u {
        return scene_color;
    }

    let sx = min(u32(clamp(in.uv.x, 0.0, 0.999999) * f32(config.screen_width)), config.screen_width - 1u);
    let sy = min(u32(clamp(in.uv.y, 0.0, 0.999999) * f32(config.screen_height)), config.screen_height - 1u);

    var frag_count: u32 = 0u;
    let coarse_px_x = max(1u, config.froxel_size_x * COARSE_FINE_TILE_EXTENT);
    let coarse_px_y = max(1u, config.froxel_size_y * COARSE_FINE_TILE_EXTENT);
    let coarse_x = min(sx / coarse_px_x, config.coarse_tiles_x - 1u);
    let coarse_y = min(sy / coarse_px_y, config.coarse_tiles_y - 1u);
    let local_coarse_tile = coarse_y * config.coarse_tiles_x + coarse_x;
    let tile_idx = config.coarse_depth_tile_base + local_coarse_tile;
    let local_x = min((sx - coarse_x * coarse_px_x) / config.froxel_size_x, COARSE_FINE_TILE_EXTENT - 1u);
    let local_y = min((sy - coarse_y * coarse_px_y) / config.froxel_size_y, COARSE_FINE_TILE_EXTENT - 1u);
    for (var coarse_z: u32 = 0u; coarse_z < COARSE_DEPTH_SLICES; coarse_z = coarse_z + 1u) {
        let page_table_idx = tile_idx * COARSE_DEPTH_SLICES + coarse_z;
        if page_table_idx < arrayLength(&coarse_count_page_table) {
            let page_handle = coarse_count_page_table[page_table_idx];
            if page_handle != 0u && page_handle != MARKED_COUNT_PAGE {
                let page_idx = page_handle - 1u;
                for (var fine_z: u32 = 0u; fine_z < COARSE_DEPTH_SLICES; fine_z = fine_z + 1u) {
                    let cell_idx = (fine_z * COARSE_FINE_TILE_EXTENT + local_y) * COARSE_FINE_TILE_EXTENT + local_x;
                    let count_idx = page_idx * COARSE_COUNT_PAGE_SIZE + cell_idx;
                    if count_idx < arrayLength(&coarse_count_pages.counts) {
                        frag_count = frag_count + coarse_count_pages.counts[count_idx];
                    }
                }
            }
        }
    }

    if frag_count == 0u {
        return scene_color;
    }

    let norm = clamp(f32(frag_count) * params.inv_norm * params.gain, 0.0, 1.0);
    let debug_color = heatmap_precise(norm);
    return vec4<f32>(mix(scene_color.rgb, debug_color, params.alpha), scene_color.a);
}
