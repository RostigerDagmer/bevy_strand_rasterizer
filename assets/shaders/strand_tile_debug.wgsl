#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import "shaders/common.wgsl"::{ calculate_froxel_index }
#import "shaders/types.wgsl"::{ FroxelConfig }

struct TileDebugParams {
    inv_norm: f32,
    gain: f32,
    alpha: f32,
    pad0: f32,
}

@group(0) @binding(#{TILE_COUNTS_BUFFER}) var<storage, read> tile_counts_buffer: array<atomic<u32>>;
@group(0) @binding(#{FROXEL_CONFIG}) var<uniform> config: FroxelConfig;
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
    let sx = min(u32(clamp(in.uv.x, 0.0, 0.999999) * f32(config.screen_width)), config.screen_width - 1u);
    let sy = min(u32(clamp(in.uv.y, 0.0, 0.999999) * f32(config.screen_height)), config.screen_height - 1u);

    let tiles_x = (config.screen_width + config.froxel_size_x - 1u) / config.froxel_size_x;
    let tiles_y = (config.screen_height + config.froxel_size_y - 1u) / config.froxel_size_y;

    let tile_coord_x = min(sx / config.froxel_size_x, tiles_x - 1u);
    let tile_coord_y = min(sy / config.froxel_size_y, tiles_y - 1u);

    var frag_count: u32 = 0u;
    for (var dz: u32 = 0u; dz < config.depth_slices; dz = dz + 1u) {
        let idx = calculate_froxel_index(tile_coord_x, tile_coord_y, dz, config);
        frag_count = frag_count + atomicLoad(&tile_counts_buffer[idx]);
    }

    if frag_count == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let norm = clamp(f32(frag_count) * params.inv_norm * params.gain, 0.0, 1.0);
    let debug_color = heatmap_precise(norm);
    return vec4<f32>(debug_color, params.alpha);
}
