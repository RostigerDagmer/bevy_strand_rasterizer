#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View

// Input textures and sampler
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;
@group(0) @binding(3) var strand_texture: texture_2d<f32>; // Input is rgba8unorm, but sampling as f32 is fine
@group(0) @binding(4) var strand_depth: texture_2d<f32>;
@group(0) @binding(5) var scene_depth: texture_depth_multisampled_2d;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Sample the main scene texture
    let scene_color = textureSample(scene_texture, texture_sampler, in.uv);
    let icoord  = in.uv * view.viewport.zw; // integer coordinates
    let scene_depth = textureLoad(scene_depth, vec2<i32>(icoord), 0);
    let strand_depth = textureSample(strand_depth, texture_sampler, in.uv).x;

    // let min_depth = max(strand_depth, scene_depth);
    // return vec4<f32>(min_depth, min_depth, min_depth, 1.0);

    if (scene_depth > strand_depth) {
        // early out if there's nothing to blend
        return scene_color;
    }

    let strand_color = textureSample(strand_texture, texture_sampler, in.uv);

    let blended_color = mix(scene_color.rgb, strand_color.rgb, strand_color.a);
    let final_alpha = scene_color.a; // Keep original scene alpha or blend if needed

    return vec4<f32>(blended_color, final_alpha);
}