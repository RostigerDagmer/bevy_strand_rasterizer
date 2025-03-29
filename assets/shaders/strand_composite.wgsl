#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

// Input textures and sampler
@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var strand_texture: texture_2d<f32>; // Input is rgba8unorm, but sampling as f32 is fine

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Sample the main scene texture
    let scene_color = textureSample(scene_texture, texture_sampler, in.uv);

    // Sample the strand heatmap texture
    // Use textureLoad if sizes/coords match exactly and no filtering needed
    // let strand_tex_size = textureDimensions(strand_texture);
    // let coord = vec2<i32>(floor(in.uv * vec2<f32>(strand_tex_size)));
    // let strand_color = textureLoad(strand_texture, coord, 0); // Load directly

    // Or sample if filtering/different coords
     let strand_color = textureSample(strand_texture, texture_sampler, in.uv);

    // --- Blending Logic ---
    // Option A: Alpha Blending (using strand_color.a)
    // Assumes strand_color has meaningful alpha (like the 0.2 you set)
    let blended_color = mix(scene_color.rgb, strand_color.rgb, strand_color.a);
    let final_alpha = scene_color.a; // Keep original scene alpha or blend if needed

    // Option B: Additive Blending (good for heatmaps)
    // let blended_color = scene_color.rgb + strand_color.rgb * strand_color.a; // Modulate by alpha
    // let final_alpha = scene_color.a;

    // Option C: Simple Overlay (if strand_color alpha is just for intensity)
    // let blended_color = scene_color.rgb + strand_color.rgb;
    // let final_alpha = scene_color.a;


    return vec4<f32>(blended_color, final_alpha);
    // Debug: Just show strand texture
    // return strand_color;
    // Debug: Just show scene texture
    // return scene_color;
}