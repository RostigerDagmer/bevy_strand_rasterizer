// strand_rasterizer.wgsl
@compute @workgroup_size(64, 1, 1)
fn rasterize_strands(@builtin(global_invocation_id) id: vec3<u32>) {
    // TODO: Implement strand rasterization
    // - Read vertices and indices
    // - Bin strands into froxels
    // - Write to output texture
}