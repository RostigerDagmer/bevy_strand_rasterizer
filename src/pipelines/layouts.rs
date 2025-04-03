pub mod rasterizer {
    use bevy::render::render_resource::ShaderDefVal;

    pub const VERTEX_BUFFER: u32 = 0;
    pub const INDEX_BUFFER: u32 = 1;
    pub const META_BUFFER: u32 = 2;
    pub const TILE_OFFSETS_BUFFER: u32 = 3;
    pub const TILE_COUNTS_BUFFER: u32 = 4;
    pub const FROXEL_TILE_BUFFER: u32 = 5;
    pub const OUTPUT_TEXTURE: u32 = 6;
    pub const FROXEL_CONFIG: u32 = 7;
    pub const VIEW_UNIFORM: u32 = 8;
    pub const SHADING_BUFFER: u32 = 9;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("VERTEX_BUFFER".into(), VERTEX_BUFFER),
            ShaderDefVal::UInt("INDEX_BUFFER".into(), INDEX_BUFFER),
            ShaderDefVal::UInt("META_BUFFER".into(), META_BUFFER),
            ShaderDefVal::UInt("TILE_OFFSETS_BUFFER".into(), TILE_OFFSETS_BUFFER),
            ShaderDefVal::UInt("TILE_COUNTS_BUFFER".into(), TILE_COUNTS_BUFFER),
            ShaderDefVal::UInt("FROXEL_TILE_BUFFER".into(), FROXEL_TILE_BUFFER),
            ShaderDefVal::UInt("OUTPUT_TEXTURE".into(), OUTPUT_TEXTURE),
            ShaderDefVal::UInt("FROXEL_CONFIG".into(), FROXEL_CONFIG),
            ShaderDefVal::UInt("VIEW_UNIFORM".into(), VIEW_UNIFORM),
            ShaderDefVal::UInt("SHADING_BUFFER".into(), SHADING_BUFFER),
        ]
    }
}

pub mod binning {
    use bevy::render::render_resource::ShaderDefVal;

    pub const VERTEX_BUFFER: u32 = 0;
    pub const INDEX_BUFFER: u32 = 1;
    pub const META_BUFFER: u32 = 2;
    pub const TILE_COUNTS_BUFFER: u32 = 3;
    pub const TILE_OFFSETS_BUFFER: u32 = 4;
    pub const CURRENT_TILE_WRITE_INDICES: u32 = 5;
    pub const FROXEL_TILE_BUFFER: u32 = 6;
    pub const FROXEL_CONFIG: u32 = 7;
    pub const VIEW_UNIFORM: u32 = 8;
    pub const GEO_BUFFER: u32 = 9;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("VERTEX_BUFFER".into(), VERTEX_BUFFER),
            ShaderDefVal::UInt("INDEX_BUFFER".into(), INDEX_BUFFER),
            ShaderDefVal::UInt("META_BUFFER".into(), META_BUFFER),
            ShaderDefVal::UInt("TILE_COUNTS_BUFFER".into(), TILE_COUNTS_BUFFER),
            ShaderDefVal::UInt("TILE_OFFSETS_BUFFER".into(), TILE_OFFSETS_BUFFER),
            ShaderDefVal::UInt(
                "CURRENT_TILE_WRITE_INDICES".into(),
                CURRENT_TILE_WRITE_INDICES,
            ),
            ShaderDefVal::UInt("FROXEL_TILE_BUFFER".into(), FROXEL_TILE_BUFFER),
            ShaderDefVal::UInt("FROXEL_CONFIG".into(), FROXEL_CONFIG),
            ShaderDefVal::UInt("VIEW_UNIFORM".into(), VIEW_UNIFORM),
            ShaderDefVal::UInt("GEO_BUFFER".into(), GEO_BUFFER),
        ]
    }
}

pub mod shading {
    use bevy::render::render_resource::ShaderDefVal;

    pub const VERTEX_BUFFER: u32 = 0;
    pub const INDEX_BUFFER: u32 = 1;
    pub const META_BUFFER: u32 = 2;
    pub const VIEW_UNIFORM: u32 = 3;
    pub const LIGHT_UNIFORM: u32 = 4;
    pub const OUTPUT_TEXTURE: u32 = 5;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("VERTEX_BUFFER".into(), VERTEX_BUFFER),
            ShaderDefVal::UInt("INDEX_BUFFER".into(), INDEX_BUFFER),
            ShaderDefVal::UInt("META_BUFFER".into(), META_BUFFER),
            ShaderDefVal::UInt("VIEW_UNIFORM".into(), VIEW_UNIFORM),
            ShaderDefVal::UInt("LIGHT_UNIFORM".into(), LIGHT_UNIFORM),
            ShaderDefVal::UInt("OUTPUT_TEXTURE".into(), OUTPUT_TEXTURE),
        ]
    }
}

pub mod post_process {
    pub const INPUT_TEXTURE: u32 = 0;
    pub const PARAMS_BUFFER: u32 = 1;
}