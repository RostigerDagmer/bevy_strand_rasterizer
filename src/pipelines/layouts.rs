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
    pub const DEEP_OPACITY_TEXTURE_O: u32 = 10;
    pub const DEEP_OPACITY_TEXTURE_D: u32 = 11;
    pub const LIGHT_UNIFORM: u32 = 12;
    pub const CLUSTER_INDICES: u32 = 13;
    pub const CLUSTER_OFFSETS_AND_COUNTS: u32 = 14;
    pub const CLUSTERABLE_OBJECTS: u32 = 15;
    pub const POINT_LIGHT_DEPTH_TEXTURE_SAMPLER: u32 = 16;
    pub const DIRECTIONAL_LIGHT_DEPTH_TEXTURE_SAMPLER: u32 = 17;
    pub const POINT_LIGHT_DEPTH_TEXTURE: u32 = 22;
    pub const DIRECTIONAL_LIGHT_DEPTH_TEXTURE: u32 = 23;
    pub const GEO_BUFFER: u32 = 18;
    pub const DEEP_OPACITY_TEXTURE_O_VIEW: u32 = 19;
    pub const DEEP_OPACITY_TEXTURE_D_VIEW: u32 = 20;
    pub const OUTPUT_DEPTH: u32 = 21;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("VERTEX_BUFFER".into(), VERTEX_BUFFER),
            ShaderDefVal::UInt("INDEX_BUFFER".into(), INDEX_BUFFER),
            ShaderDefVal::UInt("META_BUFFER".into(), META_BUFFER),
            ShaderDefVal::UInt("TILE_OFFSETS_BUFFER".into(), TILE_OFFSETS_BUFFER),
            ShaderDefVal::UInt("TILE_COUNTS_BUFFER".into(), TILE_COUNTS_BUFFER),
            ShaderDefVal::UInt("FROXEL_TILE_BUFFER".into(), FROXEL_TILE_BUFFER),
            ShaderDefVal::UInt("OUTPUT_TEXTURE".into(), OUTPUT_TEXTURE),
            ShaderDefVal::UInt("OUTPUT_DEPTH".into(), OUTPUT_DEPTH),
            ShaderDefVal::UInt("FROXEL_CONFIG".into(), FROXEL_CONFIG),
            ShaderDefVal::UInt("VIEW_UNIFORM".into(), VIEW_UNIFORM),
            ShaderDefVal::UInt("SHADING_BUFFER".into(), SHADING_BUFFER),
            ShaderDefVal::UInt("DEEP_OPACITY_TEXTURE_O".into(), DEEP_OPACITY_TEXTURE_O),
            ShaderDefVal::UInt("DEEP_OPACITY_TEXTURE_D".into(), DEEP_OPACITY_TEXTURE_D),
            ShaderDefVal::UInt("LIGHT_UNIFORM".into(), LIGHT_UNIFORM),
            ShaderDefVal::UInt("CLUSTER_INDICES".into(), CLUSTER_INDICES),
            ShaderDefVal::UInt(
                "CLUSTER_OFFSETS_AND_COUNTS".into(),
                CLUSTER_OFFSETS_AND_COUNTS,
            ),
            ShaderDefVal::UInt("CLUSTERABLE_OBJECTS".into(), CLUSTERABLE_OBJECTS),
            ShaderDefVal::UInt(
                "POINT_LIGHT_DEPTH_TEXTURE_SAMPLER".into(),
                POINT_LIGHT_DEPTH_TEXTURE_SAMPLER,
            ),
            ShaderDefVal::UInt(
                "DIRECTIONAL_LIGHT_DEPTH_TEXTURE_SAMPLER".into(),
                DIRECTIONAL_LIGHT_DEPTH_TEXTURE_SAMPLER,
            ),
            ShaderDefVal::UInt(
                "POINT_LIGHT_DEPTH_TEXTURE".into(),
                POINT_LIGHT_DEPTH_TEXTURE,
            ),
            ShaderDefVal::UInt(
                "DIRECTIONAL_LIGHT_DEPTH_TEXTURE".into(),
                DIRECTIONAL_LIGHT_DEPTH_TEXTURE,
            ),
            ShaderDefVal::UInt("GEO_BUFFER".into(), GEO_BUFFER),
        ]
    }
}

pub mod shadows {
    use bevy::render::render_resource::ShaderDefVal;

    pub const GEO_BUFFER: u32 = 0;

    pub const DEEP_OPACITY_TEXTURE_O: u32 = 1;
    pub const DEEP_OPACITY_TEXTURE_D: u32 = 2;

    pub const POINT_LIGHT_SHADOW_MAP: u32 = 3;
    pub const DIRECTIONAL_LIGHT_SHADOW_MAP: u32 = 4;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("DEEP_OPACITY_TEXTURE_O".into(), DEEP_OPACITY_TEXTURE_O),
            ShaderDefVal::UInt("DEEP_OPACITY_TEXTURE_D".into(), DEEP_OPACITY_TEXTURE_D),
            ShaderDefVal::UInt("POINT_LIGHT_SHADOW_MAP".into(), POINT_LIGHT_SHADOW_MAP),
            ShaderDefVal::UInt(
                "DIRECTIONAL_LIGHT_SHADOW_MAP".into(),
                DIRECTIONAL_LIGHT_SHADOW_MAP,
            ),
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
    pub const LIGHT_UNIFORM: u32 = 10;

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
            ShaderDefVal::UInt("LIGHT_UNIFORM".into(), LIGHT_UNIFORM),
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
    pub const CLUSTER_INDICES: u32 = 5;
    pub const CLUSTER_OFFSETS_AND_COUNTS: u32 = 6;
    pub const CLUSTERABLE_OBJECTS: u32 = 7;
    pub const POINT_LIGHT_DEPTH_TEXTURE: u32 = 8;
    pub const DIRECTIONAL_LIGHT_DEPTH_TEXTURE: u32 = 9;
    pub const OUTPUT_TEXTURE: u32 = 10;
    pub const MATERIAL_BUFFER: u32 = 11;
    pub const DEEP_OPACITY_TEXTURE_O: u32 = 12;
    pub const DEEP_OPACITY_TEXTURE_D: u32 = 13;
    pub const DEEP_OPACITY_TEXTURE_O_VIEW: u32 = 14;
    pub const DEEP_OPACITY_TEXTURE_D_VIEW: u32 = 15;
    pub const GEO_BUFFER: u32 = 16;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("VERTEX_BUFFER".into(), VERTEX_BUFFER),
            ShaderDefVal::UInt("INDEX_BUFFER".into(), INDEX_BUFFER),
            ShaderDefVal::UInt("META_BUFFER".into(), META_BUFFER),
            ShaderDefVal::UInt("VIEW_UNIFORM".into(), VIEW_UNIFORM),
            ShaderDefVal::UInt("LIGHT_UNIFORM".into(), LIGHT_UNIFORM),
            ShaderDefVal::UInt("CLUSTER_INDICES".into(), CLUSTER_INDICES),
            ShaderDefVal::UInt(
                "CLUSTER_OFFSETS_AND_COUNTS".into(),
                CLUSTER_OFFSETS_AND_COUNTS,
            ),
            ShaderDefVal::UInt("CLUSTERABLE_OBJECTS".into(), CLUSTERABLE_OBJECTS),
            ShaderDefVal::UInt(
                "POINT_LIGHT_DEPTH_TEXTURE".into(),
                POINT_LIGHT_DEPTH_TEXTURE,
            ),
            ShaderDefVal::UInt(
                "DIRECTIONAL_LIGHT_DEPTH_TEXTURE".into(),
                DIRECTIONAL_LIGHT_DEPTH_TEXTURE,
            ),
            ShaderDefVal::UInt("OUTPUT_TEXTURE".into(), OUTPUT_TEXTURE),
            ShaderDefVal::UInt("MATERIAL_BUFFER".into(), MATERIAL_BUFFER),
            ShaderDefVal::UInt("GEO_BUFFER".into(), GEO_BUFFER),
            ShaderDefVal::UInt("DEEP_OPACITY_TEXTURE_O".into(), DEEP_OPACITY_TEXTURE_O),
            ShaderDefVal::UInt("DEEP_OPACITY_TEXTURE_D".into(), DEEP_OPACITY_TEXTURE_D),
            ShaderDefVal::UInt(
                "DEEP_OPACITY_TEXTURE_O_VIEW".into(),
                DEEP_OPACITY_TEXTURE_O_VIEW,
            ),
            ShaderDefVal::UInt(
                "DEEP_OPACITY_TEXTURE_D_VIEW".into(),
                DEEP_OPACITY_TEXTURE_D_VIEW,
            ),
        ]
    }
}

pub mod post_process {
    pub const INPUT_TEXTURE: u32 = 0;
    pub const PARAMS_BUFFER: u32 = 1;
}
