pub mod prepass {
    use bevy::shader::ShaderDefVal;

    pub const PREPASS_GROUP: u32 = 2;

    // Queues
    pub const PREPASS_QUEUE: u32 = 4;
    pub const BINNING_QUEUE: u32 = 5;
    // Internals
    pub const LIGHT_UNIFORM: u32 = 6;
    pub const CLUSTER_INDICES: u32 = 7;
    pub const CLUSTER_OFFSETS_AND_COUNTS: u32 = 8;
    pub const CLUSTERABLE_OBJECTS: u32 = 9;
    pub const VIEW_UNIFORM: u32 = 10;
    // Helpers
    pub const VISIBLE_FLAGS: u32 = 11;
    pub const VISIBLE_GEO: u32 = 12;
    pub const GEO_PREFIX: u32 = 13;
    pub const INDIRECT_BUFFER: u32 = 15;
    pub const FRUSTUM_TABLE: u32 = 21;
    pub const FROXEL_BUCKET_HEADS: u32 = 22;
    pub const CHUNK_POOL: u32 = 23;
    pub const FREE_HEADS: u32 = 24;
    pub const RASTER_WORK_QUEUE: u32 = 25;
    pub const COARSE_DEPTH_LUT: u32 = 27;
    pub const COARSE_RANGE_QUEUE: u32 = 28;
    pub const COARSE_INTERVAL_HEADS: u32 = 29;
    pub const COARSE_INTERVAL_REFS: u32 = 30;
    pub const COARSE_RANGE_LOOKUP: u32 = 31;
    pub const COARSE_COUNT_PAGE_TABLE: u32 = 32;
    pub const COARSE_COUNT_PAGES: u32 = 33;
    pub const STRAND_INSTANCES: u32 = 34;
    pub const FINE_PAGE_META: u32 = 35;
    pub const FINE_CELL_OFFSETS: u32 = 36;
    pub const FINE_CELL_WRITE_CURSORS: u32 = 37;
    pub const FINE_SEG_REFS: u32 = 38;
    pub const VSMS_REQUEST_META: u32 = 41;
    pub const VSMS_REQUEST_BITS: u32 = 42;
    pub const SHADOW_DOM_SURFACE_IDS: u32 = 43;
    pub const BROAD_INSTANCE_META: u32 = 44;
    pub const OPAQUE_FINE_DEPTH_TILES: u32 = 45;
    pub const RASTER_TILE_RUN_QUEUE: u32 = 46;
    pub const RASTER_TILE_RUN_DISPATCH_ARGS: u32 = 47;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("PREPASS_GROUP".into(), PREPASS_GROUP),
            // Queues
            ShaderDefVal::UInt("PREPASS_QUEUE".into(), PREPASS_QUEUE),
            ShaderDefVal::UInt("BINNING_QUEUE".into(), BINNING_QUEUE),
            // Internals
            ShaderDefVal::UInt("LIGHT_UNIFORM".into(), LIGHT_UNIFORM),
            ShaderDefVal::UInt("CLUSTER_INDICES".into(), CLUSTER_INDICES),
            ShaderDefVal::UInt(
                "CLUSTER_OFFSETS_AND_COUNTS".into(),
                CLUSTER_OFFSETS_AND_COUNTS,
            ),
            ShaderDefVal::UInt("CLUSTERABLE_OBJECTS".into(), CLUSTERABLE_OBJECTS),
            ShaderDefVal::UInt("VIEW_UNIFORM".into(), VIEW_UNIFORM),
            // Helpers
            ShaderDefVal::UInt("VISIBLE_FLAGS".into(), VISIBLE_FLAGS),
            ShaderDefVal::UInt("VISIBLE_GEO".into(), VISIBLE_GEO),
            ShaderDefVal::UInt("GEO_PREFIX".into(), GEO_PREFIX),
            ShaderDefVal::UInt("INDIRECT_BUFFER".into(), INDIRECT_BUFFER),
            ShaderDefVal::UInt("FRUSTUM_TABLE".into(), FRUSTUM_TABLE),
            ShaderDefVal::UInt("FROXEL_BUCKET_HEADS".into(), FROXEL_BUCKET_HEADS),
            ShaderDefVal::UInt("CHUNK_POOL".into(), CHUNK_POOL),
            ShaderDefVal::UInt("FREE_HEADS".into(), FREE_HEADS),
            ShaderDefVal::UInt("RASTER_WORK_QUEUE".into(), RASTER_WORK_QUEUE),
            ShaderDefVal::UInt("COARSE_DEPTH_LUT".into(), COARSE_DEPTH_LUT),
            ShaderDefVal::UInt("COARSE_RANGE_QUEUE".into(), COARSE_RANGE_QUEUE),
            ShaderDefVal::UInt("COARSE_INTERVAL_HEADS".into(), COARSE_INTERVAL_HEADS),
            ShaderDefVal::UInt("COARSE_INTERVAL_REFS".into(), COARSE_INTERVAL_REFS),
            ShaderDefVal::UInt("COARSE_RANGE_LOOKUP".into(), COARSE_RANGE_LOOKUP),
            ShaderDefVal::UInt("COARSE_COUNT_PAGE_TABLE".into(), COARSE_COUNT_PAGE_TABLE),
            ShaderDefVal::UInt("COARSE_COUNT_PAGES".into(), COARSE_COUNT_PAGES),
            ShaderDefVal::UInt("STRAND_INSTANCES".into(), STRAND_INSTANCES),
            ShaderDefVal::UInt("FINE_PAGE_META".into(), FINE_PAGE_META),
            ShaderDefVal::UInt("FINE_CELL_OFFSETS".into(), FINE_CELL_OFFSETS),
            ShaderDefVal::UInt("FINE_CELL_WRITE_CURSORS".into(), FINE_CELL_WRITE_CURSORS),
            ShaderDefVal::UInt("FINE_SEG_REFS".into(), FINE_SEG_REFS),
            ShaderDefVal::UInt("VSMS_REQUEST_META".into(), VSMS_REQUEST_META),
            ShaderDefVal::UInt("VSMS_REQUEST_BITS".into(), VSMS_REQUEST_BITS),
            ShaderDefVal::UInt("SHADOW_DOM_SURFACE_IDS".into(), SHADOW_DOM_SURFACE_IDS),
            ShaderDefVal::UInt("BROAD_INSTANCE_META".into(), BROAD_INSTANCE_META),
            ShaderDefVal::UInt("OPAQUE_FINE_DEPTH_TILES".into(), OPAQUE_FINE_DEPTH_TILES),
            ShaderDefVal::UInt("RASTER_TILE_RUN_QUEUE".into(), RASTER_TILE_RUN_QUEUE),
            ShaderDefVal::UInt(
                "RASTER_TILE_RUN_DISPATCH_ARGS".into(),
                RASTER_TILE_RUN_DISPATCH_ARGS,
            ),
        ]
    }
}

pub mod rasterizer {
    use bevy::shader::ShaderDefVal;

    pub const RASTER_GROUP: u32 = 2;
    pub const VSMS_OPACITY_WRITE_GROUP: u32 = 3;
    pub const VSMS_DEPTH_WRITE_GROUP: u32 = 4;
    pub const VSMS_OPACITY_TABLE_GROUP: u32 = 5;
    pub const VSMS_DEPTH_TABLE_GROUP: u32 = 6;
    pub const VSMS_STORAGE_BINDING: u32 = 0;
    pub const VSMS_VIRTUAL_META_BINDING: u32 = 0;
    pub const VSMS_VIRTUAL_PAGE_TABLE_BINDING: u32 = 1;

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
    pub const GEO_BUFFER: u32 = 18;
    pub const DEEP_OPACITY_TEXTURE_O_VIEW: u32 = 19;
    pub const DEEP_OPACITY_TEXTURE_D_VIEW: u32 = 20;
    pub const OUTPUT_DEPTH: u32 = 21;
    pub const POINT_LIGHT_DEPTH_TEXTURE: u32 = 22;
    pub const DIRECTIONAL_LIGHT_DEPTH_TEXTURE: u32 = 23;
    pub const MATERIAL_BUFFER: u32 = 24;
    pub const FRUSTUM_TABLE: u32 = 25;
    pub const FROXEL_BUCKET_HEADS: u32 = 26;
    pub const CHUNK_POOL: u32 = 27;
    pub const RASTER_WORK_QUEUE: u32 = 28;
    pub const FINE_SEG_REFS: u32 = 29;
    pub const STRAND_INSTANCES: u32 = 30;
    pub const SHADOW_DOM_SURFACE_IDS: u32 = 33;
    pub const RASTER_TILE_RUN_QUEUE: u32 = 34;
    pub const SHADOW_HISTORY_TEXTURE: u32 = 35;

    pub const VSMS_POOL_PAGE_TABLE_BINDING: u32 = 0;
    pub const VSMS_POOL_TEXTURE_BINDING: u32 = 1;
    pub const VSMS_POOL_SAMPLER_BINDING: u32 = 2;
    pub const VSMS_OPACITY_POOL_TEXTURE_COUNT_DEF: &str = "VSMS_OPACITY_POOL_TEXTURE_COUNT";
    pub const VSMS_DEPTH_POOL_TEXTURE_COUNT_DEF: &str = "VSMS_DEPTH_POOL_TEXTURE_COUNT";

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("RASTER_GROUP".into(), RASTER_GROUP),
            ShaderDefVal::UInt("VSMS_OPACITY_WRITE_GROUP".into(), VSMS_OPACITY_WRITE_GROUP),
            ShaderDefVal::UInt("VSMS_DEPTH_WRITE_GROUP".into(), VSMS_DEPTH_WRITE_GROUP),
            ShaderDefVal::UInt("VSMS_OPACITY_TABLE_GROUP".into(), VSMS_OPACITY_TABLE_GROUP),
            ShaderDefVal::UInt("VSMS_DEPTH_TABLE_GROUP".into(), VSMS_DEPTH_TABLE_GROUP),
            ShaderDefVal::UInt("VSMS_STORAGE_BINDING".into(), VSMS_STORAGE_BINDING),
            ShaderDefVal::UInt(
                "VSMS_VIRTUAL_META_BINDING".into(),
                VSMS_VIRTUAL_META_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_VIRTUAL_PAGE_TABLE_BINDING".into(),
                VSMS_VIRTUAL_PAGE_TABLE_BINDING,
            ),
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
            ShaderDefVal::UInt(
                "DEEP_OPACITY_TEXTURE_O_VIEW".into(),
                DEEP_OPACITY_TEXTURE_O_VIEW,
            ),
            ShaderDefVal::UInt(
                "DEEP_OPACITY_TEXTURE_D_VIEW".into(),
                DEEP_OPACITY_TEXTURE_D_VIEW,
            ),
            ShaderDefVal::UInt("GEO_BUFFER".into(), GEO_BUFFER),
            ShaderDefVal::UInt("MATERIAL_BUFFER".into(), MATERIAL_BUFFER),
            ShaderDefVal::UInt("FRUSTUM_TABLE".into(), FRUSTUM_TABLE),
            ShaderDefVal::UInt("FROXEL_BUCKET_HEADS".into(), FROXEL_BUCKET_HEADS),
            ShaderDefVal::UInt("CHUNK_POOL".into(), CHUNK_POOL),
            ShaderDefVal::UInt("RASTER_WORK_QUEUE".into(), RASTER_WORK_QUEUE),
            ShaderDefVal::UInt("FINE_SEG_REFS".into(), FINE_SEG_REFS),
            ShaderDefVal::UInt("STRAND_INSTANCES".into(), STRAND_INSTANCES),
            ShaderDefVal::UInt("SHADOW_DOM_SURFACE_IDS".into(), SHADOW_DOM_SURFACE_IDS),
            ShaderDefVal::UInt("RASTER_TILE_RUN_QUEUE".into(), RASTER_TILE_RUN_QUEUE),
            ShaderDefVal::UInt("SHADOW_HISTORY_TEXTURE".into(), SHADOW_HISTORY_TEXTURE),
            ShaderDefVal::UInt(
                "VSMS_POOL_PAGE_TABLE_BINDING".into(),
                VSMS_POOL_PAGE_TABLE_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_POOL_TEXTURE_BINDING".into(),
                VSMS_POOL_TEXTURE_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_POOL_SAMPLER_BINDING".into(),
                VSMS_POOL_SAMPLER_BINDING,
            ),
        ]
    }
}

pub mod shadows {
    use bevy::shader::ShaderDefVal;

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

pub mod shadow_stampback {
    use bevy::shader::ShaderDefVal;

    pub const STAMP_GROUP: u32 = 0;
    pub const VSMS_DEPTH_READ_GROUP: u32 = 1;
    pub const VSMS_DEPTH_TABLE_GROUP: u32 = 2;

    pub const FRUSTUM_TABLE: u32 = 0;
    pub const SHADOW_DOM_SURFACE_IDS: u32 = 1;
    pub const PARAMS: u32 = 2;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("STAMP_GROUP".into(), STAMP_GROUP),
            ShaderDefVal::UInt("VSMS_DEPTH_READ_GROUP".into(), VSMS_DEPTH_READ_GROUP),
            ShaderDefVal::UInt("VSMS_DEPTH_TABLE_GROUP".into(), VSMS_DEPTH_TABLE_GROUP),
            ShaderDefVal::UInt("FRUSTUM_TABLE".into(), FRUSTUM_TABLE),
            ShaderDefVal::UInt("SHADOW_DOM_SURFACE_IDS".into(), SHADOW_DOM_SURFACE_IDS),
            ShaderDefVal::UInt("PARAMS".into(), PARAMS),
            ShaderDefVal::UInt(
                "VSMS_POOL_TEXTURE_BINDING".into(),
                super::rasterizer::VSMS_POOL_TEXTURE_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_POOL_SAMPLER_BINDING".into(),
                super::rasterizer::VSMS_POOL_SAMPLER_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_VIRTUAL_META_BINDING".into(),
                super::rasterizer::VSMS_VIRTUAL_META_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_VIRTUAL_PAGE_TABLE_BINDING".into(),
                super::rasterizer::VSMS_VIRTUAL_PAGE_TABLE_BINDING,
            ),
        ]
    }
}

pub mod binning {
    use bevy::shader::ShaderDefVal;

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
    use bevy::shader::ShaderDefVal;

    pub const SHADING_GROUP: u32 = 2;
    pub const VSMS_OPACITY_WRITE_GROUP: u32 = 3;
    pub const VSMS_DEPTH_WRITE_GROUP: u32 = 4;
    pub const VSMS_OPACITY_TABLE_GROUP: u32 = 5;
    pub const VSMS_DEPTH_TABLE_GROUP: u32 = 6;
    pub const VSMS_VIRTUAL_META_BINDING: u32 = 0;
    pub const VSMS_VIRTUAL_PAGE_TABLE_BINDING: u32 = 1;
    pub const VSMS_POOL_PAGE_TABLE_BINDING: u32 = 0;
    pub const VSMS_POOL_TEXTURE_BINDING: u32 = 1;
    pub const VSMS_POOL_SAMPLER_BINDING: u32 = 2;
    pub const VSMS_OPACITY_POOL_TEXTURE_COUNT_DEF: &str = "VSMS_OPACITY_POOL_TEXTURE_COUNT";
    pub const VSMS_DEPTH_POOL_TEXTURE_COUNT_DEF: &str = "VSMS_DEPTH_POOL_TEXTURE_COUNT";

    pub const VIEW_UNIFORM: u32 = 0;
    pub const LIGHT_UNIFORM: u32 = 1;
    pub const BINNING_QUEUE: u32 = 2;
    pub const FRUSTUM_TABLE: u32 = 3;
    pub const OUTPUT_TEXTURE: u32 = 4;
    pub const STRAND_INSTANCES: u32 = 5;
    pub const SHADOW_DOM_SURFACE_IDS: u32 = 6;
    pub const SHADOW_HISTORY_PREV: u32 = 7;
    pub const SHADOW_HISTORY_NEXT: u32 = 8;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("SHADING_GROUP".into(), SHADING_GROUP),
            ShaderDefVal::UInt("VSMS_OPACITY_WRITE_GROUP".into(), VSMS_OPACITY_WRITE_GROUP),
            ShaderDefVal::UInt("VSMS_DEPTH_WRITE_GROUP".into(), VSMS_DEPTH_WRITE_GROUP),
            ShaderDefVal::UInt("VSMS_OPACITY_TABLE_GROUP".into(), VSMS_OPACITY_TABLE_GROUP),
            ShaderDefVal::UInt("VSMS_DEPTH_TABLE_GROUP".into(), VSMS_DEPTH_TABLE_GROUP),
            ShaderDefVal::UInt(
                "VSMS_VIRTUAL_META_BINDING".into(),
                VSMS_VIRTUAL_META_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_VIRTUAL_PAGE_TABLE_BINDING".into(),
                VSMS_VIRTUAL_PAGE_TABLE_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_POOL_PAGE_TABLE_BINDING".into(),
                VSMS_POOL_PAGE_TABLE_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_POOL_TEXTURE_BINDING".into(),
                VSMS_POOL_TEXTURE_BINDING,
            ),
            ShaderDefVal::UInt(
                "VSMS_POOL_SAMPLER_BINDING".into(),
                VSMS_POOL_SAMPLER_BINDING,
            ),
            ShaderDefVal::UInt("VIEW_UNIFORM".into(), VIEW_UNIFORM),
            ShaderDefVal::UInt("LIGHT_UNIFORM".into(), LIGHT_UNIFORM),
            ShaderDefVal::UInt("BINNING_QUEUE".into(), BINNING_QUEUE),
            ShaderDefVal::UInt("FRUSTUM_TABLE".into(), FRUSTUM_TABLE),
            ShaderDefVal::UInt("OUTPUT_TEXTURE".into(), OUTPUT_TEXTURE),
            ShaderDefVal::UInt("STRAND_INSTANCES".into(), STRAND_INSTANCES),
            ShaderDefVal::UInt("SHADOW_DOM_SURFACE_IDS".into(), SHADOW_DOM_SURFACE_IDS),
            ShaderDefVal::UInt("SHADOW_HISTORY_PREV".into(), SHADOW_HISTORY_PREV),
            ShaderDefVal::UInt("SHADOW_HISTORY_NEXT".into(), SHADOW_HISTORY_NEXT),
        ]
    }
}

pub mod post_process {
    pub const INPUT_TEXTURE: u32 = 0;
    pub const PARAMS_BUFFER: u32 = 1;
}

pub mod tile_debug {
    use crate::{
        pipelines::task_contract::BINNING_POOL_CHUNK_SIZE,
        plugin::{COARSE_COUNT_PAGE_SIZE, COARSE_DEPTH_SLICES, COARSE_FINE_TILE_EXTENT},
    };
    use bevy::shader::ShaderDefVal;

    pub const FRUSTUM_TABLE: u32 = 0;
    pub const FROXEL_BUCKET_HEADS: u32 = 1;
    pub const CHUNK_POOL: u32 = 2;
    pub const COARSE_COUNT_PAGE_TABLE: u32 = 3;
    pub const COARSE_COUNT_PAGES: u32 = 4;
    pub const PARAMS: u32 = 5;
    pub const INPUT_TEXTURE: u32 = 6;
    pub const INPUT_SAMPLER: u32 = 7;

    pub fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("FRUSTUM_TABLE".into(), FRUSTUM_TABLE),
            ShaderDefVal::UInt("FROXEL_BUCKET_HEADS".into(), FROXEL_BUCKET_HEADS),
            ShaderDefVal::UInt("CHUNK_POOL".into(), CHUNK_POOL),
            ShaderDefVal::UInt("COARSE_COUNT_PAGE_TABLE".into(), COARSE_COUNT_PAGE_TABLE),
            ShaderDefVal::UInt("COARSE_COUNT_PAGES".into(), COARSE_COUNT_PAGES),
            ShaderDefVal::UInt("PARAMS".into(), PARAMS),
            ShaderDefVal::UInt("INPUT_TEXTURE".into(), INPUT_TEXTURE),
            ShaderDefVal::UInt("INPUT_SAMPLER".into(), INPUT_SAMPLER),
            ShaderDefVal::UInt("POOL_CHUNK_SIZE".into(), BINNING_POOL_CHUNK_SIZE),
            ShaderDefVal::UInt("COARSE_FINE_TILE_EXTENT".into(), COARSE_FINE_TILE_EXTENT),
            ShaderDefVal::UInt("COARSE_DEPTH_SLICES".into(), COARSE_DEPTH_SLICES),
            ShaderDefVal::UInt("COARSE_COUNT_PAGE_SIZE".into(), COARSE_COUNT_PAGE_SIZE),
        ]
    }
}
