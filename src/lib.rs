pub mod components;
mod nodes;
mod pipelines;
mod plugin;
pub mod resources;
mod shader_types;
pub mod strand_cache;

pub use bevy_gpu_paging_allocator as allocator;
pub use plugin::StrandRasterizerPlugin;

pub mod prelude {
    pub use crate::{
        components::{
            FroxelConfig, StrandCache, StrandGeometry, StrandInstanceTransform, StrandList,
            StrandMaterial, TieFroxelsToView,
        },
        plugin::StrandRasterizerPlugin,
        resources::{StochasticCullSettings, TileDebugSettings},
        strand_cache::{StrandCacheAsset, StrandCacheAssetLoader},
    };
}
