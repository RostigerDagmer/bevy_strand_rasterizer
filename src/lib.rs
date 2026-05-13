pub mod components;
pub mod dson;
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
            FroxelConfig, StrandAsset, StrandCache, StrandGeometry, StrandInstanceTransform,
            StrandMaterial, TieFroxelsToView,
        },
        dson::{DsonAsset, DsonAssetLoader},
        plugin::StrandRasterizerPlugin,
        resources::{StochasticCullSettings, TileDebugSettings},
        strand_cache::{StrandCacheAsset, StrandCacheAssetLoader},
    };
}
