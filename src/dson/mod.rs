pub mod model;
pub use model::*;

use bevy::{
    asset::{Asset, AssetLoader, LoadContext, io::Reader},
    reflect::TypePath,
};

/// Asset type for DSON files
#[derive(Asset, TypePath, Debug, Clone)]
pub struct DsonAsset {
    pub dson_file: DsonFile,
}

/// Asset loader for DSON files
#[derive(Default, TypePath)]
pub struct DsonAssetLoader;

impl AssetLoader for DsonAssetLoader {
    type Asset = DsonAsset;
    type Settings = ();
    type Error = anyhow::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let content = String::from_utf8(bytes)?;
        let dson_file: DsonFile = serde_json::from_str(&content)?;
        Ok(DsonAsset { dson_file })
    }

    fn extensions(&self) -> &[&str] {
        &["duf", "dsf"]
    }
}
