use bevy::{
    asset::{Asset, AssetLoader, LoadContext, io::Reader},
    reflect::TypePath,
};
use serde::{Deserialize, Serialize};

const BINARY_MAGIC: &[u8; 8] = b"SSRSTRD\0";
const BINARY_VERSION: u32 = 1;

#[derive(Asset, TypePath, Debug, Clone)]
pub struct StrandCacheAsset {
    pub cache: StrandCacheFile,
}

#[derive(Debug, Clone)]
pub struct StrandCacheFile {
    pub version: u32,
    pub source: Option<String>,
    pub scale: f32,
    pub vertices: Vec<[f32; 3]>,
    pub strands: Vec<StrandCacheStrand>,
    pub indices: Vec<u32>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct StrandCacheStrand {
    pub count: u32,
    pub offset: u32,
}

#[derive(Debug, Deserialize, Serialize)]
struct JsonStrandCacheFile {
    pub version: u32,
    pub source: Option<String>,
    #[serde(default = "unit_scale")]
    pub scale: f32,
    pub vertices: Vec<[f32; 3]>,
    pub strands: Vec<Vec<u32>>,
}

fn unit_scale() -> f32 {
    1.0
}

impl TryFrom<JsonStrandCacheFile> for StrandCacheFile {
    type Error = anyhow::Error;

    fn try_from(json: JsonStrandCacheFile) -> Result<Self, Self::Error> {
        let mut strands = Vec::with_capacity(json.strands.len());
        let mut indices = Vec::new();
        for strand in json.strands {
            let count = strand.len() as u32;
            let offset = indices.len() as u32;
            indices.extend(strand);
            strands.push(StrandCacheStrand { count, offset });
        }

        Ok(Self {
            version: json.version,
            source: json.source,
            scale: json.scale,
            vertices: json.vertices,
            strands,
            indices,
        })
    }
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact<const N: usize>(&mut self) -> anyhow::Result<[u8; N]> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or_else(|| anyhow::anyhow!("strand cache offset overflow"))?;
        let chunk = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| anyhow::anyhow!("unexpected end of strand cache"))?;
        self.offset = end;
        Ok(chunk.try_into().unwrap())
    }

    fn read_u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    fn read_u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact()?))
    }

    fn read_f32(&mut self) -> anyhow::Result<f32> {
        Ok(f32::from_le_bytes(self.read_exact()?))
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

fn parse_binary_cache(bytes: &[u8]) -> anyhow::Result<StrandCacheFile> {
    let mut reader = ByteReader::new(bytes);
    let magic = reader.read_exact::<8>()?;
    if &magic != BINARY_MAGIC {
        anyhow::bail!("invalid strand cache magic");
    }

    let version = reader.read_u32()?;
    if version != BINARY_VERSION {
        anyhow::bail!("unsupported strand cache version {version}");
    }

    let scale = reader.read_f32()?;
    let vertex_count = usize::try_from(reader.read_u64()?)?;
    let strand_count = usize::try_from(reader.read_u64()?)?;
    let index_count = usize::try_from(reader.read_u64()?)?;

    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        vertices.push([reader.read_f32()?, reader.read_f32()?, reader.read_f32()?]);
    }

    let mut strands = Vec::with_capacity(strand_count);
    for _ in 0..strand_count {
        strands.push(StrandCacheStrand {
            count: reader.read_u32()?,
            offset: reader.read_u32()?,
        });
    }

    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        indices.push(reader.read_u32()?);
    }

    if reader.remaining() != 0 {
        anyhow::bail!(
            "strand cache has {} trailing bytes after payload",
            reader.remaining()
        );
    }

    Ok(StrandCacheFile {
        version,
        source: None,
        scale,
        vertices,
        strands,
        indices,
    })
}

fn parse_json_cache(bytes: &[u8]) -> anyhow::Result<StrandCacheFile> {
    let json: JsonStrandCacheFile = serde_json::from_slice(bytes)?;
    json.try_into()
}

fn parse_cache(bytes: &[u8]) -> anyhow::Result<StrandCacheFile> {
    if bytes.starts_with(BINARY_MAGIC) {
        parse_binary_cache(bytes)
    } else {
        parse_json_cache(bytes)
    }
}

fn validate_cache(cache: &StrandCacheFile) -> anyhow::Result<()> {
    if cache.version != BINARY_VERSION {
        anyhow::bail!("unsupported strand cache version {}", cache.version);
    }

    for (strand_index, strand) in cache.strands.iter().enumerate() {
        if strand.count < 2 {
            anyhow::bail!("strand {strand_index} has fewer than two vertices");
        }

        let end = strand
            .offset
            .checked_add(strand.count)
            .ok_or_else(|| anyhow::anyhow!("strand {strand_index} index range overflows"))?;
        if end as usize > cache.indices.len() {
            anyhow::bail!(
                "strand {strand_index} range {}..{} exceeds {} cache indices",
                strand.offset,
                end,
                cache.indices.len()
            );
        }

        for index_offset in strand.offset..end {
            let index = cache.indices[index_offset as usize];
            if index as usize >= cache.vertices.len() {
                anyhow::bail!(
                    "strand {strand_index} references vertex {index}, but cache has {} vertices",
                    cache.vertices.len()
                );
            }
        }
    }

    Ok(())
}

#[derive(Default, TypePath)]
pub struct StrandCacheAssetLoader;

impl AssetLoader for StrandCacheAssetLoader {
    type Asset = StrandCacheAsset;
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
        let cache = parse_cache(&bytes)?;
        validate_cache(&cache)?;

        Ok(StrandCacheAsset { cache })
    }

    fn extensions(&self) -> &[&str] {
        &["strands", "strands.json", "json"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_f32(bytes: &mut Vec<u8>, value: f32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn parses_binary_cache() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(BINARY_MAGIC);
        push_u32(&mut bytes, BINARY_VERSION);
        push_f32(&mut bytes, 1.0);
        push_u64(&mut bytes, 3);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 3);

        for vertex in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]] {
            for component in vertex {
                push_f32(&mut bytes, component);
            }
        }

        push_u32(&mut bytes, 3);
        push_u32(&mut bytes, 0);
        for index in [0, 1, 2] {
            push_u32(&mut bytes, index);
        }

        let cache = parse_binary_cache(&bytes).unwrap();
        validate_cache(&cache).unwrap();
        assert_eq!(cache.vertices.len(), 3);
        assert_eq!(cache.strands.len(), 1);
        assert_eq!(cache.indices, vec![0, 1, 2]);
    }
}
