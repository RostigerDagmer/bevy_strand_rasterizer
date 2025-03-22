use serde::{Deserialize, Serialize};
use fluent_uri::UriRef;

pub type Url = UriRef<String>;

// Main struct to hold the parsed DSON file
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DsonFile {
    pub file_version: String,
    pub asset_info: AssetInfo,
    pub geometry_library: Option<Vec<Geometry>>,
    pub node_library: Option<Vec<Node>>,
    pub uv_set_library: Option<Vec<UVSet>>,
    pub modifier_library: Option<Vec<Modifier>>,
    pub image_library: Option<Vec<Image>>,
    pub material_library: Option<Vec<Material>>,
    pub scene: Option<Scene>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AssetInfo {
    pub id: Url,
    #[serde(rename = "type")]
    pub asset_type: Option<String>,
    pub contributor: Contributor,
    pub revision: String,
    pub modified: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Contributor {
    pub author: String,
    pub email: Option<String>,
    pub website: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum GeometryType {
    PolygonMesh,
    SubdivisionSurface,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum EdgeInterpolationMode {
    NoInterpolation,
    EdgesAndCorners,
    EdgesOnly,
}

// --- Geometry Assets ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Geometry {
    pub id: String,
    pub name: Option<String>,
    pub label: Option<String>,
    #[serde(rename = "type")]
    pub geometry_type: Option<GeometryType>, // "polygon_mesh" or "subdivision_surface"
    pub source: Option<Url>,
    pub edge_interpolation_mode: Option<EdgeInterpolationMode>, // "no_interpolation", "edges_and_corners", or "edges_only"
    pub vertices: Float3Array,
    pub polygon_groups: StringArray,
    pub polygon_material_groups: StringArray,
    pub polylist: Polylist,
    pub polyline_list: Option<PolylineList>,
    pub default_uv_set: Option<Url>,
    pub root_region: Option<Region>,
    pub graft: Option<Graft>,
    pub rigidity: Option<Rigidity>,
    pub extra: Option<Vec<Extra>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Region {
    pub id: String,
    pub label: Option<String>,
    pub display_hint: Option<String>,
    pub map: Option<IntArray>,
    pub children: Option<Vec<Region>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Graft {
    pub vertex_count: Option<u32>,
    pub poly_count: Option<u32>,
    pub vertex_pairs: Option<Int2Array>,
    pub hidden_polys: Option<IntArray>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Rigidity {
    pub weights: Option<FloatIndexedArray>,
    pub groups: Vec<RigidityGroup>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum RotationMode {
    None,
    Full,
    Primary,
    Secondary,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ScaleMode {
    None,
    Primary,
    Secondary,
    Tertiary,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RigidityGroup {
    pub id: String,
    pub rotation_mode: Option<RotationMode>, // "none", "full", "primary", or "secondary"
    pub scale_modes: Vec<ScaleMode>, // "none", "primary", "secondary", or "tertiary"
    pub reference_vertices: IntArray,
    pub mask_vertices: IntArray,
    pub reference: Option<Url>,
    pub transform_nodes: Option<Vec<Url>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct GeometryInstance {
    pub id: Option<String>,
    pub url: Url,
    pub name: Option<String>,
    pub label: Option<String>,
    #[serde(rename = "type")]
    pub geo_type: Option<GeometryType>, // "polygon_mesh" or "subdivision_surface"
    // TODO: There is more that this can override.
}

// --- UV Assets ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct UVSet {
    pub id: String,
    pub name: Option<String>,
    pub label: Option<String>,
    
    pub source: Option<Url>,
    pub vertex_count: u32,
    pub uvs: Float2Array,
    /// Each entry is [polygon_index, polygon_vertex_index, uv_index], 
    /// where polygon_vertex_index refers to the index of a vertex in the geometry
    /// that is used by the polygon at polygon_index.
    pub polygon_vertex_indices: Option<Vec<[u32; 3]>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct UVSetInstance {
    pub id: String,
    pub url: Url,
    pub parent: Option<Url>,
}

// --- Node Assets ---

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    Node,
    Bone,
    Figure,
    Camera,
    Light,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum RotationOrder {
    XYZ,
    YZX,
    ZYX,
    ZXY,
    XZY,
    YXZ,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Node {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub node_type: Option<NodeType>, // "node", "bone", "figure", "camera", "light"
    pub label: String,
    pub source: Option<Url>,
    pub parent: Option<Url>,
    pub rotation_order: Option<RotationOrder>, // "XYZ", "YZX", "ZYX", "ZXY", "XZY", or "YXZ"
    pub inherits_scale: Option<bool>,
    pub center_point: Option<Vec<ChannelFloat>>,
    pub end_point: Option<Vec<ChannelFloat>>,
    pub orientation: Option<Vec<ChannelFloat>>,
    pub rotation: Option<Vec<ChannelFloat>>,
    pub translation: Option<Vec<ChannelFloat>>,
    pub scale: Option<Vec<ChannelFloat>>,
    pub general_scale: Option<ChannelFloat>,
    pub presentation: Option<Presentation>,
    pub formulas: Option<Vec<Formula>>,
    pub extra: Option<Vec<Extra>>,
    // Camera or Light specific properties
    pub perspective: Option<CameraPerspective>,
    pub orthographic: Option<CameraOrthographic>,
    pub color: Option<[f32; 3]>,
    pub point: Option<LightPoint>,
    pub directional: Option<LightDirectional>,
    pub spot: Option<LightSpot>,
    pub on: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NodeInstance {
    pub id: String,
    pub url: Url,
    pub parent: Option<Url>,
    pub parent_in_place: Option<Url>,
    pub conform_target: Option<Url>,
    pub geometries: Option<Vec<GeometryInstance>>,
    pub preview: Option<Preview>,
    // Allow overriding most properties from Node
    pub rotation: Option<Vec<ChannelFloat>>,
    pub translation: Option<Vec<ChannelFloat>>,
    pub scale: Option<Vec<ChannelFloat>>,
    pub general_scale: Option<ChannelFloat>,
    pub label: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Preview {
    pub oriented_box: Option<OrientedBox>,
    pub center_point: Option<[f32; 3]>,
    pub end_point: Option<[f32; 3]>,
    pub rotation_order: Option<RotationOrder>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OrientedBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CameraPerspective {
    pub znear: Option<f32>,
    pub zfar: Option<f32>,
    pub yfov: Option<f32>,
    pub focal_length: Option<f32>,
    pub depth_of_field: Option<bool>,
    pub focal_distance: Option<f32>,
    pub fstop: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CameraOrthographic {
    pub znear: Option<f32>,
    pub zfar: Option<f32>,
    pub ymag: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ShadowType {
    None,
    ShadowMap,
    Raytraced,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LightPoint {
    pub intensity: Option<f32>,
    pub shadow_type: Option<ShadowType>, // "none", "shadow_map", or "raytraced"
    pub shadow_softness: Option<f32>,
    pub shadow_bias: Option<f32>,
    pub constant_attenuation: Option<f32>,
    pub linear_attenuation: Option<f32>,
    pub quadratic_attenuation: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LightDirectional {
    pub intensity: Option<f32>,
    pub shadow_type: Option<ShadowType>, // "none", "shadow_map", or "raytraced"
    pub shadow_softness: Option<f32>,
    pub shadow_bias: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LightSpot {
    pub intensity: Option<f32>,
    pub shadow_type: Option<ShadowType>, // "none", "shadow_map", or "raytraced"
    pub shadow_softness: Option<f32>,
    pub shadow_bias: Option<f32>,
    pub constant_attenuation: Option<f32>,
    pub linear_attenuation: Option<f32>,
    pub quadratic_attenuation: Option<f32>,
    pub falloff_angle: Option<f32>,
    pub falloff_exponent: Option<f32>,
}

// --- Modifier Assets ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Modifier {
    pub id: String,
    pub name: Option<String>,
    pub label: Option<String>,
    
    pub source: Option<Url>,
    
    pub parent: Option<Url>,
    pub presentation: Option<Presentation>,
    pub channel: Option<Channel>,
    pub region: Option<String>,
    pub group: Option<String>,
    pub formulas: Option<Vec<Formula>>,
    pub morph: Option<Morph>,
    pub skin: Option<SkinBinding>,
    pub extra: Option<Vec<Extra>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModifierInstance {
    pub id: String,
    pub parent: Option<Url>,
    pub url: Url,
    // Allow overriding properties from Modifier except id and presentation
    pub channel: Option<Channel>,
    pub region: Option<String>,
    pub group: Option<String>,
    pub formulas: Option<Vec<Formula>>,
    pub morph: Option<Morph>,
    pub skin: Option<SkinBinding>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Morph {
    pub vertex_count: i32,
    pub deltas: Float3IndexedArray,
    pub extra: Option<Vec<Extra>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SkinBinding {
    pub node: Url,
    pub geometry: Url,
    pub vertex_count: u32,
    pub joints: Option<Vec<WeightedJoint>>,
    pub selection_sets: Option<Vec<NamedStringMap>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct WeightedJoint {
    pub id: String,
    
    pub node: Url,
    pub node_weights: Option<FloatIndexedArray>,
    pub scale_weights: Option<FloatIndexedArray>,
    pub local_weights: Option<LocalWeights>,
    pub bulge_weights: Option<BulgeWeights>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LocalWeights {
    pub x: Option<FloatIndexedArray>,
    pub y: Option<FloatIndexedArray>,
    pub z: Option<FloatIndexedArray>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BulgeWeights {
    pub x: Option<BulgeBinding>,
    pub y: Option<BulgeBinding>,
    pub z: Option<BulgeBinding>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BulgeBinding {
    pub bulges: Vec<ChannelFloat>,
    pub left_map: FloatIndexedArray,
    pub right_map: FloatIndexedArray,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NamedStringMap {
    pub id: String,
    pub mappings: Vec<[String; 2]>, // [face group name, node name]
}

// --- Material Assets ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Material {
    pub id: String,
    pub name: Option<String>,
    pub label: Option<String>,
    pub source: Option<Url>,
    pub uv_set: Option<Url>,
    #[serde(rename = "type")]
    pub material_type: Option<String>,
    pub diffuse: Option<MaterialChannel>,
    pub diffuse_strength: Option<MaterialChannel>,
    pub specular: Option<MaterialChannel>,
    pub specular_strength: Option<MaterialChannel>,
    pub glossiness: Option<MaterialChannel>,
    pub ambient: Option<MaterialChannel>,
    pub ambient_strength: Option<MaterialChannel>,
    pub reflection: Option<MaterialChannel>,
    pub reflection_strength: Option<MaterialChannel>,
    pub refraction: Option<MaterialChannel>,
    pub refraction_strength: Option<MaterialChannel>,
    pub ior: Option<MaterialChannel>,
    pub bump: Option<MaterialChannel>,
    pub bump_min: Option<MaterialChannel>,
    pub bump_max: Option<MaterialChannel>,
    pub displacement: Option<MaterialChannel>,
    pub displacement_min: Option<MaterialChannel>,
    pub displacement_max: Option<MaterialChannel>,
    pub transparency: Option<MaterialChannel>,
    pub normal: Option<MaterialChannel>,
    pub u_offset: Option<MaterialChannel>,
    pub u_scale: Option<MaterialChannel>,
    pub v_offset: Option<MaterialChannel>,
    pub v_scale: Option<MaterialChannel>,
    pub extra: Option<Vec<Extra>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MaterialInstance {
    pub id: String,
    pub url: Url,
    pub geometry: Url,
    pub groups: Vec<String>,
    // Allow overriding most properties from Material except id and uv_set
    pub diffuse: Option<MaterialChannel>,
    pub diffuse_strength: Option<MaterialChannel>,
    pub specular: Option<MaterialChannel>,
    pub specular_strength: Option<MaterialChannel>,
    pub glossiness: Option<MaterialChannel>,
    pub ambient: Option<MaterialChannel>,
    pub ambient_strength: Option<MaterialChannel>,
    pub reflection: Option<MaterialChannel>,
    pub reflection_strength: Option<MaterialChannel>,
    pub refraction: Option<MaterialChannel>,
    pub refraction_strength: Option<MaterialChannel>,
    pub ior: Option<MaterialChannel>,
    pub bump: Option<MaterialChannel>,
    pub bump_min: Option<MaterialChannel>,
    pub bump_max: Option<MaterialChannel>,
    pub displacement: Option<MaterialChannel>,
    pub displacement_min: Option<MaterialChannel>,
    pub displacement_max: Option<MaterialChannel>,
    pub transparency: Option<MaterialChannel>,
    pub normal: Option<MaterialChannel>,
    pub u_offset: Option<MaterialChannel>,
    pub u_scale: Option<MaterialChannel>,
    pub v_offset: Option<MaterialChannel>,
    pub v_scale: Option<MaterialChannel>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MaterialChannel {
    pub channel: Option<Channel>,
    pub group: Option<String>,
    pub color: Option<[f32; 3]>,
    pub strength: Option<f32>,
    pub image: Option<Url>,
}

// --- Image Assets ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Image {
    pub id: String,
    pub name: String,
    pub source: Option<Url>,
    pub map_gamma: Option<f32>,
    pub map_size: Option<[u32; 2]>,
    pub map: Option<Vec<ImageMap>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageMap {
    pub url: Option<Url>,
    pub label: String,
    pub active: Option<bool>,
    pub color: Option<[f32; 3]>,
    pub transparency: Option<f32>,
    pub invert: Option<bool>,
    pub rotation: Option<f32>,
    pub xmirror: Option<bool>,
    pub ymirror: Option<bool>,
    pub xscale: Option<f32>,
    pub yscale: Option<f32>,
    pub xoffset: Option<f32>,
    pub yoffset: Option<f32>,
    pub operation: Option<String>,
}

// --- Scene Definition ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Scene {
    pub presentation: Option<Presentation>,
    pub nodes: Option<Vec<NodeInstance>>,
    pub uvs: Option<Vec<UVSetInstance>>,
    pub modifiers: Option<Vec<ModifierInstance>>,
    pub materials: Option<Vec<MaterialInstance>>,
    pub animations: Option<Vec<ChannelAnimation>>,
    pub current_camera: Option<Url>,
    pub extra: Option<Vec<Extra>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChannelAnimation {
    
    pub url: Url,
    pub keys: Vec<AnimationKey>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum AnimationKey {
    Simple([f32; 2]), // [time, value]
    Vector((f32, Vec<f32>)), // [time, [values...]]
    WithInterpolation((f32, f32, (String, f32, f32, f32))), // [time, value, [interpolation_type, val1, val2, val3]]
    VectorWithInterpolation((f32, Vec<f32>, Vec<serde_json::Value>)), // [time, [values...], [interpolation_type, ...]]
}

// --- Channel Types ---

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ChannelType {
    Alias,
    Bool,
    Color,
    Enum,
    Float,
    FloatColor,
    Image,
    Int,
    String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Channel {
    pub id: String,
    #[serde(rename = "type")]
    pub channel_type: ChannelType, // "alias", "bool", "color", "enum", "float", "image", "int", "string", "float_color" (genesis 9+)
    pub name: String,
    pub label: Option<String>,
    pub visible: Option<bool>,
    pub locked: Option<bool>,
    pub auto_follow: Option<bool>,
    // Type-specific fields
    pub target_channel: Option<Url>, // for alias
    pub value: Option<ChannelValue>,
    pub current_value: Option<ChannelValue>,
    pub min: Option<ChannelValue>,
    pub max: Option<ChannelValue>,
    pub clamped: Option<bool>,
    pub display_as_percent: Option<bool>,
    pub step_size: Option<f32>,
    pub mappable: Option<bool>,
    pub enum_values: Option<Vec<String>>, // for enum
    pub image_file: Option<Url>, // for image
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum ChannelValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    Color([f32; 3]),
    String(String),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChannelFloat {
    pub id: String,
    #[serde(rename = "type")]
    pub channel_type: Option<ChannelType>,
    pub name: Option<String>,
    pub label: Option<String>,
    pub visible: Option<bool>,
    pub locked: Option<bool>,
    pub auto_follow: Option<bool>,
    pub value: f32,
    pub current_value: Option<f32>,
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub clamped: Option<bool>,
    pub display_as_percent: Option<bool>,
    pub step_size: Option<f32>,
    pub mappable: Option<bool>,
}

// --- Formulas ---

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum FormulaStage {
    Multiply,
    Sum,
}

pub type FormulaUrl = String; // TODO: Implement URL parsing for formulas

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Formula {
    pub output: FormulaUrl,
    pub operations: Vec<Operation>,
    pub stage: Option<FormulaStage>, // "multiply" or "sum"
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Operation {
    pub op: String,
    pub val: Option<OperationValue>,
    pub url: Option<FormulaUrl>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum OperationValue {
    Int(i32),
    Float(f32),
    Knot2([f32; 2]),
    Knot5([f32; 5]),
}

// --- Presentation ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Presentation {
    #[serde(rename = "type")]
    pub presentation_type: String,
    pub label: String,
    pub description: String,
    pub icon_large: String,
    pub icon_small: Option<String>,
    pub colors: [[f32; 3]; 2],
}

// --- Extra for application-specific information ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Extra {
    #[serde(rename = "type")]
    pub extra_type: String,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

// --- Array Types ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Float2Array {
    pub count: u32,
    pub values: Vec<[f32; 2]>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Float3Array {
    pub count: u32,
    pub values: Vec<[f32; 3]>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StringArray {
    pub count: u32,
    pub values: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct IntArray {
    pub count: u32,
    pub values: Vec<u32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Int2Array {
    pub count: u32,
    pub values: Vec<[u32; 2]>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FloatArray {
    pub count: u32,
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FloatIndexedArray {
    pub count: u32,
    pub values: Vec<(u32, f32)>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Float3IndexedArray {
    pub count: u32,
    pub values: Vec<(u32, f32, f32, f32)>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Polylist {
    pub count: u32,
    pub values: Vec<Vec<u32>>, // [group_idx, mat_group_idx, vertex_indices...]
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PolylineList {
    pub count: u32,
    pub segment_count: Option<u32>,
    pub values: Vec<Vec<u32>>, // [group_idx, mat_group_idx, vertex_indices...]
}