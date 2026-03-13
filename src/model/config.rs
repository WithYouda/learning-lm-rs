use serde::Deserialize;

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct LlamaConfigJson {
    pub bos_token_id: u32,
    pub eos_token_id: Vec<u32>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub torch_dtype: String,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: RopeScaling,
    #[serde(default)]
    // 显式定义头维度，通常为 hidden_size / num_heads
    pub head_dim: Option<usize>, 
    #[serde(default)]
    // Llama 3 明确为 false
    pub attention_bias: bool,    
    #[serde(default)]
    // Llama 3 明确为 false
    pub mlp_bias: bool,          
}

/// llama 3 新增字段
/// 定义 RoPE Scaling 的子结构体
#[derive(Debug, serde::Serialize,Deserialize, Clone, Default)]
pub struct RopeScaling {
    pub factor: f32,   
    pub high_freq_factor: f32,
    pub low_freq_factor: f32,   
    pub original_max_position_embeddings: usize,
    pub rope_type: String,  
    // 以防万一可选字段
    #[serde(default)]
    pub short_factor: Option<Vec<f32>>, 
    #[serde(default)]
    pub long_factor: Option<Vec<f32>>,
}


#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}

#[inline(always)]
const fn default_tie_word_embeddings() -> bool {
    false
}
