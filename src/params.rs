use crate::{config::LlamaConfigJson, tensor::Tensor};
use half::f16;
use num_traits::Num;
use safetensors::SafeTensors;

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

macro_rules! data_from_bytes {
    ($bytes:expr, $P:ty) => {
        $bytes
            .chunks_exact(core::mem::size_of::<$P>())
            // from litte endian data (`_tobytes`)
            .map(|bytes| <$P>::from_le_bytes(bytes.try_into().unwrap()))
    };
}

trait GetTensorFromSafeTensors<P: Num> {
    fn get_tensor_from(tensors: &SafeTensors, name: &str) -> Result<Tensor<P>, &'static str>;
}
impl GetTensorFromSafeTensors<f32> for f32 {
    fn get_tensor_from(tensors: &SafeTensors, name: &str) -> Result<Tensor<f32>, &'static str> {
        let tensor_view = tensors.tensor(name).map_err(|e| {
            assert!(matches!(e, safetensors::SafeTensorError::TensorNotFound(_)));
            "Tensor not found"
        })?;
        let tensor = match tensor_view.dtype() {
            safetensors::Dtype::F32 => Tensor::new(
                data_from_bytes!(tensor_view.data(), f32).collect(),
                &tensor_view.shape().to_vec(),
            ),
            safetensors::Dtype::BF16 => {
                let data = data_from_bytes!(tensor_view.data(), half::bf16)
                    .map(half::bf16::to_f32)
                    .collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            }
            _ => unimplemented!(),
        };
        Ok(tensor)
    }
}
impl GetTensorFromSafeTensors<half::bf16> for half::bf16 {
    fn get_tensor_from(tensors: &SafeTensors, name: &str) -> Result<Tensor<half::bf16>, &'static str> {
        let tensor_view = tensors.tensor(name).map_err(|e| {
            assert!(matches!(e, safetensors::SafeTensorError::TensorNotFound(_)));
            "Tensor not found"
        })?;
        
        let tensor = match tensor_view.dtype() {
            safetensors::Dtype::BF16 => {
                let data = data_from_bytes!(tensor_view.data(), half::bf16).collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            }
            _ => unimplemented!(),
        };
        Ok(tensor)
    }
}

impl GetTensorFromSafeTensors<half::f16> for half::f16 {
    fn get_tensor_from(tensors: &SafeTensors, name: &str) -> Result<Tensor<half::f16>, &'static str> {
        let tensor_view = tensors.tensor(name).map_err(|e| {
            assert!(matches!(e, safetensors::SafeTensorError::TensorNotFound(_)));
            "Tensor not found"
        })?;
        
        let tensor = match tensor_view.dtype() {
            safetensors::Dtype::F16 => {
                let data = data_from_bytes!(tensor_view.data(), half::f16).collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            }
            _ => unimplemented!(),
        };
        Ok(tensor)
    }
}

macro_rules! impl_from_safetensors_for_LlamaParams {
    ($Param:ty) => {
        impl LLamaParams<$Param> {
            pub fn from_safetensors(tensors: &SafeTensors, config: &LlamaConfigJson) -> Self {
                macro_rules! get_tensor_vec {
                    ($name_pattern:literal) => {
                        (0..config.num_hidden_layers)
                            .map(|i| {
                                <$Param>::get_tensor_from(tensors, &format!($name_pattern, i))
                                    .unwrap()
                            })
                            .collect()
                    };
                }

                LLamaParams {
                    embedding_table: if config.tie_word_embeddings {
                        <$Param>::get_tensor_from(tensors, "lm_head.weight").unwrap()
                    } else {
                        <$Param>::get_tensor_from(tensors, "model.embed_tokens.weight").unwrap()
                    },

                    rms_att_w: get_tensor_vec!("model.layers.{}.input_layernorm.weight"),
                    wq: get_tensor_vec!("model.layers.{}.self_attn.q_proj.weight"),
                    wk: get_tensor_vec!("model.layers.{}.self_attn.k_proj.weight"),
                    wv: get_tensor_vec!("model.layers.{}.self_attn.v_proj.weight"),
                    wo: get_tensor_vec!("model.layers.{}.self_attn.o_proj.weight"),

                    rms_ffn_w: get_tensor_vec!("model.layers.{}.post_attention_layernorm.weight"),
                    w_up: get_tensor_vec!("model.layers.{}.mlp.up_proj.weight"),
                    w_gate: get_tensor_vec!("model.layers.{}.mlp.gate_proj.weight"),
                    w_down: get_tensor_vec!("model.layers.{}.mlp.down_proj.weight"),

                    lm_head: if config.tie_word_embeddings{
                        <$Param>::get_tensor_from(tensors, "lm_head.weight").unwrap()
                    }else{
                        <$Param>::get_tensor_from(tensors, "model.embed_tokens.weight").unwrap()
                    },
                    rms_out_w: <$Param>::get_tensor_from(tensors, "model.norm.weight").unwrap(),
                }
            }
        }
    };
}
impl_from_safetensors_for_LlamaParams!(f32);
impl_from_safetensors_for_LlamaParams!(half::f16);
impl_from_safetensors_for_LlamaParams!(half::bf16);
