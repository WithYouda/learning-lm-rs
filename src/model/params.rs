use crate::model::config::LlamaConfigJson;
use crate::core::tensor::Tensor;
use crate::formats::gguf::*;


use num_traits::Num;
use safetensors::SafeTensors;



/// 模型权重的枚举类型：支持密集张量和 GGUF 量化张量两种形式。
/// Dense: 标准 f32 张量（用于 safetensors 加载或调试）
/// GgufQ: GGUF 原生量化张量（支持 Q4K/Q6K/Q8K 等多种格式）
pub enum Weight<T> {
    Dense(Tensor<T>),
    GgufQ(QuantGGUFTensor),
}

impl<T> Weight<T> {
    /// 将权重强制解包为密集张量引用，量化权重会 panic
    #[allow(unused)]
    pub fn as_dense(&self) -> &Tensor<T> {
        match self {
            Weight::Dense(t) => t,
            Weight::GgufQ(_) => panic!("called as_dense() on quantized weight"),
        }
    }
}

/// Llama 模型的所有可训练参数集合。
/// 包含嵌入表、每层的注意力权重、FFN 权重、输出投影等。
pub struct LLamaParams<T> {
    /// 嵌入查找表 (vocab_size, hidden_size)
    pub embedding_table: Tensor<T>, 
    /// 注意力层前的 RMSNorm 权重 (hidden_size,) x layers
    pub rms_att_w: Vec<Tensor<T>>, 
    /// Q 投影权重 (n_heads * head_size, hidden_size) x layers
    pub wq: Vec<Weight<T>>,  
    /// K 投影权重 (n_kv_heads * head_size, hidden_size) x layers
    pub wk: Vec<Weight<T>>,   
    /// V 投影权重 (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Weight<T>>,  
    /// 输出投影权重 (hidden_size, n_heads * head_size) x layers
    pub wo: Vec<Weight<T>>,        
    /// FFN 层前的 RMSNorm 权重 (hidden_size,) x layers
    pub rms_ffn_w: Vec<Tensor<T>>, 
    /// FFN 上投影权重 (intermediate_size, hidden_size) x layers
    pub w_up: Vec<Weight<T>>,      
    /// FFN 门控投影权重 (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Weight<T>>,   
    /// FFN 下投影权重 (hidden_size, intermediate_size) x layers
    pub w_down: Vec<Weight<T>>,    
    /// 输出层前的 RMSNorm 权重 (hidden_size,)
    pub rms_out_w: Tensor<T>, 
    /// 输出投影矩阵 (vocab_size, hidden_size)，可能与 embedding_table 绑定
    pub lm_head: Weight<T>,   
}

/// 将一个类型的tensor转换成特定的类型的迭代器
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

/// 为支持 f32 实现了 GetTensorFromSafeTensors 特征
impl GetTensorFromSafeTensors<f32> for f32 {
    fn get_tensor_from(tensors: &SafeTensors, name: &str) -> Result<Tensor<f32>, &'static str> {
        let tensor_view = tensors.tensor(name).map_err(|e| {
            eprintln!("{} tensor not found",name);
            assert!(matches!(e, safetensors::SafeTensorError::TensorNotFound(_)));
            "Tensor not found"
        })?;
        // 若tensor的元素数据类型就是f32，则无需转换
        // 若tensor的元素类型是f16，则按照f16的大小分割后，再将每个元素转换换成f32类型
        // 若tensor的元素类型是bf16，则按照bf16的大小分割后，再将每个元素转换换成f32类型

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
            },
            safetensors::Dtype::F16 => {
                let data = data_from_bytes!(tensor_view.data(), half::f16)
                    .map(half::f16::to_f32)
                    .collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            },
            _ => unimplemented!(),
        };
        Ok(tensor)
    }
}


/// 为支持 f16 实现了 GetTensorFromSafeTensors 特征
impl GetTensorFromSafeTensors<half::f16> for half::f16 {
    fn get_tensor_from(tensors: &SafeTensors, name: &str) -> Result<Tensor<half::f16>, &'static str> {
        let tensor_view = tensors.tensor(name).map_err(|e| {
            eprintln!("{} tensor not found",name);
            assert!(matches!(e, safetensors::SafeTensorError::TensorNotFound(_)));
            "Tensor not found"
        })?;
        
        let tensor = match tensor_view.dtype() {
            safetensors::Dtype::F16 => {
                let data = data_from_bytes!(tensor_view.data(), half::f16).collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            },
            safetensors::Dtype::F32 => {
                let data = data_from_bytes!(tensor_view.data(), f32)
                    .map(|val| half::f16::from_f32(val))
                    .collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            },
            safetensors::Dtype::BF16 => {
                let data = data_from_bytes!(tensor_view.data(), half::bf16)
                    .map(|val| half::f16::from_f32(val.to_f32()))
                    .collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            },
            _ => unimplemented!(),
        };
        Ok(tensor)
    }
}


/// 为支持 bf16 实现了 GetTensorFromSafeTensors 特征
impl GetTensorFromSafeTensors<half::bf16> for half::bf16 {
    fn get_tensor_from(tensors: &SafeTensors, name: &str) -> Result<Tensor<half::bf16>, &'static str> {
        let tensor_view = tensors.tensor(name).map_err(|e| {
            eprintln!("{} tensor not found",name);
            assert!(matches!(e, safetensors::SafeTensorError::TensorNotFound(_)));
            "Tensor not found"
        })?;
        
        let tensor = match tensor_view.dtype() {
            safetensors::Dtype::BF16 => {
                let data = data_from_bytes!(tensor_view.data(), half::bf16).collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            },
            safetensors::Dtype::F32 => {
                let data = data_from_bytes!(tensor_view.data(), f32)
                    .map(|val| half::bf16::from_f32(val))
                    .collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            },
            safetensors::Dtype::F16 => {
                let data = data_from_bytes!(tensor_view.data(), half::f16)
                    .map(|val| half::bf16::from_f32(val.to_f32()))
                    .collect();
                Tensor::new(data, &tensor_view.shape().to_vec())
            },
             _ => unimplemented!(),
        };
        Ok(tensor)
    }
}

/// 从 safetensor 为每个类型生成对应的LlamaParams 结构体
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
                            .collect::<Vec<Tensor<$Param>>>()
                    };
                }

                LLamaParams {
                    embedding_table: if !config.tie_word_embeddings {
                        <$Param>::get_tensor_from(tensors, "lm_head.weight").unwrap()
                    } else {
                        <$Param>::get_tensor_from(tensors, "model.embed_tokens.weight").unwrap()
                    },

                    rms_att_w: get_tensor_vec!("model.layers.{}.input_layernorm.weight"),
                    wq: get_tensor_vec!("model.layers.{}.self_attn.q_proj.weight")
                        .into_iter()
                        .map(Weight::Dense)
                        .collect(),
                    wk: get_tensor_vec!("model.layers.{}.self_attn.k_proj.weight")
                        .into_iter()
                        .map(Weight::Dense)
                        .collect(),
                    wv: get_tensor_vec!("model.layers.{}.self_attn.v_proj.weight")
                        .into_iter()
                        .map(Weight::Dense)
                        .collect(),
                    wo: get_tensor_vec!("model.layers.{}.self_attn.o_proj.weight")
                        .into_iter()
                        .map(Weight::Dense)
                        .collect(),

                    rms_ffn_w: get_tensor_vec!("model.layers.{}.post_attention_layernorm.weight"),
                    w_up: get_tensor_vec!("model.layers.{}.mlp.up_proj.weight")
                        .into_iter()
                        .map(Weight::Dense)
                        .collect(),
                    w_gate: get_tensor_vec!("model.layers.{}.mlp.gate_proj.weight")
                        .into_iter()
                        .map(Weight::Dense)
                        .collect(),
                    w_down: get_tensor_vec!("model.layers.{}.mlp.down_proj.weight")
                        .into_iter()
                        .map(Weight::Dense)
                        .collect(),

                    lm_head: if !config.tie_word_embeddings{
                        Weight::Dense(<$Param>::get_tensor_from(tensors, "lm_head.weight").unwrap())
                    }else{
                        Weight::Dense(<$Param>::get_tensor_from(tensors, "model.embed_tokens.weight").unwrap())
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



pub trait FromF32 {
    fn from_f32_vec(v: Vec<f32>) -> Vec<Self>
    where
        Self: Sized;
}

impl FromF32 for f32 {
    fn from_f32_vec(v: Vec<f32>) -> Vec<Self> {
        v
    }
}

impl FromF32 for half::f16 {
    fn from_f32_vec(v: Vec<f32>) -> Vec<Self> {
        v.into_iter().map(half::f16::from_f32).collect()
    }
}

impl FromF32 for half::bf16 {
    fn from_f32_vec(v: Vec<f32>) -> Vec<Self> {
        v.into_iter().map(half::bf16::from_f32).collect()
    }
}

