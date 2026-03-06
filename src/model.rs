use std::fmt::Debug;
use std::fs::File;
use std::vec;
use safetensors::SafeTensors;
use std::path::Path;
use num_traits::float::Float;
use num_traits::Num;
use num_traits::FromPrimitive;
use std::time::Instant;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;

#[allow(unused)]
pub struct Llama<T: Num> {
    // vocab size —— 词汇表大小
    vocab: usize,      
    // number of layers —— transformer 解码器层数     
    n_layers: usize, 
    // number of heads for q —— Q投影的注意力头数       
    n_q_h: usize,   
    // number of heads for k and v —— K和V投影的注意力头数        
    n_kv_h: usize,    
    // dimension of hidden states —— 隐藏状态的维度      
    d: usize,        
    // length of a single q, k, or v vector —— 单个Q、K或V向量的维度       
    dqkv: usize,      
    // dimension of intermediate states —— MLP中间层的维度      
    di: usize, 
    // epsilon for RMS normalization —— RMSNorm 中的 epsilon 数值             
    eps: f32,    
    // rope theta for rope initialization —— RoPE中的基底频率参数 θ           
    rope_theta: f32,     
    // maximum sequence length —— 模型支持的最大序列长度  
    max_seq_len: usize,   
    // trained weights of this model —— 模型的所有可训练参数  
    params: LLamaParams<T>, 
    // start token id —— 起始 token id
    bos_token_id: u32,  
    // end token id —— 结束 token id    
    eos_token_id: u32,      
}

/// 生成对应的Llama 结构体的宏
macro_rules! impl_from_safetensors_for_Llama{
    ($Param:ty) =>{
        impl Llama<$Param> {
            pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
                let config = File::open(model_dir.as_ref().join("config.json")).expect("open config failed");
                let config: LlamaConfigJson = match serde_json::from_reader(config){
                    Ok(val) => val,
                    Err(e) => {
                        eprintln!("详细错误{}",e);
                        panic!("Fuck!")
                    }
                };
                let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).expect("open safetensor failed");
                let safetensor = SafeTensors::deserialize(&model_file).expect("safetensor deserialize failed");
                let params = LLamaParams::<$Param>::from_safetensors(&safetensor, &config);
                // f16模式下，强制提升 epsilon 以防下溢
                let mut eps = config.rms_norm_eps; 
                if std::any::TypeId::of::<$Param>() == std::any::TypeId::of::<half::f16>() {
                    eps = 1e-4; 
                    println!("⚠️ 检测到 f16 模式，已将 rms_norm_eps 从 {} 安全提升至 {}", config.rms_norm_eps, eps);
                }
                Self {
                    vocab: config.vocab_size,
                    n_layers: config.num_hidden_layers,
                    n_q_h: config.num_attention_heads,
                    n_kv_h: config.num_key_value_heads,
                    d: config.hidden_size,
                    dqkv: config.hidden_size / config.num_attention_heads,
                    di: config.intermediate_size,
                    eps,
                    rope_theta: config.rope_theta,
                    max_seq_len: config.max_position_embeddings,
                    params: params,
                    bos_token_id: config.bos_token_id,
                    eos_token_id: config.eos_token_id,
                }
            }
        }
    };
}

// 为f32调用impl_from_safetensors_for_Llama 宏，使得为 f32 实现 from_safetensors 方法
impl_from_safetensors_for_Llama!(f32);

// 为f16调用impl_from_safetensors_for_Llama 宏，使得为 f16 实现 from_safetensors 方法
impl_from_safetensors_for_Llama!(half::f16);

// 为bf16调用impl_from_safetensors_for_Llama 宏，使得为 bf16 实现 from_safetensors 方法
impl_from_safetensors_for_Llama!(half::bf16);


impl<T> Llama<T> 
    where  
    T: Float + Default + std::iter::Sum + num_traits::FromPrimitive 
    + std::fmt::Debug + std::ops::MulAssign + std::ops::AddAssign
    + Into<f32> + std::marker::Sync + std::marker::Send + 'static
{
    
    /// 初始化 KVCache  
    pub fn new_cache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    // 为什么input的类型是Tensor<u32>?
    // 因为input是输入的token_id序列，而token_id 是正整数
    /// 向前传播方法
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        // 当前输入的长度
        let seq_len = input.size();
        // cache 的长度
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // 为相应的层、中间结果、缓存初始化张量
        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores = Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, self.di]);

        // Embedding lookup
        // 将根据输入的token_id查询到table(嵌入表)中对应的向量复制到residual(输入嵌入向量)中
        // 开始计时 gather 操作所需时间
        let start = Instant::now();
        OP::gather(&mut residual, input, &self.params.embedding_table);
        // 结束计时 gather 操作所需时间
        let duration = start.elapsed();
        println!("gather 操作花了{:?}时间",duration);
        // 依次对每一层进行操作
        for layer in 0..self.n_layers {
            // 为每一层进行归一化，防止梯度爆炸或者消失
            // 开始计时 forward 最开始 rms_norm 操作时间
            let start = Instant::now();
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );
            // 结束计时 forward 最开始 rms_norm 操作时间
            let duration = start.elapsed();
            println!("forward 最开始 rms_norm 操作时间{:?}时间",duration);
            // 为什么 reshape 原本的形状？
            // 显式再次声明形状，确保内存布局连续！
            // (seq, n_q_h * dqkv)
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); 
            // (seq, n_kv_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); 
            // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); 

            // 分别计算出 hidden_states 在 q,k,v 上的投影并存入 q,k,v 中
            // 开始计时 forward 中分别对q,k,v 进行 matual_transb 操作总共所需时间
            let start = Instant::now();
            OP::matmul_transb(q, T::from(0.).unwrap(), &hidden_states, &self.params.wq[layer], T::from(1.).unwrap());
            OP::matmul_transb(k, T::from(0.).unwrap(), &hidden_states, &self.params.wk[layer], T::from(1.).unwrap());
            OP::matmul_transb(v, T::from(0.).unwrap(), &hidden_states, &self.params.wv[layer], T::from(1.).unwrap());
            // 结束计时 forward 中分别对q,k,v 进行 matual_transb 操作总共所需时间
            let duration = start.elapsed();
            println!("forward 中分别对q,k,v 进行 matual_transb 操作总共花了{:?}时间",duration);
            // 对q,k 进行旋转位置编码
            // 开始计时 forward 中对 q,k进行 rope 操作所需时间
            let start = Instant::now();
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            // 结束计时 forward 中对 q,k进行 rope 操作所需时间
            let duration = start.elapsed();
            println!("forward 中对 q,k进行 rope 操作花了{:?}时间",duration);
            // 生成完整的k,v
            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // 注意力计算
            // 开始计时 attention 计算所需时间
            let start = Instant::now();
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv
            );
            // 开始计时 attention 计算所需时间
            let duration = start.elapsed();
            println!("attention 计算花了{:?}时间",duration);
            // down_proj matmul and add residual;
            // residual = hidden_states @ wo.T + residual
            OP::matmul_transb(&mut residual, T::from(1.).unwrap(), &hidden_states, &self.params.wo[layer], T::from(1.).unwrap());
            // 重新初始化一份 hidden_states 张量
            hidden_states = Tensor::<T>::default(&vec![seq_len, self.d]);
            // 进行 mlp 计算
            // 开始计时 mlp 操作所需时间
            let start = Instant::now();
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps
            );
            // 结束计时 mlp 操作所需时间
            let duration = start.elapsed();
            println!("mlp操作花了{:?}时间",duration);
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<T>::default(&vec![1, self.vocab]);
        // 取hidden_states中的最后一个词的特征向量
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        // 取residual中的最后一个词的特征向量
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);
        // 对上面取出的residual切片进行归一化操作，并将结果保存在hidden_states切片中中
        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w ,
            self.eps,
        );
        // logits = hidden_states @ params.lm_head^T
        // parm.lm_head 是一个投影矩阵，每一行代表词汇表中某一个特定词的“特征模板”或者“原型向量”
        // 相乘相当于相似度匹配，每个词汇表对应位置的计算值越大，说明这个词选中的概率越大
        OP::matmul_transb(&mut logits, T::from(0.).unwrap(), &hidden_states, &self.params.lm_head, T::from(1.).unwrap());

        logits
    }

    /// 生成方法
    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        penalty: f32,
    ) -> Vec<u32>
        where T: Into<f32>
    {
        let mut result = Vec::<u32>::new();
        // step 1
        // 初始化 kvcache
        let mut cache = self.new_cache();
        // 初始化 result
        for token in token_ids{
            result.push(*token);
        }
        // 构造输入张量
        let mut input_tensor = Tensor::new(token_ids.to_vec(), &vec![1,token_ids.len()]);
        
        // 调用 random_sample 并将结果存入 result 中
        for _ in 0..max_len{
            // 开始计时 forward 操作所需时间
            let start = Instant::now();
            let forward_tensor = self.forward(&input_tensor, &mut cache);
            // 结束计时 forward 操作所需时间
            let duration = start.elapsed();
            println!("forward花了{:?}时间",duration);
            // 随机采样
            // 开始计时 随机采样所需时间
            let start = Instant::now();
            let id = OP::random_sample(&forward_tensor, &result, top_p, top_k, temperature, penalty);
            // 结束计时 随机采样所需时间
            let duration = start.elapsed();
            println!("随机采样花了{:?}时间",duration);
            if id == self.eos_token_id {
                break;
            }
            result.push(id);
            // 将这次生成的词作为新的上下文输入
            input_tensor = Tensor::new(vec![id],&vec![1, 1]);
        }
        result
    }


    pub fn chat(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        penalty: f32,
        mut cache: KVCache<T>,
    ) -> (Vec<u32>, KVCache<T>)
    where T:Into<f32>    
{
        let length = token_ids.len();
        let mut result = Vec::<u32>::new();
        let token: Vec<u32> = Vec::from(token_ids);
        for token in token_ids{
            result.push(*token);
        }
        let mut input = Tensor::<u32>::new(token, &vec![1, token_ids.len()]);
        // 循环生成直到 break
        loop {
            let id = OP::random_sample(&self.forward(&input, &mut cache), &result, top_p, top_k, temperature,penalty);
            result.push(id);
            if result.len() >= max_len || id == self.eos_token_id {
                break;
            }
            input = Tensor::<u32>::new(vec![id], &vec![1, 1]);
        }
        result = result[length..].to_vec();
        (result, cache)
    }

}

/// 注意力计算
/// 需要性能优化
fn self_attention<T>(
    // (seq, n_kv_h * n_groups * dqkv)
    hidden_states: &mut Tensor<T>,
    // (n_kv_h, n_groups, seq, total_seq) 
    att_scores: &mut Tensor<T>,   
    // (seq, n_kv_h * n_groups * dqkv) 
    q: &Tensor<T>,    
    // (total_seq, n_kv_h * dqkv)             
    k: &Tensor<T>,  
    // (total_seq, n_kv_h * dqkv)               
    v: &Tensor<T>,                 
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) 
    where T: Float + Default + FromPrimitive + std::iter::Sum + std::ops::AddAssign + Debug
        + std::marker::Sync + std::marker::Send + 'static
{
    // 第一步： 计算 score
    // socre = Q · K.T / sqrt(dim) 
    let sqrt_dim = T::from(dqkv).unwrap().sqrt();
    let scores = unsafe{att_scores.data_mut()};
    for i in 0..seq_len{
        for j in 0..total_seq_len{
            for m in 0..n_kv_h{
                for n in 0..n_groups{
                    // 定位 Q 的切片起点
                    let q_start = (m * n_groups + n) * dqkv + i * n_groups * n_kv_h * dqkv;
                    let q_= q.slice(q_start, &vec![dqkv, 1]);
                    // 定位 K 的切片切点
                    let k_start = m * dqkv + j *  n_kv_h * dqkv;
                    let k_: Tensor<T> = k.slice(k_start, &vec![dqkv, 1]);
                    // 计算 scores 
                    let value = OP::dot(&q_, &k_) / sqrt_dim;
                    // 存入 scores 正确的位置
                    scores[m * n_groups * seq_len * total_seq_len 
                        + n * seq_len * total_seq_len 
                        + i * total_seq_len
                        + j] 
                        = value;
                }
            }
        }
    }
    // 第二步：对att_scores 进行 masked_softmax 计算 
    // attn = softmax(score)
    OP::masked_softmax(att_scores);
    // 第三步：用得出的 attn 与 V 权重矩阵做矩阵乘法
    // x = attn @ V
    // attn (n_kv_head, n_group, seq_len, total_seq_len) --> n_kv_head * n_group * (seq_len, total_seq_len)
    // attn_slice = (seq_len, total_seq_len)
    // v (total_seq_len, n_kv_head * head_size) --> v.T (n_kv_head * head_size, total_seq_len)
    // v.T (n_kv_head * head_size, total_seq_len) --> n_kv_head * (head_size, total_seq_len)
    // v.T_slice = (head_size, total_seq_len)
    // matmul_transb (attn_slice , v.T_slice) = (seq_len, head_size)
    // hidden_state = attn @ V = n_kv_head * n_group * (seq_len, head_size) = (seq_len, n_kv_head * n_group * head_size) 
    let v_data = v.data();
    let hidden_len = n_kv_h * n_groups * dqkv;
    let hidden = unsafe{hidden_states.data_mut()};
    // 遍历每一个 （KV头，Group）组合
    for i in 0..n_kv_h{
        for j in 0..n_groups{
            // 对 attn 进行切片，切片形状为 seq_len X total_seq_len
            let attn_start = (i * n_groups + j) * seq_len * total_seq_len;
            let attn_slice = &att_scores.slice(attn_start, &vec![seq_len, total_seq_len]);
            // reverse v
            let mut v_rev = vec![T::from(0.).unwrap(); dqkv * total_seq_len];
            for m in 0..dqkv{
                for n in 0..total_seq_len{
                    v_rev[m * total_seq_len + n] = v_data[n * dqkv * n_kv_h + i * dqkv + m];
                }
            }
            let v_rev_tensor = Tensor::new(v_rev, &vec![dqkv, total_seq_len]);
            // matmul_transb result
            let mut mat_result = Tensor::default(&vec![seq_len, dqkv]);
            OP::matmul_transb(&mut mat_result, T::from(0.0).unwrap(), &attn_slice, &v_rev_tensor, T::from(1.).unwrap());
            // hidden_state
            let mat_data = mat_result.data();
            for row in 0..seq_len{
                for col in 0..dqkv{
                    hidden[hidden_len * row + (i * n_groups + j) * dqkv + col] = mat_data[row * dqkv + col];
                }
            }
        }
    }
}


/// mlp Multi-Layer Perceptron 多层感知机 或者称为 FFN SwiGLU Feed-Forward Network
/// 是llama机构中的前馈神经网络模块
/// 是模型记住事实、词汇含义的主要场所
fn mlp<T>
(
    residual: &mut Tensor<T>,
    hidden_states: &mut Tensor<T>,
    gate: &mut Tensor<T>,
    up: &mut Tensor<T>,
    w_up: &Tensor<T>,
    w_down: &Tensor<T>,
    w_gate: &Tensor<T>,
    rms_w: &Tensor<T>,
    eps: impl Float,
) 
    where T: Float + Default + std::iter::Sum + FromPrimitive + std::ops::MulAssign
        + std::marker::Sync + std::marker::Send + 'static
{
    // 对 residual 进行归一化操作并存储在 hidden_states中

    OP::rms_norm(hidden_states,residual,rms_w,eps);
    // 计算 hidden_states 在 gate 上的投影
    // 开始对 mlp 中的前两个 matual_transb 操作进行计时
    let start = Instant::now();
    OP::matmul_transb(gate,T::from(0.).unwrap(),hidden_states,w_gate,T::from(1.).unwrap());
    // 计算 hidden_states 在 up 上的投影
    OP::matmul_transb(up,T::from(0.).unwrap(),hidden_states,w_up,T::from(1.).unwrap());
    // 结束对 mlp 中的前两个 matual_transb 操作进行计时
    let duration = start.elapsed();
    println!("mlp 中的前两个 matual_transb 操作花了{:?}时间",duration);
    // 对up,gate进行 silu计算
    OP::silu(up,gate);
    // residual = residual + up @ down^T w_down 为输出投影
    OP::matmul_transb(residual,T::from(1.).unwrap(),up,w_down,T::from(1.).unwrap());
}



mod test{
    use crate::model::*;
    #[test]
    pub fn test_mlp() {
        let seq_len = 4;
        let d = 2;
        let di = 3;
        let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
        let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
        let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
        let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
        let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
        let eps = 1e-6;
        mlp(
            &mut residual,
            &mut hidden_states,
            &mut gate_buf,
            &mut up_buf,
            &w_up,
            &w_down,
            &w_gate,
            &rms_w,
            eps,
        );

        assert!(residual.close_to(
            &Tensor::<f32>::new(
                vec![
                    1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                    1.7290739
                ],
                &vec![seq_len, d]
            ),
            1e-3
        ))
    }

    #[test]
    pub fn test_load_safetensors() {
        use std::path::PathBuf;
        use crate::tensor::float_eq;
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("story");
        let model = Llama::<f32>::from_safetensors(model_dir);
        assert_eq!(model.vocab, 2048);
        assert_eq!(model.n_layers, 2);
        assert_eq!(model.n_q_h, 8);
        assert_eq!(model.n_kv_h, 4);
        assert_eq!(model.d, 128);
        assert_eq!(model.dqkv, 16);
        assert_eq!(model.di, 384);

        assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
        assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
        assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
        assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
        assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
        assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
        assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
        assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
        assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
        assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
        assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
        assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

    }

    #[test]
    pub fn test_self_attention() {
    use std::path::PathBuf;
    use tokenizers::Tokenizer;

        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("story");
        let llama = Llama::<f32>::from_safetensors(&model_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let input = "Once upon a time";
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        print!("\n{}", input);
        let mut cache = llama.new_cache();
        let input_tensor = Tensor::new(input_ids.to_vec(), &vec![1,input_ids.len()]);
        llama.forward(&input_tensor, &mut cache).print();
    }
}
