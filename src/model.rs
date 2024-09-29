use std::fs::File;
use std::vec;
use safetensors::SafeTensors;
use std::path::Path;
use num_traits::float::Float;
use num_traits::Num;
use num_traits::FromPrimitive;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;


#[allow(unused)]
pub struct Llama<T: Num> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

macro_rules! impl_from_safetensors_for_Llama {
    ($Param:ty) => {
        impl Llama<$Param> {
            pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
                let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
                let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
                let model_file =
                    std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
                let safetensor = SafeTensors::deserialize(&model_file).unwrap();

                assert!(config.num_attention_heads % config.num_key_value_heads == 0);
                Self {
                    vocab: config.vocab_size,
                    n_layers: config.num_hidden_layers,
                    n_q_h: config.num_attention_heads,
                    n_kv_h: config.num_key_value_heads,
                    d: config.hidden_size,
                    dqkv: config.hidden_size / config.num_attention_heads,
                    di: config.intermediate_size,
                    eps: config.rms_norm_eps,
                    rope_theta: config.rope_theta,
                    max_seq_len: config.max_position_embeddings,
                    params: LLamaParams::<$Param>::from_safetensors(&safetensor, &config),
                    bos_token_id: config.bos_token_id,
                    eos_token_id: config.eos_token_id,
                }
            }
        }
    };
}

impl_from_safetensors_for_Llama!(f32);
impl_from_safetensors_for_Llama!(half::f16);

impl <T: Float + Default + std::iter::Sum + num_traits::FromPrimitive + std::fmt::Debug + std::ops::MulAssign>Llama<T> {
    pub fn new_cache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual: Tensor<T> = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut hidden_states: Tensor<T> = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut q_buf: Tensor<T> = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores: Tensor<T> =
            Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf: Tensor<T> = Tensor::<T>::default(&vec![seq_len, self.di]);
        let mut up_buf: Tensor<T> = Tensor::<T>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_q_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, T::from(0.).unwrap(), &hidden_states, &self.params.wq[layer], T::from(1.).unwrap());
            OP::matmul_transb(k, T::from(0.).unwrap(), &hidden_states, &self.params.wk[layer], T::from(1.).unwrap());
            OP::matmul_transb(v, T::from(0.).unwrap(), &hidden_states, &self.params.wv[layer], T::from(1.).unwrap());
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

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

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

            // todo!("down_proj matmul and add residual");
            // residual = hidden_states @ wo.T + residual
            OP::matmul_transb(&mut residual, T::from(1.).unwrap(), &hidden_states, &self.params.wo[layer], T::from(1.).unwrap());
            //todo!("mlp(...)");
            hidden_states = Tensor::<T>::default(&vec![seq_len, self.d]);
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
            //
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<T>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, T::from(0.).unwrap(), &hidden_states, &self.params.lm_head, T::from(1.).unwrap());

        logits
    }

    #[allow(unused)]
    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result: Vec<u32> = Vec::<u32>::new();
        
        // todo!("实现文本生成");
        // step 1
        // initialize kvcache
        let mut cache: KVCache<T> = self.new_cache();
        // push token_ids into result
        for token in token_ids{
            result.push(*token);
        }
        let mut input_tensor: Tensor<u32> = Tensor::new_usize(token_ids.to_vec(), &vec![1,token_ids.len()]);
        
        // random_sample and push into result
        for i in 0..max_len{
            let forward_tensor: Tensor<T> = self.forward(&input_tensor, &mut cache);
            //if i == 0 {forward_tensor.print();}
            let id = OP::random_sample(&forward_tensor, top_p, top_k, temperature);
            if id == self.eos_token_id {
                break;
            }
            result.push(id);
            input_tensor = Tensor::new_usize(vec![id],&vec![1, 1]);
        }
        result
    }

    // TODO: chat function
    pub fn chat(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        mut cache: KVCache<T>,
    ) -> (Vec<u32>, KVCache<T>){
        let length = token_ids.len();
        let mut result = Vec::<u32>::new();
        let token: Vec<u32> = Vec::from(token_ids);
        for token in token_ids{
            result.push(*token);
        }
        let mut input = Tensor::<u32>::new_usize(token, &vec![1, token_ids.len()]);
        loop {
            let output = OP::random_sample(&self.forward(&input, &mut cache), top_p, top_k, temperature);
            result.push(output);
            if result.len() >= max_len || output == self.eos_token_id {
                break;
            }
            input = Tensor::<u32>::new_usize(Vec::from([output]), &vec![1, 1]);
        }
        result = result[length..].to_vec();
        (result, cache)
    }

}




fn self_attention<T>(
    hidden_states: &mut Tensor<T>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<T>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<T>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
)
    where T: Float + Default + FromPrimitive + std::iter::Sum 
{
    // step 1 ,socre = Q @ K.T / sqrt(dim) 
    let sqrt_dim = T::from(dqkv).unwrap().sqrt();
    let scores = unsafe{att_scores.data_mut()};
    for i in 0..seq_len{
        for j in 0..total_seq_len{
            for m in 0..n_kv_h{
                for n in 0..n_groups{
                    let q_start = (m * n_groups + n) * dqkv + i * n_groups * n_kv_h * dqkv;
                    let q_= q.slice(q_start, &vec![dqkv, 1]);
                    let k_start = m * dqkv + j *  n_kv_h * dqkv;
                    let k_: Tensor<T> = k.slice(k_start, &vec![dqkv, 1]);
                    let value = OP::dot(&q_, &k_) / sqrt_dim;
                    scores[m * n_groups * seq_len * total_seq_len 
                        + n * seq_len * total_seq_len 
                        + i * total_seq_len
                        + j] 
                        = value;
                }
            }
        }
    }
    // step 2, attn = softmax(score)
    OP::masked_softmax(att_scores);
    // step 3, x = attn @ V
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
    for i in 0..n_kv_h{
        for j in 0..n_groups{
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
            OP::matmul_transb(&mut mat_result, T::from(0.).unwrap(), &attn_slice, &v_rev_tensor, T::from(1.).unwrap());
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

fn mlp<T>(
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
{
    OP::rms_norm(hidden_states,residual,rms_w,eps);
    OP::matmul_transb(gate,T::from(0.).unwrap(),hidden_states,w_gate,T::from(1.).unwrap());
    OP::matmul_transb(up,T::from(0.).unwrap(),hidden_states,w_up,T::from(1.).unwrap());
    OP::silu(up,gate);
    OP::matmul_transb(residual,T::from(0.).unwrap(),up,w_down,T::from(1.).unwrap());
}

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
    let input_tensor = Tensor::new_usize(input_ids.to_vec(), &vec![1,input_ids.len()]);
    llama.forward(&input_tensor, &mut cache).print();
}
