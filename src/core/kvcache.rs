use std::{usize, vec};

use crate::core::tensor::Tensor;

/// Decode 阶段工作缓冲区，复用整套中间张量以避免每步重新分配内存。
/// 包含残差、隐藏状态、Q投影、注意力分数、gate/up 缓冲等所有 decode 所需的中间张量。
pub struct DecodeScratch<T> {
    /// 残差张量 (1, d)
    residual: Tensor<T>,
    /// 隐藏状态张量 (1, d)
    hidden_states: Tensor<T>,
    /// Q 投影缓冲 (1, n_q_h * dqkv)
    q_buf: Tensor<T>,
    /// 注意力分数张量 (attn_rows, max_seq_len)
    att_scores: Tensor<T>,
    /// FFN gate 层缓冲 (1, di)
    gate_buf: Tensor<T>,
    /// FFN up 层缓冲 (1, di)
    up_buf: Tensor<T>,
    d: usize,
    q_dim: usize,
    attn_rows: usize,
    di: usize,
    max_seq_len: usize,
}

impl<T: Default + Copy> DecodeScratch<T> {
    /// 创建新的 decode 缓冲区，按模型参数分配所有中间张量
    fn new(max_seq_len: usize, d: usize, q_dim: usize, attn_rows: usize, di: usize) -> Self {
        Self {
            residual: Tensor::default(&vec![1, d]),
            hidden_states: Tensor::default(&vec![1, d]),
            q_buf: Tensor::default(&vec![1, q_dim]),
            att_scores: Tensor::default(&vec![attn_rows, max_seq_len]),
            gate_buf: Tensor::default(&vec![1, di]),
            up_buf: Tensor::default(&vec![1, di]),
            d,
            q_dim,
            attn_rows,
            di,
            max_seq_len,
        }
    }

    /// 检查当前缓冲区参数是否与模型配置匹配，避免不必要的重建
    fn matches(&self, max_seq_len: usize, d: usize, q_dim: usize, attn_rows: usize, di: usize) -> bool {
        self.max_seq_len == max_seq_len
            && self.d == d
            && self.q_dim == q_dim
            && self.attn_rows == attn_rows
            && self.di == di
    }

    /// 获取残差张量的切片 (1, d)
    pub fn residual(&self) -> Tensor<T> {
        self.residual.slice(0, &vec![1, self.d])
    }

    /// 获取隐藏状态张量的切片 (1, d)
    pub fn hidden_states(&self) -> Tensor<T> {
        self.hidden_states.slice(0, &vec![1, self.d])
    }

    /// 获取 Q 投影缓冲的切片 (1, q_dim)
    pub fn q_buf(&self) -> Tensor<T> {
        self.q_buf.slice(0, &vec![1, self.q_dim])
    }

    /// 获取注意力分数张量的切片，宽度截取到 total_seq_len
    pub fn att_scores(&self, total_seq_len: usize) -> Tensor<T> {
        self.att_scores.slice(0, &vec![self.attn_rows, total_seq_len])
    }

    /// 获取 gate 层缓冲的切片 (1, di)
    pub fn gate_buf(&self) -> Tensor<T> {
        self.gate_buf.slice(0, &vec![1, self.di])
    }

    /// 获取 up 层缓冲的切片 (1, di)
    pub fn up_buf(&self) -> Tensor<T> {
        self.up_buf.slice(0, &vec![1, self.di])
    }
}

/// KV 缓存，保存所有层的 Key 和 Value 历史状态。
/// 多轮对话中复用已计算的 K/V，避免重复推理。
pub struct KVCache<T> {
    /// 每一层的 Key 缓存 (max_seq_len, n_kv_h * dqkv)
    k_cache: Vec<Tensor<T>>, 
    /// 每一层的 Value 缓存 (max_seq_len, n_kv_h * dqkv)
    v_cache: Vec<Tensor<T>>, 
    /// 序列的最大长度限制
    #[allow(unused)]
    max_seq_len: usize,
    /// 每个位置的 KV 维度 (n_kv_h * dqkv)
    dim: usize,
    /// 当前已缓存的序列长度
    length: usize, 
    /// Decode 阶段的复用缓冲区（懒初始化）
    decode_scratch: Option<DecodeScratch<T>>,
}

impl<T: Default + Copy> KVCache<T> {
    /// 创建新的 KV 缓存，为每一层预分配 max_seq_len x dim 的缓冲区
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len: max_seq_len,
            dim: dim,
            length: init_len,
            decode_scratch: None,
        }
    }

    /// 获取指定层的 Key 缓存切片，从 start 开始到当前 length
    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    /// 获取指定层的 Value 缓存切片，从 start 开始到当前 length
    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    /// 向前推进序列长度（每次 forward 后调用）
    pub fn increment(&mut self, seq_len: usize) {
        self.length += seq_len;
    }

    /// 返回当前已缓存的序列长度
    pub fn len(&self) -> usize {
        self.length
    }

    /// 确保 decode 缓冲区存在且参数匹配，不匹配时自动重建
    pub fn ensure_decode_scratch(
        &mut self,
        d: usize,
        q_dim: usize,
        attn_rows: usize,
        di: usize,
    ) {
        let need_rebuild = self
            .decode_scratch
            .as_ref()
            .map(|scratch| !scratch.matches(self.max_seq_len, d, q_dim, attn_rows, di))
            .unwrap_or(true);
        if need_rebuild {
            self.decode_scratch = Some(DecodeScratch::new(self.max_seq_len, d, q_dim, attn_rows, di));
        }
    }

    /// 获取 decode 缓冲区的可变引用
    pub fn decode_scratch(&mut self) -> Option<&mut DecodeScratch<T>> {
        self.decode_scratch.as_mut()
    }
}
