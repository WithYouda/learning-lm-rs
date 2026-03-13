use std::{usize, vec};

use crate::core::tensor::Tensor;

pub struct DecodeScratch<T> {
    residual: Tensor<T>,
    hidden_states: Tensor<T>,
    q_buf: Tensor<T>,
    att_scores: Tensor<T>,
    gate_buf: Tensor<T>,
    up_buf: Tensor<T>,
    d: usize,
    q_dim: usize,
    attn_rows: usize,
    di: usize,
    max_seq_len: usize,
}

impl<T: Default + Copy> DecodeScratch<T> {
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

    fn matches(&self, max_seq_len: usize, d: usize, q_dim: usize, attn_rows: usize, di: usize) -> bool {
        self.max_seq_len == max_seq_len
            && self.d == d
            && self.q_dim == q_dim
            && self.attn_rows == attn_rows
            && self.di == di
    }

    pub fn residual(&self) -> Tensor<T> {
        self.residual.slice(0, &vec![1, self.d])
    }

    pub fn hidden_states(&self) -> Tensor<T> {
        self.hidden_states.slice(0, &vec![1, self.d])
    }

    pub fn q_buf(&self) -> Tensor<T> {
        self.q_buf.slice(0, &vec![1, self.q_dim])
    }

    pub fn att_scores(&self, total_seq_len: usize) -> Tensor<T> {
        self.att_scores.slice(0, &vec![self.attn_rows, total_seq_len])
    }

    pub fn gate_buf(&self) -> Tensor<T> {
        self.gate_buf.slice(0, &vec![1, self.di])
    }

    pub fn up_buf(&self) -> Tensor<T> {
        self.up_buf.slice(0, &vec![1, self.di])
    }
}

pub struct KVCache<T> {
    // (max_seq_len, n_kv_head * dqkv) x layers
    k_cache: Vec<Tensor<T>>, 
    // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, 
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    // length of the current sequence
    length: usize, 
    decode_scratch: Option<DecodeScratch<T>>,
}

impl<T: Default + Copy> KVCache<T> {
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

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn increment(&mut self, seq_len: usize) {
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }

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

    pub fn decode_scratch(&mut self) -> Option<&mut DecodeScratch<T>> {
        self.decode_scratch.as_mut()
    }
}
