use crate::model::config::{LlamaConfigJson,RopeScaling};
use crate::core::kvcache::KVCache;
use crate::core::operators::operator as OP;
use crate::model::params::{LLamaParams, Weight};
use crate::core::tensor::Tensor;
use crate::core::sampling as sampling;
use crate::runtime::backend::{prefill_matmul_backend, PrefillMatmulBackend};
use crate::runtime::cpu;

use std::fmt::Debug;
use std::fs::File;
use std::vec;
use std::path::Path;
use std::any::TypeId;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use gemm::Parallelism;
use rayon::prelude::*;
use safetensors::SafeTensors;
use num_traits::float::Float;
use num_traits::Num;
use num_traits::FromPrimitive;
use gguf::{GGUFFile, GGUFMetadataValue};

#[derive(Default)]
struct DecodeLayerTiming {
    calls: usize,
    attn_sums: Vec<f64>,
    mlp_sums: Vec<f64>,
}

#[derive(Default, Clone, Copy)]
struct MlpTiming {
    gate_up_s: f64,
    down_s: f64,
}

#[derive(Default)]
struct PrefillLayerTiming {
    calls: usize,
    qkv_proj_sums: Vec<f64>,
    attn_core_sums: Vec<f64>,
    attn_out_sums: Vec<f64>,
    mlp_gate_up_sums: Vec<f64>,
    mlp_down_sums: Vec<f64>,
}

static DECODE_LAYER_TIMING: OnceLock<Mutex<DecodeLayerTiming>> = OnceLock::new();
static LAYER_TIMING_ENABLED: OnceLock<bool> = OnceLock::new();
static DECODE_PACKED_LAYER_FILTER: OnceLock<Option<Vec<usize>>> = OnceLock::new();
static PREFILL_LAYER_TIMING: OnceLock<Mutex<PrefillLayerTiming>> = OnceLock::new();
static PREFILL_LAYER_TIMING_ENABLED: OnceLock<bool> = OnceLock::new();

#[inline]
fn layer_timing_enabled() -> bool {
    *LAYER_TIMING_ENABLED.get_or_init(|| {
        std::env::var("LMRS_LAYER_TIMING")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[inline]
fn prefill_layer_timing_enabled() -> bool {
    *PREFILL_LAYER_TIMING_ENABLED.get_or_init(|| {
        std::env::var("LMRS_PREFILL_LAYER_TIMING")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[inline]
fn decode_packed_layer_filter() -> &'static Option<Vec<usize>> {
    DECODE_PACKED_LAYER_FILTER.get_or_init(|| {
        std::env::var("LMRS_DECODE_PACKED_LAYERS")
            .ok()
            .and_then(|v| {
                let mut out = Vec::new();
                for part in v.split(',') {
                    let s = part.trim();
                    if s.is_empty() {
                        continue;
                    }
                    if let Ok(id) = s.parse::<usize>() {
                        out.push(id);
                    }
                }
                if out.is_empty() {
                    None
                } else {
                    Some(out)
                }
            })
    })
}

#[inline]
fn decode_use_packed_kv(layer: usize) -> bool {
    match decode_packed_layer_filter() {
        None => false,
        Some(v) => v.contains(&layer),
    }
}

fn record_decode_layer_timing(layer: usize, n_layers: usize, attn_s: f64, mlp_s: f64) {
    let state = DECODE_LAYER_TIMING.get_or_init(|| Mutex::new(DecodeLayerTiming::default()));
    let mut g = state.lock().unwrap();
    if g.attn_sums.len() != n_layers {
        g.attn_sums = vec![0.0; n_layers];
        g.mlp_sums = vec![0.0; n_layers];
        g.calls = 0;
    }
    g.attn_sums[layer] += attn_s;
    g.mlp_sums[layer] += mlp_s;
    if layer + 1 == n_layers {
        g.calls += 1;
        if g.calls % 32 == 0 {
            let mut slow_layer = 0usize;
            let mut slow_total = 0.0f64;
            for i in 0..n_layers {
                let avg = (g.attn_sums[i] + g.mlp_sums[i]) / g.calls as f64;
                if avg > slow_total {
                    slow_total = avg;
                    slow_layer = i;
                }
            }
            println!("[decode-layer-timing] calls: {}", g.calls);
            println!(
                "[decode-layer-timing] slowest_layer: {} avg_total_ms: {:.3}",
                slow_layer,
                slow_total * 1000.0
            );
            for i in 0..n_layers {
                let avg_attn = g.attn_sums[i] / g.calls as f64 * 1000.0;
                let avg_mlp = g.mlp_sums[i] / g.calls as f64 * 1000.0;
                println!(
                    "[decode-layer-timing] layer: {} avg_attn_ms: {:.3} avg_mlp_ms: {:.3}",
                    i, avg_attn, avg_mlp
                );
            }
        }
    }
}

fn record_prefill_layer_timing(
    layer: usize,
    n_layers: usize,
    qkv_proj_s: f64,
    attn_core_s: f64,
    attn_out_s: f64,
    mlp_gate_up_s: f64,
    mlp_down_s: f64,
) {
    let state = PREFILL_LAYER_TIMING.get_or_init(|| Mutex::new(PrefillLayerTiming::default()));
    let mut g = state.lock().unwrap();
    if g.qkv_proj_sums.len() != n_layers {
        g.qkv_proj_sums = vec![0.0; n_layers];
        g.attn_core_sums = vec![0.0; n_layers];
        g.attn_out_sums = vec![0.0; n_layers];
        g.mlp_gate_up_sums = vec![0.0; n_layers];
        g.mlp_down_sums = vec![0.0; n_layers];
        g.calls = 0;
    }
    g.qkv_proj_sums[layer] += qkv_proj_s;
    g.attn_core_sums[layer] += attn_core_s;
    g.attn_out_sums[layer] += attn_out_s;
    g.mlp_gate_up_sums[layer] += mlp_gate_up_s;
    g.mlp_down_sums[layer] += mlp_down_s;

    if layer + 1 == n_layers {
        g.calls += 1;
        println!("[prefill-layer-timing] calls: {}", g.calls);
        let mut slow_layer = 0usize;
        let mut slow_total = 0.0f64;
        for i in 0..n_layers {
            let avg = (g.qkv_proj_sums[i]
                + g.attn_core_sums[i]
                + g.attn_out_sums[i]
                + g.mlp_gate_up_sums[i]
                + g.mlp_down_sums[i])
                / g.calls as f64;
            if avg > slow_total {
                slow_total = avg;
                slow_layer = i;
            }
        }
        println!(
            "[prefill-layer-timing] slowest_layer: {} avg_total_ms: {:.3}",
            slow_layer,
            slow_total * 1000.0
        );
        for i in 0..n_layers {
            println!(
                "[prefill-layer-timing] layer: {} avg_qkv_proj_ms: {:.3} avg_attn_core_ms: {:.3} avg_attn_out_ms: {:.3} avg_mlp_gate_up_ms: {:.3} avg_mlp_down_ms: {:.3}",
                i,
                g.qkv_proj_sums[i] / g.calls as f64 * 1000.0,
                g.attn_core_sums[i] / g.calls as f64 * 1000.0,
                g.attn_out_sums[i] / g.calls as f64 * 1000.0,
                g.mlp_gate_up_sums[i] / g.calls as f64 * 1000.0,
                g.mlp_down_sums[i] / g.calls as f64 * 1000.0,
            );
        }
    }
}


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
    pub dqkv: usize,      
    // dimension of intermediate states —— MLP中间层的维度      
    di: usize, 
    // epsilon for RMS normalization —— RMSNorm 中的 epsilon 数值             
    eps: f32,    
    // rope theta for rope initialization —— rope 对应的 角度值
    rope_theta: f32,
    // rope scaling struct —— RoPE_scaling 的结构体           
    rope_scaling: RopeScaling,     
    // maximum sequence length —— 模型支持的最大序列长度  
    max_seq_len: usize,   
    // trained weights of this model —— 模型的所有可训练参数  
    params: LLamaParams<T>, 
    // start token id —— 起始 token id
    #[allow(dead_code)]
    bos_token_id: u32,  
    // end token id —— 结束 token id    
    eos_token_id: Vec<u32>,      
}

#[inline]
fn as_f32_slice<T: 'static>(x: &[T]) -> Option<&[f32]> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: 仅在 T 为 f32 时进行类型重解释。
        Some(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) })
    } else {
        None
    }
}

#[inline]
fn as_f32_slice_mut<T: 'static>(x: &mut [T]) -> Option<&mut [f32]> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: 仅在 T 为 f32 时进行类型重解释。
        Some(unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f32, x.len()) })
    } else {
        None
    }
}

#[inline]
fn softmax_inplace_f32(x: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_v {
            max_v = v;
        }
    }

    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    let inv = 1.0f32 / sum.max(1e-12);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn masked_softmax_f32(scores: &mut [f32], groups: usize, seq_len: usize, total_seq_len: usize) {
    let stride = seq_len * total_seq_len;
    scores
        .par_chunks_mut(stride)
        .take(groups)
        .for_each(|chunk| {
            for i in 0..seq_len {
                let row = &mut chunk[i * total_seq_len..(i + 1) * total_seq_len];
                let boundary = total_seq_len - seq_len + i + 1;
                softmax_inplace_f32(&mut row[..boundary]);
                for v in &mut row[boundary..] {
                    *v = 0.0;
                }
            }
        });
}

#[inline]
fn copy_f32(dst: &mut [f32], src: &[f32]) {
    dst.copy_from_slice(src);
}

#[inline]
fn scale_f32(x: &mut [f32], s: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: 运行时已检测 avx2。
            unsafe { return scale_f32_avx2(x, s) };
        }
    }
    let mut i = 0usize;
    while i < x.len() {
        x[i] *= s;
        i += 1;
    }
}

#[inline]
fn axpy_f32(dst: &mut [f32], src: &[f32], a: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: 运行时已检测 avx2。
            unsafe { return axpy_f32_avx2(dst, src, a) };
        }
    }
    let mut i = 0usize;
    while i < dst.len() {
        dst[i] += a * src[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_f32_avx2(x: &mut [f32], s: f32) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    let n8 = x.len() / 8 * 8;
    let vs = _mm256_set1_ps(s);
    while i < n8 {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let vy = _mm256_mul_ps(vx, vs);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), vy);
        i += 8;
    }
    while i < x.len() {
        x[i] *= s;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn axpy_f32_avx2(dst: &mut [f32], src: &[f32], a: f32) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    let n8 = dst.len() / 8 * 8;
    let va = _mm256_set1_ps(a);
    while i < n8 {
        let vd = _mm256_loadu_ps(dst.as_ptr().add(i));
        let vs = _mm256_loadu_ps(src.as_ptr().add(i));
        let out = _mm256_add_ps(vd, _mm256_mul_ps(va, vs));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), out);
        i += 8;
    }
    while i < dst.len() {
        dst[i] += a * src[i];
        i += 1;
    }
}

#[inline]
fn pack_q_group(
    q_data: &[f32],
    seq_len: usize,
    q_stride: usize,
    group_base: usize,
    dqkv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * dqkv];
    for i in 0..seq_len {
        let src = i * q_stride + group_base;
        let dst = i * dqkv;
        out[dst..dst + dqkv].copy_from_slice(&q_data[src..src + dqkv]);
    }
    out
}

#[inline]
fn pack_k_head(k_data: &[f32], total_seq_len: usize, k_stride: usize, head_base: usize, dqkv: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; total_seq_len * dqkv];
    for t in 0..total_seq_len {
        let src = t * k_stride + head_base;
        let dst = t * dqkv;
        out[dst..dst + dqkv].copy_from_slice(&k_data[src..src + dqkv]);
    }
    out
}

#[inline]
fn qk_matmul_tiled(
    scores: &mut [f32],
    q_group: &[f32],
    k_head: &[f32],
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
    inv_sqrt: f32,
) {
    let t_tile = 64usize;
    let d_tile = 64usize;
    scores.fill(0.0);

    for t0 in (0..total_seq_len).step_by(t_tile) {
        let t1 = (t0 + t_tile).min(total_seq_len);
        for d0 in (0..dqkv).step_by(d_tile) {
            let d1 = (d0 + d_tile).min(dqkv);
            for i in 0..seq_len {
                let q_row = &q_group[i * dqkv + d0..i * dqkv + d1];
                let score_row = &mut scores[i * total_seq_len..(i + 1) * total_seq_len];
                for t in t0..t1 {
                    let k_row = &k_head[t * dqkv + d0..t * dqkv + d1];
                    score_row[t] += dot_f32_simd(q_row, k_row);
                }
            }
        }
    }

    for v in scores.iter_mut() {
        *v *= inv_sqrt;
    }
}

#[inline]
fn qk_matmul_gemm(scores: &mut [f32], q_group: &[f32], k_head: &[f32], seq_len: usize, total_seq_len: usize, dqkv: usize, inv_sqrt: f32) {
    // 直接在 packed 切片上做 GEMM，避免每个 group 反复构造 Tensor 与复制数据。
    unsafe {
        gemm::gemm(
            seq_len,
            total_seq_len,
            dqkv,
            scores.as_mut_ptr(),
            1,
            total_seq_len as isize,
            true,
            q_group.as_ptr(),
            1,
            dqkv as isize,
            k_head.as_ptr(),
            dqkv as isize,
            1,
            0.0,
            inv_sqrt,
            false,
            false,
            false,
            // 外层已经按 group 并行，这里再开 Rayon 会形成嵌套并行，反而拖慢。
            Parallelism::None,
        );
    }
}

#[inline]
fn av_matmul_tiled(
    out: &mut [f32],
    attn: &[f32],
    v_head: &[f32],
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let t_tile = 64usize;
    let d_tile = 64usize;
    out.fill(0.0);

    for t0 in (0..total_seq_len).step_by(t_tile) {
        let t1 = (t0 + t_tile).min(total_seq_len);
        for d0 in (0..dqkv).step_by(d_tile) {
            let d1 = (d0 + d_tile).min(dqkv);
            for i in 0..seq_len {
                let a_row = &attn[i * total_seq_len..(i + 1) * total_seq_len];
                let out_row = &mut out[i * dqkv..(i + 1) * dqkv];
                for t in t0..t1 {
                    let a = a_row[t];
                    let v_row = &v_head[t * dqkv + d0..t * dqkv + d1];
                    for (idx, vv) in v_row.iter().enumerate() {
                        out_row[d0 + idx] += a * *vv;
                    }
                }
            }
        }
    }
}

#[inline]
fn av_matmul_gemm(
    out: &mut [f32],
    attn: &[f32],
    v_head: &[f32],
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // attn: [seq_len, total_seq_len], v_head: [total_seq_len, dqkv]
    // out = attn * v_head
    unsafe {
        gemm::gemm(
            seq_len,
            dqkv,
            total_seq_len,
            out.as_mut_ptr(),
            1,
            dqkv as isize,
            true,
            attn.as_ptr(),
            1,
            total_seq_len as isize,
            v_head.as_ptr(),
            1,
            dqkv as isize,
            0.0,
            1.0,
            false,
            false,
            false,
            // 外层 group 并行已覆盖线程利用，内层 GEMM 走单线程避免过度调度。
            Parallelism::None,
        );
    }
}

#[inline]
fn fused_decode_attn_online(
    out: &mut [f32],
    q_slice: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    total_seq_len: usize,
    k_stride: usize,
    head_base: usize,
    inv_sqrt: f32,
) {
    out.fill(0.0);
    let mut max_s = f32::NEG_INFINITY;
    let mut denom = 0.0f32;
    let dqkv = out.len();

    for t in 0..total_seq_len {
        let base = t * k_stride + head_base;
        let s = dot_f32_simd(q_slice, &k_data[base..base + dqkv]) * inv_sqrt;

        if denom == 0.0 {
            max_s = s;
            denom = 1.0;
            copy_f32(out, &v_data[base..base + dqkv]);
            continue;
        }

        if s > max_s {
            let scale = (max_s - s).exp();
            scale_f32(out, scale);
            denom *= scale;
            max_s = s;
        }

        let w = (s - max_s).exp();
        denom += w;

        axpy_f32(out, &v_data[base..base + dqkv], w);
    }

    let inv = 1.0f32 / denom.max(1e-12);
    scale_f32(out, inv);
}

fn self_attention_decode_f32<T>(
    hidden_states: &mut Tensor<T>,
    q: &Tensor<T>,
    k: &Tensor<T>,
    v: &Tensor<T>,
    n_kv_h: usize,
    n_groups: usize,
    total_seq_len: usize,
    dqkv: usize,
    layer: usize,
) -> bool
where
    T: Float + Default + Copy + 'static,
{
    let q_data = match as_f32_slice(q.data()) {
        Some(s) => s,
        None => return false,
    };
    let k_data = match as_f32_slice(k.data()) {
        Some(s) => s,
        None => return false,
    };
    let v_data = match as_f32_slice(v.data()) {
        Some(s) => s,
        None => return false,
    };
    let hidden_data = unsafe { hidden_states.data_mut() };
    let hidden = match as_f32_slice_mut(hidden_data) {
        Some(s) => s,
        None => return false,
    };

    let q_stride = n_kv_h * n_groups * dqkv;
    let k_stride = n_kv_h * dqkv;
    let hidden_len = n_kv_h * n_groups * dqkv;
    let inv_sqrt = 1.0f32 / (dqkv as f32).sqrt();

    if decode_use_packed_kv(layer) {
        let k_heads: Vec<Vec<f32>> = (0..n_kv_h)
            .map(|m| pack_k_head(k_data, total_seq_len, k_stride, m * dqkv, dqkv))
            .collect();
        let v_heads: Vec<Vec<f32>> = (0..n_kv_h)
            .map(|m| pack_k_head(v_data, total_seq_len, k_stride, m * dqkv, dqkv))
            .collect();

        hidden
            .par_chunks_mut(dqkv)
            .enumerate()
            .for_each(|(hg, out_chunk)| {
                let m = hg / n_groups;
                let n = hg % n_groups;
                let q_off = (m * n_groups + n) * dqkv;
                let q_slice = &q_data[q_off..q_off + dqkv];
                fused_decode_attn_online(
                    out_chunk,
                    q_slice,
                    &k_heads[m],
                    &v_heads[m],
                    total_seq_len,
                    dqkv,
                    0,
                    inv_sqrt,
                );
            });
    } else {
        hidden
            .par_chunks_mut(dqkv)
            .enumerate()
            .for_each(|(hg, out_chunk)| {
                let m = hg / n_groups;
                let n = hg % n_groups;
                let q_off = (m * n_groups + n) * dqkv;
                let q_slice = &q_data[q_off..q_off + dqkv];
                fused_decode_attn_online(
                    out_chunk,
                    q_slice,
                    k_data,
                    v_data,
                    total_seq_len,
                    k_stride,
                    m * dqkv,
                    inv_sqrt,
                );
            });
    }

    // seq_len == 1，hidden 只有一行，长度应为 hidden_len。
    debug_assert_eq!(hidden.len(), hidden_len);
    let _ = q_stride;
    true
}

fn self_attention_prefill_f32<T>(
    hidden_states: &mut Tensor<T>,
    att_scores: &mut Tensor<T>,
    q: &Tensor<T>,
    k: &Tensor<T>,
    v: &Tensor<T>,
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) -> bool
where
    T: Float + Default + Copy + 'static,
{
    let q_data = match as_f32_slice(q.data()) {
        Some(s) => s,
        None => return false,
    };
    let k_data = match as_f32_slice(k.data()) {
        Some(s) => s,
        None => return false,
    };
    let v_data = match as_f32_slice(v.data()) {
        Some(s) => s,
        None => return false,
    };

    let scores_data = unsafe { att_scores.data_mut() };
    let scores = match as_f32_slice_mut(scores_data) {
        Some(s) => s,
        None => return false,
    };

    let hidden_data = unsafe { hidden_states.data_mut() };
    let hidden = match as_f32_slice_mut(hidden_data) {
        Some(s) => s,
        None => return false,
    };

    let backend = prefill_matmul_backend();
    let q_stride = n_kv_h * n_groups * dqkv;
    let k_stride = n_kv_h * dqkv;
    let hidden_len = n_kv_h * n_groups * dqkv;
    let inv_sqrt = 1.0f32 / (dqkv as f32).sqrt();
    let groups = n_kv_h * n_groups;
    let score_stride = seq_len * total_seq_len;

    let k_heads: Vec<Vec<f32>> = (0..n_kv_h)
        .map(|m| pack_k_head(k_data, total_seq_len, k_stride, m * dqkv, dqkv))
        .collect();
    let v_heads: Vec<Vec<f32>> = (0..n_kv_h)
        .map(|m| pack_k_head(v_data, total_seq_len, k_stride, m * dqkv, dqkv))
        .collect();

    let q_groups: Vec<Vec<f32>> = (0..groups)
        .map(|hg| {
            let m = hg / n_groups;
            let n = hg % n_groups;
            let group_base = (m * n_groups + n) * dqkv;
            pack_q_group(q_data, seq_len, q_stride, group_base, dqkv)
        })
        .collect();

    scores
        .par_chunks_mut(score_stride)
        .enumerate()
        .for_each(|(hg, chunk)| {
            let m = hg / n_groups;
            let q_group = &q_groups[hg];
            let k_head = &k_heads[m];
            match backend {
                PrefillMatmulBackend::Gemm => {
                    qk_matmul_gemm(chunk, q_group, k_head, seq_len, total_seq_len, dqkv, inv_sqrt);
                }
                PrefillMatmulBackend::Tiled => {
                    qk_matmul_tiled(chunk, q_group, k_head, seq_len, total_seq_len, dqkv, inv_sqrt);
                }
            }
        });

    masked_softmax_f32(scores, groups, seq_len, total_seq_len);

    // prefill 阶段按 group 一次算完整个 seq_len x dqkv 输出，
    // 避免之前按 row 循环时重复触发 GEMM 和 V 转置。
    let mut group_outputs = vec![0.0f32; groups * seq_len * dqkv];
    group_outputs
        .par_chunks_mut(seq_len * dqkv)
        .enumerate()
        .for_each(|(hg, out)| {
            let m = hg / n_groups;
            let attn_chunk = &scores[hg * score_stride..(hg + 1) * score_stride];
            match backend {
                PrefillMatmulBackend::Gemm => {
                    let v_head = &v_heads[m];
                    av_matmul_gemm(
                        out,
                        attn_chunk,
                        v_head,
                        seq_len,
                        total_seq_len,
                        dqkv,
                    );
                }
                PrefillMatmulBackend::Tiled => {
                    let v_head = &v_heads[m];
                    av_matmul_tiled(out, attn_chunk, v_head, seq_len, total_seq_len, dqkv);
                }
            }
        });

    for row in 0..seq_len {
        let hidden_row = &mut hidden[row * hidden_len..(row + 1) * hidden_len];
        for hg in 0..groups {
            let out_base = hg * dqkv;
            let src_base = row * dqkv;
            let group_out = &group_outputs[hg * seq_len * dqkv..(hg + 1) * seq_len * dqkv];
            hidden_row[out_base..out_base + dqkv]
                .copy_from_slice(&group_out[src_base..src_base + dqkv]);
        }
    }

    true
}

#[inline]
fn dot_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: 运行时已检测 avx512f。
            return unsafe { dot_f32_avx512(a, b) };
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: 运行时已检测 avx2。
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: 仅在 aarch64 目标上编译并调用。
        return unsafe { dot_f32_neon(a, b) };
    }

    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut i = 0usize;
    let n = a.len().min(b.len());
    let n16 = n / 16 * 16;
    let mut acc = _mm512_setzero_ps();

    while i < n16 {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        acc = _mm512_add_ps(acc, _mm512_mul_ps(va, vb));
        i += 16;
    }

    let mut tmp = [0.0f32; 16];
    _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut s = tmp.iter().sum::<f32>();
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut i = 0usize;
    let n = a.len().min(b.len());
    let n8 = n / 8 * 8;
    let mut acc = _mm256_setzero_ps();

    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        i += 8;
    }

    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut s = tmp.iter().sum::<f32>();
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut i = 0usize;
    let n = a.len().min(b.len());
    let n4 = n / 4 * 4;
    let mut acc = vdupq_n_f32(0.0);

    while i < n4 {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc = vaddq_f32(acc, vmulq_f32(va, vb));
        i += 4;
    }

    let mut tmp = [0.0f32; 4];
    vst1q_f32(tmp.as_mut_ptr(), acc);
    let mut s = tmp.iter().sum::<f32>();
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}


/// 从 safesensor 生成对应的 Llama 结构体的宏
macro_rules! impl_from_safetensors_for_Llama{
    ($Param:ty) =>{
        impl Llama<$Param> {
            pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
                // 提前初始化运行时线程池与绑核策略，确保后续并行路径使用同一组配置。
                let _ = cpu::init_runtime_tuning();
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
                    rope_scaling: config.rope_scaling,
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



/// 从 gguf 生成对应的 Llama 结构体的宏
macro_rules! impl_from_gguf_for_Llama{
    ($Param:ty) =>{
        impl Llama<$Param> {
            pub fn from_gguf(model_path: impl AsRef<Path>) -> Self {
                // GGUF 加载前就初始化运行时，避免第一次 rayon 调用时错过线程配置。
                let _ = cpu::init_runtime_tuning();
                let path = model_path.as_ref();
                let bytes = std::fs::read(path).expect("open gguf failed");
                let gguf = GGUFFile::read(&bytes)
                    .expect("gguf parse failed")
                    .expect("gguf file incomplete");
                
                let find_meta = |key: &str| -> Option<&GGUFMetadataValue> {
                    gguf.header
                        .metadata
                        .iter()
                        .find(|m| m.key == key)
                        .map(|m| &m.value)
                };

                let meta_u64 = |key: &str| -> Option<u64> {
                    match find_meta(key)? {
                        GGUFMetadataValue::Uint8(v) => Some(*v as u64),
                        GGUFMetadataValue::Uint16(v) => Some(*v as u64),
                        GGUFMetadataValue::Uint32(v) => Some(*v as u64),
                        GGUFMetadataValue::Uint64(v) => Some(*v),
                        GGUFMetadataValue::Int8(v) if *v >= 0 => Some(*v as u64),
                        GGUFMetadataValue::Int16(v) if *v >= 0 => Some(*v as u64),
                        GGUFMetadataValue::Int32(v) if *v >= 0 => Some(*v as u64),
                        GGUFMetadataValue::Int64(v) if *v >= 0 => Some(*v as u64),
                        _ => None,
                    }
                };

                let meta_f32 = |key: &str| -> Option<f32> {
                    match find_meta(key)? {
                        GGUFMetadataValue::Float32(v) => Some(*v),
                        GGUFMetadataValue::Float64(v) => Some(*v as f32),
                        GGUFMetadataValue::Uint8(v) => Some(*v as f32),
                        GGUFMetadataValue::Uint16(v) => Some(*v as f32),
                        GGUFMetadataValue::Uint32(v) => Some(*v as f32),
                        GGUFMetadataValue::Uint64(v) => Some(*v as f32),
                        GGUFMetadataValue::Int8(v) => Some(*v as f32),
                        GGUFMetadataValue::Int16(v) => Some(*v as f32),
                        GGUFMetadataValue::Int32(v) => Some(*v as f32),
                        GGUFMetadataValue::Int64(v) => Some(*v as f32),
                        _ => None,
                    }
                };

                let meta_str = |key: &str| -> Option<String> {
                    match find_meta(key)? {
                        GGUFMetadataValue::String(v) => Some(v.clone()),
                        _ => None,
                    }
                };

                let meta_u32_array = |key: &str| -> Option<Vec<u32>> {
                    match find_meta(key)? {
                        GGUFMetadataValue::Array(arr) => Some(
                            arr.value
                                .iter()
                                .filter_map(|v| match v {
                                    GGUFMetadataValue::Uint8(x) => Some(*x as u32),
                                    GGUFMetadataValue::Uint16(x) => Some(*x as u32),
                                    GGUFMetadataValue::Uint32(x) => Some(*x),
                                    GGUFMetadataValue::Uint64(x) => Some(*x as u32),
                                    GGUFMetadataValue::Int8(x) if *x >= 0 => Some(*x as u32),
                                    GGUFMetadataValue::Int16(x) if *x >= 0 => Some(*x as u32),
                                    GGUFMetadataValue::Int32(x) if *x >= 0 => Some(*x as u32),
                                    GGUFMetadataValue::Int64(x) if *x >= 0 => Some(*x as u32),
                                    _ => None,
                                })
                                .collect(),
                        ),
                        _ => None,
                    }
                };

                let vocab = meta_u64("llama.vocab_size").unwrap_or(0) as usize;
                let n_layers = meta_u64("llama.block_count").unwrap_or(0) as usize;
                let n_q_h = meta_u64("llama.attention.head_count").unwrap_or(0) as usize;
                let n_kv_h = meta_u64("llama.attention.head_count_kv")
                    .or_else(|| meta_u64("llama.attention.head_count"))
                    .unwrap_or(0) as usize;
                let d = meta_u64("llama.embedding_length").unwrap_or(0) as usize;
                let di = meta_u64("llama.feed_forward_length").unwrap_or(0) as usize;
                let max_seq_len = meta_u64("llama.context_length").unwrap_or(2048) as usize;

                // llama metadata 错误处理
                if vocab == 0 || n_layers == 0 || n_q_h == 0 || n_kv_h == 0 || d == 0 || di == 0 {
                    panic!("missing required llama metadata in gguf");
                }

                // 当数据类型为 half::f16 时，将 noem_rms 中用到的 epsilon 变为 1e-4
                let mut eps = meta_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
                if std::any::TypeId::of::<$Param>() == std::any::TypeId::of::<half::f16>() {
                    eps = eps.max(1e-4);
                }

                let rope_theta = meta_f32("llama.rope.freq_base").unwrap_or(1e4);
                let rope_type = meta_str("llama.rope.scaling.type").unwrap_or_default();
                let rope_scaling = RopeScaling {
                    factor: meta_f32("llama.rope.scaling.factor").unwrap_or_default(),
                    high_freq_factor: meta_f32("llama.rope.scaling.high_freq_factor").unwrap_or_default(),
                    low_freq_factor: meta_f32("llama.rope.scaling.low_freq_factor").unwrap_or_default(),
                    original_max_position_embeddings: meta_u64("llama.rope.scaling.original_context_length").unwrap_or_default() as usize,
                    rope_type,
                    short_factor: None,
                    long_factor: None,
                };

                let bos_token_id = meta_u64("tokenizer.ggml.bos_token_id").unwrap_or(1) as u32;
                let eos_token_id = meta_u32_array("tokenizer.ggml.eos_token_id")
                    .filter(|v| !v.is_empty())
                    .unwrap_or_else(|| vec![meta_u64("tokenizer.ggml.eos_token_id").unwrap_or(2) as u32]);

                // Infer tying behavior for lm_head.
                // Prefer explicit metadata when present; otherwise, if output.weight exists,
                // treat model as untied and use output projection.
                let tie_word_embeddings = meta_u64("llama.tie_word_embeddings")
                    .or_else(|| meta_u64("llama.embedding_tying"))
                    .map(|v| v != 0)
                    .unwrap_or_else(|| !gguf.tensors.iter().any(|t| t.name == "output.weight"));

                let params = LLamaParams::<$Param>::from_gguf(
                    &gguf,
                    &bytes,
                    n_layers,
                    n_q_h,
                    n_kv_h,
                    tie_word_embeddings,
                );

                Self {
                    vocab,
                    n_layers,
                    n_q_h,
                    n_kv_h,
                    d,
                    dqkv: d / n_q_h,
                    di,
                    eps,
                    rope_theta,
                    rope_scaling,
                    max_seq_len,
                    params,
                    bos_token_id,
                    eos_token_id,
                }
            }
        }
    };
}

impl_from_gguf_for_Llama!(f32);
impl_from_gguf_for_Llama!(half::f16);
impl_from_gguf_for_Llama!(half::bf16);



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
    // 因为input是输入的 token_id序列，而 token_id 是正整数
    /// 向前传播方法
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        // 当前输入的长度
        let seq_len = input.size();
        // cache 的长度
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;
        let use_decode_scratch = seq_len == 1;

        if use_decode_scratch {
            // decode 每次只处理一个 token，复用整套工作缓冲区能明显降低小 batch 分配抖动。
            cache.ensure_decode_scratch(
                self.d,
                self.n_q_h * self.dqkv,
                self.n_kv_h * n_groups,
                self.di,
            );
        }

        // 为相应的层、中间结果、缓存初始化张量
        // prefill 继续按需分配；decode 则走 KV cache 内部的持久化 scratch。
        let (mut residual, mut hidden_states, mut q_buf, mut att_scores, mut gate_buf, mut up_buf) =
            if use_decode_scratch {
                let scratch = cache.decode_scratch().expect("decode scratch must exist");
                (
                    scratch.residual(),
                    scratch.hidden_states(),
                    scratch.q_buf(),
                    scratch.att_scores(total_seq_len),
                    scratch.gate_buf(),
                    scratch.up_buf(),
                )
            } else {
                (
                    Tensor::<T>::default(&vec![seq_len, self.d]),
                    Tensor::<T>::default(&vec![seq_len, self.d]),
                    Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]),
                    Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]),
                    Tensor::<T>::default(&vec![seq_len, self.di]),
                    Tensor::<T>::default(&vec![seq_len, self.di]),
                )
            };

        // Embedding lookup
        // 将根据输入的token_id查询到table(嵌入表)中对应的向量复制到residual(输入嵌入向量)中
        OP::gather(&mut residual, input, &self.params.embedding_table);
        // 依次对每一层进行操作
        for layer in 0..self.n_layers {
            let enable_prefill_timing = seq_len > 1 && prefill_layer_timing_enabled();
            // 为每一层进行归一化，防止梯度爆炸或者消失
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );
            // 为什么 reshape 原本的形状？
            // 显式再次声明形状，确保内存布局连续！
            // (seq, n_q_h * dqkv)
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); 
            // (seq, n_kv_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); 
            // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); 

            let qkv_stage_start = if enable_prefill_timing {
                Some(Instant::now())
            } else {
                None
            };

            // Q/K/V 三个投影共享同一份输入隐藏状态。
            // prefill 走 batch 调度时会尽量共用输入准备和 row-tile 节奏；decode 则自动回退到并行单发。
            OP::matmul_transb_weight_batch3(
                q,
                k,
                v,
                T::from(0.).unwrap(),
                &hidden_states,
                &self.params.wq[layer],
                &self.params.wk[layer],
                &self.params.wv[layer],
                T::from(1.).unwrap(),
            );
            let qkv_proj_elapsed = qkv_stage_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            
            // 对q,k 进行旋转位置编码
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
                &self.rope_scaling,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
                &self.rope_scaling,
            );

            // 生成完整的 k,v
            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            let attn_stage_start = if (seq_len == 1 && layer_timing_enabled()) || enable_prefill_timing {
                Some(Instant::now())
            } else {
                None
            };

            // 注意力计算
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
                self.dqkv,
                layer,
            );

            let attn_core_elapsed = attn_stage_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);

            let attn_out_stage_start = if enable_prefill_timing {
                Some(Instant::now())
            } else {
                None
            };

            // down_proj matmul and add residual;
            // residual = hidden_states @ wo.T + residual
            OP::matmul_transb_weight(&mut residual, T::from(1.).unwrap(), &hidden_states, &self.params.wo[layer], T::from(1.).unwrap());
            let attn_out_elapsed = attn_out_stage_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            let attn_elapsed = if seq_len == 1 {
                attn_core_elapsed + attn_out_elapsed
            } else {
                0.0
            };
            // 层内复用缓冲区，减少 prefill 多层中的重复分配。
            unsafe { hidden_states.data_mut().fill(T::zero()); }

            let mlp_stage_start = if seq_len == 1 && layer_timing_enabled() {
                Some(Instant::now())
            } else {
                None
            };
            
            // 进行 mlp 计算
            let mlp_timing = mlp(
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

            let mlp_elapsed = mlp_stage_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            if seq_len == 1 && layer_timing_enabled() {
                record_decode_layer_timing(layer, self.n_layers, attn_elapsed, mlp_elapsed);
            }
            if enable_prefill_timing {
                record_prefill_layer_timing(
                    layer,
                    self.n_layers,
                    qkv_proj_elapsed,
                    attn_core_elapsed,
                    attn_out_elapsed,
                    mlp_timing.gate_up_s,
                    mlp_timing.down_s,
                );
            }
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
        OP::matmul_transb_weight(&mut logits, T::from(0.).unwrap(), &hidden_states, &self.params.lm_head, T::from(1.).unwrap());

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
        let mut result = Vec::<u32>::with_capacity(token_ids.len() + max_len);
        // step 1
        // 初始化 kvcache
        let mut cache = self.new_cache();
        // 初始化 result
        result.extend_from_slice(token_ids);
        let prompt_shape = vec![1, token_ids.len()];
        let one_shape = vec![1, 1];
        // 构造输入张量
        let mut input_tensor = Tensor::new(token_ids.to_vec(), &prompt_shape);
        let mut decode_token_tensor = Tensor::new(vec![0u32], &one_shape);
        
        // 调用 random_sample 并将结果存入 result 中
        for _ in 0..max_len{
            let forward_tensor = self.forward(&input_tensor, &mut cache);
            // 随机采样
            let id = sampling::random_sample(&forward_tensor, &result, top_p, top_k, temperature, penalty);
            // 检查是否始终符
            if self.eos_token_id.contains(&id) {
                break;
            }
            result.push(id);
            // decode 阶段固定只有一个 token，直接原地复用输入张量，避免每步重新分配。
            unsafe {
                decode_token_tensor.data_mut()[0] = id;
            }
            input_tensor = decode_token_tensor.slice(0, &one_shape);
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
        let mut result = Vec::<u32>::with_capacity(length + max_len);
        result.extend_from_slice(token_ids);
        let mut generated_tokens = 0usize;
        let prompt_shape = vec![1, token_ids.len()];
        let one_shape = vec![1, 1];
        let mut input = Tensor::<u32>::new(token_ids.to_vec(), &prompt_shape);
        let mut decode_token_tensor = Tensor::new(vec![0u32], &one_shape);
        // 循环生成直到 break
        let turn_end_token = loop {
            let id = sampling::random_sample(&self.forward(&input, &mut cache), &result, top_p, top_k, temperature,penalty);
            result.push(id);
            generated_tokens += 1;
            // chat 模式的 max_len 应表示“最多新生成多少 token”，而不是“提示词+输出的总长度”。
            if self.eos_token_id.contains(&id) {
                break Some(id);
            }
            if generated_tokens >= max_len {
                // 多轮对话必须把回合结束 token 写入 cache，否则下一轮 user header 会直接接在上一轮 assistant 文本后面。
                break self.eos_token_id.first().copied();
            }
            unsafe {
                decode_token_tensor.data_mut()[0] = id;
            }
            input = decode_token_tensor.slice(0, &one_shape);
        };

        if let Some(end_id) = turn_end_token {
            unsafe {
                decode_token_tensor.data_mut()[0] = end_id;
            }
            let end_input = decode_token_tensor.slice(0, &one_shape);
            let _ = self.forward(&end_input, &mut cache);
        }

        let output = result.split_off(length);
        (output, cache)
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
    layer: usize,
) 
    where T: Float + Default + FromPrimitive + std::iter::Sum + std::ops::AddAssign + Debug
        + std::marker::Sync + std::marker::Send + 'static
{
    // f32 专用内核：decode/prefill 分离，减少泛型转换与中间张量开销。
    if seq_len == 1
        && self_attention_decode_f32(
            hidden_states,
            q,
            k,
            v,
            n_kv_h,
            n_groups,
            total_seq_len,
            dqkv,
            layer,
        )
    {
        return;
    }

    if seq_len > 1
        && self_attention_prefill_f32(
            hidden_states,
            att_scores,
            q,
            k,
            v,
            n_kv_h,
            n_groups,
            seq_len,
            total_seq_len,
            dqkv,
        )
    {
        return;
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
    w_up: &Weight<T>,
    w_down: &Weight<T>,
    w_gate: &Weight<T>,
    rms_w: &Tensor<T>,
    eps: impl Float,
) -> MlpTiming
    where T: Float + Default + std::iter::Sum + FromPrimitive + std::ops::MulAssign
        + std::marker::Sync + std::marker::Send + 'static
{
    let enable_prefill_timing = hidden_states.shape()[0] > 1 && prefill_layer_timing_enabled();
    // 对 residual 进行归一化操作并存储在 hidden_states中
    OP::rms_norm(hidden_states,residual,rms_w,eps);
    let gate_up_stage_start = if enable_prefill_timing {
        Some(Instant::now())
    } else {
        None
    };
    // gate/up 共享同一份输入，prefill 优先走共享输入调度，decode 自动回退到并行单发。
    OP::matmul_transb_weight_batch2(
        gate,
        up,
        T::from(0.).unwrap(),
        hidden_states,
        w_gate,
        w_up,
        T::from(1.).unwrap(),
    );
    // 对up,gate进行 silu计算
    OP::silu(up,gate);
    let gate_up_elapsed = gate_up_stage_start
        .map(|s| s.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let down_stage_start = if enable_prefill_timing {
        Some(Instant::now())
    } else {
        None
    };
    // residual = residual + up @ down^T, w_down 为输出投影
    OP::matmul_transb_weight(residual,T::from(1.).unwrap(),up,w_down,T::from(1.).unwrap());
    MlpTiming {
        gate_up_s: gate_up_elapsed,
        down_s: down_stage_start
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0),
    }
}



#[cfg(test)]
mod test{
    use crate::core::tensor::Tensor;
    use super::{mlp, Llama, Weight};
    #[test]
    pub fn test_mlp() {
        let seq_len = 4;
        let d = 2;
        let di = 3;
        let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
        let w_up = Weight::Dense(Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]));
        let w_down = Weight::Dense(Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]));
        let w_gate = Weight::Dense(Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]));
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
        use crate::core::tensor::float_eq;

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
        assert_eq!(model.params.lm_head.as_dense().data()[10], model.params.embedding_table.data()[10]);
        assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
        assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
        assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
        assert!(float_eq(&model.params.w_down[0].as_dense().data()[100], &-0.0625, 1e-6));
        assert!(float_eq(&model.params.w_up[0].as_dense().data()[100], &1.46875, 1e-6));
        assert!(float_eq(&model.params.w_gate[1].as_dense().data()[100], &0.296875, 1e-6));
        assert!(float_eq(&model.params.wq[1].as_dense().data()[100], &0.032226563, 1e-6));
        assert!(float_eq(&model.params.wk[1].as_dense().data()[100], &-0.21386719, 1e-6));
        assert!(float_eq(&model.params.wv[0].as_dense().data()[100], &0.041015625, 1e-6));
        assert!(float_eq(&model.params.wo[0].as_dense().data()[100], &0.01965332, 1e-6));

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

    #[test]
    #[ignore = "slow GGUF parity regression"]
    pub fn test_gguf_vs_safetensors_next_token() {
        use std::path::PathBuf;
        use tokenizers::Tokenizer;

        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("test");
        let gguf_path = model_dir.join("Llama-3.2-1B-Instruct-Q4_K_L.gguf");

        let model_st = Llama::<f32>::from_safetensors(&model_dir);
        let model_gg = Llama::<f32>::from_gguf(&gguf_path);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

        let prompt = crate::chat::templates::build_llama3_prompt("hello", true);
        let enc = tokenizer.encode(prompt, false).unwrap();
        let input = Tensor::new(enc.get_ids().to_vec(), &vec![1, enc.get_ids().len()]);

        let mut cache_st = model_st.new_cache();
        let mut cache_gg = model_gg.new_cache();
        let logits_st = model_st.forward(&input, &mut cache_st);
        let logits_gg = model_gg.forward(&input, &mut cache_gg);

        let argmax = |v: &[f32]| -> usize {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &x) in v.iter().enumerate() {
                if x > best_val {
                    best_val = x;
                    best_idx = i;
                }
            }
            best_idx
        };

        let top_st = argmax(logits_st.data());
        let top_gg = argmax(logits_gg.data());
        println!("top safetensors: {top_st}, top gguf: {top_gg}");

        // For the same architecture and prompt, GGUF-dequantized logits should at least
        // agree on the top token with safetensors baseline.
        assert_eq!(top_st, top_gg);
    }

    #[test]
    #[ignore = "diagnostic only"]
    pub fn test_gguf_tensor_diagnostics() {
        use std::path::PathBuf;

        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("test");
        let gguf_path = model_dir.join("Llama-3.2-1B-Instruct-Q4_K_L.gguf");

        let model_st = Llama::<f32>::from_safetensors(&model_dir);
        let model_gg = Llama::<f32>::from_gguf(&gguf_path);

        let rms_st = model_st.params.rms_out_w.data();
        let rms_gg = model_gg.params.rms_out_w.data();
        let mut rms_max_diff = 0.0f32;
        for i in 0..rms_st.len() {
            rms_max_diff = rms_max_diff.max((rms_st[i] - rms_gg[i]).abs());
        }

        // Embedding is quantized in GGUF, so compare only a small prefix and report rough scale.
        let emb_st = model_st.params.embedding_table.data();
        let emb_gg = model_gg.params.embedding_table.data();
        let check = emb_st.len().min(1024);
        let mut emb_mae = 0.0f32;
        for i in 0..check {
            emb_mae += (emb_st[i] - emb_gg[i]).abs();
        }
        emb_mae /= check as f32;

        println!("rms_out_w max abs diff: {rms_max_diff}");
        println!("embedding first {check} mean abs diff: {emb_mae}");
    }

    #[test]
    #[ignore = "diagnostic only"]
    pub fn test_gguf_weight_mae_diagnostics() {
        use std::path::PathBuf;

        std::env::set_var("LMRS_FORCE_DENSE_GGUF", "1");

        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("test");
        let gguf_path = model_dir.join("Llama-3.2-1B-Instruct-Q4_K_L.gguf");

        let model_st = Llama::<f32>::from_safetensors(&model_dir);
        let model_gg = Llama::<f32>::from_gguf(&gguf_path);

        let mae = |a: &[f32], b: &[f32]| -> f32 {
            let mut s = 0.0f32;
            for i in 0..a.len() {
                s += (a[i] - b[i]).abs();
            }
            s / a.len() as f32
        };

        let n = model_st.n_layers;
        let mut sum_q = 0.0f32;
        let mut sum_k = 0.0f32;
        let mut sum_v = 0.0f32;
        let mut sum_o = 0.0f32;
        let mut sum_up = 0.0f32;
        let mut sum_gate = 0.0f32;
        let mut sum_down = 0.0f32;

        for i in 0..n {
            sum_q += mae(
                model_st.params.wq[i].as_dense().data(),
                model_gg.params.wq[i].as_dense().data(),
            );
            sum_k += mae(
                model_st.params.wk[i].as_dense().data(),
                model_gg.params.wk[i].as_dense().data(),
            );
            sum_v += mae(
                model_st.params.wv[i].as_dense().data(),
                model_gg.params.wv[i].as_dense().data(),
            );
            sum_o += mae(
                model_st.params.wo[i].as_dense().data(),
                model_gg.params.wo[i].as_dense().data(),
            );
            sum_up += mae(
                model_st.params.w_up[i].as_dense().data(),
                model_gg.params.w_up[i].as_dense().data(),
            );
            sum_gate += mae(
                model_st.params.w_gate[i].as_dense().data(),
                model_gg.params.w_gate[i].as_dense().data(),
            );
            sum_down += mae(
                model_st.params.w_down[i].as_dense().data(),
                model_gg.params.w_down[i].as_dense().data(),
            );
        }

        println!("avg layer MAE wq: {}", sum_q / n as f32);
        println!("avg layer MAE wk: {}", sum_k / n as f32);
        println!("avg layer MAE wv: {}", sum_v / n as f32);
        println!("avg layer MAE wo: {}", sum_o / n as f32);
        println!("avg layer MAE w_up: {}", sum_up / n as f32);
        println!("avg layer MAE w_gate: {}", sum_gate / n as f32);
        println!("avg layer MAE w_down: {}", sum_down / n as f32);

        let lm_mae = mae(
            model_st.params.lm_head.as_dense().data(),
            model_gg.params.lm_head.as_dense().data(),
        );
        println!("lm_head MAE: {lm_mae}");

        std::env::remove_var("LMRS_FORCE_DENSE_GGUF");
    }

    #[test]
    #[ignore = "diagnostic only"]
    pub fn test_gguf_tensor_type_diagnostics() {
        use gguf::GGUFFile;
        use std::collections::BTreeMap;
        use std::path::PathBuf;

        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("test");
        let gguf_path = model_dir.join("Llama-3.2-1B-Instruct-Q4_K_L.gguf");
        let bytes = std::fs::read(&gguf_path).unwrap();
        let gguf = GGUFFile::read(&bytes).unwrap().unwrap();

        let mut counts = BTreeMap::<String, usize>::new();
        for t in &gguf.tensors {
            let k = format!("{:?}", t.tensor_type);
            *counts.entry(k).or_insert(0) += 1;
        }
        println!("GGUF tensor type counts: {counts:?}");

        let show = |name: &str| {
            let ty = gguf
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| format!("{:?}", t.tensor_type))
                .unwrap_or_else(|| "<missing>".to_string());
            println!("{name}: {ty}");
        };

        show("blk.0.attn_q.weight");
        show("blk.0.attn_k.weight");
        show("blk.0.attn_v.weight");
        show("blk.0.attn_output.weight");
        show("blk.0.ffn_up.weight");
        show("blk.0.ffn_gate.weight");
        show("blk.0.ffn_down.weight");
        show("token_embd.weight");
        show("output.weight");
    }
}
