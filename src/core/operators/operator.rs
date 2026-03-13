use crate::model::config::RopeScaling;
use crate::core::tensor::Tensor;
use crate::model::params::Weight;
use crate::runtime::cpu;
use crate::core::operators::quant::generic::{
    matmul_transb_gguf_quant,
    matmul_transb_gguf_quant_batch2,
    matmul_transb_gguf_quant_batch3,
};
use num_traits::float::Float;
use num_traits::Num;
use num_traits::{FromPrimitive, ToPrimitive};
use gemm::{Parallelism};
use rayon::prelude::*;

use std::fmt::Debug;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct RopeFreqKey {
    d: usize,
    theta_bits: u32,
    use_llama3_scaling: bool,
    factor_bits: u32,
    orig_max_pos: usize,
    low_freq_factor_bits: u32,
    high_freq_factor_bits: u32,
}

static ROPE_INV_FREQ_CACHE: OnceLock<Mutex<HashMap<RopeFreqKey, Arc<Vec<f32>>>>> = OnceLock::new();

#[inline]
fn build_rope_inv_freqs(
    d: usize,
    theta_f32: f32,
    use_llama3_scaling: bool,
    factor: f32,
    orig_max_pos: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
) -> Vec<f32> {
    let half = d / 2;
    let mut out = vec![0.0f32; half];
    let d_f32 = d as f32;
    let two_f32 = 2.0_f32;

    let low_freq_wavelen = if use_llama3_scaling {
        orig_max_pos / low_freq_factor
    } else {
        0.0
    };
    let high_freq_wavelen = if use_llama3_scaling {
        orig_max_pos / high_freq_factor
    } else {
        0.0
    };

    for i in 0..half {
        let i_f32 = i as f32;
        let exponent = (two_f32 * i_f32) / d_f32;
        let freq_base = theta_f32.powf(-exponent);
        out[i] = if use_llama3_scaling {
            let wavelen = (2.0 * std::f32::consts::PI) / freq_base;
            if wavelen < high_freq_wavelen {
                freq_base
            } else if wavelen > low_freq_wavelen {
                freq_base / factor
            } else {
                let denom = high_freq_factor - low_freq_factor;
                let smooth = if denom.abs() > 1e-6 {
                    (orig_max_pos / wavelen - low_freq_factor) / denom
                } else {
                    0.0
                };
                (1.0 - smooth) * (freq_base / factor) + smooth * freq_base
            }
        } else {
            freq_base
        };
    }

    out
}

#[inline]
fn rope_inv_freqs_cached(
    d: usize,
    theta_f32: f32,
    use_llama3_scaling: bool,
    factor: f32,
    orig_max_pos: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
) -> Arc<Vec<f32>> {
    let key = RopeFreqKey {
        d,
        theta_bits: theta_f32.to_bits(),
        use_llama3_scaling,
        factor_bits: factor.to_bits(),
        orig_max_pos: orig_max_pos as usize,
        low_freq_factor_bits: low_freq_factor.to_bits(),
        high_freq_factor_bits: high_freq_factor.to_bits(),
    };

    let cache = ROPE_INV_FREQ_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(v) = guard.get(&key) {
        return Arc::clone(v);
    }

    let built = Arc::new(build_rope_inv_freqs(
        d,
        theta_f32,
        use_llama3_scaling,
        factor,
        orig_max_pos,
        low_freq_factor,
        high_freq_factor,
    ));
    guard.insert(key, Arc::clone(&built));
    built
}

/// RMS norm 的应用步骤 SIMD 化：y[i] = x[i] * w[i] * inv_rms
#[inline]
fn rms_norm_apply_f32_simd(y: &mut [f32], x: &[f32], w: &[f32], inv_rms: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { rms_norm_apply_f32_avx2(y, x, w, inv_rms) };
            return;
        }
    }
    for i in 0..y.len() {
        y[i] = x[i] * w[i] * inv_rms;
    }
}

/// AVX2 加速的 RMS norm 应用：将 x*w*inv_rms 向量化。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rms_norm_apply_f32_avx2(y: &mut [f32], x: &[f32], w: &[f32], inv_rms: f32) {
    use std::arch::x86_64::*;
    let n = y.len();
    let n8 = n / 8 * 8;
    let v_inv_rms = _mm256_set1_ps(inv_rms);
    let mut i = 0usize;
    while i < n8 {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let vw = _mm256_loadu_ps(w.as_ptr().add(i));
        let prod = _mm256_mul_ps(_mm256_mul_ps(vx, vw), v_inv_rms);
        _mm256_storeu_ps(y.as_mut_ptr().add(i), prod);
        i += 8;
    }
    while i < n {
        y[i] = x[i] * w[i] * inv_rms;
        i += 1;
    }
}

#[inline]
fn sum_squares_f32_simd(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return unsafe { sum_squares_f32_avx512(x) };
        }
        // FMA + AVX2: 使用 fmadd_ps 将 v*v+acc 合并为单条指令
        if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2") {
            return unsafe { sum_squares_f32_avx2_fma(x) };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { sum_squares_f32_avx2(x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { sum_squares_f32_neon(x) };
    }

    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_squares_f32_avx512(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut i = 0usize;
    let n = x.len();
    let n16 = n / 16 * 16;
    let mut acc = _mm512_setzero_ps();
    while i < n16 {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        acc = _mm512_add_ps(acc, _mm512_mul_ps(v, v));
        i += 16;
    }
    let mut tmp = [0.0f32; 16];
    _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut s = tmp.iter().sum::<f32>();
    while i < n {
        s += x[i] * x[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_squares_f32_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut i = 0usize;
    let n = x.len();
    let n8 = n / 8 * 8;
    let mut acc = _mm256_setzero_ps();
    while i < n8 {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
        i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut s = tmp.iter().sum::<f32>();
    while i < n {
        s += x[i] * x[i];
        i += 1;
    }
    s
}

/// FMA 加速的平方和：使用 _mm256_fmadd_ps(v, v, acc) 替代分离的 mul+add。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_squares_f32_avx2_fma(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut i = 0usize;
    let n = x.len();
    let n8 = n / 8 * 8;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    // 双累加器减少数据依赖
    while i + 16 <= n8 {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        i += 16;
    }
    while i < n8 {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(v, v, acc0);
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc0);
    let mut s = tmp.iter().sum::<f32>();
    while i < n {
        s += x[i] * x[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "aarch64")]
unsafe fn sum_squares_f32_neon(x: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut i = 0usize;
    let n = x.len();
    let n4 = n / 4 * 4;
    let mut acc = vdupq_n_f32(0.0);
    while i < n4 {
        let v = vld1q_f32(x.as_ptr().add(i));
        acc = vaddq_f32(acc, vmulq_f32(v, v));
        i += 4;
    }
    let mut tmp = [0.0f32; 4];
    vst1q_f32(tmp.as_mut_ptr(), acc);
    let mut s = tmp.iter().sum::<f32>();
    while i < n {
        s += x[i] * x[i];
        i += 1;
    }
    s
}


/// 可性能优化！
/// 
/// get (row) vectors from a 2D table given a list of indices
pub fn gather<T>(y: &mut Tensor<T>, indices: &Tensor<u32>, table: &Tensor<T>) 
    where T: Default + Num + Copy + Float
{
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

/// f32 转换计算改造完成
/// 改造 llama 3 完成
/// RoPE: Rotary Positional Embedding 旋转位置编码
/// 需要性能优化
pub fn rope<T>(y: &mut Tensor<T>, start_pos: usize, theta: f32, scaling: &RopeScaling) 
    where T: Float + Default + FromPrimitive +  Copy + Into<f32>
{
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    let theta_f32: f32 = theta.to_f32().unwrap_or(0.0);
    let use_llama3_scaling = scaling.rope_type == "llama3"
        && scaling.factor > 0.0
        && scaling.original_max_position_embeddings > 0
        && scaling.low_freq_factor > 0.0
        && scaling.high_freq_factor > 0.0;
    let factor = scaling.factor;
    let orig_max_pos = scaling.original_max_position_embeddings as f32;
    let low_freq_factor = scaling.low_freq_factor;
    let high_freq_factor = scaling.high_freq_factor;
    let half = d / 2;
    let inv_freqs = rope_inv_freqs_cached(
        d,
        theta_f32,
        use_llama3_scaling,
        factor,
        orig_max_pos,
        low_freq_factor,
        high_freq_factor,
    );
    let mut sin_cache = vec![0.0f32; half];
    let mut cos_cache = vec![0.0f32; half];

    for tok in 0..seq_len {
        let pos = start_pos + tok;
        let pos_f32 = pos as f32;
        for i in 0..half {
            let phase = pos_f32 * inv_freqs[i];
            let (sin, cos) = phase.sin_cos();
            sin_cache[i] = sin;
            cos_cache[i] = cos;
        }

        for head in 0..n_heads {
            let base_offset = tok * n_heads * d + head * d;
            for i in 0..half {
                let idx_a = base_offset + i;
                let idx_b = base_offset + i + half;
                let a_f32: f32 = data[idx_a].into();
                let b_f32: f32 = data[idx_b].into();
                let sin = sin_cache[i];
                let cos = cos_cache[i];
                let new_a = a_f32* cos - b_f32 * sin;
                let new_b = b_f32 * cos + a_f32 * sin;
                data[idx_a] = T::from(new_a).unwrap_or(T::zero());
                data[idx_b] = T::from(new_b).unwrap_or(T::zero());
            }
        }
    }
}

/// f32 计算改造完成
/// softmax(x) = exp(x - max) / sum(exp(x - max))
/// y = softmax(mask(x))
/// 带有 mask 的 softmax 
/// 先 mask 后进行 softmax 
/// 如何保障f16计算正确？计算前先将 T 类型全部转换成 f32 来计算？那么你精度转换是何意味？
/// 精度转换是为了计算时不出现溢出等异常现象，最后会统一将 f32 转换回 T 格式！减少内存占用！
pub fn masked_softmax<T>(y: &mut Tensor<T>) 
    where T: Float + Default + std::iter::Sum + FromPrimitive + Debug + 'static
{
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };

    // f32 快路径：避免反复 to_f32/from_f32 和临时 exp buffer。
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let data_f32 = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, data.len()) };
        data_f32
            .par_chunks_mut(total_seq_len)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let i = row_idx % seq_len;
                let boundary = total_seq_len - seq_len + i + 1;

                let mut max = row[0];
                for &v in &row[..boundary] {
                    if v > max {
                        max = v;
                    }
                }

                let mut sum_exp = 0.0f32;
                for v in &mut row[..boundary] {
                    *v = (*v - max).exp();
                    sum_exp += *v;
                }

                let inv = 1.0f32 / sum_exp.max(1e-12);
                for v in &mut row[..boundary] {
                    *v *= inv;
                }
                for v in &mut row[boundary..] {
                    *v = 0.0;
                }
            });
        return;
    }

    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            // 设置边界，使矩阵称为下三角矩阵，这就是 mask 的核心
            let boundary = total_seq_len - seq_len + i + 1;
            // 使用 fold 来找出特定范围内的最大值
            let max = data[offset..offset + boundary]
                .iter()
                .map(|val| val.to_f32().unwrap()) 
                .fold(data[offset].to_f32().unwrap(), |a, b| a.max(b));
            // 计算指数并累加
            let mut sum_exp: f32 = 0.0;
            let mut exp_vals:Vec<f32> = Vec::with_capacity(boundary);
            for j in 0..boundary{
                let val_f32 = data[offset + j].to_f32().unwrap_or(0.0);
                let e = (val_f32 - max).exp();
                sum_exp += e;
                //data[offset + j] = T::from_f32(e).unwrap_or(T::zero());
                exp_vals.push(e);
            }
            // softmax 归一化，使得全部和为 1,并将值全部转回 T
            (0..boundary).for_each(|j| data[offset + j] = T::from_f32(exp_vals[j] / sum_exp).unwrap_or(T::zero()));
            // 将 boundary 后面的所有位置全部强制设为 0.0 
            (boundary..total_seq_len).for_each(|j| data[offset + j] = T::zero());
        }
    }
}

/// f32 转换计算改造完成
/// LLaMA 的输入通常形状为 [Batch_Size, Seq_Len, Hidden_Dim]
/// RMSNorm 是沿着 最后一个维度 (Hidden_Dim) 进行的。也就是说，对于每一个 token 的向量，我们要独立地计算它的 RMS 并归一化
/// RMSNorm 它强制每一层的输出分布保持统计特性一致（均方根为 1）。
/// 这样，第 N 层就不需要关心第 N-1 层的具体数值范围是多少，它只需要处理标准化的数据。这就解耦了层与层之间的依赖，让深层网络的训练成为可能
/// 对输入的 x 进行归一化操作，并将结果保存在 y 中
pub fn rms_norm<T>(y: &mut Tensor<T>, x: &Tensor<T>, w: &Tensor<T>, epsilon: impl Float) 
    where T: Float + std::iter::Sum + Default + 'static
{
    assert!(y.size() == x.size());
    // 获取维度数
    let ndim = y.shape().len();
    // 确保至少有2个维度
    assert!(ndim >= 2);
    // 序列长度
    let seq_len = y.shape()[ndim - 2];
    // 隐藏层维度
    let hidden_size = y.shape()[ndim - 1];
    // 获取权重维度数
    let wdim = w.shape().len();
    // 确保权重维度数只有1个维度
    assert!(wdim == 1);
    // 确保权重长度必须等于隐藏层维度
    assert!(w.size() == hidden_size);
    // 批次数量
    let batch = y.size() / (seq_len * hidden_size);
    // 获取数据的引用
    let y = unsafe { y.data_mut() };
    let x = x.data();
    let w = w.data();

    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let y_f32 = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len()) };
        let x_f32 = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let w_f32 = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len()) };
        let eps = epsilon.to_f32().unwrap_or(0.0);
        let inv_hidden = 1.0f32 / hidden_size as f32;

        for b in 0..batch {
            let base = b * seq_len * hidden_size;
            for l in 0..seq_len {
                let offset = base + l * hidden_size;
                let x_row = &x_f32[offset..offset + hidden_size];
                let y_row = &mut y_f32[offset..offset + hidden_size];
                let sum_sq = sum_squares_f32_simd(x_row);
                let inv_rms = 1.0f32 / (sum_sq * inv_hidden + eps).sqrt();

                // RMS norm 内循环：y[i] = x[i] * w[i] * inv_rms，使用 SIMD 加速
                rms_norm_apply_f32_simd(y_row, x_row, w_f32, inv_rms);
            }
        }
        return;
    }

    // 遍历每个批次
    for b in 0..batch {
        // 当前批次的基索引
        let base = b * seq_len * hidden_size;
        // 遍历批次中的每个序列
        for l in 0..seq_len {
            // 当前序列的偏移量
            let offset = base + l * hidden_size;
            // 平方和
            let mut sum_sq: f32 = 0.0;
            for i in 0..hidden_size{
                let val = x[offset + i].to_f32().unwrap_or(0.0);
                sum_sq += val * val;
            }
            let total_hidden_size = hidden_size as f32;
            let sqrt:f32 = (sum_sq / total_hidden_size  + epsilon.to_f32().unwrap_or(0.0)).sqrt();
            // 计算并储存结果
            for i in 0..hidden_size {
                let w_val = w[i].to_f32().unwrap_or(0.0);
                let x_val = x[offset + i].to_f32().unwrap_or(0.0);
                let res = (w_val * x_val) / sqrt;
                y[offset + i] = T::from(res).unwrap_or(T::zero());
            }
        }
    }
}

/// f32 转换计算改造完成
/// y = sigmoid(x) * x * y
/// Swish/SiLU 激活函数
pub fn silu<T>(y: &mut Tensor<T>, x: &Tensor<T>) 
    where T: Float + Default + FromPrimitive + ToPrimitive + Copy + 'static
{
    debug_assert!(y.size() == x.size());
    let y_data = unsafe {y.data_mut()};
    let x_data = x.data();

    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let y_f32 = unsafe { std::slice::from_raw_parts_mut(y_data.as_mut_ptr() as *mut f32, y_data.len()) };
        let x_f32 = unsafe { std::slice::from_raw_parts(x_data.as_ptr() as *const f32, x_data.len()) };

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: 运行时已检测 avx2。
                unsafe {
                    use std::arch::x86_64::*;
                    let mut i = 0usize;
                    let n = y_f32.len();
                    while i + 8 <= n {
                        let mut s = [0.0f32; 8];
                        for j in 0..8 {
                            let xv = x_f32[i + j];
                            s[j] = xv / (1.0 + (-xv).exp());
                        }
                        let vy = _mm256_loadu_ps(y_f32.as_ptr().add(i));
                        let vs = _mm256_loadu_ps(s.as_ptr());
                        let out = _mm256_mul_ps(vy, vs);
                        _mm256_storeu_ps(y_f32.as_mut_ptr().add(i), out);
                        i += 8;
                    }
                    while i < n {
                        let xv = x_f32[i];
                        y_f32[i] *= xv / (1.0 + (-xv).exp());
                        i += 1;
                    }
                }
                return;
            }
        }

        for i in 0..y_f32.len() {
            let xv = x_f32[i];
            y_f32[i] *= xv / (1.0 + (-xv).exp());
        }
        return;
    }

    y_data.iter_mut().zip(x_data.iter()).for_each(|(y_val,x_val)|{
        let x_f32 = x_val.to_f32().unwrap_or(0.0);
        let exp_neg_x = (-x_f32).exp();
        let silu_val_f32 = x_f32 / (1.0 + exp_neg_x);
        let silu_val = T::from(silu_val_f32).unwrap_or(T::zero());
        *y_val = *y_val * silu_val;
    });
}

/// Matrix multiply with weight dispatch.
///
/// This is the key entry for real quantized inference:
/// - dense weights -> existing float matmul path
pub fn matmul_transb_weight<T>(
    c: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    b: &Weight<T>,
    alpha: T,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    match b {
        Weight::Dense(w) => matmul_transb(c, beta, a, w, alpha),
        Weight::GgufQ(wq) => matmul_transb_gguf_quant(c, beta, a, wq, alpha),
    }
}

pub fn matmul_transb_weight_batch2<T>(
    c0: &mut Tensor<T>,
    c1: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    b0: &Weight<T>,
    b1: &Weight<T>,
    alpha: T,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    match (b0, b1) {
        // gate/up 若已在加载期构建条带布局，则默认直接走 batch2 量化路径。
        // 旧 dense workset 仅保留为实验开关兜底。
        (Weight::GgufQ(wq0), Weight::GgufQ(wq1))
            if a.shape().len() == 2
                && a.shape()[0] > 1
                && wq0.tensor_type == wq1.tensor_type
                && wq0.cols() == wq1.cols()
                && (wq0.prefill_packed.is_some()
                    || wq1.prefill_packed.is_some()
                    || wq0.prefill_q8k_interleave.is_some()
                    || wq1.prefill_q8k_interleave.is_some()
                    || wq0.prefill_stripes.is_some()
                    || std::env::var("LMRS_PREFILL_GATEUP_WORKSET")
                        .ok()
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false)) =>
        {
            matmul_transb_gguf_quant_batch2(c0, c1, beta, a, wq0, wq1, alpha)
        }
        _ => {
            rayon::join(
                || matmul_transb_weight(c0, beta, a, b0, alpha),
                || matmul_transb_weight(c1, beta, a, b1, alpha),
            );
        }
    }
}

pub fn matmul_transb_weight_batch3<T>(
    c0: &mut Tensor<T>,
    c1: &mut Tensor<T>,
    c2: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    b0: &Weight<T>,
    b1: &Weight<T>,
    b2: &Weight<T>,
    alpha: T,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    match (b0, b1, b2) {
        // QKV 的阶段九默认路径同样改为“量化预打包工作集 + 大 GEMM”。
        // 实验开关只负责更激进的并行展开，不再决定是否启用这条主路径。
        (Weight::GgufQ(wq0), Weight::GgufQ(wq1), Weight::GgufQ(wq2))
            if a.shape().len() == 2
                && a.shape()[0] > 1
                && wq0.tensor_type == wq1.tensor_type
                && wq0.tensor_type == wq2.tensor_type
                && wq0.cols() == wq1.cols()
                && wq0.cols() == wq2.cols() =>
        {
            matmul_transb_gguf_quant_batch3(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha)
        }
        _ => {
            rayon::join(
                || matmul_transb_weight(c0, beta, a, b0, alpha),
                || {
                    rayon::join(
                        || matmul_transb_weight(c1, beta, a, b1, alpha),
                        || matmul_transb_weight(c2, beta, a, b2, alpha),
                    );
                },
            );
        }
    }
}

/// gemm 库深度优化版本
/// f32 每次操作大概需要 7 us
/// 只对 f32 进行优化计算 
pub fn matmul_transb<T>(c: &mut Tensor<T>, beta: T, a: &Tensor<T>, b: &Tensor<T>, alpha: T) 
    where T: Float + Default + Copy + std::iter::Sum + 'static
{
    // 确保 A 和 B 能进行矩阵乘法
    assert!(a.shape().len() == b.shape().len());
    // 确保 A 和 C 能进行矩阵加法
    assert!(a.shape().len() == c.shape().len());

    let ndim = a.shape().len();
    assert!(ndim >= 2);
    let a_row = a.shape()[ndim - 2];
    let a_col = a.shape()[ndim - 1];

    let b_row = b.shape()[ndim - 2];
    let b_col = b.shape()[ndim - 1];

    let c_row = c.shape()[ndim - 2];
    let c_col = c.shape()[ndim - 1];

    let c = unsafe { c.data_mut() };
    let a = a.data();
    let b = b.data();

    assert!(a_col == b_col);
    assert!(c_col == b_row);
    assert!(a_row == c_row);

    // Fast path: call gemm for f32.
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            gemm::gemm(
                c_row,
                c_col,
                a_col,
                c.as_mut_ptr() as *mut f32,
                1,
                c_col as isize,
                true,
                a.as_ptr() as *const f32,
                1,
                a_col as isize,
                b.as_ptr() as *const f32,
                b_col as isize,
                1,
                beta.to_f32().unwrap_or(0.0),
                alpha.to_f32().unwrap_or(0.0),
                false,
                false,
                false,
                Parallelism::Rayon(cpu::prefill_threads()),
            );
        }
        return;
    }
    // // Fast path: call gemm for bf16, f16.
    // if TypeId::of::<T>() == TypeId::of::<half::f16>() || TypeId::of::<T>() == TypeId::of::<half::bf16>() {
        
    //     let total_a = a_row * a_col;
    //     let total_b = b_row * a_col;
    //     let total_c = a_row * b_row;

    //     // 分配临时的 f32 缓冲区
    //     let mut a_f32 = vec![0.0f32; total_a];
    //     let mut b_f32 = vec![0.0f32; total_b];
    //     let mut c_f32 = vec![0.0f32; total_c];

    //     let alpha_f32 = alpha.to_f32().unwrap_or(1.0);
    //     let beta_f32 = beta.to_f32().unwrap_or(0.0);

    //     // --- Step 3.1: 并行转换 A 和 B 到 f32 ---
    //     if TypeId::of::<T>() == TypeId::of::<half::f16>() {
    //         // 安全地将原始数据视为 f16 切片
    //         let a_slice_half = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const half::f16, total_a) };
    //         let b_slice_half = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const half::f16, total_b) };
            
    //         // 现在 a_slice_half 是 &[f16]，它是 Sync 的，可以安全地在 rayon 中共享
    //         a_f32.par_iter_mut().enumerate().for_each(|(i, x)| {
    //             *x = a_slice_half[i].to_f32();
    //         });
    //         b_f32.par_iter_mut().enumerate().for_each(|(i, x)| {
    //             *x = b_slice_half[i].to_f32();
    //         });
    //     } else {
    //         // bf16
    //         let a_slice_bf = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const half::bf16, total_a) };
    //         let b_slice_bf = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const half::bf16, total_b) };

    //         a_f32.par_iter_mut().enumerate().for_each(|(i, x)| {
    //             *x = a_slice_bf[i].to_f32();
    //         });
    //         b_f32.par_iter_mut().enumerate().for_each(|(i, x)| {
    //             *x = b_slice_bf[i].to_f32();
    //         });
    //     }

    //     if beta_f32 != 0.0 {
    //         if TypeId::of::<T>() == TypeId::of::<half::f16>() {
    //             let c_slice_half = unsafe { std::slice::from_raw_parts(c.as_ptr() as *const half::f16, total_c) };
    //             c_f32.par_iter_mut().enumerate().for_each(|(i, x)| {
    //                 *x = c_slice_half[i].to_f32();
    //             });
    //         } else {
    //             let c_slice_bf = unsafe { std::slice::from_raw_parts(c.as_ptr() as *const half::bf16, total_c) };
    //             c_f32.par_iter_mut().enumerate().for_each(|(i, x)| {
    //                 *x = c_slice_bf[i].to_f32();
    //             });
    //         }
    //     } 
    //     unsafe {
    //         gemm::gemm(
    //             a_row, 
    //             b_row, 
    //             a_col,
    //             c_f32.as_mut_ptr(),
    //             1, 
    //             b_row as isize,
    //             false,
    //             a_f32.as_ptr(),
    //             1, 
    //             a_col as isize,
    //             b_f32.as_ptr(),
    //             a_col as isize, 
    //             1, 
    //             beta_f32,
    //             alpha_f32,
    //             false,
    //             false,
    //             false,
    //             Parallelism::Rayon(0),
    //         );
    //     }

    //     if TypeId::of::<T>() == TypeId::of::<half::f16>() {
    //         // 创建目标可变切片
    //         let c_out_slice = unsafe { 
    //             std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut half::f16, total_c) 
    //         };
            
    //         // zip: 将 (f32值, &mut f16位置) 配对
    //         // par_iter_mut(): 允许并行修改 c_out_slice
    //         c_f32.par_iter().zip(c_out_slice.par_iter_mut()).for_each(|(&val, dest)| {
    //             *dest = half::f16::from_f32(val);
    //         });
            
    //     } else {
    //         // bf16
    //         let c_out_slice = unsafe { 
    //             std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut half::bf16, total_c) 
    //         };
            
    //         c_f32.par_iter().zip(c_out_slice.par_iter_mut()).for_each(|(&val, dest)| {
    //             *dest = half::bf16::from_f32(val);
    //         });
    //     }

    //     return;
    // }
    
    // Generic fallback for non-f32/f64 element types.
    let beta_f32 = beta.to_f32().unwrap_or(0.0);
    let alpha_f32 = alpha.to_f32().unwrap_or(0.0);
    for l in 0..c_row {
        for i in 0..c_col {
            let sum: f32 = (0..a_col)
                .map(|j| {
                    let val_a = a[l * a_col + j].to_f32().unwrap_or(0.0);
                    let val_b = b[i * b_col + j].to_f32().unwrap_or(0.0);
                    val_a * val_b
                })
                .sum();
            let c_idx = l * c_col + i;
            let c_old_f32 = c[c_idx].to_f32().unwrap_or(0.0);
            let result_f32 = beta_f32 * c_old_f32 + alpha_f32 * sum;
            c[c_idx] = T::from(result_f32).unwrap_or(T::zero());
        }
    }
}



/// f32 转换计算改造完成
/// 无任何优化版本
/// C = beta * C + alpha * A @ B^T，@指的是矩阵乘法
/// 需要性能优化
/// 一次计算大概需要 30us 时间
// pub fn matmul_transb<T>(c: &mut Tensor<T>, beta: T, a: &Tensor<T>, b: &Tensor<T>, alpha: T) 
//     where T: Float + Default + Copy + std::iter::Sum + 'static
// {
//     // 确保 A 和 B 能进行矩阵乘法
//     assert!(a.shape().len() == b.shape().len());
//     // 确保 A 和 C 能进行矩阵加法
//     assert!(a.shape().len() == c.shape().len());

//     let ndim = a.shape().len();
//     assert!(ndim >= 2);
//     let a_row = a.shape()[ndim - 2];
//     let a_col = a.shape()[ndim - 1];

//     let b_row = b.shape()[ndim - 2];
//     let b_col = b.shape()[ndim - 1];

//     let c_row = c.shape()[ndim - 2];
//     let c_col = c.shape()[ndim - 1];

//     let c = unsafe { c.data_mut() };
//     let a = a.data();
//     let b = b.data();

//     assert!(a_col == b_col);
//     assert!(c_col == b_row);
//     assert!(a_row == c_row);
//     // Generic fallback for non-f32/f64 element types.
//     let beta_f32 = beta.to_f32().unwrap_or(0.0);
//     let alpha_f32 = alpha.to_f32().unwrap_or(0.0);
//     for l in 0..c_row {
//         for i in 0..c_col {
//             let sum: f32 = (0..a_col)
//                 .map(|j| {
//                     let val_a = a[l * a_col + j].to_f32().unwrap_or(0.0);
//                     let val_b = b[i * b_col + j].to_f32().unwrap_or(0.0);
//                     val_a * val_b
//                 })
//                 .sum();
//             let c_idx = l * c_col + i;
//             let c_old_f32 = c[c_idx].to_f32().unwrap_or(0.0);
//             let result_f32 = beta_f32 * c_old_f32 + alpha_f32 * sum;
//             c[c_idx] = T::from(result_f32).unwrap_or(T::zero());
//         }
//     }
// }

/// 使用 rayon 并行计算版本
/// 一次大概需要 150us 时间
// pub fn matmul_transb<T>(c: &mut Tensor<T>, beta: T, a: &Tensor<T>, b: &Tensor<T>, alpha: T) 
//     where T: Float + Default + Copy + std::iter::Sum + num_traits::ToPrimitive + num_traits::FromPrimitive + Sync + Send
// {
//     // 确保 A 和 B 能进行矩阵乘法
//     assert!(a.shape().len() == b.shape().len());
//     // 确保 A 和 C 能进行矩阵加法
//     assert!(a.shape().len() == c.shape().len());

//     let ndim = a.shape().len();
//     assert!(ndim >= 2);
//     let a_row = a.shape()[ndim - 2];
//     let a_col = a.shape()[ndim - 1];

//     let b_row = b.shape()[ndim - 2];
//     let b_col = b.shape()[ndim - 1];

//     let c_row = c.shape()[ndim - 2];
//     let c_col = c.shape()[ndim - 1];

//     let c = unsafe { c.data_mut() };
//     let a = a.data();
//     let b = b.data();

//     assert!(a_col == b_col);
//     assert!(c_col == b_row);
//     assert!(a_row == c_row);

//     let beta_f32 = beta.to_f32().unwrap_or(0.0);
//     let alpha_f32 = alpha.to_f32().unwrap_or(0.0);
//     // 并行加速开始
//     c.par_chunks_mut(c_col).enumerate().for_each(|(l,c_row_slice)|{
//         for i in 0..c_col{
//             let sum: f32 = (0..a_col)
//                 .map(|j|{
//                     let val_a = a[l * a_col + j].to_f32().unwrap_or(0.0);
//                     let val_b = b[i * b_col + j].to_f32().unwrap_or(0.0);
//                     val_a * val_b
//                 })
//                 .sum(); 
//             let c_old_val = c_row_slice[i]; 
//             let c_old_f32 = c_old_val.to_f32().unwrap_or(0.0);
//             let result_f32 = beta_f32 * c_old_f32 + alpha_f32 * sum;
//             c_row_slice[i] = T::from(result_f32).unwrap_or(T::zero());
//         }
//     });
// }


/// f32 转换计算改造完成
/// Dot product of two tensors (treated as vectors)
/// 两个矩阵点乘
/// 需要性能优化
pub fn dot<T>(x: &Tensor<T>, y: &Tensor<T>) -> T 
    where T: Float + Default + ToPrimitive + FromPrimitive + std::ops::AddAssign + Copy
{
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum: f32 = 0.0;
    for i in 0..len {
        let vx = x_[i].to_f32().expect("Failed to convert to f32");
        let vy = y_[i].to_f32().expect("Failed to convert to f32");
        sum += vx * vy;
    }
    T::from(sum).unwrap()
}




#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    c.print();
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}


#[test]
fn test_matmul_transb_weight_ggufq_q4_0() {
    use crate::formats::gguf::QuantGGUFTensor;
    use gguf::GGMLType;
    //use crate::core::operators::quant::generic::matmul_transb_weight;
    use crate::model::params::Weight;

    // 走通用 GGUF 量化 kernel 的最小用例（Q4_0）。
    // d=1，qs 全 0x98 => 低4位=8, 高4位=9，Q4_0 会减去 8，得到重复 [0, 1]。
    let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
    let mut raw = Vec::<u8>::new();
    raw.extend_from_slice(&d_bits);
    raw.extend_from_slice(&[0x98u8; 16]);

    let wq = QuantGGUFTensor {
        raw,
        shape: vec![1, 32],
        tensor_type: GGMLType::Q4_0,
        row_map: None,
        prefill_workset: None,
        prefill_packed: None,
        prefill_k_metadata: None,
        prefill_q8k_interleave: None,
        prefill_stripes: None,
        hot_layer: false,
    };

    let a = Tensor::<f32>::new(vec![1.0; 32], &vec![1, 32]);
    let mut c = Tensor::<f32>::default(&vec![1, 1]);
    matmul_transb_weight(&mut c, 0.0, &a, &Weight::GgufQ(wq), 1.0);

    // 16 * (0 + 1) = 16
    assert!(c.close_to(&Tensor::<f32>::new(vec![16.0], &vec![1, 1]), 1e-6));
}
