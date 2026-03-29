use crate::core::tensor::Tensor;
use crate::formats::gguf::{
    QuantGGUFTensor,
    QuantPrefillKMetadata,
    QuantPrefillPackedLayout,
    QuantPrefillPackedLayoutKind,
    QuantPrefillQ8KInterleaveLayout,
    QuantPrefillStripeLayout,
    QuantQ4KPrefillMetadata,
    QuantQ6KPrefillMetadata,
};

use gemm::Parallelism;
use gguf::GGMLType;
use num_traits::Float;
use crate::runtime::threadpool;
use std::any::TypeId;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

/// 量化布局抽象：
/// - `block_size/qk` 描述块大小与块覆盖元素数；
/// - `decode_block_dot` 用于不落地 dense 的块点积；
/// - `decode_block_into` 用于需要行级缓存时的批量反量化。
pub trait QuantLayout {
    fn block_size() -> usize;
    fn qk() -> usize;
    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::F32
    }
    fn vec_dot_type() -> PrefillActivationKind {
        Self::prefill_activation_kind()
    }
    fn nrows() -> usize {
        1
    }
    fn supports_prefill_q8k_x4() -> bool {
        false
    }
    fn max_prefill_q8k_col_tile() -> usize {
        1
    }
    fn uses_q8k_activation() -> bool {
        false
    }
    fn supports_prefill_packed_q8k() -> bool {
        false
    }
    fn supports_interleaved_prefill_q8k() -> bool {
        Self::supports_prefill_packed_q8k()
    }
    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32;
    fn decode_block_dot_q80(_raw: &[u8], _a: &QuantQ80Block) -> f32 {
        unreachable!("当前量化布局未实现 Q8_0 激活点积")
    }
    fn decode_block_dot_q8k(_raw: &[u8], _a: &QuantQ8KBlock) -> f32 {
        unreachable!("当前量化布局未实现 Q8_K 激活点积")
    }
    fn accumulate_block_dot_q8k_x4(_raw: &[u8], _a: &QuantQ8KBlockX4, _out: &mut [f32; 4]) {
        unreachable!("当前量化布局未实现 Q8_K x4 微内核块点积")
    }
    fn decode_block_into(raw: &[u8], out: &mut [f32]);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefillActivationKind {
    F32,
    Q80,
    Q8K,
}

#[derive(Clone, Copy)]
struct QuantTypeTraits {
    vec_dot_type: PrefillActivationKind,
    nrows: usize,
    supports_prefill_q8k_x4: bool,
    max_prefill_q8k_col_tile: usize,
}

/// 收集指定量化布局的关键属性（点积类型、行数、x4 支持等），供调度逻辑使用
#[inline]
fn quant_type_traits<L: QuantLayout>() -> QuantTypeTraits {
    QuantTypeTraits {
        vec_dot_type: L::vec_dot_type(),
        nrows: L::nrows(),
        supports_prefill_q8k_x4: L::supports_prefill_q8k_x4(),
        max_prefill_q8k_col_tile: L::max_prefill_q8k_col_tile(),
    }
}

#[derive(Clone, Copy)]
enum PrefillQ8KKernelShape {
    X1,
    X4,
    X8,
}

impl PrefillQ8KKernelShape {
    fn col_tile(self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }

    fn blocklen(self) -> usize {
        self.col_tile()
    }
}

/// 选择 Q8K 微内核形状（x1/x4/x8），根据权重行数和列数自动决定
#[inline]
fn prefill_q8k_kernel_shape(n: usize, m: usize, traits: QuantTypeTraits) -> Option<PrefillQ8KKernelShape> {
    // 允许 m 不被 nrows 整除：对齐部分走 x4 微内核，尾部走单行回退
    if !traits.supports_prefill_q8k_x4 || traits.nrows == 0 || m < traits.nrows {
        return None;
    }
    Some(if traits.max_prefill_q8k_col_tile >= 8 && n >= 8 {
        PrefillQ8KKernelShape::X8
    } else if traits.max_prefill_q8k_col_tile >= 4 && n >= 4 {
        PrefillQ8KKernelShape::X4
    } else {
        PrefillQ8KKernelShape::X1
    })
}

/// 从字节切片小端读取 u16
#[inline]
fn read_u16_le(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}

/// 从字节切片小端读取 u32
#[inline]
fn read_u32_le(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

/// 从 Q4K scales 中提取指定子块的 scale 和 min
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// 从 12 字节编码中解包 Q3K 的 16 个 scale 值（含偏移减 32）
#[inline]
fn unpack_q3k_scales(scales12: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let mut aux = [0u32; 4];
    aux[0] = read_u32_le(scales12, 0);
    aux[1] = read_u32_le(scales12, 4);
    aux[2] = read_u32_le(scales12, 8);

    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

    let mut b = [0u8; 16];
    for i in 0..4 {
        let t = aux[i].to_le_bytes();
        b[i * 4..i * 4 + 4].copy_from_slice(&t);
    }

    let mut out = [0i8; 16];
    for i in 0..16 {
        out[i] = b[i] as i8 - 32;
    }
    out
}

/// f32 向量点积 SIMD 分发：自动选择 AVX512/FMA+AVX2/AVX2/NEON/标量路径
#[inline]
fn dot_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return unsafe { dot_f32_avx512(a, b) };
        }
        // FMA + AVX2: 使用 _mm256_fmadd_ps 将 mul+add 合并为单条指令
        if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_f32_avx2_fma(a, b) };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
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
        let vm = _mm512_mul_ps(va, vb);
        acc = _mm512_add_ps(acc, vm);
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
        let vm = _mm256_mul_ps(va, vb);
        acc = _mm256_add_ps(acc, vm);
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

/// FMA 加速的 f32 向量点积：使用 _mm256_fmadd_ps 将乘法和累加合并为单条指令，
/// 减少指令数并提高吞吐量。对 hot matrix cache 路径特别有效。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut i = 0usize;
    let n = a.len().min(b.len());
    let n8 = n / 8 * 8;
    // 使用两个累加器减少指令依赖，提高流水线利用率
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while i + 16 <= n8 {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        i += 16;
    }
    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc0);
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

struct Q40;
struct Q41;
struct Q50;
struct Q51;
struct Q80;
struct Q2K;
struct Q3K;
struct Q4K;
struct Q5K;
struct Q6K;

/// Q8_0 块与 f32 向量的点积（先反量化再调 SIMD 点积）
#[inline]
fn q80_decode_block_dot_simd(raw: &[u8], a: &[f32]) -> f32 {
    let mut vals = [0.0f32; 32];
    Q80::decode_block_into(raw, &mut vals);
    dot_f32_simd(a, &vals)
}

/// Q4K 块与 f32 向量的 legacy 点积（先反量化再 SIMD 点积）
#[inline]
fn q4k_decode_block_dot_legacy(raw: &[u8], a: &[f32]) -> f32 {
    let mut vals = [0.0f32; 256];
    Q4K::decode_block_into(raw, &mut vals);
    dot_f32_simd(a, &vals)
}

/// Q4K 块与 f32 向量的 direct 点积（不落地反量化，逐子块计算）
#[inline]
fn q4k_decode_block_dot_direct(raw: &[u8], a: &[f32]) -> f32 {
    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
    let scales = &raw[4..16];
    let qs = &raw[16..144];

    let mut sum = 0.0f32;
    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut a_off = 0usize;
    for _ in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let d2 = d * sc2 as f32;
        let m1f = dmin * m1 as f32;
        let m2f = dmin * m2 as f32;

        let mut vals1 = [0.0f32; 32];
        let mut vals2 = [0.0f32; 32];
        let mut sum_a1 = 0.0f32;
        let mut sum_a2 = 0.0f32;
        for l in 0..32 {
            let qv = qs[q_off + l];
            let a1 = a[a_off + l];
            let a2 = a[a_off + 32 + l];
            vals1[l] = (qv & 0x0f) as f32;
            sum_a1 += a1;
            vals2[l] = (qv >> 4) as f32;
            sum_a2 += a2;
        }

        sum += d1 * dot_f32_simd(&a[a_off..a_off + 32], &vals1) - m1f * sum_a1;
        sum += d2 * dot_f32_simd(&a[a_off + 32..a_off + 64], &vals2) - m2f * sum_a2;
        is += 2;
        q_off += 32;
        a_off += 64;
    }
    sum
}

/// Q6K 块与 f32 向量的 legacy 点积
#[inline]
fn q6k_decode_block_dot_legacy(raw: &[u8], a: &[f32]) -> f32 {
    let mut vals = [0.0f32; 256];
    Q6K::decode_block_into(raw, &mut vals);
    dot_f32_simd(a, &vals)
}

/// Q6K 块与 f32 向量的 direct 点积
#[inline]
fn q6k_decode_block_dot_direct(raw: &[u8], a: &[f32]) -> f32 {
    let mut ql = &raw[0..128];
    let mut qh = &raw[128..192];
    let mut sc = &raw[192..208];
    let d = half::f16::from_bits(read_u16_le(raw, 208)).to_f32();

    let mut sum = 0.0f32;
    let mut a_off = 0usize;
    for _ in (0..256).step_by(128) {
        let mut vals1 = [0.0f32; 32];
        let mut vals2 = [0.0f32; 32];
        let mut vals3 = [0.0f32; 32];
        let mut vals4 = [0.0f32; 32];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i8 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8 - 32;

            let s1 = d * (sc[is + 0] as i8) as f32;
            let s2 = d * (sc[is + 2] as i8) as f32;
            let s3 = d * (sc[is + 4] as i8) as f32;
            let s4 = d * (sc[is + 6] as i8) as f32;

            vals1[l] = s1 * q1 as f32;
            vals2[l] = s2 * q2 as f32;
            vals3[l] = s3 * q3 as f32;
            vals4[l] = s4 * q4 as f32;
        }
        sum += dot_f32_simd(&a[a_off..a_off + 32], &vals1);
        sum += dot_f32_simd(&a[a_off + 32..a_off + 64], &vals2);
        sum += dot_f32_simd(&a[a_off + 64..a_off + 96], &vals3);
        sum += dot_f32_simd(&a[a_off + 96..a_off + 128], &vals4);
        a_off += 128;
        ql = &ql[64..];
        qh = &qh[32..];
        sc = &sc[8..];
    }
    sum
}

/// llama.cpp 的 K-quant 主路径会先把激活量化成 `Q8_K`，
/// 再让 `Q4_K/Q6_K` 直接和 `Q8_K` 做点积。
///
/// 当前阶段先把这层基础设施补到 decode 主路径：
/// - 每个 256 元素激活块只量化一次；
/// - 保留 `bsums`，为后续更接近 llama.cpp 的 gemv/gemm micro-kernel 做准备；
/// - 先覆盖 `Q4_K/Q6_K`，因为它们是当前模型里最关键的 K-quant 类型。
#[derive(Clone)]
pub(crate) struct QuantQ80Block {
    d: f32,
    qs: [i8; 32],
    sum: i16,
}

/// 将 32 个 f32 激活值量化为 Q8_0 块
#[inline]
fn quantize_activation_block_q80(x: &[f32]) -> QuantQ80Block {
    debug_assert!(x.len() == 32);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let mut out = QuantQ80Block {
        d: 0.0,
        qs: [0; 32],
        sum: 0,
    };
    if amax == 0.0 {
        return out;
    }

    let iscale = 127.0f32 / amax;
    out.d = 1.0 / iscale;
    for (idx, &v) in x.iter().enumerate() {
        let q = (v * iscale).round() as i32;
        let q = q.clamp(-127, 127) as i8;
        out.qs[idx] = q;
        out.sum += q as i16;
    }
    out
}

/// Q4_0 块 × Q8_0 激活块的点积
#[inline]
fn q40_decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let qs = &raw[2..18];
    let mut dot = 0i32;
    for j in 0..16 {
        dot += (qs[j] & 0x0f) as i32 * a.qs[2 * j] as i32;
        dot += (qs[j] >> 4) as i32 * a.qs[2 * j + 1] as i32;
    }
    d * a.d * (dot as f32 - 8.0 * a.sum as f32)
}

/// Q4_1 块 × Q8_0 激活块的点积
#[inline]
fn q41_decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let m0 = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
    let qs = &raw[4..20];
    let mut dot = 0i32;
    for j in 0..16 {
        dot += (qs[j] & 0x0f) as i32 * a.qs[2 * j] as i32;
        dot += (qs[j] >> 4) as i32 * a.qs[2 * j + 1] as i32;
    }
    a.d * (d * dot as f32 + m0 * a.sum as f32)
}

/// Q5_0 块 × Q8_0 激活块的点积
#[inline]
fn q50_decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let qh = read_u32_le(raw, 2);
    let qs = &raw[6..22];
    let mut dot = 0i32;
    for j in 0..16 {
        let x0 = ((qs[j] & 0x0f) as i32) | ((((qh >> j) & 0x01) as i32) << 4);
        let x1 = ((qs[j] >> 4) as i32) | ((((qh >> (j + 16)) & 0x01) as i32) << 4);
        dot += x0 * a.qs[2 * j] as i32;
        dot += x1 * a.qs[2 * j + 1] as i32;
    }
    d * a.d * (dot as f32 - 16.0 * a.sum as f32)
}

/// Q5_1 块 × Q8_0 激活块的点积
#[inline]
fn q51_decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let m0 = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
    let qh = read_u32_le(raw, 4);
    let qs = &raw[8..24];
    let mut dot = 0i32;
    for j in 0..16 {
        let x0 = ((qs[j] & 0x0f) as i32) | ((((qh >> j) & 0x01) as i32) << 4);
        let x1 = ((qs[j] >> 4) as i32) | ((((qh >> (j + 16)) & 0x01) as i32) << 4);
        dot += x0 * a.qs[2 * j] as i32;
        dot += x1 * a.qs[2 * j + 1] as i32;
    }
    a.d * (d * dot as f32 + m0 * a.sum as f32)
}

/// Q8_0 块 × Q8_0 激活块的点积
#[inline]
fn q80_decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let mut dot = 0i32;
    for i in 0..32 {
        dot += i8::from_le_bytes([raw[2 + i]]) as i32 * a.qs[i] as i32;
    }
    d * a.d * dot as f32
}

#[derive(Clone)]
pub(crate) struct QuantQ8KBlock {
    d: f32,
    qs: [i8; 256],
    bsums: [i16; 16],
}

#[derive(Clone)]
pub(crate) struct QuantQ8KBlockX4 {
    d: [f32; 4],
    qs: [i8; 1024],
    bsums: [i16; 64],
}

/// 从 x4 块中按行号和元素索引读取量化值。
/// 采用非交错（行主序）布局：qs[row * 256 + idx]。
/// 这样每行的 256 个值是连续的，便于 AVX2 向量化读取。
#[inline]
fn q8k_x4_q(a: &QuantQ8KBlockX4, row: usize, idx: usize) -> i8 {
    a.qs[row * 256 + idx]
}

/// 将 4 个 QuantQ8KBlock 打包为一个 QuantQ8KBlockX4。
/// 采用非交错（行主序拼接）布局：
///   qs = [row0 的 256 字节 | row1 的 256 字节 | row2 的 256 字节 | row3 的 256 字节]
/// 优点：每行数据连续存储，AVX2 可直接 loadu 256 位加载 32 字节，
///       且可直接复用单行 AVX2 点积内核。
/// 将 4 行 Q8K 激活块打包为 x4 向量化块（行主序拼接）
#[inline]
fn pack_q8k_block_x4(rows: &[QuantQ8KBlock], _blocklen: usize) -> QuantQ8KBlockX4 {
    debug_assert!(rows.len() == 4);
    let mut out = QuantQ8KBlockX4 {
        d: [0.0; 4],
        qs: [0; 1024],
        bsums: [0; 64],
    };
    for row in 0..4 {
        out.d[row] = rows[row].d;
        // bsums 按行拼接：[row0: 16][row1: 16][row2: 16][row3: 16]
        for i in 0..16 {
            out.bsums[row * 16 + i] = rows[row].bsums[i];
        }
        // qs 按行拼接：每行 256 字节连续
        let dst_base = row * 256;
        out.qs[dst_base..dst_base + 256].copy_from_slice(&rows[row].qs);
    }
    out
}

/// 将多行 f32 激活值按 Q8K x4 格式打包（4 行一组）
#[inline]
fn pack_activation_panel_block_q8k_x4(
    a_rows_f32: &[f32],
    m: usize,
    k: usize,
    col_start: usize,
    block_cnt: usize,
    blocklen: usize,
) -> Vec<QuantQ8KBlockX4> {
    debug_assert!(m % 4 == 0);
    let mut out = Vec::with_capacity((m / 4) * block_cnt);
    let mut scratch = Vec::with_capacity(4);
    for row_group in 0..(m / 4) {
        let row_start = row_group * 4;
        for blk in 0..block_cnt {
            scratch.clear();
            for row in 0..4 {
                let base = (row_start + row) * k + col_start + blk * 256;
                scratch.push(quantize_activation_block_q8k(&a_rows_f32[base..base + 256]));
            }
            out.push(pack_q8k_block_x4(&scratch, blocklen));
        }
    }
    out
}

#[inline]
/// 将 256 个 f32 激活值量化为 Q8K 块。
/// 有 AVX2+FMA 时走 SIMD 路径，否则用标量回退。
fn quantize_activation_block_q8k(x: &[f32]) -> QuantQ8KBlock {
    debug_assert!(x.len() == 256);

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { quantize_activation_block_q8k_avx2(x) };
        }
    }
    quantize_activation_block_q8k_scalar(x)
}

/// Q8K 量化标量路径。
fn quantize_activation_block_q8k_scalar(x: &[f32]) -> QuantQ8KBlock {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let av = v.abs();
        if av > amax {
            amax = av;
            max = v;
        }
    }

    let mut out = QuantQ8KBlock {
        d: 0.0,
        qs: [0; 256],
        bsums: [0; 16],
    };
    if amax == 0.0 {
        return out;
    }

    let iscale = -127.0f32 / max;
    out.d = 1.0 / iscale;
    for (idx, &v) in x.iter().enumerate() {
        let q = (v * iscale).round() as i32;
        let q = q.clamp(-127, 127) as i8;
        out.qs[idx] = q;
        out.bsums[idx / 16] += q as i16;
    }
    out
}

/// Q8K 量化 AVX2 加速路径：使用 SIMD 加速 max 查找和量化循环。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_activation_block_q8k_avx2(x: &[f32]) -> QuantQ8KBlock {
    use std::arch::x86_64::*;

    // 第一步：用 AVX2 找最大绝对值
    let mut max_abs_vec = _mm256_setzero_ps();
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF_u32 as i32));
    for i in (0..256).step_by(8) {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let abs_v = _mm256_and_ps(v, sign_mask);
        max_abs_vec = _mm256_max_ps(max_abs_vec, abs_v);
    }
    // 水平归约最大值
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), max_abs_vec);
    let amax = tmp.iter().cloned().fold(0.0f32, f32::max);

    let mut out = QuantQ8KBlock {
        d: 0.0,
        qs: [0; 256],
        bsums: [0; 16],
    };
    if amax == 0.0 {
        return out;
    }

    // 找到产生 amax 的原始值（带符号）
    let mut max = 0.0f32;
    for &v in x {
        if v.abs() >= amax {
            max = v;
            break;
        }
    }

    let iscale = -127.0f32 / max;
    out.d = 1.0 / iscale;

    // 第二步：用 AVX2 向量化量化循环
    let vscale = _mm256_set1_ps(iscale);
    let vmin = _mm256_set1_ps(-127.0);
    let vmax = _mm256_set1_ps(127.0);

    for blk16 in 0..16 {
        let base = blk16 * 16;
        let mut bsum = 0i16;

        // 处理 16 个元素（2 组 × 8 个 AVX2 向量）
        for sub in 0..2 {
            let off = base + sub * 8;
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let scaled = _mm256_mul_ps(v, vscale);
            let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let clamped = _mm256_min_ps(_mm256_max_ps(rounded, vmin), vmax);
            let ints = _mm256_cvtps_epi32(clamped);

            // 提取 8 个 i32 并转为 i8
            let mut vals = [0i32; 8];
            _mm256_storeu_si256(vals.as_mut_ptr() as *mut __m256i, ints);
            for j in 0..8 {
                let q = vals[j] as i8;
                out.qs[off + j] = q;
                bsum += q as i16;
            }
        }
        out.bsums[blk16] = bsum;
    }
    out
}

/// 将一行 f32 激活值量化为 Q8K 块序列
#[inline]
fn quantize_activation_row_q8k(a: &[f32]) -> Vec<QuantQ8KBlock> {
    debug_assert!(a.len() % 256 == 0);
    let mut out = Vec::with_capacity(a.len() / 256);
    for blk in 0..a.len() / 256 {
        let base = blk * 256;
        out.push(quantize_activation_block_q8k(&a[base..base + 256]));
    }
    out
}

/// Q4K 块 × Q8K 激活块的点积（自动选择 AVX2 或标量路径）
#[inline]
fn q4k_decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: 运行时已检测到 avx2 支持。
            return unsafe { q4k_decode_block_dot_q8k_avx2(raw, a) };
        }
    }
    q4k_decode_block_dot_q8k_scalar(raw, a)
}

/// Q4K × Q8K 点积的标量回退路径。
#[inline]
fn q4k_decode_block_dot_q8k_scalar(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
    let scales = &raw[4..16];
    let qs = &raw[16..144];

    let mut sum = 0.0f32;
    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut a_off = 0usize;
    for sub in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let d2 = d * sc2 as f32;
        let m1f = dmin * m1 as f32;
        let m2f = dmin * m2 as f32;

        let mut dot1 = 0i32;
        let mut dot2 = 0i32;
        for l in 0..32 {
            let qv = qs[q_off + l];
            dot1 += (qv & 0x0f) as i32 * a.qs[a_off + l] as i32;
            dot2 += (qv >> 4) as i32 * a.qs[a_off + 32 + l] as i32;
        }

        let sum1 = a.bsums[sub * 4 + 0] as i32 + a.bsums[sub * 4 + 1] as i32;
        let sum2 = a.bsums[sub * 4 + 2] as i32 + a.bsums[sub * 4 + 3] as i32;
        sum += a.d * (d1 * dot1 as f32 - m1f * sum1 as f32);
        sum += a.d * (d2 * dot2 as f32 - m2f * sum2 as f32);

        is += 2;
        q_off += 32;
        a_off += 64;
    }
    sum
}

/// Q4K × Q8K 点积 AVX2 加速版。
/// 对照 llama.cpp 的 `ggml_vec_dot_q4_K_q8_K` AVX2 路径实现。
/// 每个 sub-block（32 字节 Q4K × 64 字节 Q8K）用 256-bit SIMD 一次性处理：
/// - `_mm256_maddubs_epi16`：无符号 × 有符号字节乘加到 i16
/// - `_mm256_madd_epi16`：i16 对累加到 i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_decode_block_dot_q8k_avx2(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
    use std::arch::x86_64::*;

    if a.d == 0.0 {
        return 0.0;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
    let scales = &raw[4..16];
    let qs = &raw[16..144];

    let low_mask = _mm256_set1_epi8(0x0f);
    let ones_16 = _mm256_set1_epi16(1);

    let mut sum = 0.0f32;
    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut a_off = 0usize;

    for sub in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);

        // 加载 32 字节 Q4K 量化值
        let q_raw = _mm256_loadu_si256(qs.as_ptr().add(q_off) as *const __m256i);
        // 低 4 位 = 前半 sub-block
        let q_lo = _mm256_and_si256(q_raw, low_mask);
        // 高 4 位 = 后半 sub-block
        let q_hi = _mm256_srli_epi16(q_raw, 4);
        let q_hi = _mm256_and_si256(q_hi, low_mask);

        // 加载 64 字节 Q8K 激活（分两个 32 字节块）
        let a_lo = _mm256_loadu_si256(a.qs.as_ptr().add(a_off) as *const __m256i);
        let a_hi = _mm256_loadu_si256(a.qs.as_ptr().add(a_off + 32) as *const __m256i);

        // _mm256_maddubs_epi16：第一个操作数为无符号字节，第二个为有符号字节
        // q_lo (0..15 无符号) × a_lo (i8) → i16 对累加
        let dot_lo = _mm256_maddubs_epi16(q_lo, a_lo);
        let dot_hi = _mm256_maddubs_epi16(q_hi, a_hi);

        // _mm256_madd_epi16：i16 对累加到 i32
        let isum_lo = _mm256_madd_epi16(dot_lo, ones_16);
        let isum_hi = _mm256_madd_epi16(dot_hi, ones_16);

        // 水平归约 i32 → 标量
        let dot1 = hsum_i32_avx2(isum_lo);
        let dot2 = hsum_i32_avx2(isum_hi);

        let sum1 = a.bsums[sub * 4] as i32 + a.bsums[sub * 4 + 1] as i32;
        let sum2 = a.bsums[sub * 4 + 2] as i32 + a.bsums[sub * 4 + 3] as i32;

        sum += a.d * (d * sc1 as f32 * dot1 as f32 - dmin * m1 as f32 * sum1 as f32)
             + a.d * (d * sc2 as f32 * dot2 as f32 - dmin * m2 as f32 * sum2 as f32);

        is += 2;
        q_off += 32;
        a_off += 64;
    }

    sum
}

/// AVX2 水平归约 8 × i32 → 单个 i32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    // 把高 128 位加到低 128 位
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);
    // 水平归约 4 × i32
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0x01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

/// Q6K 块 × Q8K 激活块的点积（自动选择 AVX2 或标量路径）
#[inline]
fn q6k_decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { q6k_decode_block_dot_q8k_avx2(raw, a) };
        }
    }
    q6k_decode_block_dot_q8k_scalar(raw, a)
}

/// Q6K × Q8K 点积的标量回退路径。
#[inline]
fn q6k_decode_block_dot_q8k_scalar(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let mut ql = &raw[0..128];
    let mut qh = &raw[128..192];
    let mut sc = &raw[192..208];
    let d = half::f16::from_bits(read_u16_le(raw, 208)).to_f32();

    let mut sum = 0.0f32;
    let mut a_off = 0usize;
    for _ in (0..256).step_by(128) {
        let mut dot1 = 0i32;
        let mut dot2 = 0i32;
        let mut dot3 = 0i32;
        let mut dot4 = 0i32;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32;

            dot1 += q1 * a.qs[a_off + l + 0] as i32 * (sc[is + 0] as i8 as i32);
            dot2 += q2 * a.qs[a_off + l + 32] as i32 * (sc[is + 2] as i8 as i32);
            dot3 += q3 * a.qs[a_off + l + 64] as i32 * (sc[is + 4] as i8 as i32);
            dot4 += q4 * a.qs[a_off + l + 96] as i32 * (sc[is + 6] as i8 as i32);
        }
        sum += a.d * d * (dot1 + dot2 + dot3 + dot4) as f32;
        a_off += 128;
        ql = &ql[64..];
        qh = &qh[32..];
        sc = &sc[8..];
    }
    sum
}

/// Q6K × Q8K 点积 AVX2 加速版。
/// 对照 llama.cpp 的 `ggml_vec_dot_q6_K_q8_K` 实现。
/// Q6K 布局：128 字节 ql + 64 字节 qh + 16 字节 sc + 2 字节 d。
/// 每 128 个元素（4 组 × 32 元素）一轮，6-bit 权重通过
/// ql[l] 低/高 4 位 + qh[l] 的 2 位拼成。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q6k_decode_block_dot_q8k_avx2(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
    use std::arch::x86_64::*;

    if a.d == 0.0 {
        return 0.0;
    }

    let ql_all = &raw[0..128];
    let qh_all = &raw[128..192];
    let sc_all = &raw[192..208];
    let d = half::f16::from_bits(read_u16_le(raw, 208)).to_f32();

    let low_mask = _mm256_set1_epi8(0x0f);
    let m32 = _mm256_set1_epi8(32);

    let mut sum = 0.0f32;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    let mut a_off = 0usize;

    for _ in 0..2 {
        // 加载 ql: 64 字节分两个 32-byte load
        let ql_0 = _mm256_loadu_si256(ql_all.as_ptr().add(ql_off) as *const __m256i);
        let ql_1 = _mm256_loadu_si256(ql_all.as_ptr().add(ql_off + 32) as *const __m256i);
        // 加载 qh: 32 字节
        let qh = _mm256_loadu_si256(qh_all.as_ptr().add(qh_off) as *const __m256i);

        // 组合 6-bit 量化值：
        // q1 = (ql_0 & 0xf) | ((qh & 0x03) << 4)
        let q1 = _mm256_or_si256(
            _mm256_and_si256(ql_0, low_mask),
            _mm256_slli_epi16(_mm256_and_si256(qh, _mm256_set1_epi8(0x03)), 4),
        );
        // q2 = (ql_1 & 0xf) | (((qh >> 2) & 0x03) << 4)
        let q2 = _mm256_or_si256(
            _mm256_and_si256(ql_1, low_mask),
            _mm256_slli_epi16(
                _mm256_and_si256(_mm256_srli_epi16(qh, 2), _mm256_set1_epi8(0x03)),
                4,
            ),
        );
        // q3 = (ql_0 >> 4) | (((qh >> 4) & 0x03) << 4)
        let q3 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_0, 4), low_mask),
            _mm256_slli_epi16(
                _mm256_and_si256(_mm256_srli_epi16(qh, 4), _mm256_set1_epi8(0x03)),
                4,
            ),
        );
        // q4 = (ql_1 >> 4) | (((qh >> 6) & 0x03) << 4)
        let q4 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_1, 4), low_mask),
            _mm256_slli_epi16(
                _mm256_and_si256(_mm256_srli_epi16(qh, 6), _mm256_set1_epi8(0x03)),
                4,
            ),
        );

        // 减去偏移 32 得到有符号值
        let q1s = _mm256_sub_epi8(q1, m32);
        let q2s = _mm256_sub_epi8(q2, m32);
        let q3s = _mm256_sub_epi8(q3, m32);
        let q4s = _mm256_sub_epi8(q4, m32);

        // 加载 Q8K 激活
        let a1 = _mm256_loadu_si256(a.qs.as_ptr().add(a_off) as *const __m256i);
        let a2 = _mm256_loadu_si256(a.qs.as_ptr().add(a_off + 32) as *const __m256i);
        let a3 = _mm256_loadu_si256(a.qs.as_ptr().add(a_off + 64) as *const __m256i);
        let a4 = _mm256_loadu_si256(a.qs.as_ptr().add(a_off + 96) as *const __m256i);

        // Q6K 的 scale 是 per-16-element 的，需要把 32 字节点积分成
        // 低 16 字节和高 16 字节两部分，分别乘以对应的 scale。
        let sc0 = sc_all[sc_off] as i8 as i32;
        let sc1 = sc_all[sc_off + 1] as i8 as i32;
        let sc2 = sc_all[sc_off + 2] as i8 as i32;
        let sc3 = sc_all[sc_off + 3] as i8 as i32;
        let sc4 = sc_all[sc_off + 4] as i8 as i32;
        let sc5 = sc_all[sc_off + 5] as i8 as i32;
        let sc6 = sc_all[sc_off + 6] as i8 as i32;
        let sc7 = sc_all[sc_off + 7] as i8 as i32;

        // 每组 32 字节 SIMD 点积，拆分成低/高 16 字节各自归约
        let (dot1_lo, dot1_hi) = split_hsum_i32_lo_hi_avx2(q1s, a1);
        let (dot2_lo, dot2_hi) = split_hsum_i32_lo_hi_avx2(q2s, a2);
        let (dot3_lo, dot3_hi) = split_hsum_i32_lo_hi_avx2(q3s, a3);
        let (dot4_lo, dot4_hi) = split_hsum_i32_lo_hi_avx2(q4s, a4);

        let total = (dot1_lo * sc0 + dot1_hi * sc1)
                  + (dot2_lo * sc2 + dot2_hi * sc3)
                  + (dot3_lo * sc4 + dot3_hi * sc5)
                  + (dot4_lo * sc6 + dot4_hi * sc7);

        sum += a.d * d * total as f32;

        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
        a_off += 128;
    }
    sum
}

/// Q6K 用的 SIMD 辅助：返回 32 字节 i8×i8 点积的低 16 元素和与高 16 元素和。
/// 用于应用 per-16-element scale。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn split_hsum_i32_lo_hi_avx2(
    w: std::arch::x86_64::__m256i,
    a: std::arch::x86_64::__m256i,
) -> (i32, i32) {
    use std::arch::x86_64::*;

    // 分拆 i8×i8 为无符号×有符号：
    // 通过偏移把 w 的有符号值变成无符号：w_u = w + 128，然后 maddubs，再减去 128 * sum(a)
    let m128 = _mm256_set1_epi8(-128i8);
    let w_u = _mm256_sub_epi8(w, m128); // w + 128 (有符号减 -128 = 加 128)
    let ones_16 = _mm256_set1_epi16(1);

    let prod = _mm256_maddubs_epi16(w_u, a); // (w+128) * a → i16
    let isum = _mm256_madd_epi16(prod, ones_16); // → 8 × i32

    // 修正项：128 * sum(a_lo_16) 和 128 * sum(a_hi_16)
    // 先算 a 的绝对求和：把 a 当作有符号字节，求 16 元素的 i16 和
    let a_extend = _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_set1_epi8(1), a),
        ones_16,
    );

    let corrected = _mm256_sub_epi32(isum, _mm256_slli_epi32(a_extend, 7));

    // 低 128 位是前 16 个字节的结果，高 128 位是后 16 个字节
    let lo128 = _mm256_castsi256_si128(corrected);
    let hi128 = _mm256_extracti128_si256(corrected, 1);

    // 归约各自的 4 × i32
    let lo_sum = hsum_i32_sse(lo128);
    let hi_sum = hsum_i32_sse(hi128);

    (lo_sum, hi_sum)
}

/// SSE 水平归约 4 × i32 → 单个 i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32_sse(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::*;
    let hi64 = _mm_unpackhi_epi64(v, v);
    let sum64 = _mm_add_epi32(v, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0x01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

/// Q2K 块 × Q8K 激活块的点积
#[inline]
fn q2k_decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let scales = &raw[0..16];
    let mut qp = &raw[16..80];
    let d = half::f16::from_bits(read_u16_le(raw, 80)).to_f32();
    let dmin = half::f16::from_bits(read_u16_le(raw, 82)).to_f32();

    let mut sum = 0.0f32;
    let mut is = 0usize;
    let mut a_off = 0usize;
    for _ in (0..256).step_by(128) {
        let mut shift = 0usize;
        for _ in 0..4 {
            let sc0 = scales[is];
            is += 1;
            let dl0 = d * (sc0 & 0x0f) as f32;
            let ml0 = dmin * (sc0 >> 4) as f32;
            let mut dot0 = 0i32;
            for l in 0..16 {
                dot0 += ((qp[l] >> shift) & 0x03) as i32 * a.qs[a_off + l] as i32;
            }
            sum += a.d * (dl0 * dot0 as f32 - ml0 * a.bsums[a_off / 16] as f32);
            a_off += 16;

            let sc1 = scales[is];
            is += 1;
            let dl1 = d * (sc1 & 0x0f) as f32;
            let ml1 = dmin * (sc1 >> 4) as f32;
            let mut dot1 = 0i32;
            for l in 0..16 {
                dot1 += ((qp[l + 16] >> shift) & 0x03) as i32 * a.qs[a_off + l] as i32;
            }
            sum += a.d * (dl1 * dot1 as f32 - ml1 * a.bsums[a_off / 16] as f32);
            a_off += 16;
            shift += 2;
        }
        qp = &qp[32..];
    }
    sum
}

/// Q3K 块 × Q8K 激活块的点积
#[inline]
fn q3k_decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let hm = &raw[0..32];
    let mut q = &raw[32..96];
    let scales = unpack_q3k_scales(&raw[96..108]);
    let d_all = half::f16::from_bits(read_u16_le(raw, 108)).to_f32();

    let mut sum = 0.0f32;
    let mut is = 0usize;
    let mut mbit: u8 = 1;
    let mut a_off = 0usize;
    for _ in (0..256).step_by(128) {
        let mut shift = 0usize;
        for _ in 0..4 {
            let dl0 = d_all * scales[is] as f32;
            is += 1;
            let mut dot0 = 0i32;
            for l in 0..16 {
                let lo = ((q[l] >> shift) & 0x03) as i32;
                let hi = if (hm[l] & mbit) != 0 { 0 } else { 4 };
                dot0 += (lo - hi) * a.qs[a_off + l] as i32;
            }
            sum += a.d * dl0 * dot0 as f32;
            a_off += 16;

            let dl1 = d_all * scales[is] as f32;
            is += 1;
            let mut dot1 = 0i32;
            for l in 0..16 {
                let lo = ((q[l + 16] >> shift) & 0x03) as i32;
                let hi = if (hm[l + 16] & mbit) != 0 { 0 } else { 4 };
                dot1 += (lo - hi) * a.qs[a_off + l] as i32;
            }
            sum += a.d * dl1 * dot1 as f32;
            a_off += 16;

            shift += 2;
            mbit <<= 1;
        }
        q = &q[32..];
    }
    sum
}

/// Q5K 块 × Q8K 激活块的点积
#[inline]
fn q5k_decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
    if a.d == 0.0 {
        return 0.0;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
    let scales = &raw[4..16];
    let qh = &raw[16..48];
    let mut ql = &raw[48..176];

    let mut sum = 0.0f32;
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    let mut a_off = 0usize;
    for _ in (0..256).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        is += 1;
        let (sc2, m2) = get_scale_min_k4(is, scales);
        is += 1;
        let d1 = d * sc1 as f32;
        let d2 = d * sc2 as f32;
        let mm1 = dmin * m1 as f32;
        let mm2 = dmin * m2 as f32;

        let mut dot1 = 0i32;
        for l in 0..32 {
            let v = (ql[l] & 0x0f) as i32 + if (qh[l] & u1) != 0 { 16 } else { 0 };
            dot1 += v * a.qs[a_off + l] as i32;
        }
        let sum1 = a.bsums[a_off / 16] as i32 + a.bsums[a_off / 16 + 1] as i32;
        sum += a.d * (d1 * dot1 as f32 - mm1 * sum1 as f32);
        a_off += 32;

        let mut dot2 = 0i32;
        for l in 0..32 {
            let v = (ql[l] >> 4) as i32 + if (qh[l] & u2) != 0 { 16 } else { 0 };
            dot2 += v * a.qs[a_off + l] as i32;
        }
        let sum2 = a.bsums[a_off / 16] as i32 + a.bsums[a_off / 16 + 1] as i32;
        sum += a.d * (d2 * dot2 as f32 - mm2 * sum2 as f32);
        a_off += 32;

        ql = &ql[32..];
        u1 <<= 2;
        u2 <<= 2;
    }
    sum
}

/// Q2K 块 × Q8K x4 激活块的 4 路累加点积
#[inline]
fn q2k_accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
    if a.d.iter().all(|&d| d == 0.0) {
        return;
    }

    let scales = &raw[0..16];
    let mut qp = &raw[16..80];
    let d = half::f16::from_bits(read_u16_le(raw, 80)).to_f32();
    let dmin = half::f16::from_bits(read_u16_le(raw, 82)).to_f32();

    let mut is = 0usize;
    let mut a_off = 0usize;
    for _ in (0..256).step_by(128) {
        let mut shift = 0usize;
        for _ in 0..4 {
            let sc0 = scales[is];
            is += 1;
            let dl0 = d * (sc0 & 0x0f) as f32;
            let ml0 = dmin * (sc0 >> 4) as f32;
            let mut dot0 = [0i32; 4];
            for l in 0..16 {
                let q = ((qp[l] >> shift) & 0x03) as i32;
                for row in 0..4 {
                    dot0[row] += q * q8k_x4_q(a, row, a_off + l) as i32;
                }
            }
            for row in 0..4 {
                let bsums_idx = row * 16 + a_off / 16;
                out[row] += a.d[row] * (dl0 * dot0[row] as f32 - ml0 * a.bsums[bsums_idx] as f32);
            }
            a_off += 16;

            let sc1 = scales[is];
            is += 1;
            let dl1 = d * (sc1 & 0x0f) as f32;
            let ml1 = dmin * (sc1 >> 4) as f32;
            let mut dot1 = [0i32; 4];
            for l in 0..16 {
                let q = ((qp[l + 16] >> shift) & 0x03) as i32;
                for row in 0..4 {
                    dot1[row] += q * q8k_x4_q(a, row, a_off + l) as i32;
                }
            }
            for row in 0..4 {
                let bsums_idx = row * 16 + a_off / 16;
                out[row] += a.d[row] * (dl1 * dot1[row] as f32 - ml1 * a.bsums[bsums_idx] as f32);
            }
            a_off += 16;
            shift += 2;
        }
        qp = &qp[32..];
    }
}

/// Q3K 块 × Q8K x4 激活块的 4 路累加点积
#[inline]
fn q3k_accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
    if a.d.iter().all(|&d| d == 0.0) {
        return;
    }

    let hm = &raw[0..32];
    let mut q = &raw[32..96];
    let scales = unpack_q3k_scales(&raw[96..108]);
    let d_all = half::f16::from_bits(read_u16_le(raw, 108)).to_f32();

    let mut is = 0usize;
    let mut mbit: u8 = 1;
    let mut a_off = 0usize;
    for _ in (0..256).step_by(128) {
        let mut shift = 0usize;
        for _ in 0..4 {
            let dl0 = d_all * scales[is] as f32;
            is += 1;
            let mut dot0 = [0i32; 4];
            for l in 0..16 {
                let lo = ((q[l] >> shift) & 0x03) as i32;
                let hi = if (hm[l] & mbit) != 0 { 0 } else { 4 };
                let v = lo - hi;
                for row in 0..4 {
                    dot0[row] += v * q8k_x4_q(a, row, a_off + l) as i32;
                }
            }
            for row in 0..4 {
                out[row] += a.d[row] * dl0 * dot0[row] as f32;
            }
            a_off += 16;

            let dl1 = d_all * scales[is] as f32;
            is += 1;
            let mut dot1 = [0i32; 4];
            for l in 0..16 {
                let lo = ((q[l + 16] >> shift) & 0x03) as i32;
                let hi = if (hm[l + 16] & mbit) != 0 { 0 } else { 4 };
                let v = lo - hi;
                for row in 0..4 {
                    dot1[row] += v * q8k_x4_q(a, row, a_off + l) as i32;
                }
            }
            for row in 0..4 {
                out[row] += a.d[row] * dl1 * dot1[row] as f32;
            }
            a_off += 16;

            shift += 2;
            mbit <<= 1;
        }
        q = &q[32..];
    }
}

/// Q5K 块 × Q8K x4 激活块的 4 路累加点积
#[inline]
fn q5k_accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
    if a.d.iter().all(|&d| d == 0.0) {
        return;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
    let scales = &raw[4..16];
    let qh = &raw[16..48];
    let mut ql = &raw[48..176];

    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    let mut a_off = 0usize;
    for _ in (0..256).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        is += 1;
        let (sc2, m2) = get_scale_min_k4(is, scales);
        is += 1;
        let d1 = d * sc1 as f32;
        let d2 = d * sc2 as f32;
        let mm1 = dmin * m1 as f32;
        let mm2 = dmin * m2 as f32;

        let mut dot1 = [0i32; 4];
        for l in 0..32 {
            let v = (ql[l] & 0x0f) as i32 + if (qh[l] & u1) != 0 { 16 } else { 0 };
            for row in 0..4 {
                dot1[row] += v * q8k_x4_q(a, row, a_off + l) as i32;
            }
        }
        for row in 0..4 {
            let base = row * 16 + a_off / 16;
            let sum1 = a.bsums[base] as i32 + a.bsums[base + 1] as i32;
            out[row] += a.d[row] * (d1 * dot1[row] as f32 - mm1 * sum1 as f32);
        }
        a_off += 32;

        let mut dot2 = [0i32; 4];
        for l in 0..32 {
            let v = (ql[l] >> 4) as i32 + if (qh[l] & u2) != 0 { 16 } else { 0 };
            for row in 0..4 {
                dot2[row] += v * q8k_x4_q(a, row, a_off + l) as i32;
            }
        }
        for row in 0..4 {
            let base = row * 16 + a_off / 16;
            let sum2 = a.bsums[base] as i32 + a.bsums[base + 1] as i32;
            out[row] += a.d[row] * (d2 * dot2[row] as f32 - mm2 * sum2 as f32);
        }
        a_off += 32;

        ql = &ql[32..];
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Q4K × Q8K x4 累加：对 4 行激活分别执行 Q4K 点积并累加到 out。
/// 非交错布局下每行 256 字节连续，可直接复用单行 AVX2 内核。
/// Q4K 块 × Q8K x4 激活块的 4 路累加点积（自动选择 AVX2 或标量）
#[inline]
fn q4k_accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { q4k_accumulate_block_dot_q8k_x4_avx2(raw, a, out) };
            return;
        }
    }
    q4k_accumulate_block_dot_q8k_x4_scalar(raw, a, out);
}

/// Q4K × Q8K x4 标量回退路径（非交错布局）。
#[inline]
fn q4k_accumulate_block_dot_q8k_x4_scalar(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
    if a.d.iter().all(|&d| d == 0.0) {
        return;
    }

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
    let scales = &raw[4..16];
    let qs = &raw[16..144];

    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut a_off = 0usize;
    for sub in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let d2 = d * sc2 as f32;
        let m1f = dmin * m1 as f32;
        let m2f = dmin * m2 as f32;

        let mut dot1 = [0i32; 4];
        let mut dot2 = [0i32; 4];
        for l in 0..32 {
            let qv = qs[q_off + l];
            let lo = (qv & 0x0f) as i32;
            let hi = (qv >> 4) as i32;
            for row in 0..4 {
                dot1[row] += lo * q8k_x4_q(a, row, a_off + l) as i32;
                dot2[row] += hi * q8k_x4_q(a, row, a_off + 32 + l) as i32;
            }
        }

        for row in 0..4 {
            let base = row * 16 + sub * 4;
            let sum1 = a.bsums[base + 0] as i32 + a.bsums[base + 1] as i32;
            let sum2 = a.bsums[base + 2] as i32 + a.bsums[base + 3] as i32;
            out[row] += a.d[row] * (d1 * dot1[row] as f32 - m1f * sum1 as f32);
            out[row] += a.d[row] * (d2 * dot2[row] as f32 - m2f * sum2 as f32);
        }

        is += 2;
        q_off += 32;
        a_off += 64;
    }
}

/// Q4K × Q8K x4 AVX2 加速版。
/// 非交错布局下，每行 qs 数据连续存储在 a.qs[row*256..(row+1)*256]，
/// 可直接复用单行 AVX2 的 maddubs/madd 逻辑。
/// 权重数据只解析一次，4 行激活分别执行 SIMD 点积。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_accumulate_block_dot_q8k_x4_avx2(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
    use std::arch::x86_64::*;

    let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
    let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
    let scales = &raw[4..16];
    let qs = &raw[16..144];

    let low_mask = _mm256_set1_epi8(0x0f);
    let ones_16 = _mm256_set1_epi16(1);

    // 逐行处理：权重只加载一次，4 行激活分别做 SIMD 点积
    for row in 0..4 {
        if a.d[row] == 0.0 {
            continue;
        }
        let a_qs_base = row * 256; // 非交错布局：每行 256 字节连续
        let a_bsums_base = row * 16;
        let mut sum = 0.0f32;
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut a_off = 0usize;

        for sub in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

            // 加载 32 字节 Q4K 权重
            let q_raw = _mm256_loadu_si256(qs.as_ptr().add(q_off) as *const __m256i);
            let q_lo = _mm256_and_si256(q_raw, low_mask);
            let q_hi = _mm256_and_si256(_mm256_srli_epi16(q_raw, 4), low_mask);

            // 加载该行对应的 64 字节 Q8K 激活
            let a_lo = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off) as *const __m256i);
            let a_hi = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off + 32) as *const __m256i);

            // maddubs + madd → i32 累加
            let dot_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(q_lo, a_lo), ones_16);
            let dot_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(q_hi, a_hi), ones_16);

            let isum_lo = hsum_i32_avx2(dot_lo);
            let isum_hi = hsum_i32_avx2(dot_hi);

            let sum1 = a.bsums[a_bsums_base + sub * 4] as i32
                + a.bsums[a_bsums_base + sub * 4 + 1] as i32;
            let sum2 = a.bsums[a_bsums_base + sub * 4 + 2] as i32
                + a.bsums[a_bsums_base + sub * 4 + 3] as i32;

            sum += a.d[row]
                * (d * sc1 as f32 * isum_lo as f32 - dmin * m1 as f32 * sum1 as f32)
                + a.d[row]
                    * (d * sc2 as f32 * isum_hi as f32 - dmin * m2 as f32 * sum2 as f32);

            is += 2;
            q_off += 32;
            a_off += 64;
        }
        out[row] += sum;
    }
}

/// Q4K × Q8K x4 累加 (meta 预提取版)：使用预解析的 d/dmin/scales/mins 元数据。
/// 非交错布局下每行连续，有 AVX2 时走 SIMD 路径。
#[inline]
fn q4k_accumulate_block_dot_q8k_x4_meta(
    raw: &[u8],
    meta: &QuantQ4KPrefillMetadata,
    linear_idx: usize,
    a: &QuantQ8KBlockX4,
    out: &mut [f32; 4],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { q4k_accumulate_block_dot_q8k_x4_meta_avx2(raw, meta, linear_idx, a, out) };
            return;
        }
    }
    q4k_accumulate_block_dot_q8k_x4_meta_scalar(raw, meta, linear_idx, a, out);
}

/// Q4K × Q8K x4 meta 标量回退路径。
#[inline]
fn q4k_accumulate_block_dot_q8k_x4_meta_scalar(
    raw: &[u8],
    meta: &QuantQ4KPrefillMetadata,
    linear_idx: usize,
    a: &QuantQ8KBlockX4,
    out: &mut [f32; 4],
) {
    if a.d.iter().all(|&d| d == 0.0) {
        return;
    }

    let qs = &raw[16..144];
    let d = meta.d[linear_idx];
    let dmin = meta.dmin[linear_idx];
    let meta_off = linear_idx * 8;

    let mut q_off = 0usize;
    let mut a_off = 0usize;
    for sub in 0..4 {
        let sc1 = meta.scales[meta_off + sub * 2] as f32;
        let m1 = meta.mins[meta_off + sub * 2] as f32;
        let sc2 = meta.scales[meta_off + sub * 2 + 1] as f32;
        let m2 = meta.mins[meta_off + sub * 2 + 1] as f32;
        let d1 = d * sc1;
        let d2 = d * sc2;
        let m1f = dmin * m1;
        let m2f = dmin * m2;

        let mut dot1 = [0i32; 4];
        let mut dot2 = [0i32; 4];
        for l in 0..32 {
            let qv = qs[q_off + l];
            let lo = (qv & 0x0f) as i32;
            let hi = (qv >> 4) as i32;
            for row in 0..4 {
                dot1[row] += lo * q8k_x4_q(a, row, a_off + l) as i32;
                dot2[row] += hi * q8k_x4_q(a, row, a_off + 32 + l) as i32;
            }
        }

        for row in 0..4 {
            let base = row * 16 + sub * 4;
            let sum1 = a.bsums[base] as i32 + a.bsums[base + 1] as i32;
            let sum2 = a.bsums[base + 2] as i32 + a.bsums[base + 3] as i32;
            out[row] += a.d[row] * (d1 * dot1[row] as f32 - m1f * sum1 as f32);
            out[row] += a.d[row] * (d2 * dot2[row] as f32 - m2f * sum2 as f32);
        }

        q_off += 32;
        a_off += 64;
    }
}

/// Q4K × Q8K x4 meta AVX2 加速版。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_accumulate_block_dot_q8k_x4_meta_avx2(
    raw: &[u8],
    meta: &QuantQ4KPrefillMetadata,
    linear_idx: usize,
    a: &QuantQ8KBlockX4,
    out: &mut [f32; 4],
) {
    use std::arch::x86_64::*;

    let qs = &raw[16..144];
    let d = meta.d[linear_idx];
    let dmin = meta.dmin[linear_idx];
    let meta_off = linear_idx * 8;

    let low_mask = _mm256_set1_epi8(0x0f);
    let ones_16 = _mm256_set1_epi16(1);

    for row in 0..4 {
        if a.d[row] == 0.0 {
            continue;
        }
        let a_qs_base = row * 256;
        let a_bsums_base = row * 16;
        let mut sum = 0.0f32;
        let mut q_off = 0usize;
        let mut a_off = 0usize;

        for sub in 0..4 {
            let sc1 = meta.scales[meta_off + sub * 2] as f32;
            let m1 = meta.mins[meta_off + sub * 2] as f32;
            let sc2 = meta.scales[meta_off + sub * 2 + 1] as f32;
            let m2 = meta.mins[meta_off + sub * 2 + 1] as f32;

            // 加载 32 字节 Q4K 权重
            let q_raw = _mm256_loadu_si256(qs.as_ptr().add(q_off) as *const __m256i);
            let q_lo = _mm256_and_si256(q_raw, low_mask);
            let q_hi = _mm256_and_si256(_mm256_srli_epi16(q_raw, 4), low_mask);

            let a_lo = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off) as *const __m256i);
            let a_hi = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off + 32) as *const __m256i);

            let dot_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(q_lo, a_lo), ones_16);
            let dot_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(q_hi, a_hi), ones_16);

            let isum_lo = hsum_i32_avx2(dot_lo);
            let isum_hi = hsum_i32_avx2(dot_hi);

            let sum1 = a.bsums[a_bsums_base + sub * 4] as i32
                + a.bsums[a_bsums_base + sub * 4 + 1] as i32;
            let sum2 = a.bsums[a_bsums_base + sub * 4 + 2] as i32
                + a.bsums[a_bsums_base + sub * 4 + 3] as i32;

            sum += a.d[row] * (d * sc1 * isum_lo as f32 - dmin * m1 * sum1 as f32)
                + a.d[row] * (d * sc2 * isum_hi as f32 - dmin * m2 * sum2 as f32);

            q_off += 32;
            a_off += 64;
        }
        out[row] += sum;
    }
}

/// Q6K × Q8K x4 累加：对 4 行激活分别执行 Q6K 点积并累加到 out。
/// 非交错布局下每行 256 字节连续，可直接复用单行 AVX2 内核逻辑。
/// Q6K 块 × Q8K x4 激活块的 4 路累加点积（自动选择 AVX2 或标量）
#[inline]
fn q6k_accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { q6k_accumulate_block_dot_q8k_x4_avx2(raw, a, out) };
            return;
        }
    }
    q6k_accumulate_block_dot_q8k_x4_scalar(raw, a, out);
}

/// Q6K × Q8K x4 标量回退路径（非交错布局）。
#[inline]
fn q6k_accumulate_block_dot_q8k_x4_scalar(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
    if a.d.iter().all(|&d| d == 0.0) {
        return;
    }

    let mut ql = &raw[0..128];
    let mut qh = &raw[128..192];
    let mut sc = &raw[192..208];
    let d = half::f16::from_bits(read_u16_le(raw, 208)).to_f32();

    let mut a_off = 0usize;
    for _ in (0..256).step_by(128) {
        let mut dot1 = [0i32; 4];
        let mut dot2 = [0i32; 4];
        let mut dot3 = [0i32; 4];
        let mut dot4 = [0i32; 4];
        for l in 0..32 {
            let q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32;
            for row in 0..4 {
                dot1[row] += q1 * q8k_x4_q(a, row, a_off + l + 0) as i32;
                dot2[row] += q2 * q8k_x4_q(a, row, a_off + l + 32) as i32;
                dot3[row] += q3 * q8k_x4_q(a, row, a_off + l + 64) as i32;
                dot4[row] += q4 * q8k_x4_q(a, row, a_off + l + 96) as i32;
            }
        }
        for row in 0..4 {
            out[row] += a.d[row] * d * (sc[0] as i8 as f32 * dot1[row] as f32);
            out[row] += a.d[row] * d * (sc[2] as i8 as f32 * dot2[row] as f32);
            out[row] += a.d[row] * d * (sc[4] as i8 as f32 * dot3[row] as f32);
            out[row] += a.d[row] * d * (sc[6] as i8 as f32 * dot4[row] as f32);
        }
        a_off += 128;
        ql = &ql[64..];
        qh = &qh[32..];
        sc = &sc[8..];
    }
}

/// Q6K × Q8K x4 AVX2 加速版。
/// 非交错布局下，每行 qs 数据连续存储在 a.qs[row*256..(row+1)*256]，
/// 权重只解码一次，4 行激活分别执行 SIMD 点积。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q6k_accumulate_block_dot_q8k_x4_avx2(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
    use std::arch::x86_64::*;

    let ql_all = &raw[0..128];
    let qh_all = &raw[128..192];
    let sc_all = &raw[192..208];
    let d = half::f16::from_bits(read_u16_le(raw, 208)).to_f32();

    let low_mask = _mm256_set1_epi8(0x0f);
    let m32 = _mm256_set1_epi8(32);

    // 逐行处理：权重只解码一次，4 行激活分别 SIMD 点积
    for row in 0..4 {
        if a.d[row] == 0.0 {
            continue;
        }
        let a_qs_base = row * 256;
        let mut sum = 0.0f32;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut a_off = 0usize;

        for _ in 0..2 {
            // 加载 ql (64字节) 和 qh (32字节)
            let ql_0 = _mm256_loadu_si256(ql_all.as_ptr().add(ql_off) as *const __m256i);
            let ql_1 = _mm256_loadu_si256(ql_all.as_ptr().add(ql_off + 32) as *const __m256i);
            let qh = _mm256_loadu_si256(qh_all.as_ptr().add(qh_off) as *const __m256i);

            // 组合 6-bit 量化值
            let q1 = _mm256_or_si256(
                _mm256_and_si256(ql_0, low_mask),
                _mm256_slli_epi16(_mm256_and_si256(qh, _mm256_set1_epi8(0x03)), 4),
            );
            let q2 = _mm256_or_si256(
                _mm256_and_si256(ql_1, low_mask),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 2), _mm256_set1_epi8(0x03)),
                    4,
                ),
            );
            let q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_0, 4), low_mask),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 4), _mm256_set1_epi8(0x03)),
                    4,
                ),
            );
            let q4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_1, 4), low_mask),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 6), _mm256_set1_epi8(0x03)),
                    4,
                ),
            );

            // 减去偏移 32 得到有符号值
            let q1s = _mm256_sub_epi8(q1, m32);
            let q2s = _mm256_sub_epi8(q2, m32);
            let q3s = _mm256_sub_epi8(q3, m32);
            let q4s = _mm256_sub_epi8(q4, m32);

            // 加载该行的 128 字节 Q8K 激活
            let a1 = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off) as *const __m256i);
            let a2 = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off + 32) as *const __m256i);
            let a3 = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off + 64) as *const __m256i);
            let a4 = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off + 96) as *const __m256i);

            // scale 系数
            let sc0 = sc_all[sc_off] as i8 as i32;
            let sc1 = sc_all[sc_off + 1] as i8 as i32;
            let sc2 = sc_all[sc_off + 2] as i8 as i32;
            let sc3 = sc_all[sc_off + 3] as i8 as i32;
            let sc4 = sc_all[sc_off + 4] as i8 as i32;
            let sc5 = sc_all[sc_off + 5] as i8 as i32;
            let sc6 = sc_all[sc_off + 6] as i8 as i32;
            let sc7 = sc_all[sc_off + 7] as i8 as i32;

            // 拆分低/高 16 字节的 SIMD 点积
            let (dot1_lo, dot1_hi) = split_hsum_i32_lo_hi_avx2(q1s, a1);
            let (dot2_lo, dot2_hi) = split_hsum_i32_lo_hi_avx2(q2s, a2);
            let (dot3_lo, dot3_hi) = split_hsum_i32_lo_hi_avx2(q3s, a3);
            let (dot4_lo, dot4_hi) = split_hsum_i32_lo_hi_avx2(q4s, a4);

            let total = (dot1_lo * sc0 + dot1_hi * sc1)
                + (dot2_lo * sc2 + dot2_hi * sc3)
                + (dot3_lo * sc4 + dot3_hi * sc5)
                + (dot4_lo * sc6 + dot4_hi * sc7);

            sum += a.d[row] * d * total as f32;

            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            a_off += 128;
        }
        out[row] += sum;
    }
}

/// Q6K × Q8K x4 累加 (meta 预提取版)：使用预解析的 d/scales 元数据。
/// 非交错布局下每行连续，有 AVX2 时走 SIMD 路径。
#[inline]
fn q6k_accumulate_block_dot_q8k_x4_meta(
    raw: &[u8],
    meta: &QuantQ6KPrefillMetadata,
    linear_idx: usize,
    a: &QuantQ8KBlockX4,
    out: &mut [f32; 4],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { q6k_accumulate_block_dot_q8k_x4_meta_avx2(raw, meta, linear_idx, a, out) };
            return;
        }
    }
    q6k_accumulate_block_dot_q8k_x4_meta_scalar(raw, meta, linear_idx, a, out);
}

/// Q6K × Q8K x4 meta 标量回退路径。
#[inline]
fn q6k_accumulate_block_dot_q8k_x4_meta_scalar(
    raw: &[u8],
    meta: &QuantQ6KPrefillMetadata,
    linear_idx: usize,
    a: &QuantQ8KBlockX4,
    out: &mut [f32; 4],
) {
    if a.d.iter().all(|&d| d == 0.0) {
        return;
    }

    let mut ql = &raw[0..128];
    let mut qh = &raw[128..192];
    let d = meta.d[linear_idx];
    let scales = &meta.scales[linear_idx * 16..(linear_idx + 1) * 16];

    let mut a_off = 0usize;
    for half in 0..2 {
        let sc = &scales[half * 8..half * 8 + 8];
        let mut dot1 = [0i32; 4];
        let mut dot2 = [0i32; 4];
        let mut dot3 = [0i32; 4];
        let mut dot4 = [0i32; 4];
        for l in 0..32 {
            let q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32;
            for row in 0..4 {
                dot1[row] += q1 * q8k_x4_q(a, row, a_off + l) as i32;
                dot2[row] += q2 * q8k_x4_q(a, row, a_off + l + 32) as i32;
                dot3[row] += q3 * q8k_x4_q(a, row, a_off + l + 64) as i32;
                dot4[row] += q4 * q8k_x4_q(a, row, a_off + l + 96) as i32;
            }
        }
        for row in 0..4 {
            out[row] += a.d[row] * d * (sc[0] as f32 * dot1[row] as f32);
            out[row] += a.d[row] * d * (sc[2] as f32 * dot2[row] as f32);
            out[row] += a.d[row] * d * (sc[4] as f32 * dot3[row] as f32);
            out[row] += a.d[row] * d * (sc[6] as f32 * dot4[row] as f32);
        }
        a_off += 128;
        ql = &ql[64..];
        qh = &qh[32..];
    }
}

/// Q6K × Q8K x4 meta AVX2 加速版。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q6k_accumulate_block_dot_q8k_x4_meta_avx2(
    raw: &[u8],
    meta: &QuantQ6KPrefillMetadata,
    linear_idx: usize,
    a: &QuantQ8KBlockX4,
    out: &mut [f32; 4],
) {
    use std::arch::x86_64::*;

    let ql_all = &raw[0..128];
    let qh_all = &raw[128..192];
    let d = meta.d[linear_idx];
    let scales = &meta.scales[linear_idx * 16..(linear_idx + 1) * 16];

    let low_mask = _mm256_set1_epi8(0x0f);
    let m32 = _mm256_set1_epi8(32);

    for row in 0..4 {
        if a.d[row] == 0.0 {
            continue;
        }
        let a_qs_base = row * 256;
        let mut sum = 0.0f32;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut a_off = 0usize;

        for half in 0..2 {
            let sc = &scales[half * 8..half * 8 + 8];

            let ql_0 = _mm256_loadu_si256(ql_all.as_ptr().add(ql_off) as *const __m256i);
            let ql_1 = _mm256_loadu_si256(ql_all.as_ptr().add(ql_off + 32) as *const __m256i);
            let qh = _mm256_loadu_si256(qh_all.as_ptr().add(qh_off) as *const __m256i);

            let q1 = _mm256_or_si256(
                _mm256_and_si256(ql_0, low_mask),
                _mm256_slli_epi16(_mm256_and_si256(qh, _mm256_set1_epi8(0x03)), 4),
            );
            let q2 = _mm256_or_si256(
                _mm256_and_si256(ql_1, low_mask),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 2), _mm256_set1_epi8(0x03)),
                    4,
                ),
            );
            let q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_0, 4), low_mask),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 4), _mm256_set1_epi8(0x03)),
                    4,
                ),
            );
            let q4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_1, 4), low_mask),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 6), _mm256_set1_epi8(0x03)),
                    4,
                ),
            );

            let q1s = _mm256_sub_epi8(q1, m32);
            let q2s = _mm256_sub_epi8(q2, m32);
            let q3s = _mm256_sub_epi8(q3, m32);
            let q4s = _mm256_sub_epi8(q4, m32);

            let a1 = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off) as *const __m256i);
            let a2 = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off + 32) as *const __m256i);
            let a3 = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off + 64) as *const __m256i);
            let a4 = _mm256_loadu_si256(a.qs.as_ptr().add(a_qs_base + a_off + 96) as *const __m256i);

            let sc0 = sc[0] as f32;
            let sc1 = sc[1] as f32; // 注意：meta 里的 scale 是 f32，不是 i8
            let sc2 = sc[2] as f32;
            let sc3 = sc[3] as f32;
            let sc4 = sc[4] as f32;
            let sc5 = sc[5] as f32;
            let sc6 = sc[6] as f32;
            let sc7 = sc[7] as f32;

            let (dot1_lo, dot1_hi) = split_hsum_i32_lo_hi_avx2(q1s, a1);
            let (dot2_lo, dot2_hi) = split_hsum_i32_lo_hi_avx2(q2s, a2);
            let (dot3_lo, dot3_hi) = split_hsum_i32_lo_hi_avx2(q3s, a3);
            let (dot4_lo, dot4_hi) = split_hsum_i32_lo_hi_avx2(q4s, a4);

            let total = (dot1_lo as f32 * sc0 + dot1_hi as f32 * sc1)
                + (dot2_lo as f32 * sc2 + dot2_hi as f32 * sc3)
                + (dot3_lo as f32 * sc4 + dot3_hi as f32 * sc5)
                + (dot4_lo as f32 * sc6 + dot4_hi as f32 * sc7);

            sum += a.d[row] * d * total;

            ql_off += 64;
            qh_off += 32;
            a_off += 128;
        }
        out[row] += sum;
    }
}

impl QuantLayout for Q80 {
    fn block_size() -> usize {
        34
    }

    fn qk() -> usize {
        32
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q80
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        q80_decode_block_dot_simd(raw, a)
    }

    fn decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
        q80_decode_block_dot_q80(raw, a)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        for i in 0..32 {
            out[i] = d * (i8::from_le_bytes([raw[2 + i]]) as f32);
        }
    }
}

impl QuantLayout for Q40 {
    fn block_size() -> usize {
        18
    }

    fn qk() -> usize {
        32
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q80
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let qs = &raw[2..18];
        let mut s = 0.0f32;
        for j in 0..16 {
            let x0 = (qs[j] & 0x0f) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            s += a[2 * j] * (d * x0 as f32);
            s += a[2 * j + 1] * (d * x1 as f32);
        }
        s
    }

    fn decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
        q40_decode_block_dot_q80(raw, a)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let qs = &raw[2..18];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0f) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            out[2 * j] = d * x0 as f32;
            out[2 * j + 1] = d * x1 as f32;
        }
    }
}

impl QuantLayout for Q41 {
    fn block_size() -> usize {
        20
    }

    fn qk() -> usize {
        32
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q80
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let m0 = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
        let qs = &raw[4..20];
        let mut s = 0.0f32;
        for j in 0..16 {
            let x0 = (qs[j] & 0x0f) as f32;
            let x1 = (qs[j] >> 4) as f32;
            s += a[2 * j] * (d * x0 + m0);
            s += a[2 * j + 1] * (d * x1 + m0);
        }
        s
    }

    fn decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
        q41_decode_block_dot_q80(raw, a)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let m0 = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
        let qs = &raw[4..20];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0f) as f32;
            let x1 = (qs[j] >> 4) as f32;
            out[2 * j] = d * x0 + m0;
            out[2 * j + 1] = d * x1 + m0;
        }
    }
}

impl QuantLayout for Q50 {
    fn block_size() -> usize {
        22
    }

    fn qk() -> usize {
        32
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q80
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let qh = read_u32_le(raw, 2);
        let qs = &raw[6..22];
        let mut s = 0.0f32;
        for j in 0..16 {
            let xh0 = (((qh >> (j + 0)) << 4) & 0x10) as i32;
            let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
            let x0 = ((qs[j] & 0x0f) as i32 | xh0) - 16;
            let x1 = ((qs[j] >> 4) as i32 | xh1) - 16;
            s += a[2 * j] * (d * x0 as f32);
            s += a[2 * j + 1] * (d * x1 as f32);
        }
        s
    }

    fn decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
        q50_decode_block_dot_q80(raw, a)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let qh = read_u32_le(raw, 2);
        let qs = &raw[6..22];
        for j in 0..16 {
            let xh0 = (((qh >> (j + 0)) << 4) & 0x10) as i32;
            let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
            let x0 = ((qs[j] & 0x0f) as i32 | xh0) - 16;
            let x1 = ((qs[j] >> 4) as i32 | xh1) - 16;
            out[2 * j] = d * x0 as f32;
            out[2 * j + 1] = d * x1 as f32;
        }
    }
}

impl QuantLayout for Q51 {
    fn block_size() -> usize {
        24
    }

    fn qk() -> usize {
        32
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q80
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let m0 = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
        let qh = read_u32_le(raw, 4);
        let qs = &raw[8..24];
        let mut s = 0.0f32;
        for j in 0..16 {
            let xh0 = (((qh >> (j + 0)) << 4) & 0x10) as i32;
            let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
            let x0 = (qs[j] & 0x0f) as i32 | xh0;
            let x1 = (qs[j] >> 4) as i32 | xh1;
            s += a[2 * j] * (d * x0 as f32 + m0);
            s += a[2 * j + 1] * (d * x1 as f32 + m0);
        }
        s
    }

    fn decode_block_dot_q80(raw: &[u8], a: &QuantQ80Block) -> f32 {
        q51_decode_block_dot_q80(raw, a)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let m0 = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
        let qh = read_u32_le(raw, 4);
        let qs = &raw[8..24];
        for j in 0..16 {
            let xh0 = (((qh >> (j + 0)) << 4) & 0x10) as i32;
            let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
            let x0 = (qs[j] & 0x0f) as i32 | xh0;
            let x1 = (qs[j] >> 4) as i32 | xh1;
            out[2 * j] = d * x0 as f32 + m0;
            out[2 * j + 1] = d * x1 as f32 + m0;
        }
    }
}

impl QuantLayout for Q4K {
    fn block_size() -> usize {
        144
    }

    fn qk() -> usize {
        256
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn vec_dot_type() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn nrows() -> usize {
        4
    }

    fn supports_prefill_q8k_x4() -> bool {
        true
    }

    fn max_prefill_q8k_col_tile() -> usize {
        8
    }

    fn uses_q8k_activation() -> bool {
        true
    }

    fn supports_prefill_packed_q8k() -> bool {
        true
    }

    fn supports_interleaved_prefill_q8k() -> bool {
        true
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        if direct_qk_vecdot_enabled() {
            q4k_decode_block_dot_direct(raw, a)
        } else {
            q4k_decode_block_dot_legacy(raw, a)
        }
    }

    fn decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
        q4k_decode_block_dot_q8k(raw, a)
    }

    fn accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
        q4k_accumulate_block_dot_q8k_x4(raw, a, out)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
        let scales = &raw[4..16];
        let qs = &raw[16..144];

        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut out_off = 0usize;
        for _sub in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let m1f = dmin * m1 as f32;
            let m2f = dmin * m2 as f32;

            for l in 0..32 {
                let qv = qs[q_off + l];
                out[out_off + l] = d1 * (qv & 0x0f) as f32 - m1f;
            }
            for l in 0..32 {
                let qv = qs[q_off + l];
                out[out_off + 32 + l] = d2 * (qv >> 4) as f32 - m2f;
            }

            q_off += 32;
            out_off += 64;
            is += 2;
        }
    }
}

impl QuantLayout for Q2K {
    fn block_size() -> usize {
        84
    }

    fn qk() -> usize {
        256
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn vec_dot_type() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn nrows() -> usize {
        4
    }

    fn supports_prefill_q8k_x4() -> bool {
        q235k_prefill_q8k_x4_enabled()
    }

    fn max_prefill_q8k_col_tile() -> usize {
        4
    }

    fn uses_q8k_activation() -> bool {
        true
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        let mut vals = [0.0f32; 256];
        Self::decode_block_into(raw, &mut vals);
        let mut s = 0.0f32;
        for i in 0..256 {
            s += a[i] * vals[i];
        }
        s
    }

    fn decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
        // 先补齐与 llama.cpp 一致的 `Q8_K` 激活入口，
        // 当前阶段只要求这三个类型先接入统一的 `x1 vec_dot` 主路径。
        q2k_decode_block_dot_q8k(raw, a)
    }

    fn accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
        q2k_accumulate_block_dot_q8k_x4(raw, a, out)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let scales = &raw[0..16];
        let q = &raw[16..80];
        let d = half::f16::from_bits(read_u16_le(raw, 80)).to_f32();
        let dmin = half::f16::from_bits(read_u16_le(raw, 82)).to_f32();

        let mut is = 0usize;
        let mut qp = q;
        let mut out_off = 0usize;
        for _ in (0..256).step_by(128) {
            let mut shift = 0usize;
            for _ in 0..4 {
                let sc0 = scales[is];
                is += 1;
                let dl0 = d * (sc0 & 0x0f) as f32;
                let ml0 = dmin * (sc0 >> 4) as f32;
                for l in 0..16 {
                    out[out_off] = dl0 * ((qp[l] >> shift) & 0x03) as f32 - ml0;
                    out_off += 1;
                }

                let sc1 = scales[is];
                is += 1;
                let dl1 = d * (sc1 & 0x0f) as f32;
                let ml1 = dmin * (sc1 >> 4) as f32;
                for l in 0..16 {
                    out[out_off] = dl1 * ((qp[l + 16] >> shift) & 0x03) as f32 - ml1;
                    out_off += 1;
                }
                shift += 2;
            }
            qp = &qp[32..];
        }
    }
}

impl QuantLayout for Q3K {
    fn block_size() -> usize {
        110
    }

    fn qk() -> usize {
        256
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn vec_dot_type() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn nrows() -> usize {
        4
    }

    fn supports_prefill_q8k_x4() -> bool {
        q235k_prefill_q8k_x4_enabled()
    }

    fn max_prefill_q8k_col_tile() -> usize {
        4
    }

    fn uses_q8k_activation() -> bool {
        true
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        let mut s = 0.0f32;
        let hm = &raw[0..32];
        let mut q = &raw[32..96];
        let scales = unpack_q3k_scales(&raw[96..108]);
        let d_all = half::f16::from_bits(read_u16_le(raw, 108)).to_f32();

        let mut is = 0usize;
        let mut mbit: u8 = 1;
        let mut a_off = 0usize;
        for _ in (0..256).step_by(128) {
            let mut shift = 0usize;
            for _ in 0..4 {
                let dl0 = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let lo = ((q[l] >> shift) & 0x03) as i8;
                    let hi = if (hm[l] & mbit) != 0 { 0 } else { 4 };
                    s += a[a_off] * (dl0 * (lo - hi) as f32);
                    a_off += 1;
                }

                let dl1 = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let lo = ((q[l + 16] >> shift) & 0x03) as i8;
                    let hi = if (hm[l + 16] & mbit) != 0 { 0 } else { 4 };
                    s += a[a_off] * (dl1 * (lo - hi) as f32);
                    a_off += 1;
                }

                shift += 2;
                mbit <<= 1;
            }
            q = &q[32..];
        }
        s
    }

    fn decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
        q3k_decode_block_dot_q8k(raw, a)
    }

    fn accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
        q3k_accumulate_block_dot_q8k_x4(raw, a, out)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let hm = &raw[0..32];
        let mut q = &raw[32..96];
        let scales = unpack_q3k_scales(&raw[96..108]);
        let d_all = half::f16::from_bits(read_u16_le(raw, 108)).to_f32();

        let mut is = 0usize;
        let mut mbit: u8 = 1;
        let mut out_off = 0usize;
        for _ in (0..256).step_by(128) {
            let mut shift = 0usize;
            for _ in 0..4 {
                let dl0 = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let lo = ((q[l] >> shift) & 0x03) as i8;
                    let hi = if (hm[l] & mbit) != 0 { 0 } else { 4 };
                    out[out_off] = dl0 * (lo - hi) as f32;
                    out_off += 1;
                }

                let dl1 = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let lo = ((q[l + 16] >> shift) & 0x03) as i8;
                    let hi = if (hm[l + 16] & mbit) != 0 { 0 } else { 4 };
                    out[out_off] = dl1 * (lo - hi) as f32;
                    out_off += 1;
                }

                shift += 2;
                mbit <<= 1;
            }
            q = &q[32..];
        }
    }
}

impl QuantLayout for Q5K {
    fn block_size() -> usize {
        176
    }

    fn qk() -> usize {
        256
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn vec_dot_type() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn nrows() -> usize {
        4
    }

    fn supports_prefill_q8k_x4() -> bool {
        q235k_prefill_q8k_x4_enabled()
    }

    fn max_prefill_q8k_col_tile() -> usize {
        4
    }

    fn uses_q8k_activation() -> bool {
        true
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        let mut s = 0.0f32;
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
        let scales = &raw[4..16];
        let qh = &raw[16..48];
        let mut ql = &raw[48..176];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut a_off = 0usize;
        for _ in (0..256).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            is += 1;
            let (sc2, m2) = get_scale_min_k4(is, scales);
            is += 1;
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let mm1 = dmin * m1 as f32;
            let mm2 = dmin * m2 as f32;

            for l in 0..32 {
                let v = (ql[l] & 0x0f) as i32 + if (qh[l] & u1) != 0 { 16 } else { 0 };
                s += a[a_off] * (d1 * v as f32 - mm1);
                a_off += 1;
            }
            for l in 0..32 {
                let v = (ql[l] >> 4) as i32 + if (qh[l] & u2) != 0 { 16 } else { 0 };
                s += a[a_off] * (d2 * v as f32 - mm2);
                a_off += 1;
            }

            ql = &ql[32..];
            u1 <<= 2;
            u2 <<= 2;
        }
        s
    }

    fn decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
        q5k_decode_block_dot_q8k(raw, a)
    }

    fn accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
        q5k_accumulate_block_dot_q8k_x4(raw, a, out)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let d = half::f16::from_bits(read_u16_le(raw, 0)).to_f32();
        let dmin = half::f16::from_bits(read_u16_le(raw, 2)).to_f32();
        let scales = &raw[4..16];
        let qh = &raw[16..48];
        let mut ql = &raw[48..176];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut out_off = 0usize;
        for _ in (0..256).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            is += 1;
            let (sc2, m2) = get_scale_min_k4(is, scales);
            is += 1;
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let mm1 = dmin * m1 as f32;
            let mm2 = dmin * m2 as f32;

            for l in 0..32 {
                let v = (ql[l] & 0x0f) as i32 + if (qh[l] & u1) != 0 { 16 } else { 0 };
                out[out_off] = d1 * v as f32 - mm1;
                out_off += 1;
            }
            for l in 0..32 {
                let v = (ql[l] >> 4) as i32 + if (qh[l] & u2) != 0 { 16 } else { 0 };
                out[out_off] = d2 * v as f32 - mm2;
                out_off += 1;
            }

            ql = &ql[32..];
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

impl QuantLayout for Q6K {
    fn block_size() -> usize {
        210
    }

    fn qk() -> usize {
        256
    }

    fn prefill_activation_kind() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn vec_dot_type() -> PrefillActivationKind {
        PrefillActivationKind::Q8K
    }

    fn nrows() -> usize {
        4
    }

    fn supports_prefill_q8k_x4() -> bool {
        true
    }

    fn max_prefill_q8k_col_tile() -> usize {
        8
    }

    fn uses_q8k_activation() -> bool {
        true
    }

    fn supports_prefill_packed_q8k() -> bool {
        true
    }

    fn supports_interleaved_prefill_q8k() -> bool {
        true
    }

    fn decode_block_dot(raw: &[u8], a: &[f32]) -> f32 {
        if direct_qk_vecdot_enabled() {
            q6k_decode_block_dot_direct(raw, a)
        } else {
            q6k_decode_block_dot_legacy(raw, a)
        }
    }

    fn decode_block_dot_q8k(raw: &[u8], a: &QuantQ8KBlock) -> f32 {
        q6k_decode_block_dot_q8k(raw, a)
    }

    fn accumulate_block_dot_q8k_x4(raw: &[u8], a: &QuantQ8KBlockX4, out: &mut [f32; 4]) {
        q6k_accumulate_block_dot_q8k_x4(raw, a, out)
    }

    fn decode_block_into(raw: &[u8], out: &mut [f32]) {
        let mut ql = &raw[0..128];
        let mut qh = &raw[128..192];
        let mut sc = &raw[192..208];
        let d = half::f16::from_bits(read_u16_le(raw, 208)).to_f32();

        let mut out_off = 0usize;
        for _ in (0..256).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8 - 32;

                out[out_off + l + 0] = d * (sc[is + 0] as i8) as f32 * q1 as f32;
                out[out_off + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                out[out_off + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                out[out_off + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
            out_off += 128;
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct MatrixCacheKey {
    raw_ptr: usize,
    ty: u32,
}

struct MatrixCacheState {
    map: HashMap<MatrixCacheKey, MatrixCacheEntry>,
    lru: VecDeque<MatrixCacheKey>,
    max_bytes: usize,
    total_bytes: usize,
    hits: u64,
    misses: u64,
}

struct MatrixCacheEntry {
    value: Arc<Vec<f32>>,
    bytes: usize,
}

impl MatrixCacheState {
    fn new(max_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            lru: VecDeque::new(),
            max_bytes,
            total_bytes: 0,
            hits: 0,
            misses: 0,
        }
    }

    fn touch(&mut self, key: MatrixCacheKey) {
        if let Some(pos) = self.lru.iter().position(|k| *k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key);
    }

    fn get(&mut self, key: MatrixCacheKey) -> Option<Arc<Vec<f32>>> {
        let value = self.map.get(&key).map(|entry| entry.value.clone());
        if value.is_some() {
            self.hits += 1;
            self.touch(key);
        } else {
            self.misses += 1;
        }
        value
    }

    fn insert(&mut self, key: MatrixCacheKey, value: Arc<Vec<f32>>) {
        let bytes = value
            .len()
            .saturating_mul(std::mem::size_of::<f32>());
        if bytes == 0 || bytes > self.max_bytes {
            return;
        }

        if self.map.contains_key(&key) {
            self.touch(key);
            return;
        }

        while self.total_bytes.saturating_add(bytes) > self.max_bytes {
            if let Some(old) = self.lru.pop_front() {
                if let Some(old_entry) = self.map.remove(&old) {
                    self.total_bytes = self.total_bytes.saturating_sub(old_entry.bytes);
                }
            } else {
                break;
            }
        }

        if self.total_bytes.saturating_add(bytes) > self.max_bytes {
            return;
        }

        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.map.insert(key, MatrixCacheEntry { value, bytes });
        self.lru.push_back(key);
    }

    fn stats(&self) -> (u64, u64, usize) {
        (self.hits, self.misses, self.map.len())
    }

    fn clear(&mut self) {
        self.map.clear();
        self.lru.clear();
        self.total_bytes = 0;
        self.hits = 0;
        self.misses = 0;
    }
}

static PREFILL_PREDECODE_BUDGET_MB: OnceLock<usize> = OnceLock::new();
static PREFILL_TILE_BUDGET_MB: OnceLock<usize> = OnceLock::new();
static PREFILL_K_TILE_BUDGET_MB: OnceLock<usize> = OnceLock::new();
static PREFILL_WORKSET_BUDGET_MB: OnceLock<usize> = OnceLock::new();
static PREFILL_WORKSET_MIN_ROWS: OnceLock<usize> = OnceLock::new();
static PREFILL_LONGPROMPT_TILING_ENABLED: OnceLock<bool> = OnceLock::new();
static HOT_MATRIX_CACHE_BUDGET_MB: OnceLock<usize> = OnceLock::new();
static HOT_MATRIX_CACHE: OnceLock<Mutex<MatrixCacheState>> = OnceLock::new();
static PREFILL_BATCH_PROJ_ENABLED: OnceLock<bool> = OnceLock::new();
static DIRECT_QK_VECDOT_ENABLED: OnceLock<bool> = OnceLock::new();

/// 获取全局热矩阵缓存实例
#[inline]
fn hot_matrix_cache() -> &'static Mutex<MatrixCacheState> {
    HOT_MATRIX_CACHE.get_or_init(|| Mutex::new(MatrixCacheState::new(hot_matrix_cache_budget_bytes())))
}

/// 从热矩阵缓存中查找已反量化的 dense 矩阵
#[inline]
fn hot_matrix_cache_get(key: MatrixCacheKey) -> Option<Arc<Vec<f32>>> {
    let mut guard = hot_matrix_cache().lock().unwrap();
    guard.get(key)
}

/// 将反量化后的 dense 矩阵插入热矩阵缓存
#[inline]
fn hot_matrix_cache_insert(key: MatrixCacheKey, value: Arc<Vec<f32>>) {
    let mut guard = hot_matrix_cache().lock().unwrap();
    guard.insert(key, value);
}

/// 获取 prefill 预解码工作集内存预算（默认 64MB，通过 LMRS_PREFILL_PREDECODE_MB 配置）
#[inline]
fn prefill_predecode_budget_bytes() -> usize {
    *PREFILL_PREDECODE_BUDGET_MB.get_or_init(|| {
        std::env::var("LMRS_PREFILL_PREDECODE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(64)
    }) * 1024 * 1024
}

/// 获取 prefill tile 内存预算（默认 8MB，通过 LMRS_PREFILL_TILE_MB 配置）
#[inline]
fn prefill_tile_budget_bytes() -> usize {
    *PREFILL_TILE_BUDGET_MB.get_or_init(|| {
        std::env::var("LMRS_PREFILL_TILE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(8)
    }) * 1024 * 1024
}

/// 获取 prefill K 方向分块内存预算（默认 2MB，通过 LMRS_PREFILL_K_TILE_MB 配置）
#[inline]
fn prefill_k_tile_budget_bytes() -> usize {
    *PREFILL_K_TILE_BUDGET_MB.get_or_init(|| {
        std::env::var("LMRS_PREFILL_K_TILE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(2)
    }) * 1024 * 1024
}

/// 获取 prefill workset 内存预算（默认 256MB，通过 LMRS_PREFILL_WORKSET_MB 配置）
#[inline]
fn prefill_workset_budget_bytes() -> usize {
    *PREFILL_WORKSET_BUDGET_MB.get_or_init(|| {
        std::env::var("LMRS_PREFILL_WORKSET_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(256)
    }) * 1024 * 1024
}

/// 获取 workset 使用的最小行数阈值（默认 8，通过 LMRS_PREFILL_WORKSET_MIN_ROWS 配置）
#[inline]
fn prefill_workset_min_rows() -> usize {
    *PREFILL_WORKSET_MIN_ROWS.get_or_init(|| {
        std::env::var("LMRS_PREFILL_WORKSET_MIN_ROWS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(8)
    })
}

/// 获取热矩阵缓存内存限额（默认 2048MB，通过 LMRS_HOT_MATRIX_CACHE_MB 配置）
#[inline]
fn hot_matrix_cache_budget_bytes() -> usize {
    *HOT_MATRIX_CACHE_BUDGET_MB.get_or_init(|| {
        std::env::var("LMRS_HOT_MATRIX_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(2048)
    }) * 1024 * 1024
}

/// 是否启用长 prompt 的 K 方向分块（默认关闭，通过 LMRS_PREFILL_LONGPROMPT_TILING 启用）
#[inline]
fn prefill_longprompt_tiling_enabled() -> bool {
    *PREFILL_LONGPROMPT_TILING_ENABLED.get_or_init(|| {
        std::env::var("LMRS_PREFILL_LONGPROMPT_TILING")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// 是否启用 batch 投影调度（默认关闭，通过 LMRS_PREFILL_BATCH_PROJ 启用）
#[inline]
fn prefill_batch_proj_enabled() -> bool {
    *PREFILL_BATCH_PROJ_ENABLED.get_or_init(|| {
        std::env::var("LMRS_PREFILL_BATCH_PROJ")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// 是否启用直接 QK 向量点积（默认关闭，通过 LMRS_DIRECT_QK_VECDOT 启用）
#[inline]
fn direct_qk_vecdot_enabled() -> bool {
    *DIRECT_QK_VECDOT_ENABLED.get_or_init(|| {
        std::env::var("LMRS_DIRECT_QK_VECDOT")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[inline]
fn q235k_prefill_q8k_x4_enabled() -> bool {
    // `Q2_K/Q3_K/Q5_K -> Q8_K 4x1/4x4` 当前先保留为实验链路。
    std::env::var("LMRS_Q235K_Q8K_X4")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// 根据工作集大小和输入列数计算 prefill 的行方向 tile 大小
#[inline]
fn prefill_row_tile(k: usize, n: usize, m: usize) -> usize {
    if !prefill_longprompt_tiling_enabled() {
        let bytes_per_row = k.saturating_mul(std::mem::size_of::<f32>()).max(1);
        let cache_friendly_budget = prefill_tile_budget_bytes();
        let working_budget = prefill_predecode_budget_bytes().min(cache_friendly_budget).max(bytes_per_row);
        let by_budget = (working_budget / bytes_per_row).max(1);
        let row_cap = if k >= 8192 {
            256
        } else if k >= 4096 {
            384
        } else {
            512
        };
        return by_budget.clamp(1, n.max(1)).min(row_cap).max(1);
    }

    if m < 256 {
        let bytes_per_row = k.saturating_mul(std::mem::size_of::<f32>()).max(1);
        let cache_friendly_budget = prefill_tile_budget_bytes();
        let working_budget = prefill_predecode_budget_bytes().min(cache_friendly_budget).max(bytes_per_row);
        let by_budget = (working_budget / bytes_per_row).max(1);
        let row_cap = if k >= 8192 {
            256
        } else if k >= 4096 {
            384
        } else {
            512
        };
        return by_budget.clamp(1, n.max(1)).min(row_cap).max(1);
    }

    // row tile 现在同时考虑解码后的权重条带和输出子矩阵大小。
    // prompt 越长（m 越大），每次处理的列块就越要保守，避免 `O/down` 的工作集把 LLC 顶满。
    let bytes_per_row = k
        .saturating_add(m)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    let cache_friendly_budget = prefill_tile_budget_bytes();
    let working_budget = prefill_predecode_budget_bytes()
        .min(cache_friendly_budget)
        .max(bytes_per_row);
    let by_budget = (working_budget / bytes_per_row).max(1);
    let row_cap = if k >= 8192 {
        if m >= 256 { 96 } else if m >= 128 { 128 } else { 192 }
    } else if k >= 4096 {
        if m >= 256 { 128 } else if m >= 128 { 192 } else { 320 }
    } else if m >= 256 {
        192
    } else if m >= 128 {
        256
    } else {
        512
    };
    by_budget.clamp(1, n.max(1)).min(row_cap).max(1)
}

/// 根据工作集大小计算 prefill 的 K 方向 tile（仅长 prompt 时生效）
#[inline]
fn prefill_k_tile(qk: usize, k: usize, row_cnt: usize, m: usize) -> usize {
    if !prefill_longprompt_tiling_enabled() {
        return k;
    }

    // 短 prompt 下，额外切分 k 方向只会增加 GEMM 次数，通常得不偿失；
    // 仅在长 prompt / 大工作集时才启用真正的 k 分块。
    if m < 256 || k <= 4096 || row_cnt < 128 {
        return k;
    }
    let bytes_per_k = row_cnt
        .saturating_add(m)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    let working_budget = prefill_k_tile_budget_bytes().max(qk.saturating_mul(bytes_per_k));
    let by_budget = (working_budget / bytes_per_k).max(qk);
    let k_cap = if m >= 256 {
        2048
    } else if m >= 128 {
        3072
    } else {
        4096
    };
    let aligned = (by_budget / qk).max(1) * qk;
    aligned.min(k).min(k_cap.max(qk))
}

/// 获取热矩阵缓存的统计数据：(命中次数, 缺失次数, 缓存条目数)
pub fn quant_row_cache_stats() -> (u64, u64, usize) {
    let matrix_cache = hot_matrix_cache().lock().unwrap();
    matrix_cache.stats()
}

/// 清空热矩阵缓存
pub fn quant_row_cache_clear() {
    let mut matrix_cache = hot_matrix_cache().lock().unwrap();
    matrix_cache.clear();
}

/// 将量化类型映射为唯一整数标签（用于缓存 key 构建）
#[inline]
fn ty_tag(ty: GGMLType) -> u32 {
    ty as u32
}

enum PreparedActivation<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrefillKernelProfile {
    // `gate/up` 属于高扩张比的小 k 热点，适合更大的 row tile 并尽量一次吃完整个 k。
    ExpansionWide,
    // `QKV/attn_out` 在当前模型里大多满足 `k<=2048`，完整吃掉 k 往往比保守切块更划算。
    FixedSmallK,
    // `down/kv` 这类更窄的投影仍然保守，避免把 LLC 顶满。
    ProjectionNarrow,
    ProjectionDefault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefillKernelConfig {
    row_tile: usize,
    block_tile: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrefillKernelKind {
    // `gate/up` 属于扩张型投影，输出行数通常大于输入列数，
    // 更适合保持完整 k 面板并放宽 row tile。
    Expansion,
    // `O/down/QKV` 属于回投影或等维投影，
    // 工作集更容易把 LLC 顶满，需要更保守地限制 tile。
    Projection,
}

impl<'a> PreparedActivation<'a> {
    #[inline]
    fn as_slice(&self) -> &[f32] {
        match self {
            PreparedActivation::Borrowed(s) => s,
            PreparedActivation::Owned(v) => v.as_slice(),
        }
    }
}

/// 将泛型激活数据转换为 f32 切片（若已为 f32 则零拷贝借用）
fn prepare_activation_rows_f32<'a, T>(a_data: &'a [T], m: usize, k: usize) -> PreparedActivation<'a>
where
    T: Float + Copy + Send + Sync + 'static,
{
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        PreparedActivation::Borrowed(unsafe {
            std::slice::from_raw_parts(a_data.as_ptr() as *const f32, m * k)
        })
    } else {
        let mut rows = vec![0.0f32; m * k];
        threadpool::parallel_chunks_mut(&mut rows, k, |row_a, dst| {
                let src = &a_data[row_a * k..(row_a + 1) * k];
                for i in 0..k {
                    dst[i] = src[i].to_f32().unwrap_or(0.0);
                }
            });
        PreparedActivation::Owned(rows)
    }
}

/// 根据输出/输入维度比例判断 prefill 内核类型（Expansion vs Projection）
#[inline]
fn prefill_kernel_kind(n: usize, k: usize) -> PrefillKernelKind {
    if n > k {
        PrefillKernelKind::Expansion
    } else {
        PrefillKernelKind::Projection
    }
}

/// 根据内核类型和矩阵维度确定具体的 prefill 内核配置
#[inline]
fn prefill_kernel_profile(kind: PrefillKernelKind, n: usize, k: usize, m: usize) -> PrefillKernelProfile {
    match kind {
        PrefillKernelKind::Expansion => PrefillKernelProfile::ExpansionWide,
        PrefillKernelKind::Projection if k <= 2048 && m >= 32 => PrefillKernelProfile::FixedSmallK,
        PrefillKernelKind::Projection if n.saturating_mul(2) <= k => PrefillKernelProfile::ProjectionNarrow,
        PrefillKernelKind::Projection => PrefillKernelProfile::ProjectionDefault,
    }
}

#[inline]
/// 根据内核 profile 调整 row tile 大小
fn prefill_row_tile_for_profile(profile: PrefillKernelProfile, k: usize, n: usize, m: usize) -> usize {
    let base = prefill_row_tile(k, n, m);
    let tuned = match profile {
        PrefillKernelProfile::ExpansionWide => {
            if m >= 128 {
                base.saturating_mul(2)
            } else if m >= 64 {
                base.saturating_mul(3) / 2
            } else {
                base
            }
        }
        PrefillKernelProfile::FixedSmallK => {
            if m >= 64 {
                base.saturating_mul(2)
            } else if m >= 32 {
                base.saturating_mul(3) / 2
            } else {
                base
            }
        }
        PrefillKernelProfile::ProjectionNarrow => base.min(256),
        PrefillKernelProfile::ProjectionDefault => base,
    };
    tuned.min(n.max(1)).max(1)
}

#[inline]
/// 根据内核 profile 调整 K 方向 tile 大小
fn prefill_k_tile_for_profile(
    profile: PrefillKernelProfile,
    qk: usize,
    k: usize,
    row_cnt: usize,
    m: usize,
) -> usize {
    match profile {
        PrefillKernelProfile::ExpansionWide => k,
        // 小 k 的 QKV/attn_out 只保留更积极的 row-tile，
        // 不再强制 full-k，避免在冷热工作集切换时放大 prefill 波动。
        PrefillKernelProfile::FixedSmallK => prefill_k_tile(qk, k, row_cnt, m),
        PrefillKernelProfile::ProjectionNarrow | PrefillKernelProfile::ProjectionDefault => {
            prefill_k_tile(qk, k, row_cnt, m)
        }
    }
}

/// 综合计算 prefill 内核配置（row_tile + block_tile）
#[inline]
fn prefill_kernel_config(qk: usize, n: usize, k: usize, m: usize) -> PrefillKernelConfig {
    let kind = prefill_kernel_kind(n, k);
    let profile = prefill_kernel_profile(kind, n, k, m);
    let row_tile = prefill_row_tile_for_profile(profile, k, n, m);
    let k_tile = prefill_k_tile_for_profile(profile, qk, k, row_tile.min(n), m);
    PrefillKernelConfig {
        row_tile,
        block_tile: (k_tile / qk).max(1),
    }
}

/// 将激活行按 f32 格式打包为 panel。并行拷贝各行指定列范围的数据
fn pack_activation_panel_block(
    a_rows_f32: &[f32],
    m: usize,
    k: usize,
    col_start: usize,
    col_len: usize,
) -> Vec<f32> {
    let mut panel = vec![0.0f32; m * col_len];
    threadpool::parallel_chunks_mut(&mut panel, col_len, |row, dst| {
            let src_off = row * k + col_start;
            dst.copy_from_slice(&a_rows_f32[src_off..src_off + col_len]);
        });
    panel
}

enum PrefillActivationPanel {
    F32(Vec<f32>),
    Q80 {
        blocks: Vec<QuantQ80Block>,
        block_cnt: usize,
    },
    Q8K {
        blocks: Vec<QuantQ8KBlock>,
        block_cnt: usize,
    },
}

/// 将激活行按 Q8_0 格式打包为 panel
#[inline]
fn pack_activation_panel_block_q80(
    a_rows_f32: &[f32],
    m: usize,
    k: usize,
    col_start: usize,
    block_cnt: usize,
) -> Vec<QuantQ80Block> {
    let mut out = Vec::with_capacity(m * block_cnt);
    for row in 0..m {
        let row_base = row * k + col_start;
        for blk in 0..block_cnt {
            let base = row_base + blk * 32;
            out.push(quantize_activation_block_q80(&a_rows_f32[base..base + 32]));
        }
    }
    out
}

/// 将激活行按 Q8K 格式打包为 panel
#[inline]
fn pack_activation_panel_block_q8k(
    a_rows_f32: &[f32],
    m: usize,
    k: usize,
    col_start: usize,
    block_cnt: usize,
) -> Vec<QuantQ8KBlock> {
    let mut out = Vec::with_capacity(m * block_cnt);
    for row in 0..m {
        let row_base = row * k + col_start;
        for blk in 0..block_cnt {
            let base = row_base + blk * 256;
            out.push(quantize_activation_block_q8k(&a_rows_f32[base..base + 256]));
        }
    }
    out
}

/// 根据量化布局自动选择 F32/Q80/Q8K 格式打包激活 panel
#[inline]
fn pack_prefill_activation_panel<L: QuantLayout>(
    a_rows_f32: &[f32],
    m: usize,
    k: usize,
    col_start: usize,
    block_cnt: usize,
) -> PrefillActivationPanel {
    match L::prefill_activation_kind() {
        PrefillActivationKind::F32 => PrefillActivationPanel::F32(pack_activation_panel_block(
            a_rows_f32,
            m,
            k,
            col_start,
            block_cnt * L::qk(),
        )),
        PrefillActivationKind::Q80 => PrefillActivationPanel::Q80 {
            blocks: pack_activation_panel_block_q80(a_rows_f32, m, k, col_start, block_cnt),
            block_cnt,
        },
        PrefillActivationKind::Q8K => PrefillActivationPanel::Q8K {
            blocks: pack_activation_panel_block_q8k(a_rows_f32, m, k, col_start, block_cnt),
            block_cnt,
        },
    }
}

/// 计算 panel 中指定行和块的索引
#[inline]
fn prefill_quant_panel_index(row_a: usize, blk: usize, block_cnt: usize) -> usize {
    row_a * block_cnt + blk
}

/// 将权重矩阵的一行反量化到 f32 输出缓冲区
#[inline]
fn decode_weight_row_into<L: QuantLayout>(
    wq: &QuantGGUFTensor,
    blocks_per_row: usize,
    row_w: usize,
    out_row: &mut [f32],
) {
    decode_weight_row_range_into::<L>(wq, blocks_per_row, row_w, 0, blocks_per_row, out_row);
}

/// 将权重矩阵的一行指定块范围反量化到 f32 输出缓冲区
#[inline]
fn decode_weight_row_range_into<L: QuantLayout>(
    wq: &QuantGGUFTensor,
    blocks_per_row: usize,
    row_w: usize,
    block_start: usize,
    block_cnt: usize,
    out_row: &mut [f32],
) {
    let phys_row = wq.physical_row(row_w);
    let row_base = phys_row * blocks_per_row * L::block_size();
    for blk in 0..block_cnt {
        let src_blk = block_start + blk;
        let base = row_base + src_blk * L::block_size();
        let out_off = blk * L::qk();
        L::decode_block_into(
            &wq.raw[base..base + L::block_size()],
            &mut out_row[out_off..out_off + L::qk()],
        );
    }
}

/// 计算 dense 矩阵所需字节数
#[inline]
fn dense_matrix_bytes(rows: usize, cols: usize) -> usize {
    rows.saturating_mul(cols)
        .saturating_mul(std::mem::size_of::<f32>())
}

/// 返回 stripe 布局的目标字节大小（256KB）
#[inline]
fn stripe_target_bytes() -> usize {
    256 * 1024
}

/// 根据列数计算 stripe 布局的行方向 tile
#[inline]
fn stripe_row_tile(rows: usize, cols: usize) -> usize {
    let base = if cols >= 8192 {
        96
    } else if cols >= 4096 {
        128
    } else {
        192
    };
    base.min(rows.max(1)).max(1)
}

/// 根据行 tile 和块大小计算 stripe 布局的块方向 tile
#[inline]
fn stripe_block_tile(blocks_per_row: usize, row_tile: usize, block_size: usize) -> usize {
    let bytes_per_block_row = row_tile.saturating_mul(block_size).max(1);
    let target = stripe_target_bytes() / bytes_per_block_row;
    target.max(1).min(blocks_per_row.max(1))
}

/// 构建指定量化布局的 prefill 条带布局实现
fn build_prefill_quant_stripe_layout_impl<L: QuantLayout>(wq: &QuantGGUFTensor) -> QuantPrefillStripeLayout {
    let rows = wq.rows();
    let cols = wq.cols();
    let blocks_per_row = cols / L::qk();
    let row_tile = stripe_row_tile(rows, cols);
    let block_tile = stripe_block_tile(blocks_per_row, row_tile, L::block_size());
    let mut stripes = Vec::with_capacity(rows.div_ceil(row_tile) * blocks_per_row.div_ceil(block_tile));

    for block_start in (0..blocks_per_row).step_by(block_tile) {
        let block_cnt = (block_start + block_tile).min(blocks_per_row) - block_start;
        for row_start in (0..rows).step_by(row_tile) {
            let row_cnt = (row_start + row_tile).min(rows) - row_start;
            let mut stripe = vec![0u8; row_cnt * block_cnt * L::block_size()];
            for row_idx in 0..row_cnt {
                let phys_row = wq.physical_row(row_start + row_idx);
                let src_base = (phys_row * blocks_per_row + block_start) * L::block_size();
                let dst_base = row_idx * block_cnt * L::block_size();
                let src = &wq.raw[src_base..src_base + block_cnt * L::block_size()];
                stripe[dst_base..dst_base + src.len()].copy_from_slice(src);
            }
            stripes.push(stripe);
        }
    }

    QuantPrefillStripeLayout {
        row_tile,
        block_tile,
        rows,
        blocks_per_row,
        stripes,
    }
}

/// 构建 prefill 阶段的量化条带常驻布局。
///
/// 目标：
/// - 维持量化原生字节表示，不落整块 dense；
/// - 让 gate/up/down 这类大矩阵在 prefill 时按条带顺序访问；
/// - 保持默认主路径比 dense workset 更轻量、更稳。
pub fn build_prefill_quant_stripe_layout(wq: &QuantGGUFTensor) -> Option<QuantPrefillStripeLayout> {
    if wq.cols() == 0 {
        return None;
    }

    match wq.tensor_type {
        GGMLType::Q4_0 => Some(build_prefill_quant_stripe_layout_impl::<Q40>(wq)),
        GGMLType::Q4_1 => Some(build_prefill_quant_stripe_layout_impl::<Q41>(wq)),
        GGMLType::Q5_0 => Some(build_prefill_quant_stripe_layout_impl::<Q50>(wq)),
        GGMLType::Q5_1 => Some(build_prefill_quant_stripe_layout_impl::<Q51>(wq)),
        GGMLType::Q8_0 => Some(build_prefill_quant_stripe_layout_impl::<Q80>(wq)),
        GGMLType::Q2K => Some(build_prefill_quant_stripe_layout_impl::<Q2K>(wq)),
        GGMLType::Q3K => Some(build_prefill_quant_stripe_layout_impl::<Q3K>(wq)),
        GGMLType::Q4K => Some(build_prefill_quant_stripe_layout_impl::<Q4K>(wq)),
        GGMLType::Q5K => Some(build_prefill_quant_stripe_layout_impl::<Q5K>(wq)),
        GGMLType::Q6K => Some(build_prefill_quant_stripe_layout_impl::<Q6K>(wq)),
        _ => None,
    }
}

/// 构建指定量化布局的 Q8K 交错布局实现（按块主序重排）
fn build_prefill_q8k_interleave_layout_impl<L: QuantLayout>(
    wq: &QuantGGUFTensor,
) -> QuantPrefillQ8KInterleaveLayout {
    let rows = wq.rows();
    let blocks_per_row = wq.cols() / L::qk();
    let block_size = L::block_size();
    let mut packed = vec![0u8; wq.raw.len()];

    for block_idx in 0..blocks_per_row {
        for logical_row in 0..rows {
            let phys_row = wq.physical_row(logical_row);
            let src_base = (phys_row * blocks_per_row + block_idx) * block_size;
            let dst_base = (block_idx * rows + logical_row) * block_size;
            packed[dst_base..dst_base + block_size]
                .copy_from_slice(&wq.raw[src_base..src_base + block_size]);
        }
    }

    QuantPrefillQ8KInterleaveLayout {
        rows,
        blocks_per_row,
        block_size,
        packed,
    }
}

/// 构建 Q8K 交错 prefill 布局（仅支持 Q4K/Q6K 量化类型）
pub fn build_prefill_q8k_interleave_layout(
    wq: &QuantGGUFTensor,
) -> Option<QuantPrefillQ8KInterleaveLayout> {
    if wq.cols() == 0 {
        return None;
    }

    match wq.tensor_type {
        GGMLType::Q4K => Some(build_prefill_q8k_interleave_layout_impl::<Q4K>(wq)),
        GGMLType::Q6K => Some(build_prefill_q8k_interleave_layout_impl::<Q6K>(wq)),
        _ => None,
    }
}

/// 构建 packed prefill 布局（将量化权重预打包以提高 cache 利用率）
pub fn build_prefill_packed_layout(wq: &QuantGGUFTensor) -> Option<QuantPrefillPackedLayout> {
    if wq.cols() == 0 {
        return None;
    }

    match wq.tensor_type {
        GGMLType::Q4K => Some(QuantPrefillPackedLayout {
            rows: wq.rows(),
            blocks_per_row: wq.cols() / Q4K::qk(),
            block_size: Q4K::block_size(),
            kind: QuantPrefillPackedLayoutKind::BlockMajorQ8K,
            packed: build_prefill_q8k_interleave_layout_impl::<Q4K>(wq).packed,
            metadata: Some(QuantPrefillKMetadata::Q4K(build_prefill_q4k_metadata_layout(wq))),
        }),
        GGMLType::Q6K => Some(QuantPrefillPackedLayout {
            rows: wq.rows(),
            blocks_per_row: wq.cols() / Q6K::qk(),
            block_size: Q6K::block_size(),
            kind: QuantPrefillPackedLayoutKind::BlockMajorQ8K,
            packed: build_prefill_q8k_interleave_layout_impl::<Q6K>(wq).packed,
            metadata: Some(QuantPrefillKMetadata::Q6K(build_prefill_q6k_metadata_layout(wq))),
        }),
        _ => None,
    }
}

/// 拐取 Q4K 量化权重的 d/dmin/scales/mins 元数据，用于预计算 prefill
fn build_prefill_q4k_metadata_layout(wq: &QuantGGUFTensor) -> QuantQ4KPrefillMetadata {
    let rows = wq.rows();
    let blocks_per_row = wq.cols() / Q4K::qk();
    let mut d = Vec::with_capacity(rows * blocks_per_row);
    let mut dmin = Vec::with_capacity(rows * blocks_per_row);
    let mut scales = Vec::with_capacity(rows * blocks_per_row * 8);
    let mut mins = Vec::with_capacity(rows * blocks_per_row * 8);

    for logical_row in 0..rows {
        let phys_row = wq.physical_row(logical_row);
        for block_idx in 0..blocks_per_row {
            let base = (phys_row * blocks_per_row + block_idx) * Q4K::block_size();
            let raw = &wq.raw[base..base + Q4K::block_size()];
            let raw_scales = &raw[4..16];
            d.push(half::f16::from_bits(read_u16_le(raw, 0)).to_f32());
            dmin.push(half::f16::from_bits(read_u16_le(raw, 2)).to_f32());
            for sub in 0..8 {
                let (scale, min) = get_scale_min_k4(sub, raw_scales);
                scales.push(scale);
                mins.push(min);
            }
        }
    }

    QuantQ4KPrefillMetadata {
        rows,
        blocks_per_row,
        d,
        dmin,
        scales,
        mins,
    }
}

/// 拐取 Q6K 量化权重的 d/scales 元数据，用于预计算 prefill
fn build_prefill_q6k_metadata_layout(wq: &QuantGGUFTensor) -> QuantQ6KPrefillMetadata {
    let rows = wq.rows();
    let blocks_per_row = wq.cols() / Q6K::qk();
    let mut d = Vec::with_capacity(rows * blocks_per_row);
    let mut scales = Vec::with_capacity(rows * blocks_per_row * 16);

    for logical_row in 0..rows {
        let phys_row = wq.physical_row(logical_row);
        for block_idx in 0..blocks_per_row {
            let base = (phys_row * blocks_per_row + block_idx) * Q6K::block_size();
            let raw = &wq.raw[base..base + Q6K::block_size()];
            d.push(half::f16::from_bits(read_u16_le(raw, 208)).to_f32());
            scales.extend(raw[192..208].iter().map(|&v| v as i8));
        }
    }

    QuantQ6KPrefillMetadata {
        rows,
        blocks_per_row,
        d,
        scales,
    }
}

/// 构建 K 量化 prefill 元数据布局（提取 super-block scale/min 用于加速 prefill 点积）
pub fn build_prefill_k_metadata_layout(wq: &QuantGGUFTensor) -> Option<QuantPrefillKMetadata> {
    if wq.cols() == 0 {
        return None;
    }

    match wq.tensor_type {
        GGMLType::Q4K => Some(QuantPrefillKMetadata::Q4K(build_prefill_q4k_metadata_layout(wq))),
        GGMLType::Q6K => Some(QuantPrefillKMetadata::Q6K(build_prefill_q6k_metadata_layout(wq))),
        _ => None,
    }
}

#[inline]
fn get_or_build_hot_dense_matrix<L: QuantLayout>(
    wq: &QuantGGUFTensor,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) -> Arc<Vec<f32>> {
    let cache_key = MatrixCacheKey {
        raw_ptr: wq.raw.as_ptr() as usize,
        ty: ty_tag(wq.tensor_type),
    };
    hot_matrix_cache_get(cache_key).unwrap_or_else(|| {
        // 热点矩阵转成连续 dense 后，decode 可以跨 token 持续复用，
        // 当前默认主路径不再让 prefill 冷启动时主动构建它，避免重蹈阶段六的回退。
        let mut dense = vec![0.0f32; n * k];
        threadpool::parallel_chunks_mut(&mut dense, k, |row_w, out_row| {
                decode_weight_row_into::<L>(wq, blocks_per_row, row_w, out_row);
            });
        let dense = Arc::new(dense);
        hot_matrix_cache_insert(cache_key, dense.clone());
        dense
    })
}

fn apply_prefill_quant_row_range<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    a_panel_f32: &[f32],
    k_len: usize,
    wq: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    blocks_per_row: usize,
    block_start: usize,
    block_cnt: usize,
    row_start: usize,
    row_end: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let row_cnt = row_end - row_start;
    let c_f32 = if TypeId::of::<T>() == TypeId::of::<f32>() {
        Some(unsafe {
            std::slice::from_raw_parts_mut(c_data.as_mut_ptr() as *mut f32, c_data.len())
        })
    } else {
        None
    };
    let mut decoded_rows = vec![0.0f32; row_cnt * k_len];
    threadpool::parallel_chunks_mut(&mut decoded_rows, k_len, |i, out_row| {
            decode_weight_row_range_into::<L>(
                wq,
                blocks_per_row,
                row_start + i,
                block_start,
                block_cnt,
                out_row,
            );
        });

    if let Some(c_f32) = c_f32 {
        unsafe {
            gemm::gemm(
                m,
                row_cnt,
                k_len,
                c_f32.as_mut_ptr().add(row_start),
                1,
                n as isize,
                true,
                a_panel_f32.as_ptr(),
                1,
                k_len as isize,
                decoded_rows.as_ptr(),
                k_len as isize,
                1,
                beta_f32,
                alpha_f32,
                false,
                false,
                false,
                Parallelism::None,
            );
        }
    } else {
        threadpool::parallel_chunks_mut(c_data, n, |row_a, c_row| {
                let a_row_f32 = &a_panel_f32[row_a * k_len..(row_a + 1) * k_len];
                for i in 0..row_cnt {
                    let row_w = row_start + i;
                    let sum = dot_f32_simd(a_row_f32, &decoded_rows[i * k_len..(i + 1) * k_len]);
                    let old = c_row[row_w].to_f32().unwrap_or(0.0);
                    c_row[row_w] = T::from(beta_f32 * old + alpha_f32 * sum).unwrap_or(T::zero());
                }
            });
    }
}

fn apply_prefill_panel_to_output<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    a_panel_f32: &[f32],
    k_len: usize,
    wq: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    blocks_per_row: usize,
    block_start: usize,
    block_cnt: usize,
    row_tile: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    for row_start in (0..n).step_by(row_tile) {
        let row_end = (row_start + row_tile).min(n);
        apply_prefill_quant_row_range::<T, L>(
            c_data,
            n,
            m,
            a_panel_f32,
            k_len,
            wq,
            beta_f32,
            alpha_f32,
            blocks_per_row,
            block_start,
            block_cnt,
            row_start,
            row_end,
        );
    }
}

fn apply_prefill_quant_row_range_q80<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    _m: usize,
    a_panel_q80: &[QuantQ80Block],
    wq: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    blocks_per_row: usize,
    block_start: usize,
    block_cnt: usize,
    row_start: usize,
    row_end: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let row_cnt = row_end - row_start;
    threadpool::parallel_chunks_mut(c_data, n, |row_a, c_row| {
            for i in 0..row_cnt {
                let row_w = row_start + i;
                let phys_row = wq.physical_row(row_w);
                let row_base = phys_row * blocks_per_row * L::block_size();
                let mut s = 0.0f32;
                for blk in 0..block_cnt {
                    let base = row_base + (block_start + blk) * L::block_size();
                    let block = &wq.raw[base..base + L::block_size()];
                    let a_blk = &a_panel_q80[prefill_quant_panel_index(row_a, blk, block_cnt)];
                    s += L::decode_block_dot_q80(block, a_blk);
                }
                let old = c_row[row_w].to_f32().unwrap_or(0.0);
                c_row[row_w] = T::from(beta_f32 * old + alpha_f32 * s).unwrap_or(T::zero());
            }
        });
}

fn apply_prefill_quant_panel_to_output_q80<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    a_panel_q80: &[QuantQ80Block],
    wq: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    blocks_per_row: usize,
    block_start: usize,
    block_cnt: usize,
    row_tile: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    for row_start in (0..n).step_by(row_tile) {
        let row_end = (row_start + row_tile).min(n);
        apply_prefill_quant_row_range_q80::<T, L>(
            c_data,
            n,
            m,
            a_panel_q80,
            wq,
            beta_f32,
            alpha_f32,
            blocks_per_row,
            block_start,
            block_cnt,
            row_start,
            row_end,
        );
    }
}

fn apply_prefill_quant_row_range_q8k<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    _m: usize,
    a_panel_q8k: &[QuantQ8KBlock],
    wq: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    blocks_per_row: usize,
    block_start: usize,
    block_cnt: usize,
    row_start: usize,
    row_end: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let row_cnt = row_end - row_start;
    threadpool::parallel_chunks_mut(c_data, n, |row_a, c_row| {
            for i in 0..row_cnt {
                let row_w = row_start + i;
                let phys_row = wq.physical_row(row_w);
                let row_base = phys_row * blocks_per_row * L::block_size();
                let mut s = 0.0f32;
                for blk in 0..block_cnt {
                    let base = row_base + (block_start + blk) * L::block_size();
                    let block = &wq.raw[base..base + L::block_size()];
                    let a_blk = &a_panel_q8k[prefill_quant_panel_index(row_a, blk, block_cnt)];
                    s += L::decode_block_dot_q8k(block, a_blk);
                }
                let old = c_row[row_w].to_f32().unwrap_or(0.0);
                c_row[row_w] = T::from(beta_f32 * old + alpha_f32 * s).unwrap_or(T::zero());
            }
        });
}

fn apply_prefill_quant_panel_to_output_q8k<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    a_panel_q8k: &[QuantQ8KBlock],
    wq: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    blocks_per_row: usize,
    block_start: usize,
    block_cnt: usize,
    row_tile: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    for row_start in (0..n).step_by(row_tile) {
        let row_end = (row_start + row_tile).min(n);
        apply_prefill_quant_row_range_q8k::<T, L>(
            c_data,
            n,
            m,
            a_panel_q8k,
            wq,
            beta_f32,
            alpha_f32,
            blocks_per_row,
            block_start,
            block_cnt,
            row_start,
            row_end,
        );
    }
}

/// x4 微内核：对 m/4 组激活行（每组 4 行）× n 权重行执行量化矩阵乘法。
/// 核心优化：对权重行维度 (n) 使用 rayon 并行化，避免之前完全串行的性能瓶颈。
/// 配合 AVX2 加速的 x4 累加内核，大幅提升 prefill 吞吐。
fn apply_prefill_q8k_x4_microkernel<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    a_panel_q8k_x4: &[QuantQ8KBlockX4],
    wq: &QuantGGUFTensor,
    shape: PrefillQ8KKernelShape,
    beta_f32: f32,
    alpha_f32: f32,
    blocks_per_row: usize,
    block_start: usize,
    block_cnt: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    debug_assert!(m % 4 == 0);
    let row_groups = m / 4;
    let _col_tile = shape.col_tile();
    let packed_layout = wq.prefill_packed.as_deref();
    let legacy_interleave = wq.prefill_q8k_interleave.as_deref();
    let k_metadata = packed_layout
        .and_then(|layout| layout.metadata())
        .or_else(|| wq.prefill_k_metadata.as_deref());

    for row_group in 0..row_groups {
        let row_base = row_group * 4;
        let a_base = row_group * block_cnt;

        // 对权重行维度 (n) 并行化：每个线程独立计算一批权重行的 4 路点积
        let mut results: Vec<[f32; 4]> = vec![[0.0f32; 4]; n];
        // SAFETY: 每个 logical_row 写入独立的不重叠元素
        #[derive(Clone, Copy)]
        struct SyncPtr(*mut [f32; 4]);
        unsafe impl Send for SyncPtr {}
        unsafe impl Sync for SyncPtr {}
        impl SyncPtr {
            #[inline(always)]
            fn get(self) -> *mut [f32; 4] { self.0 }
        }
        let results_ptr = SyncPtr(results.as_mut_ptr());
        threadpool::pool().parallel_for(n, |logical_row| {
                let mut sums = [0.0f32; 4];
                for blk in 0..block_cnt {
                    let a_blk = &a_panel_q8k_x4[a_base + blk];
                    let block = if let Some(layout) = packed_layout {
                        layout.block(block_start + blk, logical_row).unwrap_or_else(|| {
                            panic!(
                                "invalid packed q8k access: block={} row={}",
                                block_start + blk,
                                logical_row
                            )
                        })
                    } else if let Some(layout) = legacy_interleave {
                        layout.block(block_start + blk, logical_row).unwrap_or_else(|| {
                            panic!(
                                "invalid packed q8k interleave access: block={} row={}",
                                block_start + blk,
                                logical_row
                            )
                        })
                    } else {
                        let phys_row = wq.physical_row(logical_row);
                        let base =
                            (phys_row * blocks_per_row + block_start + blk) * L::block_size();
                        &wq.raw[base..base + L::block_size()]
                    };
                    let linear_idx = logical_row * blocks_per_row + block_start + blk;
                    match k_metadata {
                        Some(QuantPrefillKMetadata::Q4K(meta))
                            if wq.tensor_type == GGMLType::Q4K =>
                        {
                            q4k_accumulate_block_dot_q8k_x4_meta(
                                block, meta, linear_idx, a_blk, &mut sums,
                            )
                        }
                        Some(QuantPrefillKMetadata::Q6K(meta))
                            if wq.tensor_type == GGMLType::Q6K =>
                        {
                            q6k_accumulate_block_dot_q8k_x4_meta(
                                block, meta, linear_idx, a_blk, &mut sums,
                            )
                        }
                        _ => L::accumulate_block_dot_q8k_x4(block, a_blk, &mut sums),
                    }
                }
                // SAFETY: 每个 logical_row 独立写入不重叠的元素
                unsafe { *results_ptr.get().add(logical_row) = sums; }
            });

        // 将并行计算的结果写回输出矩阵
        for (col, sums) in results.iter().enumerate() {
            for row in 0..4 {
                let idx = (row_base + row) * n + col;
                let old = c_data[idx].to_f32().unwrap_or(0.0);
                c_data[idx] =
                    T::from(beta_f32 * old + alpha_f32 * sums[row]).unwrap_or(T::zero());
            }
        }
    }
}

/// Q8K 交错布局的 prefill 矩阵乘法（使用预打包权重）
fn matmul_prefill_with_layout_q8k_interleaved<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    k: usize,
    a_rows_f32: &[f32],
    wq: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    config: PrefillKernelConfig,
    shape: PrefillQ8KKernelShape,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let blocks_per_row = k / L::qk();
    let block_tile = config.block_tile;
    let blocklen = shape.blocklen();
    // m_aligned: 能被 4 整除的部分走 x4 微内核，m_rem: 尾部走单行 Q8K 回退
    let m_aligned = (m / 4) * 4;
    let m_rem = m - m_aligned;
    for block_start in (0..blocks_per_row).step_by(block_tile) {
        let block_cnt = (block_start + block_tile).min(blocks_per_row) - block_start;

        // 对齐部分：用 x4 微内核处理（AVX2 加速 + 权重行并行）
        if m_aligned > 0 {
            let a_panel_q8k_x4 = pack_activation_panel_block_q8k_x4(
                a_rows_f32,
                m_aligned,
                k,
                block_start * L::qk(),
                block_cnt,
                blocklen,
            );
            apply_prefill_q8k_x4_microkernel::<T, L>(
                c_data,
                n,
                m_aligned,
                &a_panel_q8k_x4,
                wq,
                shape,
                if block_start == 0 { beta_f32 } else { 1.0 },
                alpha_f32,
                blocks_per_row,
                block_start,
                block_cnt,
            );
        }

        // 尾部剩余行：用单行 Q8K 内核处理（仍然有 AVX2 加速）
        if m_rem > 0 {
            let rem_a_rows = &a_rows_f32[m_aligned * k..];
            let rem_c_data = &mut c_data[m_aligned * n..];
            let a_panel_q8k = pack_activation_panel_block_q8k(
                rem_a_rows,
                m_rem,
                k,
                block_start * L::qk(),
                block_cnt,
            );
            apply_prefill_quant_panel_to_output_q8k::<T, L>(
                rem_c_data,
                n,
                m_rem,
                &a_panel_q8k,
                wq,
                if block_start == 0 { beta_f32 } else { 1.0 },
                alpha_f32,
                blocks_per_row,
                block_start,
                block_cnt,
                config.row_tile,
            );
        }
    }
}

/// Q8K 交错布局的 2 矩阵 prefill 并行乘法
fn matmul_prefill_batch2_with_layout_q8k_interleaved<T, L: QuantLayout>(
    c0_data: &mut [T],
    c1_data: &mut [T],
    n: usize,
    m: usize,
    k: usize,
    a_rows_f32: &[f32],
    wq0: &QuantGGUFTensor,
    wq1: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    config: PrefillKernelConfig,
    shape: PrefillQ8KKernelShape,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let blocks_per_row = k / L::qk();
    let block_tile = config.block_tile;
    let blocklen = shape.blocklen();
    let m_aligned = (m / 4) * 4;
    let m_rem = m - m_aligned;
    for block_start in (0..blocks_per_row).step_by(block_tile) {
        let block_cnt = (block_start + block_tile).min(blocks_per_row) - block_start;

        // 对齐部分：x4 微内核 + 双权重矩阵并行
        if m_aligned > 0 {
            let a_panel_q8k_x4 = pack_activation_panel_block_q8k_x4(
                a_rows_f32,
                m_aligned,
                k,
                block_start * L::qk(),
                block_cnt,
                blocklen,
            );
            threadpool::join(
                || {
                    apply_prefill_q8k_x4_microkernel::<T, L>(
                        c0_data,
                        n,
                        m_aligned,
                        &a_panel_q8k_x4,
                        wq0,
                        shape,
                        if block_start == 0 { beta_f32 } else { 1.0 },
                        alpha_f32,
                        blocks_per_row,
                        block_start,
                        block_cnt,
                    )
                },
                || {
                    apply_prefill_q8k_x4_microkernel::<T, L>(
                        c1_data,
                        n,
                        m_aligned,
                        &a_panel_q8k_x4,
                        wq1,
                        shape,
                        if block_start == 0 { beta_f32 } else { 1.0 },
                        alpha_f32,
                        blocks_per_row,
                        block_start,
                        block_cnt,
                    )
                },
            );
        }

        // 尾部剩余行
        if m_rem > 0 {
            let rem_a_rows = &a_rows_f32[m_aligned * k..];
            let rem_c0 = &mut c0_data[m_aligned * n..];
            let rem_c1 = &mut c1_data[m_aligned * n..];
            let a_panel_q8k = pack_activation_panel_block_q8k(
                rem_a_rows,
                m_rem,
                k,
                block_start * L::qk(),
                block_cnt,
            );
            let b = if block_start == 0 { beta_f32 } else { 1.0 };
            threadpool::join(
                || {
                    apply_prefill_quant_panel_to_output_q8k::<T, L>(
                        rem_c0, n, m_rem, &a_panel_q8k, wq0, b, alpha_f32,
                        blocks_per_row, block_start, block_cnt, config.row_tile,
                    );
                },
                || {
                    apply_prefill_quant_panel_to_output_q8k::<T, L>(
                        rem_c1, n, m_rem, &a_panel_q8k, wq1, b, alpha_f32,
                        blocks_per_row, block_start, block_cnt, config.row_tile,
                    );
                },
            );
        }
    }
}

fn apply_prefill_quant_row_range_from_stripe<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    a_panel_f32: &[f32],
    k_len: usize,
    stripe_bytes: &[u8],
    beta_f32: f32,
    alpha_f32: f32,
    row_start: usize,
    row_cnt: usize,
    block_cnt: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let c_f32 = if TypeId::of::<T>() == TypeId::of::<f32>() {
        Some(unsafe {
            std::slice::from_raw_parts_mut(c_data.as_mut_ptr() as *mut f32, c_data.len())
        })
    } else {
        None
    };
    let mut decoded_rows = vec![0.0f32; row_cnt * k_len];
    threadpool::parallel_chunks_mut(&mut decoded_rows, k_len, |i, out_row| {
            let row_base = i * block_cnt * L::block_size();
            for blk in 0..block_cnt {
                let src_base = row_base + blk * L::block_size();
                let out_off = blk * L::qk();
                L::decode_block_into(
                    &stripe_bytes[src_base..src_base + L::block_size()],
                    &mut out_row[out_off..out_off + L::qk()],
                );
            }
        });

    if let Some(c_f32) = c_f32 {
        unsafe {
            gemm::gemm(
                m,
                row_cnt,
                k_len,
                c_f32.as_mut_ptr().add(row_start),
                1,
                n as isize,
                true,
                a_panel_f32.as_ptr(),
                1,
                k_len as isize,
                decoded_rows.as_ptr(),
                k_len as isize,
                1,
                beta_f32,
                alpha_f32,
                false,
                false,
                false,
                Parallelism::None,
            );
        }
    } else {
        threadpool::parallel_chunks_mut(c_data, n, |row_a, c_row| {
                let a_row_f32 = &a_panel_f32[row_a * k_len..(row_a + 1) * k_len];
                for i in 0..row_cnt {
                    let row_w = row_start + i;
                    let sum = dot_f32_simd(a_row_f32, &decoded_rows[i * k_len..(i + 1) * k_len]);
                    let old = c_row[row_w].to_f32().unwrap_or(0.0);
                    c_row[row_w] = T::from(beta_f32 * old + alpha_f32 * sum).unwrap_or(T::zero());
                }
            });
    }
}

fn apply_prefill_quant_row_range_from_stripe_q80<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    _m: usize,
    a_panel_q80: &[QuantQ80Block],
    stripe_bytes: &[u8],
    beta_f32: f32,
    alpha_f32: f32,
    row_start: usize,
    row_cnt: usize,
    block_cnt: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    threadpool::parallel_chunks_mut(c_data, n, |row_a, c_row| {
            for i in 0..row_cnt {
                let mut s = 0.0f32;
                let row_base = i * block_cnt * L::block_size();
                for blk in 0..block_cnt {
                    let src_base = row_base + blk * L::block_size();
                    let block = &stripe_bytes[src_base..src_base + L::block_size()];
                    let a_blk = &a_panel_q80[prefill_quant_panel_index(row_a, blk, block_cnt)];
                    s += L::decode_block_dot_q80(block, a_blk);
                }
                let row_w = row_start + i;
                let old = c_row[row_w].to_f32().unwrap_or(0.0);
                c_row[row_w] = T::from(beta_f32 * old + alpha_f32 * s).unwrap_or(T::zero());
            }
        });
}

fn apply_prefill_quant_row_range_from_stripe_q8k<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    _m: usize,
    a_panel_q8k: &[QuantQ8KBlock],
    stripe_bytes: &[u8],
    beta_f32: f32,
    alpha_f32: f32,
    row_start: usize,
    row_cnt: usize,
    block_cnt: usize,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    threadpool::parallel_chunks_mut(c_data, n, |row_a, c_row| {
            for i in 0..row_cnt {
                let mut s = 0.0f32;
                let row_base = i * block_cnt * L::block_size();
                for blk in 0..block_cnt {
                    let src_base = row_base + blk * L::block_size();
                    let block = &stripe_bytes[src_base..src_base + L::block_size()];
                    let a_blk = &a_panel_q8k[prefill_quant_panel_index(row_a, blk, block_cnt)];
                    s += L::decode_block_dot_q8k(block, a_blk);
                }
                let row_w = row_start + i;
                let old = c_row[row_w].to_f32().unwrap_or(0.0);
                c_row[row_w] = T::from(beta_f32 * old + alpha_f32 * s).unwrap_or(T::zero());
            }
        });
}

/// 量化条带布局的 prefill 矩阵乘法（逐条带并行计算）
fn matmul_prefill_with_stripes<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    k: usize,
    a_rows_f32: &[f32],
    stripes: &QuantPrefillStripeLayout,
    beta_f32: f32,
    alpha_f32: f32,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    for block_start in (0..stripes.blocks_per_row).step_by(stripes.block_tile) {
        let block_cnt = (block_start + stripes.block_tile).min(stripes.blocks_per_row) - block_start;
        let k_len = block_cnt * L::qk();
        let a_panel = pack_prefill_activation_panel::<L>(a_rows_f32, m, k, block_start * L::qk(), block_cnt);

        for row_start in (0..n).step_by(stripes.row_tile) {
            let row_cnt = (row_start + stripes.row_tile).min(n) - row_start;
            if let Some(stripe_bytes) = stripes.stripe(block_start, row_start) {
                match &a_panel {
                    PrefillActivationPanel::F32(a_panel_f32) => {
                        apply_prefill_quant_row_range_from_stripe::<T, L>(
                            c_data,
                            n,
                            m,
                            a_panel_f32,
                            k_len,
                            stripe_bytes,
                            if block_start == 0 { beta_f32 } else { 1.0 },
                            alpha_f32,
                            row_start,
                            row_cnt,
                            block_cnt,
                        );
                    }
                    PrefillActivationPanel::Q80 { blocks, .. } => {
                        apply_prefill_quant_row_range_from_stripe_q80::<T, L>(
                            c_data,
                            n,
                            m,
                            blocks,
                            stripe_bytes,
                            if block_start == 0 { beta_f32 } else { 1.0 },
                            alpha_f32,
                            row_start,
                            row_cnt,
                            block_cnt,
                        );
                    }
                    PrefillActivationPanel::Q8K { blocks, .. } => {
                        apply_prefill_quant_row_range_from_stripe_q8k::<T, L>(
                            c_data,
                            n,
                            m,
                            blocks,
                            stripe_bytes,
                            if block_start == 0 { beta_f32 } else { 1.0 },
                            alpha_f32,
                            row_start,
                            row_cnt,
                            block_cnt,
                        );
                    }
                }
            }
        }
    }
}

/// 量化条带布局的 2 矩阵 prefill 并行乘法
fn matmul_prefill_batch2_with_stripes<T, L: QuantLayout>(
    c0_data: &mut [T],
    c1_data: &mut [T],
    n: usize,
    m: usize,
    k: usize,
    a_rows_f32: &[f32],
    stripes0: &QuantPrefillStripeLayout,
    stripes1: &QuantPrefillStripeLayout,
    beta_f32: f32,
    alpha_f32: f32,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    for block_start in (0..stripes0.blocks_per_row).step_by(stripes0.block_tile) {
        let block_cnt = (block_start + stripes0.block_tile).min(stripes0.blocks_per_row) - block_start;
        let k_len = block_cnt * L::qk();
        let a_panel = pack_prefill_activation_panel::<L>(a_rows_f32, m, k, block_start * L::qk(), block_cnt);

        for row_start in (0..n).step_by(stripes0.row_tile) {
            let row_cnt = (row_start + stripes0.row_tile).min(n) - row_start;
            let stripe0 = stripes0
                .stripe(block_start, row_start)
                .expect("missing stripe for batch2 tensor 0");
            let stripe1 = stripes1
                .stripe(block_start, row_start)
                .expect("missing stripe for batch2 tensor 1");
            match &a_panel {
                PrefillActivationPanel::F32(a_panel_f32) => {
                    threadpool::join(
                        || {
                            apply_prefill_quant_row_range_from_stripe::<T, L>(
                                c0_data,
                                n,
                                m,
                                a_panel_f32,
                                k_len,
                                stripe0,
                                if block_start == 0 { beta_f32 } else { 1.0 },
                                alpha_f32,
                                row_start,
                                row_cnt,
                                block_cnt,
                            )
                        },
                        || {
                            apply_prefill_quant_row_range_from_stripe::<T, L>(
                                c1_data,
                                n,
                                m,
                                a_panel_f32,
                                k_len,
                                stripe1,
                                if block_start == 0 { beta_f32 } else { 1.0 },
                                alpha_f32,
                                row_start,
                                row_cnt,
                                block_cnt,
                            )
                        },
                    );
                }
                PrefillActivationPanel::Q80 { blocks, .. } => {
                    threadpool::join(
                        || {
                            apply_prefill_quant_row_range_from_stripe_q80::<T, L>(
                                c0_data,
                                n,
                                m,
                                blocks,
                                stripe0,
                                if block_start == 0 { beta_f32 } else { 1.0 },
                                alpha_f32,
                                row_start,
                                row_cnt,
                                block_cnt,
                            )
                        },
                        || {
                            apply_prefill_quant_row_range_from_stripe_q80::<T, L>(
                                c1_data,
                                n,
                                m,
                                blocks,
                                stripe1,
                                if block_start == 0 { beta_f32 } else { 1.0 },
                                alpha_f32,
                                row_start,
                                row_cnt,
                                block_cnt,
                            )
                        },
                    );
                }
                PrefillActivationPanel::Q8K { blocks, .. } => {
                    threadpool::join(
                        || {
                            apply_prefill_quant_row_range_from_stripe_q8k::<T, L>(
                                c0_data,
                                n,
                                m,
                                blocks,
                                stripe0,
                                if block_start == 0 { beta_f32 } else { 1.0 },
                                alpha_f32,
                                row_start,
                                row_cnt,
                                block_cnt,
                            )
                        },
                        || {
                            apply_prefill_quant_row_range_from_stripe_q8k::<T, L>(
                                c1_data,
                                n,
                                m,
                                blocks,
                                stripe1,
                                if block_start == 0 { beta_f32 } else { 1.0 },
                                alpha_f32,
                                row_start,
                                row_cnt,
                                block_cnt,
                            )
                        },
                    );
                }
            }
        }
    }
}

fn build_prefill_dense_workset<L: QuantLayout>(
    wq: &QuantGGUFTensor,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) -> Vec<f32> {
    let mut workset = vec![0.0f32; n * k];
    threadpool::parallel_chunks_mut(&mut workset, k, |row_w, out_row| {
            decode_weight_row_into::<L>(wq, blocks_per_row, row_w, out_row);
        });
    workset
}

fn apply_prefill_dense_workset_to_output<T>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    k: usize,
    a_rows_f32: &[f32],
    packed_weight_f32: &[f32],
    beta_f32: f32,
    alpha_f32: f32,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let c_f32 = unsafe {
            std::slice::from_raw_parts_mut(c_data.as_mut_ptr() as *mut f32, c_data.len())
        };
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                c_f32.as_mut_ptr(),
                1,
                n as isize,
                true,
                a_rows_f32.as_ptr(),
                1,
                k as isize,
                packed_weight_f32.as_ptr(),
                k as isize,
                1,
                beta_f32,
                alpha_f32,
                false,
                false,
                false,
                Parallelism::None,
            );
        }
    } else {
        threadpool::parallel_chunks_mut(c_data, n, |row_a, c_row| {
                let a_row_f32 = &a_rows_f32[row_a * k..(row_a + 1) * k];
                for row_w in 0..n {
                    let row_vals = &packed_weight_f32[row_w * k..(row_w + 1) * k];
                    let sum = dot_f32_simd(a_row_f32, row_vals);
                    let old = c_row[row_w].to_f32().unwrap_or(0.0);
                    c_row[row_w] = T::from(beta_f32 * old + alpha_f32 * sum).unwrap_or(T::zero());
                }
            });
    }
}

#[inline]
fn prefill_workset_eligible(m: usize, total_bytes: usize) -> bool {
    m >= prefill_workset_min_rows() && total_bytes <= prefill_workset_budget_bytes()
}

/// 通用 prefill 矩阵乘法：分块反量化 + 并行点积累加
fn matmul_prefill_with_layout<T, L: QuantLayout>(
    c_data: &mut [T],
    n: usize,
    m: usize,
    k: usize,
    a_rows_f32: &[f32],
    wq: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    config: PrefillKernelConfig,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let traits = quant_type_traits::<L>();
    if let Some(shape) = prefill_q8k_kernel_shape(n, m, traits) {
        matmul_prefill_with_layout_q8k_interleaved::<T, L>(
            c_data,
            n,
            m,
            k,
            a_rows_f32,
            wq,
            beta_f32,
            alpha_f32,
            config,
            shape,
        );
        return;
    }

    let blocks_per_row = k / L::qk();
    let block_tile = config.block_tile;

    for block_start in (0..blocks_per_row).step_by(block_tile) {
        let block_cnt = (block_start + block_tile).min(blocks_per_row) - block_start;
        let k_len = block_cnt * L::qk();
        // prefill 在每个 k 分块只打包一次激活面板，
        // 后续所有 row tile 直接复用这块连续 panel，避免反复从原始 A 行切片。
        let a_panel = pack_prefill_activation_panel::<L>(a_rows_f32, m, k, block_start * L::qk(), block_cnt);
        match &a_panel {
            PrefillActivationPanel::F32(a_panel_f32) => {
                apply_prefill_panel_to_output::<T, L>(
                    c_data,
                    n,
                    m,
                    a_panel_f32,
                    k_len,
                    wq,
                    if block_start == 0 { beta_f32 } else { 1.0 },
                    alpha_f32,
                    blocks_per_row,
                    block_start,
                    block_cnt,
                    config.row_tile,
                );
            }
            PrefillActivationPanel::Q80 { blocks, block_cnt } => {
                apply_prefill_quant_panel_to_output_q80::<T, L>(
                    c_data,
                    n,
                    m,
                    blocks,
                    wq,
                    if block_start == 0 { beta_f32 } else { 1.0 },
                    alpha_f32,
                    blocks_per_row,
                    block_start,
                    *block_cnt,
                    config.row_tile,
                );
            }
            PrefillActivationPanel::Q8K { blocks, block_cnt } => {
                apply_prefill_quant_panel_to_output_q8k::<T, L>(
                    c_data,
                    n,
                    m,
                    blocks,
                    wq,
                    if block_start == 0 { beta_f32 } else { 1.0 },
                    alpha_f32,
                    blocks_per_row,
                    block_start,
                    *block_cnt,
                    config.row_tile,
                );
            }
        }
    }
}

/// 量化 panel 路径的 2 矩阵 prefill 并行乘法
fn matmul_prefill_batch2_with_quant_panel<T, L: QuantLayout>(
    c0_data: &mut [T],
    c1_data: &mut [T],
    n: usize,
    m: usize,
    k: usize,
    a_rows_f32: &[f32],
    wq0: &QuantGGUFTensor,
    wq1: &QuantGGUFTensor,
    beta_f32: f32,
    alpha_f32: f32,
    config: PrefillKernelConfig,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let blocks_per_row = k / L::qk();
    let block_tile = config.block_tile;
    for block_start in (0..blocks_per_row).step_by(block_tile) {
        let block_cnt = (block_start + block_tile).min(blocks_per_row) - block_start;
        let a_panel = pack_prefill_activation_panel::<L>(a_rows_f32, m, k, block_start * L::qk(), block_cnt);
        match &a_panel {
            PrefillActivationPanel::F32(a_panel_f32) => {
                let k_len = block_cnt * L::qk();
                threadpool::join(
                    || {
                        apply_prefill_panel_to_output::<T, L>(
                            c0_data,
                            n,
                            m,
                            a_panel_f32,
                            k_len,
                            wq0,
                            if block_start == 0 { beta_f32 } else { 1.0 },
                            alpha_f32,
                            blocks_per_row,
                            block_start,
                            block_cnt,
                            config.row_tile,
                        )
                    },
                    || {
                        apply_prefill_panel_to_output::<T, L>(
                            c1_data,
                            n,
                            m,
                            a_panel_f32,
                            k_len,
                            wq1,
                            if block_start == 0 { beta_f32 } else { 1.0 },
                            alpha_f32,
                            blocks_per_row,
                            block_start,
                            block_cnt,
                            config.row_tile,
                        )
                    },
                );
            }
            PrefillActivationPanel::Q80 { blocks, block_cnt } => {
                threadpool::join(
                    || {
                        apply_prefill_quant_panel_to_output_q80::<T, L>(
                            c0_data,
                            n,
                            m,
                            blocks,
                            wq0,
                            if block_start == 0 { beta_f32 } else { 1.0 },
                            alpha_f32,
                            blocks_per_row,
                            block_start,
                            *block_cnt,
                            config.row_tile,
                        )
                    },
                    || {
                        apply_prefill_quant_panel_to_output_q80::<T, L>(
                            c1_data,
                            n,
                            m,
                            blocks,
                            wq1,
                            if block_start == 0 { beta_f32 } else { 1.0 },
                            alpha_f32,
                            blocks_per_row,
                            block_start,
                            *block_cnt,
                            config.row_tile,
                        )
                    },
                );
            }
            PrefillActivationPanel::Q8K { blocks, block_cnt } => {
                threadpool::join(
                    || {
                        apply_prefill_quant_panel_to_output_q8k::<T, L>(
                            c0_data,
                            n,
                            m,
                            blocks,
                            wq0,
                            if block_start == 0 { beta_f32 } else { 1.0 },
                            alpha_f32,
                            blocks_per_row,
                            block_start,
                            *block_cnt,
                            config.row_tile,
                        )
                    },
                    || {
                        apply_prefill_quant_panel_to_output_q8k::<T, L>(
                            c1_data,
                            n,
                            m,
                            blocks,
                            wq1,
                            if block_start == 0 { beta_f32 } else { 1.0 },
                            alpha_f32,
                            blocks_per_row,
                            block_start,
                            *block_cnt,
                            config.row_tile,
                        )
                    },
                );
            }
        }
    }
}

/// 2 矩阵共享输入的 prefill 并行矩阵乘法（自动选择最优布局路径）
fn matmul_prefill_batch2_with_layout<T, L: QuantLayout>(
    c0: &mut Tensor<T>,
    c1: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    wq0: &QuantGGUFTensor,
    wq1: &QuantGGUFTensor,
    alpha: T,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let a_shape = a.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let blocks_per_row = k / L::qk();
    let beta_f32 = beta.to_f32().unwrap_or(0.0);
    let alpha_f32 = alpha.to_f32().unwrap_or(0.0);
    let use_quant_panel = L::prefill_activation_kind() != PrefillActivationKind::F32;
    let prepared_a = prepare_activation_rows_f32(a.data(), m, k);
    let a_rows_f32 = prepared_a.as_slice();
    let c0_data = unsafe { c0.data_mut() };
    let c1_data = unsafe { c1.data_mut() };

    if let (Some(stripes0), Some(stripes1)) = (wq0.prefill_stripes.as_deref(), wq1.prefill_stripes.as_deref()) {
        matmul_prefill_batch2_with_stripes::<T, L>(
            c0_data,
            c1_data,
            wq0.rows(),
            m,
            k,
            a_rows_f32,
            stripes0,
            stripes1,
            beta_f32,
            alpha_f32,
        );
        return;
    }

    let traits = quant_type_traits::<L>();
    if let Some(shape) = prefill_q8k_kernel_shape(wq0.rows(), m, traits) {
        let config = prefill_kernel_config(L::qk(), wq0.rows(), k, m);
        matmul_prefill_batch2_with_layout_q8k_interleaved::<T, L>(
            c0_data,
            c1_data,
            wq0.rows(),
            m,
            k,
            a_rows_f32,
            wq0,
            wq1,
            beta_f32,
            alpha_f32,
            config,
            shape,
        );
        return;
    }

    if use_quant_panel {
        let config = prefill_kernel_config(L::qk(), wq0.rows(), k, m);
        matmul_prefill_batch2_with_quant_panel::<T, L>(
            c0_data,
            c1_data,
            wq0.rows(),
            m,
            k,
            a_rows_f32,
            wq0,
            wq1,
            beta_f32,
            alpha_f32,
            config,
        );
        return;
    }

    let total_workset_bytes = dense_matrix_bytes(wq0.rows(), k)
        .saturating_add(dense_matrix_bytes(wq1.rows(), k));

    if let (Some(packed0), Some(packed1)) = (wq0.prefill_workset.as_deref(), wq1.prefill_workset.as_deref()) {
        if prefill_batch_proj_enabled() {
            threadpool::join(
                || apply_prefill_dense_workset_to_output(c0_data, wq0.rows(), m, k, a_rows_f32, packed0, beta_f32, alpha_f32),
                || apply_prefill_dense_workset_to_output(c1_data, wq1.rows(), m, k, a_rows_f32, packed1, beta_f32, alpha_f32),
            );
        } else {
            apply_prefill_dense_workset_to_output(c0_data, wq0.rows(), m, k, a_rows_f32, packed0, beta_f32, alpha_f32);
            apply_prefill_dense_workset_to_output(c1_data, wq1.rows(), m, k, a_rows_f32, packed1, beta_f32, alpha_f32);
        }
        return;
    }

    if prefill_batch_proj_enabled() && prefill_workset_eligible(m, total_workset_bytes) {
        let packed0 = build_prefill_dense_workset::<L>(wq0, wq0.rows(), k, blocks_per_row);
        let packed1 = build_prefill_dense_workset::<L>(wq1, wq1.rows(), k, blocks_per_row);
        threadpool::join(
            || apply_prefill_dense_workset_to_output(c0_data, wq0.rows(), m, k, a_rows_f32, &packed0, beta_f32, alpha_f32),
            || apply_prefill_dense_workset_to_output(c1_data, wq1.rows(), m, k, a_rows_f32, &packed1, beta_f32, alpha_f32),
        );
        return;
    }

    matmul_with_layout::<T, L>(c0, beta, a, wq0, alpha);
    matmul_with_layout::<T, L>(c1, beta, a, wq1, alpha);
}

/// 3 个权重矩阵共享同一输入的 prefill 并行矩阵乘法（QKV 投影）
fn matmul_prefill_batch3_with_layout<T, L: QuantLayout>(
    c0: &mut Tensor<T>,
    c1: &mut Tensor<T>,
    c2: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    wq0: &QuantGGUFTensor,
    wq1: &QuantGGUFTensor,
    wq2: &QuantGGUFTensor,
    alpha: T,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let a_shape = a.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let blocks_per_row = k / L::qk();
    let beta_f32 = beta.to_f32().unwrap_or(0.0);
    let alpha_f32 = alpha.to_f32().unwrap_or(0.0);
    let use_quant_panel = L::prefill_activation_kind() != PrefillActivationKind::F32;
    let prepared_a = prepare_activation_rows_f32(a.data(), m, k);
    let a_rows_f32 = prepared_a.as_slice();
    let c0_data = unsafe { c0.data_mut() };
    let c1_data = unsafe { c1.data_mut() };
    let c2_data = unsafe { c2.data_mut() };

    if use_quant_panel {
        matmul_with_layout::<T, L>(c0, beta, a, wq0, alpha);
        matmul_with_layout::<T, L>(c1, beta, a, wq1, alpha);
        matmul_with_layout::<T, L>(c2, beta, a, wq2, alpha);
        return;
    }

    let total_workset_bytes = dense_matrix_bytes(wq0.rows(), k)
        .saturating_add(dense_matrix_bytes(wq1.rows(), k))
        .saturating_add(dense_matrix_bytes(wq2.rows(), k));

    if let (Some(packed0), Some(packed1), Some(packed2)) = (
        wq0.prefill_workset.as_deref(),
        wq1.prefill_workset.as_deref(),
        wq2.prefill_workset.as_deref(),
    ) {
        if prefill_batch_proj_enabled() {
            threadpool::join(
                || apply_prefill_dense_workset_to_output(c0_data, wq0.rows(), m, k, a_rows_f32, packed0, beta_f32, alpha_f32),
                || {
                    threadpool::join(
                        || apply_prefill_dense_workset_to_output(c1_data, wq1.rows(), m, k, a_rows_f32, packed1, beta_f32, alpha_f32),
                        || apply_prefill_dense_workset_to_output(c2_data, wq2.rows(), m, k, a_rows_f32, packed2, beta_f32, alpha_f32),
                    );
                },
            );
        } else {
            apply_prefill_dense_workset_to_output(c0_data, wq0.rows(), m, k, a_rows_f32, packed0, beta_f32, alpha_f32);
            apply_prefill_dense_workset_to_output(c1_data, wq1.rows(), m, k, a_rows_f32, packed1, beta_f32, alpha_f32);
            apply_prefill_dense_workset_to_output(c2_data, wq2.rows(), m, k, a_rows_f32, packed2, beta_f32, alpha_f32);
        }
        return;
    }

    if prefill_batch_proj_enabled() && prefill_workset_eligible(m, total_workset_bytes) {
        let packed0 = build_prefill_dense_workset::<L>(wq0, wq0.rows(), k, blocks_per_row);
        let packed1 = build_prefill_dense_workset::<L>(wq1, wq1.rows(), k, blocks_per_row);
        let packed2 = build_prefill_dense_workset::<L>(wq2, wq2.rows(), k, blocks_per_row);
        threadpool::join(
            || apply_prefill_dense_workset_to_output(c0_data, wq0.rows(), m, k, a_rows_f32, &packed0, beta_f32, alpha_f32),
            || {
                threadpool::join(
                    || apply_prefill_dense_workset_to_output(c1_data, wq1.rows(), m, k, a_rows_f32, &packed1, beta_f32, alpha_f32),
                    || apply_prefill_dense_workset_to_output(c2_data, wq2.rows(), m, k, a_rows_f32, &packed2, beta_f32, alpha_f32),
                );
            },
        );
        return;
    }

    matmul_with_layout::<T, L>(c0, beta, a, wq0, alpha);
    matmul_with_layout::<T, L>(c1, beta, a, wq1, alpha);
    matmul_with_layout::<T, L>(c2, beta, a, wq2, alpha);
}

/// 量化矩阵乘法主入口：根据 seq_len 自动分发到 decode（单行）或 prefill（多行）路径
fn matmul_with_layout<T, L: QuantLayout>(c: &mut Tensor<T>, beta: T, a: &Tensor<T>, wq: &QuantGGUFTensor, alpha: T)
where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let a_shape = a.shape();
    let c_shape = c.shape();
    assert!(a_shape.len() == 2, "GGUF quant kernel currently supports 2D A only");
    assert!(c_shape.len() == 2, "GGUF quant kernel currently supports 2D C only");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = wq.rows();
    assert!(wq.cols() == k, "quant weight cols must match A cols");
    assert!(c_shape[0] == m && c_shape[1] == n, "C shape mismatch in quant kernel");

    assert!(k % L::qk() == 0, "k must be divisible by qk for quant kernel");
    let blocks_per_row = k / L::qk();
    let beta_f32 = beta.to_f32().unwrap_or(0.0);
    let alpha_f32 = alpha.to_f32().unwrap_or(0.0);
    let traits = quant_type_traits::<L>();
    let use_hot_matrix_cache = wq.hot_layer
        && m == 1
        && dense_matrix_bytes(n, k) <= hot_matrix_cache_budget_bytes();

    let a_data = a.data();
    let c_data = unsafe { c.data_mut() };

    if m == 1 {
        let a_row_owned;
        let a_row_f32: &[f32] = if TypeId::of::<T>() == TypeId::of::<f32>() {
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const f32, k) }
        } else {
            a_row_owned = a_data[0..k]
                .iter()
                .map(|v| v.to_f32().unwrap_or(0.0))
                .collect::<Vec<_>>();
            &a_row_owned
        };

        // 第一阶段先把 llama.cpp 式 `Q4_K/Q6_K × Q8_K` 激活量化主路径
        // 接到 decode（m=1）上。这样每个 256 元素激活块只量化一次，
        // 后续行遍历直接复用量化块，而不是反复拿 f32 子切片做点积。
        let q8k_blocks = if matches!(traits.vec_dot_type, PrefillActivationKind::Q8K) {
            Some(quantize_activation_row_q8k(a_row_f32))
        } else {
            None
        };

        let c_row = &mut c_data[0..n];
        if use_hot_matrix_cache {
            let dense_matrix = get_or_build_hot_dense_matrix::<L>(wq, n, k, blocks_per_row);

            threadpool::parallel_iter_mut(c_row, |row_w, out| {
                let row_vals = &dense_matrix[row_w * k..(row_w + 1) * k];
                let sum = dot_f32_simd(&a_row_f32, row_vals);
                let old = out.to_f32().unwrap_or(0.0);
                *out = T::from(beta_f32 * old + alpha_f32 * sum).unwrap_or(T::zero());
            });
        } else {
            // decode 路径：每个权重行的点积独立计算，par_iter 并行化
            threadpool::parallel_iter_mut(c_row, |row_w, out| {
                let phys_row = wq.physical_row(row_w);
                let row_base = phys_row * blocks_per_row * L::block_size();
                let mut s = 0.0f32;
                for blk in 0..blocks_per_row {
                    let base = row_base + blk * L::block_size();
                    let block = &wq.raw[base..base + L::block_size()];

                    // 软件预取：在处理当前块时预取下一个块的数据到 L1 缓存
                    #[cfg(target_arch = "x86_64")]
                    {
                        if blk + 1 < blocks_per_row {
                            let next_base = row_base + (blk + 1) * L::block_size();
                            unsafe {
                                use std::arch::x86_64::*;
                                let ptr = wq.raw.as_ptr().add(next_base);
                                _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
                                // 对大块（>64字节）再预取后半部分
                                if L::block_size() > 64 {
                                    _mm_prefetch(ptr.add(64) as *const i8, _MM_HINT_T0);
                                }
                            }
                        }
                    }

                    if let Some(q8k_blocks) = q8k_blocks.as_ref() {
                        s += L::decode_block_dot_q8k(block, &q8k_blocks[blk]);
                        continue;
                    }

                    let a_base = blk * L::qk();
                    let av = &a_row_f32[a_base..a_base + L::qk()];
                    s += L::decode_block_dot(block, av);
                }
                let old = out.to_f32().unwrap_or(0.0);
                *out = T::from(beta_f32 * old + alpha_f32 * s).unwrap_or(T::zero());
            });
        }
    } else {
        // prefill 模式：
        // 1. 先把输入激活统一转成 f32；
        // 2. 再按固定形状 profile 决定 row/k tile；
        // 3. 每个 k 分块只打包一次 activation panel，给量化权重内核直接消费。
        let prepared_a = prepare_activation_rows_f32(a_data, m, k);
        let a_rows_f32 = prepared_a.as_slice();
        if let Some(stripes) = wq.prefill_stripes.as_deref() {
            matmul_prefill_with_stripes::<T, L>(
                c_data,
                n,
                m,
                k,
                a_rows_f32,
                stripes,
                beta_f32,
                alpha_f32,
            );
            return;
        }
        let config = prefill_kernel_config(L::qk(), n, k, m);
        matmul_prefill_with_layout::<T, L>(
            c_data,
            n,
            m,
            k,
            a_rows_f32,
            wq,
            beta_f32,
            alpha_f32,
            config,
        );
    }
}

/// 通用 GGUF 量化 kernel：
/// - 覆盖 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2K/Q3K/Q4K/Q5K/Q6K
/// - 采用 QuantLayout trait 做类型分发，避免内层 match 巨分支
/// - Q8_0/Q4K/Q6K 点积走 SIMD 路径（AVX2/NEON 自动探测）
pub fn matmul_transb_gguf_quant<T>(
    c: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    wq: &QuantGGUFTensor,
    alpha: T,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    match wq.tensor_type {
        GGMLType::Q4_0 => matmul_with_layout::<T, Q40>(c, beta, a, wq, alpha),
        GGMLType::Q4_1 => matmul_with_layout::<T, Q41>(c, beta, a, wq, alpha),
        GGMLType::Q5_0 => matmul_with_layout::<T, Q50>(c, beta, a, wq, alpha),
        GGMLType::Q5_1 => matmul_with_layout::<T, Q51>(c, beta, a, wq, alpha),
        GGMLType::Q8_0 => matmul_with_layout::<T, Q80>(c, beta, a, wq, alpha),
        GGMLType::Q2K => matmul_with_layout::<T, Q2K>(c, beta, a, wq, alpha),
        GGMLType::Q3K => matmul_with_layout::<T, Q3K>(c, beta, a, wq, alpha),
        GGMLType::Q4K => matmul_with_layout::<T, Q4K>(c, beta, a, wq, alpha),
        GGMLType::Q5K => matmul_with_layout::<T, Q5K>(c, beta, a, wq, alpha),
        GGMLType::Q6K => matmul_with_layout::<T, Q6K>(c, beta, a, wq, alpha),
        _ => panic!("unsupported GGUF quant type in kernel: {:?}", wq.tensor_type),
    }
}

/// 双矩阵批量量化矩阵乘法，prefill 阶段 2 个权重矩阵共享同一输入的并行计算
pub fn matmul_transb_gguf_quant_batch2<T>(
    c0: &mut Tensor<T>,
    c1: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    wq0: &QuantGGUFTensor,
    wq1: &QuantGGUFTensor,
    alpha: T,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let a_shape = a.shape();
    if a_shape.len() != 2 || a_shape[0] <= 1 || wq0.tensor_type != wq1.tensor_type || wq0.cols() != wq1.cols() {
        matmul_transb_gguf_quant(c0, beta, a, wq0, alpha);
        matmul_transb_gguf_quant(c1, beta, a, wq1, alpha);
        return;
    }

    match wq0.tensor_type {
        GGMLType::Q4_0 => matmul_prefill_batch2_with_layout::<T, Q40>(c0, c1, beta, a, wq0, wq1, alpha),
        GGMLType::Q4_1 => matmul_prefill_batch2_with_layout::<T, Q41>(c0, c1, beta, a, wq0, wq1, alpha),
        GGMLType::Q5_0 => matmul_prefill_batch2_with_layout::<T, Q50>(c0, c1, beta, a, wq0, wq1, alpha),
        GGMLType::Q5_1 => matmul_prefill_batch2_with_layout::<T, Q51>(c0, c1, beta, a, wq0, wq1, alpha),
        GGMLType::Q8_0 => matmul_prefill_batch2_with_layout::<T, Q80>(c0, c1, beta, a, wq0, wq1, alpha),
        GGMLType::Q2K => matmul_prefill_batch2_with_layout::<T, Q2K>(c0, c1, beta, a, wq0, wq1, alpha),
        GGMLType::Q3K => matmul_prefill_batch2_with_layout::<T, Q3K>(c0, c1, beta, a, wq0, wq1, alpha),
        GGMLType::Q4K => matmul_prefill_batch2_with_layout::<T, Q4K>(c0, c1, beta, a, wq0, wq1, alpha),
        GGMLType::Q5K => matmul_prefill_batch2_with_layout::<T, Q5K>(c0, c1, beta, a, wq0, wq1, alpha),
        GGMLType::Q6K => matmul_prefill_batch2_with_layout::<T, Q6K>(c0, c1, beta, a, wq0, wq1, alpha),
        _ => panic!("unsupported GGUF quant type in batch2 kernel: {:?}", wq0.tensor_type),
    }
}

/// 三矩阵批量量化矩阵乘法，prefill 阶段 3 个权重矩阵共享同一输入的并行计算
pub fn matmul_transb_gguf_quant_batch3<T>(
    c0: &mut Tensor<T>,
    c1: &mut Tensor<T>,
    c2: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    wq0: &QuantGGUFTensor,
    wq1: &QuantGGUFTensor,
    wq2: &QuantGGUFTensor,
    alpha: T,
) where
    T: Float + Default + Copy + std::iter::Sum + Send + Sync + 'static,
{
    let a_shape = a.shape();
    if a_shape.len() != 2
        || a_shape[0] <= 1
        || wq0.tensor_type != wq1.tensor_type
        || wq0.tensor_type != wq2.tensor_type
        || wq0.cols() != wq1.cols()
        || wq0.cols() != wq2.cols()
    {
        matmul_transb_gguf_quant(c0, beta, a, wq0, alpha);
        matmul_transb_gguf_quant(c1, beta, a, wq1, alpha);
        matmul_transb_gguf_quant(c2, beta, a, wq2, alpha);
        return;
    }

    match wq0.tensor_type {
        GGMLType::Q4_0 => matmul_prefill_batch3_with_layout::<T, Q40>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        GGMLType::Q4_1 => matmul_prefill_batch3_with_layout::<T, Q41>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        GGMLType::Q5_0 => matmul_prefill_batch3_with_layout::<T, Q50>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        GGMLType::Q5_1 => matmul_prefill_batch3_with_layout::<T, Q51>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        GGMLType::Q8_0 => matmul_prefill_batch3_with_layout::<T, Q80>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        GGMLType::Q2K => matmul_prefill_batch3_with_layout::<T, Q2K>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        GGMLType::Q3K => matmul_prefill_batch3_with_layout::<T, Q3K>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        GGMLType::Q4K => matmul_prefill_batch3_with_layout::<T, Q4K>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        GGMLType::Q5K => matmul_prefill_batch3_with_layout::<T, Q5K>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        GGMLType::Q6K => matmul_prefill_batch3_with_layout::<T, Q6K>(c0, c1, c2, beta, a, wq0, wq1, wq2, alpha),
        _ => panic!("unsupported GGUF quant type in batch3 kernel: {:?}", wq0.tensor_type),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_prefill_k_metadata_layout,
        build_prefill_packed_layout,
        build_prefill_q8k_interleave_layout,
        matmul_transb_gguf_quant_batch2,
        matmul_transb_gguf_quant,
        pack_q8k_block_x4,
        q40_decode_block_dot_q80,
        q2k_accumulate_block_dot_q8k_x4,
        q2k_decode_block_dot_q8k,
        q3k_accumulate_block_dot_q8k_x4,
        q3k_decode_block_dot_q8k,
        q4k_decode_block_dot_q8k,
        q5k_accumulate_block_dot_q8k_x4,
        q5k_decode_block_dot_q8k,
        q6k_decode_block_dot_q8k,
        quantize_activation_block_q80,
        quantize_activation_block_q8k,
    };
    use crate::core::tensor::Tensor;
    use crate::formats::gguf::QuantGGUFTensor;
    use gguf::GGMLType;
    use std::sync::{Arc, Mutex, OnceLock};

    fn with_q235k_q8k_x4_enabled<R>(f: impl FnOnce() -> R) -> R {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        std::env::set_var("LMRS_Q235K_Q8K_X4", "1");
        let result = f();
        std::env::remove_var("LMRS_Q235K_Q8K_X4");
        result
    }

    fn spec(ty: GGMLType) -> (usize, usize) {
        match ty {
            GGMLType::Q4_0 => (32, 18),
            GGMLType::Q4_1 => (32, 20),
            GGMLType::Q5_0 => (32, 22),
            GGMLType::Q5_1 => (32, 24),
            GGMLType::Q8_0 => (32, 34),
            GGMLType::Q2K => (256, 84),
            GGMLType::Q3K => (256, 110),
            GGMLType::Q4K => (256, 144),
            GGMLType::Q5K => (256, 176),
            GGMLType::Q6K => (256, 210),
            _ => panic!("unsupported in test"),
        }
    }

    fn run_zero_block_case(ty: GGMLType) {
        let (qk, block_size) = spec(ty);
        let wq = QuantGGUFTensor {
            raw: vec![0u8; block_size],
            shape: vec![1, qk],
            tensor_type: ty,
            row_map: None,
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };

        let a = Tensor::<f32>::new(vec![1.0; qk], &vec![1, qk]);
        let mut c = Tensor::<f32>::default(&vec![1, 1]);
        matmul_transb_gguf_quant(&mut c, 0.0, &a, &wq, 1.0);

        assert!(c.close_to(&Tensor::<f32>::new(vec![0.0], &vec![1, 1]), 1e-6));
    }

    fn run_zero_block_prefill_case(ty: GGMLType) {
        let (qk, block_size) = spec(ty);
        let wq = QuantGGUFTensor {
            raw: vec![0u8; block_size],
            shape: vec![1, qk],
            tensor_type: ty,
            row_map: None,
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };

        let a = Tensor::<f32>::new(vec![1.0; qk * 2], &vec![2, qk]);
        let mut c = Tensor::<f32>::default(&vec![2, 1]);
        matmul_transb_gguf_quant(&mut c, 0.0, &a, &wq, 1.0);

        assert!(c.close_to(&Tensor::<f32>::new(vec![0.0, 0.0], &vec![2, 1]), 1e-6));
    }

    fn run_zero_block_batch2_case(ty: GGMLType) {
        let (qk, block_size) = spec(ty);
        let wq0 = QuantGGUFTensor {
            raw: vec![0u8; block_size],
            shape: vec![1, qk],
            tensor_type: ty,
            row_map: None,
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };
        let wq1 = QuantGGUFTensor {
            raw: vec![0u8; block_size],
            shape: vec![1, qk],
            tensor_type: ty,
            row_map: None,
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };

        let a = Tensor::<f32>::new(vec![1.0; qk * 2], &vec![2, qk]);
        let mut c0 = Tensor::<f32>::default(&vec![2, 1]);
        let mut c1 = Tensor::<f32>::default(&vec![2, 1]);
        matmul_transb_gguf_quant_batch2(&mut c0, &mut c1, 0.0, &a, &wq0, &wq1, 1.0);

        assert!(c0.close_to(&Tensor::<f32>::new(vec![0.0, 0.0], &vec![2, 1]), 1e-6));
        assert!(c1.close_to(&Tensor::<f32>::new(vec![0.0, 0.0], &vec![2, 1]), 1e-6));
    }

    fn run_zero_block_interleaved_prefill_case(ty: GGMLType) {
        let (qk, block_size) = spec(ty);
        let rows = 8;
        let wq = QuantGGUFTensor {
            raw: vec![0u8; rows * block_size],
            shape: vec![rows, qk],
            tensor_type: ty,
            row_map: None,
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };

        let a = Tensor::<f32>::new(vec![1.0; qk * 4], &vec![4, qk]);
        let mut c = Tensor::<f32>::default(&vec![4, rows]);
        matmul_transb_gguf_quant(&mut c, 0.0, &a, &wq, 1.0);

        assert!(c.close_to(&Tensor::<f32>::new(vec![0.0; 4 * rows], &vec![4, rows]), 1e-6));
    }

    fn run_zero_block_q8k_x4_prefill_case(ty: GGMLType) {
        with_q235k_q8k_x4_enabled(|| {
            let (qk, block_size) = spec(ty);
            let rows = 4;
            let wq = QuantGGUFTensor {
                raw: vec![0u8; rows * block_size],
                shape: vec![rows, qk],
                tensor_type: ty,
                row_map: None,
                prefill_workset: None,
                prefill_packed: None,
                prefill_k_metadata: None,
                prefill_q8k_interleave: None,
                prefill_stripes: None,
                hot_layer: false,
            };

            let a = Tensor::<f32>::new(vec![1.0; qk * 4], &vec![4, qk]);
            let mut c = Tensor::<f32>::default(&vec![4, rows]);
            matmul_transb_gguf_quant(&mut c, 0.0, &a, &wq, 1.0);

            assert!(c.close_to(&Tensor::<f32>::new(vec![0.0; 4 * rows], &vec![4, rows]), 1e-6));
        });
    }

    fn run_zero_block_q8k_x4_batch2_case(ty: GGMLType) {
        with_q235k_q8k_x4_enabled(|| {
            let (qk, block_size) = spec(ty);
            let rows = 4;
            let wq0 = QuantGGUFTensor {
                raw: vec![0u8; rows * block_size],
                shape: vec![rows, qk],
                tensor_type: ty,
                row_map: None,
                prefill_workset: None,
                prefill_packed: None,
                prefill_k_metadata: None,
                prefill_q8k_interleave: None,
                prefill_stripes: None,
                hot_layer: false,
            };
            let wq1 = QuantGGUFTensor {
                raw: vec![0u8; rows * block_size],
                shape: vec![rows, qk],
                tensor_type: ty,
                row_map: None,
                prefill_workset: None,
                prefill_packed: None,
                prefill_k_metadata: None,
                prefill_q8k_interleave: None,
                prefill_stripes: None,
                hot_layer: false,
            };

            let a = Tensor::<f32>::new(vec![1.0; qk * 4], &vec![4, qk]);
            let mut c0 = Tensor::<f32>::default(&vec![4, rows]);
            let mut c1 = Tensor::<f32>::default(&vec![4, rows]);
            matmul_transb_gguf_quant_batch2(&mut c0, &mut c1, 0.0, &a, &wq0, &wq1, 1.0);

            assert!(c0.close_to(&Tensor::<f32>::new(vec![0.0; 4 * rows], &vec![4, rows]), 1e-6));
            assert!(c1.close_to(&Tensor::<f32>::new(vec![0.0; 4 * rows], &vec![4, rows]), 1e-6));
        });
    }

    fn run_zero_block_interleaved_batch2_case(ty: GGMLType) {
        let (qk, block_size) = spec(ty);
        let rows = 8;
        let wq0 = QuantGGUFTensor {
            raw: vec![0u8; rows * block_size],
            shape: vec![rows, qk],
            tensor_type: ty,
            row_map: None,
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };
        let wq1 = QuantGGUFTensor {
            raw: vec![0u8; rows * block_size],
            shape: vec![rows, qk],
            tensor_type: ty,
            row_map: None,
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };

        let a = Tensor::<f32>::new(vec![1.0; qk * 4], &vec![4, qk]);
        let mut c0 = Tensor::<f32>::default(&vec![4, rows]);
        let mut c1 = Tensor::<f32>::default(&vec![4, rows]);
        matmul_transb_gguf_quant_batch2(&mut c0, &mut c1, 0.0, &a, &wq0, &wq1, 1.0);

        assert!(c0.close_to(&Tensor::<f32>::new(vec![0.0; 4 * rows], &vec![4, rows]), 1e-6));
        assert!(c1.close_to(&Tensor::<f32>::new(vec![0.0; 4 * rows], &vec![4, rows]), 1e-6));
    }

    fn run_zero_block_interleaved_prefill_with_packed_layout_case(ty: GGMLType) {
        let (qk, block_size) = spec(ty);
        let rows = 8;
        let mut wq = QuantGGUFTensor {
            raw: vec![0u8; rows * block_size],
            shape: vec![rows, qk],
            tensor_type: ty,
            row_map: Some(vec![6, 4, 2, 0, 7, 5, 3, 1]),
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };
        let layout = build_prefill_packed_layout(&wq).expect("packed layout should be built");
        wq.prefill_packed = Some(Arc::new(layout));

        let a = Tensor::<f32>::new(vec![1.0; qk * 4], &vec![4, qk]);
        let mut c = Tensor::<f32>::default(&vec![4, rows]);
        matmul_transb_gguf_quant(&mut c, 0.0, &a, &wq, 1.0);

        assert!(c.close_to(&Tensor::<f32>::new(vec![0.0; 4 * rows], &vec![4, rows]), 1e-6));
    }

    fn run_q8k_interleave_layout_rowmap_case(ty: GGMLType) {
        let (_qk, block_size) = spec(ty);
        let rows = 4;
        let blocks_per_row = 2;
        let mut raw = vec![0u8; rows * blocks_per_row * block_size];
        let row_map = vec![2usize, 0, 3, 1];

        for phys_row in 0..rows {
            for block in 0..blocks_per_row {
                let base = (phys_row * blocks_per_row + block) * block_size;
                raw[base..base + block_size].fill((phys_row * 10 + block) as u8);
            }
        }

        let wq = QuantGGUFTensor {
            raw,
            shape: vec![rows, blocks_per_row * spec(ty).0],
            tensor_type: ty,
            row_map: Some(row_map),
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };

        let layout = build_prefill_q8k_interleave_layout(&wq).expect("interleave layout should be built");
        assert_eq!(layout.rows, rows);
        assert_eq!(layout.blocks_per_row, blocks_per_row);
        assert_eq!(layout.block_size, block_size);

        for logical_row in 0..rows {
            let phys_row = wq.physical_row(logical_row);
            for block in 0..blocks_per_row {
                let expected = (phys_row * 10 + block) as u8;
                let packed = layout.block(block, logical_row).unwrap();
                assert!(packed.iter().all(|&v| v == expected));
            }
        }

        let packed_layout = build_prefill_packed_layout(&wq).expect("packed layout should be built");
        assert_eq!(packed_layout.rows, rows);
        assert_eq!(packed_layout.blocks_per_row, blocks_per_row);
        assert_eq!(packed_layout.block_size, block_size);
        for logical_row in 0..rows {
            let phys_row = wq.physical_row(logical_row);
            for block in 0..blocks_per_row {
                let expected = (phys_row * 10 + block) as u8;
                let packed = packed_layout.block(block, logical_row).unwrap();
                assert!(packed.iter().all(|&v| v == expected));
            }
        }
    }

    fn run_zero_block_packed_prefill_x1_case(ty: GGMLType) {
        let (qk, block_size) = spec(ty);
        let mut wq = QuantGGUFTensor {
            raw: vec![0u8; block_size],
            shape: vec![1, qk],
            tensor_type: ty,
            row_map: None,
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };
        let layout = build_prefill_packed_layout(&wq).expect("packed layout should be built");
        wq.prefill_packed = Some(Arc::new(layout));

        let a = Tensor::<f32>::new(vec![1.0; qk * 4], &vec![4, qk]);
        let mut c = Tensor::<f32>::default(&vec![4, 1]);
        matmul_transb_gguf_quant(&mut c, 0.0, &a, &wq, 1.0);

        assert!(c.close_to(&Tensor::<f32>::new(vec![0.0; 4], &vec![4, 1]), 1e-6));
    }

    #[test]
    fn test_quant_q4k_prefill_k_metadata_layout() {
        let rows = 2;
        let blocks_per_row = 1;
        let mut raw = vec![0u8; rows * 144];

        let block0 = &mut raw[0..144];
        block0[0..2].copy_from_slice(&half::f16::from_f32(1.5).to_bits().to_le_bytes());
        block0[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
        block0[4..16].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        let block1 = &mut raw[144..288];
        block1[0..2].copy_from_slice(&half::f16::from_f32(2.5).to_bits().to_le_bytes());
        block1[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        block1[4..16].copy_from_slice(&[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]);

        let expected_row0: Vec<(u8, u8)> = (0..8)
            .map(|idx| super::get_scale_min_k4(idx, &raw[148..160]))
            .collect();
        let expected_row1: Vec<(u8, u8)> = (0..8)
            .map(|idx| super::get_scale_min_k4(idx, &raw[4..16]))
            .collect();

        let wq = QuantGGUFTensor {
            raw,
            shape: vec![rows, blocks_per_row * 256],
            tensor_type: GGMLType::Q4K,
            row_map: Some(vec![1, 0]),
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };

        let meta = build_prefill_k_metadata_layout(&wq).expect("metadata should be built");
        let meta = match meta {
            super::QuantPrefillKMetadata::Q4K(meta) => meta,
            _ => panic!("unexpected metadata kind"),
        };

        assert_eq!(meta.rows, rows);
        assert_eq!(meta.blocks_per_row, blocks_per_row);
        assert_eq!(meta.d, vec![2.5, 1.5]);
        assert_eq!(meta.dmin, vec![0.5, 0.25]);

        let expected: Vec<(u8, u8)> = expected_row0.into_iter().chain(expected_row1).collect();

        for (idx, (scale, min)) in expected.into_iter().enumerate() {
            assert_eq!(meta.scales[idx], scale);
            assert_eq!(meta.mins[idx], min);
        }
    }

    #[test]
    fn test_quant_q6k_prefill_k_metadata_layout() {
        let rows = 2;
        let blocks_per_row = 1;
        let mut raw = vec![0u8; rows * 210];

        let block0 = &mut raw[0..210];
        block0[192..208].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136]);
        block0[208..210].copy_from_slice(&half::f16::from_f32(1.25).to_bits().to_le_bytes());

        let block1 = &mut raw[210..420];
        block1[192..208].copy_from_slice(&[11, 12, 13, 14, 15, 16, 17, 18, 139, 140, 141, 142, 143, 144, 145, 146]);
        block1[208..210].copy_from_slice(&half::f16::from_f32(2.25).to_bits().to_le_bytes());

        let wq = QuantGGUFTensor {
            raw,
            shape: vec![rows, blocks_per_row * 256],
            tensor_type: GGMLType::Q6K,
            row_map: Some(vec![1, 0]),
            prefill_workset: None,
            prefill_packed: None,
            prefill_k_metadata: None,
            prefill_q8k_interleave: None,
            prefill_stripes: None,
            hot_layer: false,
        };

        let meta = build_prefill_k_metadata_layout(&wq).expect("metadata should be built");
        let meta = match meta {
            super::QuantPrefillKMetadata::Q6K(meta) => meta,
            _ => panic!("unexpected metadata kind"),
        };

        assert_eq!(meta.rows, rows);
        assert_eq!(meta.blocks_per_row, blocks_per_row);
        assert_eq!(meta.d, vec![2.25, 1.25]);
        assert_eq!(
            &meta.scales[0..16],
            &[11, 12, 13, 14, 15, 16, 17, 18, -117, -116, -115, -114, -113, -112, -111, -110]
        );
        assert_eq!(
            &meta.scales[16..32],
            &[1, 2, 3, 4, 5, 6, 7, 8, -127, -126, -125, -124, -123, -122, -121, -120]
        );
    }

    #[test]
    fn test_quant_q4_0_zero() {
        run_zero_block_case(GGMLType::Q4_0);
    }

    #[test]
    fn test_quant_q4_1_zero() {
        run_zero_block_case(GGMLType::Q4_1);
    }

    #[test]
    fn test_quant_q5_0_zero() {
        run_zero_block_case(GGMLType::Q5_0);
    }

    #[test]
    fn test_quant_q5_1_zero() {
        run_zero_block_case(GGMLType::Q5_1);
    }

    #[test]
    fn test_quant_q8_0_zero() {
        run_zero_block_case(GGMLType::Q8_0);
    }

    #[test]
    fn test_quant_q2k_zero() {
        run_zero_block_case(GGMLType::Q2K);
    }

    #[test]
    fn test_quant_q3k_zero() {
        run_zero_block_case(GGMLType::Q3K);
    }

    #[test]
    fn test_quant_q4k_zero() {
        run_zero_block_case(GGMLType::Q4K);
    }

    #[test]
    fn test_quant_q5k_zero() {
        run_zero_block_case(GGMLType::Q5K);
    }

    #[test]
    fn test_quant_q6k_zero() {
        run_zero_block_case(GGMLType::Q6K);
    }

    #[test]
    fn test_quantize_q8k_zero_block() {
        let blk = quantize_activation_block_q8k(&[0.0; 256]);
        assert_eq!(blk.d, 0.0);
        assert!(blk.qs.iter().all(|&v| v == 0));
        assert!(blk.bsums.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantize_q80_zero_block() {
        let blk = quantize_activation_block_q80(&[0.0; 32]);
        assert_eq!(blk.d, 0.0);
        assert!(blk.qs.iter().all(|&v| v == 0));
        assert_eq!(blk.sum, 0);
    }

    #[test]
    fn test_q40_q80_zero_dot() {
        let blk = quantize_activation_block_q80(&[1.0; 32]);
        let raw = vec![0u8; 18];
        assert_eq!(q40_decode_block_dot_q80(&raw, &blk), 0.0);
    }

    #[test]
    fn test_q4k_q8k_zero_dot() {
        let blk = quantize_activation_block_q8k(&[1.0; 256]);
        let raw = vec![0u8; 144];
        assert_eq!(q4k_decode_block_dot_q8k(&raw, &blk), 0.0);
    }

    #[test]
    fn test_q2k_q8k_zero_dot() {
        let blk = quantize_activation_block_q8k(&[1.0; 256]);
        let raw = vec![0u8; 84];
        assert_eq!(q2k_decode_block_dot_q8k(&raw, &blk), 0.0);
    }

    #[test]
    fn test_q3k_q8k_zero_dot() {
        let blk = quantize_activation_block_q8k(&[1.0; 256]);
        let raw = vec![0u8; 110];
        assert_eq!(q3k_decode_block_dot_q8k(&raw, &blk), 0.0);
    }

    #[test]
    fn test_q5k_q8k_zero_dot() {
        let blk = quantize_activation_block_q8k(&[1.0; 256]);
        let raw = vec![0u8; 176];
        assert_eq!(q5k_decode_block_dot_q8k(&raw, &blk), 0.0);
    }

    #[test]
    fn test_q2k_q8k_x4_zero_dot() {
        let rows = vec![quantize_activation_block_q8k(&[1.0; 256]); 4];
        let blk = pack_q8k_block_x4(&rows, 4);
        let raw = vec![0u8; 84];
        let mut out = [1.0; 4];
        q2k_accumulate_block_dot_q8k_x4(&raw, &blk, &mut out);
        assert_eq!(out, [1.0; 4]);
    }

    #[test]
    fn test_q3k_q8k_x4_zero_dot() {
        let rows = vec![quantize_activation_block_q8k(&[1.0; 256]); 4];
        let blk = pack_q8k_block_x4(&rows, 4);
        let raw = vec![0u8; 110];
        let mut out = [1.0; 4];
        q3k_accumulate_block_dot_q8k_x4(&raw, &blk, &mut out);
        assert_eq!(out, [1.0; 4]);
    }

    #[test]
    fn test_q5k_q8k_x4_zero_dot() {
        let rows = vec![quantize_activation_block_q8k(&[1.0; 256]); 4];
        let blk = pack_q8k_block_x4(&rows, 4);
        let raw = vec![0u8; 176];
        let mut out = [1.0; 4];
        q5k_accumulate_block_dot_q8k_x4(&raw, &blk, &mut out);
        assert_eq!(out, [1.0; 4]);
    }

    #[test]
    fn test_q6k_q8k_zero_dot() {
        let blk = quantize_activation_block_q8k(&[1.0; 256]);
        let raw = vec![0u8; 210];
        assert_eq!(q6k_decode_block_dot_q8k(&raw, &blk), 0.0);
    }

    #[test]
    fn test_quant_q2k_prefill_zero() {
        run_zero_block_prefill_case(GGMLType::Q2K);
    }

    #[test]
    fn test_quant_q3k_prefill_zero() {
        run_zero_block_prefill_case(GGMLType::Q3K);
    }

    #[test]
    fn test_quant_q4k_prefill_zero() {
        run_zero_block_prefill_case(GGMLType::Q4K);
    }

    #[test]
    fn test_quant_q5k_prefill_zero() {
        run_zero_block_prefill_case(GGMLType::Q5K);
    }

    #[test]
    fn test_quant_q2k_q8k_x4_prefill_zero() {
        run_zero_block_q8k_x4_prefill_case(GGMLType::Q2K);
    }

    #[test]
    fn test_quant_q3k_q8k_x4_prefill_zero() {
        run_zero_block_q8k_x4_prefill_case(GGMLType::Q3K);
    }

    #[test]
    fn test_quant_q5k_q8k_x4_prefill_zero() {
        run_zero_block_q8k_x4_prefill_case(GGMLType::Q5K);
    }

    #[test]
    fn test_quant_q2k_q8k_x4_batch2_zero() {
        run_zero_block_q8k_x4_batch2_case(GGMLType::Q2K);
    }

    #[test]
    fn test_quant_q3k_q8k_x4_batch2_zero() {
        run_zero_block_q8k_x4_batch2_case(GGMLType::Q3K);
    }

    #[test]
    fn test_quant_q5k_q8k_x4_batch2_zero() {
        run_zero_block_q8k_x4_batch2_case(GGMLType::Q5K);
    }

    #[test]
    fn test_quant_q40_prefill_zero() {
        run_zero_block_prefill_case(GGMLType::Q4_0);
    }

    #[test]
    fn test_quant_q4k_batch2_zero() {
        run_zero_block_batch2_case(GGMLType::Q4K);
    }

    #[test]
    fn test_quant_q4k_interleaved_prefill_zero() {
        run_zero_block_interleaved_prefill_case(GGMLType::Q4K);
    }

    #[test]
    fn test_quant_q6k_interleaved_prefill_zero() {
        run_zero_block_interleaved_prefill_case(GGMLType::Q6K);
    }

    #[test]
    fn test_quant_q4k_interleaved_prefill_packed_layout_zero() {
        run_zero_block_interleaved_prefill_with_packed_layout_case(GGMLType::Q4K);
    }

    #[test]
    fn test_quant_q6k_interleaved_prefill_packed_layout_zero() {
        run_zero_block_interleaved_prefill_with_packed_layout_case(GGMLType::Q6K);
    }

    #[test]
    fn test_quant_q4k_interleaved_batch2_zero() {
        run_zero_block_interleaved_batch2_case(GGMLType::Q4K);
    }

    #[test]
    fn test_quant_q4k_q8k_interleave_layout_rowmap() {
        run_q8k_interleave_layout_rowmap_case(GGMLType::Q4K);
    }

    #[test]
    fn test_quant_q6k_q8k_interleave_layout_rowmap() {
        run_q8k_interleave_layout_rowmap_case(GGMLType::Q6K);
    }

    #[test]
    fn test_quant_q4k_packed_prefill_x1_zero() {
        run_zero_block_packed_prefill_x1_case(GGMLType::Q4K);
    }

    #[test]
    fn test_quant_q6k_packed_prefill_x1_zero() {
        run_zero_block_packed_prefill_x1_case(GGMLType::Q6K);
    }
}
