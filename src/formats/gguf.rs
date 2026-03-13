use crate::core::operators::quant::generic::{
    build_prefill_k_metadata_layout,
    build_prefill_packed_layout,
    build_prefill_q8k_interleave_layout,
    build_prefill_quant_stripe_layout,
};
//use crate::model::llama::*;
use crate::core::tensor::Tensor;
use crate::model::params::*;
use crate::runtime::cpu;

use gguf::{GGMLType, GGUFFile, GGUFMetadataValue, GGUfMetadataValueType};
use std::sync::Arc;


/// prefill 阶段的量化条带常驻布局。
///
/// 说明：
/// - 不把整块权重落成 f32 dense；
/// - 仅把量化原始块按「block tile -> row tile」的访问顺序重排，
///   让 gate/up/down 的 prefill 内核更接近顺序访存；
/// - 总字节量与原始 quant raw 同量级，远小于 dense workset。
#[derive(Debug)]
pub struct QuantPrefillStripeLayout {
    pub row_tile: usize,
    pub block_tile: usize,
    pub rows: usize,
    pub blocks_per_row: usize,
    pub stripes: Vec<Vec<u8>>,
}

impl QuantPrefillStripeLayout {
    pub fn row_tiles(&self) -> usize {
        self.rows.div_ceil(self.row_tile.max(1))
    }

    pub fn stripe_index(&self, block_start: usize, row_start: usize) -> Option<usize> {
        if block_start % self.block_tile != 0 || row_start % self.row_tile != 0 {
            return None;
        }
        let block_idx = block_start / self.block_tile;
        let row_idx = row_start / self.row_tile;
        let row_tiles = self.row_tiles();
        let block_tiles = self.blocks_per_row.div_ceil(self.block_tile.max(1));
        if row_idx >= row_tiles || block_idx >= block_tiles {
            return None;
        }
        Some(block_idx * row_tiles + row_idx)
    }

    pub fn stripe(&self, block_start: usize, row_start: usize) -> Option<&[u8]> {
        self.stripe_index(block_start, row_start)
            .and_then(|idx| self.stripes.get(idx).map(|stripe| stripe.as_slice()))
    }

    pub fn bytes(&self) -> usize {
        self.stripes.iter().map(|stripe| stripe.len()).sum()
    }
}

/// 面向 `Q8_K x4` prefill 微内核的加载期重排布局。
///
/// 说明：
/// - 原始 `raw` 仍保留 GGUF 行主序块布局；
/// - 这里额外缓存一份按 `block -> logical row` 顺序重排后的只读视图；
/// - 这样 prefill 微内核在固定 `block` 下扫描输出列时，可直接顺序读连续块，
///   同时把 `row_map` 的 q/k 反排列也前移到加载期消化掉。
#[derive(Debug)]
pub struct QuantPrefillQ8KInterleaveLayout {
    pub rows: usize,
    pub blocks_per_row: usize,
    pub block_size: usize,
    pub packed: Vec<u8>,
}

impl QuantPrefillQ8KInterleaveLayout {
    pub fn block(&self, block_idx: usize, logical_row: usize) -> Option<&[u8]> {
        if block_idx >= self.blocks_per_row || logical_row >= self.rows {
            return None;
        }
        let base = (block_idx * self.rows + logical_row) * self.block_size;
        Some(&self.packed[base..base + self.block_size])
    }

    pub fn bytes(&self) -> usize {
        self.packed.len()
    }
}

#[derive(Debug)]
pub struct QuantQ4KPrefillMetadata {
    pub rows: usize,
    pub blocks_per_row: usize,
    pub d: Vec<f32>,
    pub dmin: Vec<f32>,
    pub scales: Vec<u8>,
    pub mins: Vec<u8>,
}

#[derive(Debug)]
pub struct QuantQ6KPrefillMetadata {
    pub rows: usize,
    pub blocks_per_row: usize,
    pub d: Vec<f32>,
    pub scales: Vec<i8>,
}

#[derive(Debug)]
pub enum QuantPrefillKMetadata {
    Q4K(QuantQ4KPrefillMetadata),
    Q6K(QuantQ6KPrefillMetadata),
}

impl QuantPrefillKMetadata {
    pub fn bytes(&self) -> usize {
        match self {
            Self::Q4K(meta) => {
                meta.d.len() * std::mem::size_of::<f32>()
                    + meta.dmin.len() * std::mem::size_of::<f32>()
                    + meta.scales.len() * std::mem::size_of::<u8>()
                    + meta.mins.len() * std::mem::size_of::<u8>()
            }
            Self::Q6K(meta) => {
                meta.d.len() * std::mem::size_of::<f32>()
                    + meta.scales.len() * std::mem::size_of::<i8>()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPrefillPackedLayoutKind {
    BlockMajorQ8K,
}

/// prefill 阶段的主布局。
///
/// 说明：
/// - 阶段 H 起，优先把 `Q4_K/Q6_K` 的 prefill 常驻信息收敛到一份布局里；
/// - `packed` 保存给微内核直接消费的 block-major 顺序；
/// - metadata 与 packed 绑定，避免再拆成独立 side-car 管理。
#[derive(Debug)]
pub struct QuantPrefillPackedLayout {
    pub rows: usize,
    pub blocks_per_row: usize,
    pub block_size: usize,
    pub kind: QuantPrefillPackedLayoutKind,
    pub packed: Vec<u8>,
    pub metadata: Option<QuantPrefillKMetadata>,
}

impl QuantPrefillPackedLayout {
    pub fn block(&self, block_idx: usize, logical_row: usize) -> Option<&[u8]> {
        if block_idx >= self.blocks_per_row || logical_row >= self.rows {
            return None;
        }
        match self.kind {
            QuantPrefillPackedLayoutKind::BlockMajorQ8K => {
                let base = (block_idx * self.rows + logical_row) * self.block_size;
                Some(&self.packed[base..base + self.block_size])
            }
        }
    }

    pub fn metadata(&self) -> Option<&QuantPrefillKMetadata> {
        self.metadata.as_ref()
    }

    pub fn bytes(&self) -> usize {
        self.packed.len()
            + self
                .metadata
                .as_ref()
                .map(|meta| meta.bytes())
                .unwrap_or(0)
    }
}


/// 通用 GGUF 量化张量容器。
///
/// 说明：
/// - raw 仍保持 GGUF 原始块布局；
/// - tensor_type 指定具体量化格式（Q4_0/Q5K/Q6K...）；
/// - row_map 用于 q/k 的行反排列（逻辑行 -> 物理行）。
pub struct QuantGGUFTensor {
    pub raw: Vec<u8>,
    pub shape: Vec<usize>,
    pub tensor_type: GGMLType,
    pub row_map: Option<Vec<usize>>,
    pub prefill_workset: Option<Arc<Vec<f32>>>,
    pub prefill_packed: Option<Arc<QuantPrefillPackedLayout>>,
    pub prefill_k_metadata: Option<Arc<QuantPrefillKMetadata>>,
    pub prefill_q8k_interleave: Option<Arc<QuantPrefillQ8KInterleaveLayout>>,
    pub prefill_stripes: Option<Arc<QuantPrefillStripeLayout>>,
    // 标记热点层：仅在 decode 阶段（m=1）启用行级预解码缓存。
    pub hot_layer: bool,
}

impl QuantGGUFTensor {
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    pub fn physical_row(&self, logical_row: usize) -> usize {
        self.row_map
            .as_ref()
            .map(|m| m[logical_row])
            .unwrap_or(logical_row)
    }
}


fn find_meta_value<'a>(gguf: &'a GGUFFile, key: &str) -> Option<&'a GGUFMetadataValue> {
    gguf.header
        .metadata
        .iter()
        .find(|m| m.key == key)
        .map(|m| &m.value)
}

fn meta_u64(gguf: &GGUFFile, key: &str) -> Option<u64> {
    match find_meta_value(gguf, key)? {
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
}

fn read_le_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn read_le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_le_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn align_up(x: usize, a: usize) -> usize {
    if a == 0 {
        return x;
    }
    (x + a - 1) & !(a - 1)
}

fn skip_gguf_string(bytes: &[u8], idx: &mut usize) {
    let len = read_le_u64(bytes, *idx) as usize;
    *idx += 8;
    *idx += len;
}

fn skip_metadata_value(bytes: &[u8], idx: &mut usize, value_type: GGUfMetadataValueType) {
    match value_type {
        GGUfMetadataValueType::Uint8 | GGUfMetadataValueType::Int8 | GGUfMetadataValueType::Bool => {
            *idx += 1;
        }
        GGUfMetadataValueType::Uint16 | GGUfMetadataValueType::Int16 => {
            *idx += 2;
        }
        GGUfMetadataValueType::Uint32
        | GGUfMetadataValueType::Int32
        | GGUfMetadataValueType::Float32 => {
            *idx += 4;
        }
        GGUfMetadataValueType::Uint64
        | GGUfMetadataValueType::Int64
        | GGUfMetadataValueType::Float64 => {
            *idx += 8;
        }
        GGUfMetadataValueType::String => {
            skip_gguf_string(bytes, idx);
        }
        GGUfMetadataValueType::Array => {
            let inner = read_le_u32(bytes, *idx);
            *idx += 4;
            let len = read_le_u64(bytes, *idx) as usize;
            *idx += 8;
            let inner_ty = GGUfMetadataValueType::try_from(inner).unwrap();
            for _ in 0..len {
                skip_metadata_value(bytes, idx, inner_ty);
            }
        }
    }
}

fn gguf_tensor_data_start(gguf: &GGUFFile, bytes: &[u8]) -> usize {
    let mut idx = 0usize;
    assert!(&bytes[idx..idx + 4] == b"GGUF");
    idx += 4; // magic
    idx += 4; // version
    let tensor_count = read_le_u64(bytes, idx) as usize;
    idx += 8;
    let metadata_count = read_le_u64(bytes, idx) as usize;
    idx += 8;

    for _ in 0..metadata_count {
        skip_gguf_string(bytes, &mut idx); // key
        let ty = read_le_u32(bytes, idx);
        idx += 4;
        let value_type = GGUfMetadataValueType::try_from(ty).unwrap();
        skip_metadata_value(bytes, &mut idx, value_type);
    }

    for _ in 0..tensor_count {
        skip_gguf_string(bytes, &mut idx); // name
        let n_dims = read_le_u32(bytes, idx) as usize;
        idx += 4;
        idx += 8 * n_dims; // dimensions
        idx += 4; // tensor type
        idx += 8; // offset
    }

    let alignment = meta_u64(gguf, "general.alignment").unwrap_or(32) as usize;
    align_up(idx, alignment)
}

// GGUF q/k projections are stored in a row-permuted layout for RoPE-friendly kernels.
// Convert back to the canonical row order expected by this implementation.
fn unpermute_qk_rows<T: Copy + Default>(src: Tensor<T>, n_heads: usize) -> Tensor<T> {
    let shape = src.shape().to_vec();
    assert!(shape.len() == 2, "q/k weight must be 2D");
    let rows = shape[0];
    let cols = shape[1];
    assert!(n_heads > 0 && rows % n_heads == 0, "invalid q/k rows or head count");

    let head_dim = rows / n_heads;
    assert!(head_dim % 2 == 0, "head_dim must be even for q/k unpermute");
    let g = head_dim / 2;

    let in_data = src.data();
    let mut out = vec![T::default(); rows * cols];

    for h in 0..n_heads {
        for i in 0..g {
            for t in 0..2 {
                let rp = h * (2 * g) + i * 2 + t;
                let ro = h * (2 * g) + t * g + i;
                let src_row = &in_data[rp * cols..(rp + 1) * cols];
                let dst_row = &mut out[ro * cols..(ro + 1) * cols];
                dst_row.copy_from_slice(src_row);
            }
        }
    }

    Tensor::new(out, &shape)
}

// 构建 q/k 行反排列映射：逻辑行 -> GGUF 物理行。
// 这样做后，量化权重无需先解码成 dense 再重排，kernel 可直接按映射访问。
fn build_qk_unpermute_row_map(rows: usize, n_heads: usize) -> Vec<usize> {
    assert!(n_heads > 0 && rows % n_heads == 0, "invalid q/k rows or head count");
    let head_dim = rows / n_heads;
    assert!(head_dim % 2 == 0, "head_dim must be even for q/k unpermute");
    let g = head_dim / 2;

    let mut map = vec![0usize; rows];
    for h in 0..n_heads {
        for t in 0..2 {
            for i in 0..g {
                let ro = h * (2 * g) + t * g + i;
                let rp = h * (2 * g) + i * 2 + t;
                map[ro] = rp;
            }
        }
    }
    map
}

#[inline]
fn gguf_prefill_workset_budget_bytes() -> usize {
    std::env::var("LMRS_PREFILL_WORKSET_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1024)
        * 1024
        * 1024
}

#[inline]
fn gguf_prefill_stripe_budget_bytes() -> usize {
    std::env::var("LMRS_PREFILL_STRIPE_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(256)
        * 1024
        * 1024
}

#[inline]
fn gguf_prefill_packed_budget_bytes() -> usize {
    std::env::var("LMRS_PREFILL_PACKED_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or_else(|| {
            gguf_prefill_q8k_interleave_budget_bytes()
                .saturating_add(gguf_prefill_k_metadata_budget_bytes())
        })
}

    #[inline]
    fn gguf_prefill_q8k_interleave_budget_bytes() -> usize {
        // 阶段 D 先保留为实验能力：
        // 只有连续口径和交错口径都稳定改善时，才允许回到默认主路径。
        std::env::var("LMRS_PREFILL_Q8K_INTERLEAVE_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
        * 1024
        * 1024
    }

    #[inline]
    fn gguf_prefill_k_metadata_budget_bytes() -> usize {
        // 阶段 G 已实现，但 5 轮结果显示它不能默认保留在主路径；
        // 只有显式给出非零预算时才启用。
        std::env::var("LMRS_PREFILL_K_METADATA_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
        * 1024
        * 1024
    }

#[inline]
fn gguf_gateup_workset_enabled() -> bool {
    std::env::var("LMRS_PREFILL_GATEUP_WORKSET")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[inline]
fn gguf_ffn_stripes_enabled() -> bool {
    std::env::var("LMRS_PREFILL_FFN_STRIPES")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// 反量化方法，将量化的格式全部转为 f32
fn read_gguf_tensor_as_f32(
    gguf: &GGUFFile,
    bytes: &[u8],
    tensor_data_start: usize,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let info = gguf
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("tensor not found in gguf: {name}"))?;

    let shape: Vec<usize> = info.dimensions.iter().rev().map(|d| *d as usize).collect();
    let n_elem = info
        .dimensions
        .iter()
        .fold(1usize, |acc, d| acc.saturating_mul(*d as usize));

    let start = tensor_data_start + info.offset as usize;

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

    #[inline]
    fn read_u32_le(bytes: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
    }

    #[inline]
    fn unpack_q3k_scales(scales12: &[u8]) -> [i8; 16] {
        // 对齐 ggml dequantize_row_q3_K
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

    let data = match info.tensor_type {
        GGMLType::F32 => {
            let byte_len = n_elem * 4;
            let end = start + byte_len;
            let mut out = Vec::with_capacity(n_elem);
            for chunk in bytes[start..end].chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            out
        }
        GGMLType::F16 => {
            let byte_len = n_elem * 2;
            let end = start + byte_len;
            let mut out = Vec::with_capacity(n_elem);
            for chunk in bytes[start..end].chunks_exact(2) {
                let h = read_le_u16(chunk, 0);
                out.push(half::f16::from_bits(h).to_f32());
            }
            out
        }
        GGMLType::Q8_0 => {
            // ggml block_q8_0 layout:
            // d: fp16 scale (2 bytes), qs: 32 x i8 (32 bytes)
            const QK8_0: usize = 32;
            const BLOCK_SIZE: usize = 2 + QK8_0;
            if n_elem % QK8_0 != 0 {
                return Err(format!(
                    "invalid q8_0 tensor element count for {name}: {n_elem} is not divisible by {QK8_0}"
                ));
            }
            let n_blocks = n_elem / QK8_0;
            let byte_len = n_blocks * BLOCK_SIZE;
            let end = start + byte_len;
            let raw = &bytes[start..end];

            let mut out = Vec::with_capacity(n_elem);
            for b in 0..n_blocks {
                let base = b * BLOCK_SIZE;
                let d_bits = read_le_u16(raw, base);
                let d = half::f16::from_bits(d_bits).to_f32();
                for i in 0..QK8_0 {
                    let q = i8::from_le_bytes([raw[base + 2 + i]]) as f32;
                    out.push(d * q);
                }
            }
            out
        }
        GGMLType::Q4K => {
            // ggml block_q4_K layout (QK_K = 256):
            // d: fp16, dmin: fp16, scales[12], qs[128]
            const QK_K: usize = 256;
            const SCALES_SIZE: usize = 12;
            const QS_SIZE: usize = QK_K / 2;
            const BLOCK_SIZE: usize = 2 + 2 + SCALES_SIZE + QS_SIZE;

            if n_elem % QK_K != 0 {
                return Err(format!(
                    "invalid q4_k tensor element count for {name}: {n_elem} is not divisible by {QK_K}"
                ));
            }

            let n_blocks = n_elem / QK_K;
            let byte_len = n_blocks * BLOCK_SIZE;
            let end = start + byte_len;
            let raw = &bytes[start..end];

            let mut out = Vec::with_capacity(n_elem);
            for b in 0..n_blocks {
                let base = b * BLOCK_SIZE;
                let d = half::f16::from_bits(read_le_u16(raw, base)).to_f32();
                let dmin = half::f16::from_bits(read_le_u16(raw, base + 2)).to_f32();
                let scales = &raw[base + 4..base + 4 + SCALES_SIZE];
                let mut q = &raw[base + 4 + SCALES_SIZE..base + BLOCK_SIZE];

                let mut is = 0usize;
                for _ in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let d1 = d * sc1 as f32;
                    let m1f = dmin * m1 as f32;

                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc2 as f32;
                    let m2f = dmin * m2 as f32;

                    for l in 0..32 {
                        let v = (q[l] & 0x0f) as f32;
                        out.push(d1 * v - m1f);
                    }
                    for l in 0..32 {
                        let v = (q[l] >> 4) as f32;
                        out.push(d2 * v - m2f);
                    }

                    q = &q[32..];
                    is += 2;
                }
            }
            out
        },
        GGMLType::Q4_0 => {
            const QK: usize = 32;
            const BLOCK: usize = 2 + QK / 2; // d(fp16) + qs[16]
            if n_elem % QK != 0 {
                return Err(format!("invalid q4_0 tensor element count for {name}: {n_elem}"));
            }
            let n_blocks = n_elem / QK;
            let raw = &bytes[start..start + n_blocks * BLOCK];
            let mut out = Vec::with_capacity(n_elem);

            for b in 0..n_blocks {
                let base = b * BLOCK;
                let d = half::f16::from_bits(read_le_u16(raw, base)).to_f32();
                let qs = &raw[base + 2..base + BLOCK];
                for j in 0..(QK / 2) {
                    let x0 = (qs[j] & 0x0f) as i32 - 8;
                    let x1 = (qs[j] >> 4) as i32 - 8;
                    out.push(d * x0 as f32);
                    out.push(d * x1 as f32);
                }
            }
            out
        }
        GGMLType::Q4_1 => {
            const QK: usize = 32;
            const BLOCK: usize = 2 + 2 + QK / 2; // d,m,qs[16]
            if n_elem % QK != 0 {
                return Err(format!("invalid q4_1 tensor element count for {name}: {n_elem}"));
            }
            let n_blocks = n_elem / QK;
            let raw = &bytes[start..start + n_blocks * BLOCK];
            let mut out = Vec::with_capacity(n_elem);

            for b in 0..n_blocks {
                let base = b * BLOCK;
                let d = half::f16::from_bits(read_le_u16(raw, base)).to_f32();
                let m = half::f16::from_bits(read_le_u16(raw, base + 2)).to_f32();
                let qs = &raw[base + 4..base + BLOCK];
                for j in 0..(QK / 2) {
                    let x0 = (qs[j] & 0x0f) as f32;
                    let x1 = (qs[j] >> 4) as f32;
                    out.push(d * x0 + m);
                    out.push(d * x1 + m);
                }
            }
            out
        }
        GGMLType::Q5_0 => {
            const QK: usize = 32;
            const BLOCK: usize = 2 + 4 + QK / 2; // d + qh(u32) + qs[16]
            if n_elem % QK != 0 {
                return Err(format!("invalid q5_0 tensor element count for {name}: {n_elem}"));
            }
            let n_blocks = n_elem / QK;
            let raw = &bytes[start..start + n_blocks * BLOCK];
            let mut out = Vec::with_capacity(n_elem);

            for b in 0..n_blocks {
                let base = b * BLOCK;
                let d = half::f16::from_bits(read_le_u16(raw, base)).to_f32();
                let qh = read_u32_le(raw, base + 2);
                let qs = &raw[base + 6..base + BLOCK];

                for j in 0..(QK / 2) {
                    let xh0 = (((qh >> (j + 0)) << 4) & 0x10) as i32;
                    let xh1 = (((qh >> (j + 12)) & 0x10)) as i32;
                    let x0 = ((qs[j] & 0x0f) as i32 | xh0) - 16;
                    let x1 = ((qs[j] >> 4) as i32 | xh1) - 16;
                    out.push(d * x0 as f32);
                    out.push(d * x1 as f32);
                }
            }
            out
        }
        GGMLType::Q5_1 => {
            const QK: usize = 32;
            const BLOCK: usize = 2 + 2 + 4 + QK / 2; // d,m,qh,qs
            if n_elem % QK != 0 {
                return Err(format!("invalid q5_1 tensor element count for {name}: {n_elem}"));
            }
            let n_blocks = n_elem / QK;
            let raw = &bytes[start..start + n_blocks * BLOCK];
            let mut out = Vec::with_capacity(n_elem);

            for b in 0..n_blocks {
                let base = b * BLOCK;
                let d = half::f16::from_bits(read_le_u16(raw, base)).to_f32();
                let m = half::f16::from_bits(read_le_u16(raw, base + 2)).to_f32();
                let qh = read_u32_le(raw, base + 4);
                let qs = &raw[base + 8..base + BLOCK];

                for j in 0..(QK / 2) {
                    let xh0 = (((qh >> (j + 0)) << 4) & 0x10) as i32;
                    let xh1 = (((qh >> (j + 12)) & 0x10)) as i32;
                    let x0 = (qs[j] & 0x0f) as i32 | xh0;
                    let x1 = (qs[j] >> 4) as i32 | xh1;
                    out.push(d * x0 as f32 + m);
                    out.push(d * x1 as f32 + m);
                }
            }
            out
        }
        GGMLType::Q2K => {
            // block_q2_K: scales[16], qs[64], d(fp16), dmin(fp16)
            const QK: usize = 256;
            const BLOCK: usize = 16 + 64 + 2 + 2;
            if n_elem % QK != 0 {
                return Err(format!("invalid q2k tensor element count for {name}: {n_elem}"));
            }
            let n_blocks = n_elem / QK;
            let raw = &bytes[start..start + n_blocks * BLOCK];
            let mut out = Vec::with_capacity(n_elem);

            for b in 0..n_blocks {
                let base = b * BLOCK;
                let scales = &raw[base..base + 16];
                let q = &raw[base + 16..base + 80];
                let d = half::f16::from_bits(read_le_u16(raw, base + 80)).to_f32();
                let dmin = half::f16::from_bits(read_le_u16(raw, base + 82)).to_f32();

                let mut is = 0usize;
                let mut qp = q;
                for _n in (0..QK).step_by(128) {
                    let mut shift = 0usize;
                    for _j in 0..4 {
                        let sc0 = scales[is];
                        is += 1;
                        let dl0 = d * (sc0 & 0x0f) as f32;
                        let ml0 = dmin * (sc0 >> 4) as f32;
                        for l in 0..16 {
                            out.push(dl0 * ((qp[l] >> shift) & 0x03) as f32 - ml0);
                        }

                        let sc1 = scales[is];
                        is += 1;
                        let dl1 = d * (sc1 & 0x0f) as f32;
                        let ml1 = dmin * (sc1 >> 4) as f32;
                        for l in 0..16 {
                            out.push(dl1 * ((qp[l + 16] >> shift) & 0x03) as f32 - ml1);
                        }

                        shift += 2;
                    }
                    qp = &qp[32..];
                }
            }
            out
        }
        GGMLType::Q3K => {
            // block_q3_K: hmask[32], qs[64], scales[12], d(fp16)
            const QK: usize = 256;
            const BLOCK: usize = 32 + 64 + 12 + 2;
            if n_elem % QK != 0 {
                return Err(format!("invalid q3k tensor element count for {name}: {n_elem}"));
            }
            let n_blocks = n_elem / QK;
            let raw = &bytes[start..start + n_blocks * BLOCK];
            let mut out = Vec::with_capacity(n_elem);

            for b in 0..n_blocks {
                let base = b * BLOCK;
                let hm = &raw[base..base + 32];
                let mut q = &raw[base + 32..base + 96];
                let scales = unpack_q3k_scales(&raw[base + 96..base + 108]);
                let d_all = half::f16::from_bits(read_le_u16(raw, base + 108)).to_f32();

                let mut is = 0usize;
                let mut m: u8 = 1;
                for _n in (0..QK).step_by(128) {
                    let mut shift = 0usize;
                    for _j in 0..4 {
                        let dl0 = d_all * scales[is] as f32;
                        is += 1;
                        for l in 0..16 {
                            let lo = ((q[l] >> shift) & 0x03) as i8;
                            let hi = if (hm[l] & m) != 0 { 0 } else { 4 };
                            out.push(dl0 * (lo - hi) as f32);
                        }

                        let dl1 = d_all * scales[is] as f32;
                        is += 1;
                        for l in 0..16 {
                            let lo = ((q[l + 16] >> shift) & 0x03) as i8;
                            let hi = if (hm[l + 16] & m) != 0 { 0 } else { 4 };
                            out.push(dl1 * (lo - hi) as f32);
                        }

                        shift += 2;
                        m <<= 1;
                    }
                    q = &q[32..];
                }
            }
            out
        }
        GGMLType::Q5K => {
            // block_q5_K: d(fp16), dmin(fp16), scales[12], qh[32], qs[128]
            const QK: usize = 256;
            const BLOCK: usize = 2 + 2 + 12 + 32 + 128;
            if n_elem % QK != 0 {
                return Err(format!("invalid q5k tensor element count for {name}: {n_elem}"));
            }
            let n_blocks = n_elem / QK;
            let raw = &bytes[start..start + n_blocks * BLOCK];
            let mut out = Vec::with_capacity(n_elem);

            for b in 0..n_blocks {
                let base = b * BLOCK;
                let d = half::f16::from_bits(read_le_u16(raw, base)).to_f32();
                let dmin = half::f16::from_bits(read_le_u16(raw, base + 2)).to_f32();
                let scales = &raw[base + 4..base + 16];
                let qh = &raw[base + 16..base + 48];
                let mut ql = &raw[base + 48..base + BLOCK];

                let mut is = 0usize;
                let mut u1: u8 = 1;
                let mut u2: u8 = 2;
                for _j in (0..QK).step_by(64) {
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
                        out.push(d1 * v as f32 - mm1);
                    }
                    for l in 0..32 {
                        let v = (ql[l] >> 4) as i32 + if (qh[l] & u2) != 0 { 16 } else { 0 };
                        out.push(d2 * v as f32 - mm2);
                    }

                    ql = &ql[32..];
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            out
        }
        GGMLType::Q6K => {
            // block_q6_K: ql[128], qh[64], scales[16], d(fp16)
            const QK: usize = 256;
            const BLOCK: usize = 128 + 64 + 16 + 2;
            if n_elem % QK != 0 {
                return Err(format!("invalid q6k tensor element count for {name}: {n_elem}"));
            }
            let n_blocks = n_elem / QK;
            let raw = &bytes[start..start + n_blocks * BLOCK];
            let mut out = Vec::with_capacity(n_elem);

            for b in 0..n_blocks {
                let base = b * BLOCK;
                let mut ql = &raw[base..base + 128];
                let mut qh = &raw[base + 128..base + 192];
                let mut sc = &raw[base + 192..base + 208];
                let d = half::f16::from_bits(read_le_u16(raw, base + 208)).to_f32();

                for _n in (0..QK).step_by(128) {
                    let mut block = [0.0f32; 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i8 - 32;
                        let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8 - 32;

                        block[l + 0] = d * (sc[is + 0] as i8) as f32 * q1 as f32;
                        block[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        block[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        block[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                    out.extend_from_slice(&block);
                    ql = &ql[64..];
                    qh = &qh[32..];
                    sc = &sc[8..];
                }
            }
            out
        },
        _ => {
            return Err(format!(
                "unsupported gguf tensor type for {name}: {:?} (need F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2K/Q3K/Q4K/Q5K/Q6K )",
                info.tensor_type
            ));
        }
    };

    Ok((data, shape))
}

fn gguf_quant_block_size(t: GGMLType) -> Option<usize> {
    match t {
        GGMLType::Q4_0 => Some(2 + 16),
        GGMLType::Q4_1 => Some(2 + 2 + 16),
        GGMLType::Q5_0 => Some(2 + 4 + 16),
        GGMLType::Q5_1 => Some(2 + 2 + 4 + 16),
        GGMLType::Q8_0 => Some(2 + 32),
        GGMLType::Q2K => Some(16 + 64 + 2 + 2),
        GGMLType::Q3K => Some(32 + 64 + 12 + 2),
        GGMLType::Q4K => Some(2 + 2 + 12 + 128),
        GGMLType::Q5K => Some(2 + 2 + 12 + 32 + 128),
        GGMLType::Q6K => Some(128 + 64 + 16 + 2),
        _ => None,
    }
}

fn gguf_quant_qk(t: GGMLType) -> Option<usize> {
    match t {
        GGMLType::Q4_0 | GGMLType::Q4_1 | GGMLType::Q5_0 | GGMLType::Q5_1 | GGMLType::Q8_0 => Some(32),
        GGMLType::Q2K | GGMLType::Q3K | GGMLType::Q4K | GGMLType::Q5K | GGMLType::Q6K => Some(256),
        _ => None,
    }
}

fn read_gguf_generic_quant_tensor(
    gguf: &GGUFFile,
    bytes: &[u8],
    tensor_data_start: usize,
    name: &str,
    row_map: Option<Vec<usize>>,
    prefill_workset: Option<Arc<Vec<f32>>>,
    hot_layer: bool,
) -> Result<QuantGGUFTensor, String> {
    let info = gguf
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("tensor not found in gguf: {name}"))?;

    let qk = gguf_quant_qk(info.tensor_type)
        .ok_or_else(|| format!("tensor {name} is not supported quant type: {:?}", info.tensor_type))?;
    let block_size = gguf_quant_block_size(info.tensor_type)
        .ok_or_else(|| format!("tensor {name} block_size unknown for type: {:?}", info.tensor_type))?;

    let shape: Vec<usize> = info.dimensions.iter().rev().map(|d| *d as usize).collect();
    if shape.len() != 2 {
        return Err(format!("quant tensor {name} is not 2D, shape={shape:?}"));
    }

    let n_elem = info
        .dimensions
        .iter()
        .fold(1usize, |acc, d| acc.saturating_mul(*d as usize));
    if n_elem % qk != 0 {
        return Err(format!(
            "invalid quant tensor element count for {name}: {n_elem} is not divisible by {qk}"
        ));
    }

    let n_blocks = n_elem / qk;
    let byte_len = n_blocks * block_size;
    let start = tensor_data_start + info.offset as usize;
    let end = start + byte_len;

    Ok(QuantGGUFTensor {
        raw: bytes[start..end].to_vec(),
        shape,
        tensor_type: info.tensor_type,
        row_map,
        prefill_workset,
        prefill_packed: None,
        prefill_k_metadata: None,
        prefill_q8k_interleave: None,
        prefill_stripes: None,
        hot_layer,
    })
}

macro_rules! impl_from_gguf_for_LlamaParams {
    ($Param:ty) => {
        impl LLamaParams<$Param> {
            pub fn from_gguf(
                gguf: &GGUFFile,
                bytes: &[u8],
                n_layers: usize,
                n_q_h: usize,
                n_kv_h: usize,
                tie_word_embeddings: bool,
            ) -> Self {
                let tensor_data_start = gguf_tensor_data_start(gguf, bytes);
                let force_dense = std::env::var("LMRS_FORCE_DENSE_GGUF").is_ok();
                let decode_hot_layer = std::env::var("LMRS_DECODE_HOT_LAYER")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok());
                let total_prefill_workset_budget = gguf_prefill_workset_budget_bytes();
                let mut total_prefill_stripe_budget = gguf_prefill_stripe_budget_bytes();
                let mut total_prefill_packed_budget = gguf_prefill_packed_budget_bytes();
                let mut total_prefill_k_metadata_budget = gguf_prefill_k_metadata_budget_bytes();
                let mut total_prefill_q8k_interleave_budget = gguf_prefill_q8k_interleave_budget_bytes();
                let mut qkv_prefill_workset_budget = total_prefill_workset_budget / 2;
                let mut gate_up_prefill_workset_budget = total_prefill_workset_budget / 2;

                let load = |name: &str| -> Tensor<$Param> {
                    let (vals, shape) = read_gguf_tensor_as_f32(gguf, bytes, tensor_data_start, name)
                        .unwrap_or_else(|e| panic!("{e}"));
                    Tensor::new(<$Param as FromF32>::from_f32_vec(vals), &shape)
                };

                // For linear layers we try to keep known quantized formats in compressed form.
                // If tensor type is not supported yet, we fallback to dense loading.
                let mut load_weight = |name: &str| -> Weight<$Param> {
                    // q/k 在 GGUF 中通常是行重排布局，这里只构建映射，不做 dense 反量化。
                    let is_q = name.contains(".attn_q.weight");
                    let is_k = name.contains(".attn_k.weight");
                    let is_v = name.contains(".attn_v.weight");
                    let is_o = name.contains(".attn_output.weight");
                    let is_gate = name.contains(".ffn_gate.weight");
                    let is_up = name.contains(".ffn_up.weight");
                    let is_down = name.contains(".ffn_down.weight");

                    if force_dense {
                        // 调试模式下仍可走 dense，用于与量化路径做 A/B 对照。
                        if is_q {
                            return Weight::Dense(unpermute_qk_rows(load(name), n_q_h));
                        }
                        if is_k {
                            return Weight::Dense(unpermute_qk_rows(load(name), n_kv_h));
                        }
                        return Weight::Dense(load(name));
                    }

                    let info = gguf
                        .tensors
                        .iter()
                        .find(|t| t.name == name)
                        .unwrap_or_else(|| panic!("tensor not found in gguf: {name}"));

                    // GGUF 的 dimensions 对 2D 权重通常是 [cols, rows]，
                    // 这里取最后一维作为逻辑 rows，避免把 cols 误当 rows。
                    let logical_rows = *info
                        .dimensions
                        .last()
                        .unwrap_or_else(|| panic!("invalid tensor dims for {name}")) as usize;

                    let row_map = if is_q {
                        Some(build_qk_unpermute_row_map(
                            logical_rows,
                            n_q_h,
                        ))
                    } else if is_k {
                        Some(build_qk_unpermute_row_map(
                            logical_rows,
                            n_kv_h,
                        ))
                    } else {
                        None
                    };

                    let enable_gateup_workset = gguf_gateup_workset_enabled();
                    let allow_prefill_workset = is_q || is_k || is_v || (enable_gateup_workset && (is_gate || is_up));

                    let prefill_workset = if allow_prefill_workset {
                        let workset_bytes = logical_rows
                            .saturating_mul(*info.dimensions.first().unwrap_or(&0) as usize)
                            .saturating_mul(std::mem::size_of::<f32>());
                        let budget_slot = if is_gate || is_up {
                            &mut gate_up_prefill_workset_budget
                        } else {
                            &mut qkv_prefill_workset_budget
                        };

                        if workset_bytes > 0 && workset_bytes <= *budget_slot {
                            let (vals, shape) = read_gguf_tensor_as_f32(gguf, bytes, tensor_data_start, name)
                                .unwrap_or_else(|e| panic!("{e}"));
                            let vals = if is_q {
                                unpermute_qk_rows(Tensor::new(vals, &shape), n_q_h).data().to_vec()
                            } else if is_k {
                                unpermute_qk_rows(Tensor::new(vals, &shape), n_kv_h).data().to_vec()
                            } else {
                                vals
                            };
                            *budget_slot = budget_slot.saturating_sub(workset_bytes);
                            Some(Arc::new(vals))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // 热点层策略：优先覆盖 decode 阶段占比高、开销大的矩阵。
                    let mut hot_layer = name.contains(".attn_v.weight")
                        || name.contains(".attn_output.weight")
                        || name.contains(".ffn_down.weight");

                    if let Some(target_layer) = decode_hot_layer {
                        if let Some(rest) = name.strip_prefix("blk.") {
                            if let Some((id_str, _)) = rest.split_once('.') {
                                if let Ok(id) = id_str.parse::<usize>() {
                                    if id == target_layer {
                                        hot_layer = true;
                                    }
                                }
                            }
                        }
                    }

                    if gguf_quant_qk(info.tensor_type).is_some() {
                        let mut qt = read_gguf_generic_quant_tensor(
                            gguf,
                            bytes,
                            tensor_data_start,
                            name,
                            row_map,
                            prefill_workset,
                            hot_layer,
                        )
                        .unwrap_or_else(|e| panic!("{e}"));

                        cpu::maybe_advise_hugepage(qt.raw.as_mut_ptr(), qt.raw.len());
                        cpu::maybe_prepare_prefill_weight_pages(qt.raw.as_ptr(), qt.raw.len());

                        let allow_prefill_packed = is_q || is_k || is_v || is_o || is_gate || is_up || is_down;
                        if allow_prefill_packed {
                            if let Some(mut layout) = build_prefill_packed_layout(&qt) {
                                let layout_bytes = layout.bytes();
                                if layout_bytes > 0 && layout_bytes <= total_prefill_packed_budget {
                                    total_prefill_packed_budget = total_prefill_packed_budget
                                        .saturating_sub(layout_bytes);
                                    cpu::maybe_advise_hugepage(layout.packed.as_mut_ptr(), layout.packed.len());
                                    cpu::maybe_prepare_prefill_weight_pages(layout.packed.as_ptr(), layout.packed.len());
                                    qt.prefill_packed = Some(Arc::new(layout));
                                }
                            }
                        }

                        if let Some(meta) = build_prefill_k_metadata_layout(&qt) {
                            let meta_bytes = meta.bytes();
                            if meta_bytes > 0 && meta_bytes <= total_prefill_k_metadata_budget {
                                total_prefill_k_metadata_budget = total_prefill_k_metadata_budget
                                    .saturating_sub(meta_bytes);
                                qt.prefill_k_metadata = Some(Arc::new(meta));
                            }
                        }

                        let allow_prefill_q8k_interleave = is_q || is_k || is_v || is_o || is_gate || is_up || is_down;
                        let raw_bytes = qt.raw.len();
                        if allow_prefill_q8k_interleave
                            && raw_bytes > 0
                            && raw_bytes <= total_prefill_q8k_interleave_budget
                        {
                            if let Some(layout) = build_prefill_q8k_interleave_layout(&qt) {
                                let layout_bytes = layout.bytes();
                                if layout_bytes > 0 && layout_bytes <= total_prefill_q8k_interleave_budget {
                                    total_prefill_q8k_interleave_budget = total_prefill_q8k_interleave_budget
                                        .saturating_sub(layout_bytes);
                                    cpu::maybe_advise_hugepage(layout.packed.as_ptr() as *mut u8, layout.packed.len());
                                    cpu::maybe_prepare_prefill_weight_pages(layout.packed.as_ptr(), layout.packed.len());
                                    qt.prefill_q8k_interleave = Some(Arc::new(layout));
                                }
                            }
                        }

                        // FFN 的 gate/up/down 改走轻量条带常驻布局：
                        // 只复制量化原始块并按 prefill 访问顺序重排，不再整块落成 dense。
                        if gguf_ffn_stripes_enabled() && (is_gate || is_up || is_down) {
                            if raw_bytes > 0 && raw_bytes <= total_prefill_stripe_budget {
                                if let Some(stripes) = build_prefill_quant_stripe_layout(&qt) {
                                    let stripe_bytes = stripes.bytes();
                                    if stripe_bytes > 0 && stripe_bytes <= total_prefill_stripe_budget {
                                        total_prefill_stripe_budget = total_prefill_stripe_budget.saturating_sub(stripe_bytes);
                                        qt.prefill_stripes = Some(Arc::new(stripes));
                                    }
                                }
                            }
                        }

                        Weight::GgufQ(qt)
                    } else {
                        Weight::Dense(load(name))
                    }
                };

                let load_layer = |pattern: &str| -> Vec<Tensor<$Param>> {
                    (0..n_layers)
                        .map(|i| {
                            let name = pattern.replace("{}", &i.to_string());
                            load(&name)
                        })
                        .collect()
                };

                let embedding = load("token_embd.weight");
                let lm_head = if tie_word_embeddings {
                    load_weight("token_embd.weight")
                } else if gguf.tensors.iter().any(|t| t.name == "output.weight") {
                    load_weight("output.weight")
                } else {
                    load_weight("token_embd.weight")
                };
                let rms_att_w = load_layer("blk.{}.attn_norm.weight");
                let wq = (0..n_layers)
                    .map(|i| load_weight(&format!("blk.{}.attn_q.weight", i)))
                    .collect();
                let wk = (0..n_layers)
                    .map(|i| load_weight(&format!("blk.{}.attn_k.weight", i)))
                    .collect();
                let wv = (0..n_layers)
                    .map(|i| load_weight(&format!("blk.{}.attn_v.weight", i)))
                    .collect();
                let wo = (0..n_layers)
                    .map(|i| load_weight(&format!("blk.{}.attn_output.weight", i)))
                    .collect();
                let rms_ffn_w = load_layer("blk.{}.ffn_norm.weight");
                let w_up = (0..n_layers)
                    .map(|i| load_weight(&format!("blk.{}.ffn_up.weight", i)))
                    .collect();
                let w_gate = (0..n_layers)
                    .map(|i| load_weight(&format!("blk.{}.ffn_gate.weight", i)))
                    .collect();
                let w_down = (0..n_layers)
                    .map(|i| load_weight(&format!("blk.{}.ffn_down.weight", i)))
                    .collect();
                let rms_out_w = load("output_norm.weight");

                LLamaParams {
                    embedding_table: embedding,
                    rms_att_w,
                    wq,
                    wk,
                    wv,
                    wo,
                    rms_ffn_w,
                    w_up,
                    w_gate,
                    w_down,
                    rms_out_w,
                    lm_head,
                }
            }
        }
    };
}

impl_from_gguf_for_LlamaParams!(f32);
impl_from_gguf_for_LlamaParams!(half::f16);
impl_from_gguf_for_LlamaParams!(half::bf16);
