use crate::tensor::Tensor;
use num_traits::float::Float;
use num_traits::Num;
use num_traits::{FromPrimitive, ToPrimitive};
use std::cmp::Ordering;
use std::fmt::Debug;
use gemm::{Parallelism};
use std::any::TypeId;
use rayon::prelude::*;

// 可性能优化！
// get (row) vectors from a 2D table given a list of indices
/// 为什么这里的 indices 的类型是 Tensor<u32> 呢？
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
/// RoPE: Rotary Positional Embedding 旋转位置编码
/// 需要性能优化
pub fn rope<T>(y: &mut Tensor<T>, start_pos: usize, theta: impl Float) 
    where T: Float + Default + FromPrimitive +  Copy + Into<f32>
{
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    let d_f32 = d as f32;
    let two_f32 = 2.0_f32;
    let theta_f32: f32 = theta.to_f32().unwrap_or(0.0);
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        let pos_f32 = pos as f32;
        for head in 0..n_heads {
            let base_offset = tok * n_heads * d + head * d;
            for i in 0..d / 2 {
                let i_f32 = i as f32;
                let idx_a = base_offset + i;
                let idx_b = base_offset + i + (d/2);
                let a_f32: f32 = data[idx_a].into();
                let b_f32: f32 = data[idx_b].into();
                let exponent = (two_f32 * i_f32) / d_f32;
                let freq = pos_f32 / theta_f32.powf(exponent);
                let (sin, cos) = freq.sin_cos();
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
pub fn masked_softmax<T>(y: &mut Tensor<T>) 
    where T: Float + Default + std::iter::Sum + FromPrimitive + Debug
{
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
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
    where T: Float + std::iter::Sum + Default
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
    where T: Float + Default + FromPrimitive + ToPrimitive + Copy
{
    debug_assert!(y.size() == x.size());
    let y_data = unsafe {y.data_mut()};
    let x_data = x.data();
    y_data.iter_mut().zip(x_data.iter()).for_each(|(y_val,x_val)|{
        let x_f32 = x_val.to_f32().unwrap_or(0.0);
        let exp_neg_x = (-x_f32).exp();
        let silu_val_f32 = x_f32 / (1.0 + exp_neg_x);
        let silu_val = T::from(silu_val_f32).unwrap_or(T::zero());
        *y_val = *y_val * silu_val;
    });
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
                Parallelism::Rayon(0),
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

/// f32 转换计算改造完成
/// Sample a index from a tensor (treated as a probability vector)
/// 从张量中根据温度、top-k和top-p随机采样出一个词元的索引
/// top-p 累计概率阈值，只保留累计概率达到 top-p 的这些词
/// top-k 只保留概率最高的 k 个词
/// temperature: 温度系数。
/// temp > 1: 增加随机性（分布更平坦）。
/// temp < 1: 减少随机性（分布更尖锐，倾向于高概率词）。
/// temp = 0 或极低：退化为贪婪搜索（Greedy Search），只选概率最大的词。
pub fn random_sample<T>(x: &Tensor<T>, history: &Vec::<u32>, top_p: f32, top_k: u32, temperature: f32, penalty: f32) -> u32 
    where T: Float + Default + FromPrimitive + Into<f32>
{
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    // 当采样参数无效或温度为零时，退化为“贪婪模式”（Greedy Search），直接返回概率（或 Logits）最大的那个词的索引
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .filter(|(_, i)| i.is_normal() || i.is_zero())
            .max_by(|(_, &a), (_, &b)| {
                if a > b {
                    Ordering::Greater
                }else if a < b{
                    Ordering::Less
                }else{
                    Ordering::Equal
                }
            })
            .unwrap()
            .0 as u32;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability <T: Float + Copy>{
        val: T,
        tok: u32,
    }
    impl<T: Float + Copy> Eq for Probability<T>{}
    impl<T: Float + Copy> PartialOrd for Probability<T> {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl<T: Float + Copy> Ord for Probability<T>{
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            if self.val > other.val {
                std::cmp::Ordering::Less
            } else if self.val < other.val {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        }
    }
    impl<T: Float + Copy> From<(usize, &T)> for Probability<T> {
        #[inline]
        fn from((i, p): (usize, &T)) -> Self {
            Self {
                val: *p,
                tok: i as u32,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .filter(|(_, i)| i.is_normal() || i.is_zero())
        .map(Probability::from)
        .collect::<Vec<_>>();

    // 惩罚
    if penalty != 1.0 && !history.is_empty(){
        for item in logits.iter_mut(){
            if history.contains(&item.tok){
                let mut val_f32: f32 = item.val.into();
                if val_f32 > 0.0{
                    val_f32 /= penalty;
                }else{
                    val_f32 *= penalty;
                }
                item.val = T::from(val_f32).unwrap_or(T::zero());
            }
        }
    }

    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, T::from(1.).unwrap());
    // softmax & sum
    let temperature = T::from(temperature).unwrap();
    logits.iter_mut().skip(1).fold(T::one(), |prev, p| {
        p.val = prev + ((p.val - max) / temperature).exp();
        p.val
    });
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * T::from(top_p).unwrap();
    let plimit = T::from(rand::random::<f32>()).unwrap() * T::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
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
