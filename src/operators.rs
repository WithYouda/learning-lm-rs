use crate::tensor::Tensor;
use num_traits::float::Float;
use num_traits::Num;
use num_traits::{FromPrimitive, ToPrimitive};
use std::cmp::Ordering;
// get (row) vectors from a 2D table given a list of indices
pub fn gather<T>(y: &mut Tensor<T>, indices: &Tensor<u32>, table: &Tensor<T>) 
    where T:  Default + Num + Copy + Float
{
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data_usize()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope<T>(y: &mut Tensor<T>, start_pos: usize, theta: impl Float) 
    where T: Float + Default + FromPrimitive
{
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = T::from(pos).unwrap()  / T::from(theta).unwrap().powf(T::from(i * 2).unwrap() / T::from(d).unwrap());
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax<T>(y: &mut Tensor<T>) 
    where T: Float + Default + std::iter::Sum + FromPrimitive
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
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<T>();

            (0..boundary).for_each(|j| data[offset + j] =  data[offset + j] / sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = T::from(0.0).unwrap());
        }
    }
}

pub fn rms_norm<T>(y: &mut Tensor<T>, x: &Tensor<T>, w: &Tensor<T>, epsilon: impl Float) 
    where T: Float + std::iter::Sum + Default 
{
    assert!(y.size() == x.size());
    // 获取维度数
    let ndim = y.shape().len();
    // 确保至少有2个维度
    assert!(ndim >= 2);
    // 序列的数量
    let seq_len = y.shape()[ndim - 2];
    // 每个序列的特征长度
    let total_seq_len = y.shape()[ndim - 1];
    // 获取维度数
    let wdim = w.shape().len();
    // 确保只有1个维度
    assert!(wdim == 1);
    // 确保长度相同
    assert!(w.size() == total_seq_len);
    // 批次数量
    let batch = y.size() / (seq_len * total_seq_len);
    // 获取数据的引用
    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    let _w = w.data();
    // 遍历每个批次
    for b in 0..batch {
        // 当前批次的基索引
        let base = b * seq_len * total_seq_len;
        // 遍历批次中的每个序列
        for l in 0..seq_len {
            // 当前序列的偏移量
            let offset = base + l * total_seq_len;
            // 平方和
            let s: T = _x[offset..offset + total_seq_len]
                .iter()
                .map(|f| f.powi(2))
                .sum();
            let total_seq_len_t = T::from(total_seq_len as f64).unwrap();
            let sqrt = (s / total_seq_len_t + T::from(epsilon).unwrap()).sqrt();
            // 计算并储存结果
            for i in 0..total_seq_len {
                _y[offset + i] = _w[i] * _x[offset + i] / sqrt;
            }
        }
    }
    
  // 修改前的代码
  /*
  let ndim = y.shape().len();
  assert!(ndim >= 2);

  let _y = unsafe{y.data_mut()};
  let _x = x.data();
  let _w = w.data();
  let seq = x.shape()[ndim - 2];
  let cow = x.shape()[ndim - 1];

  for i in 0..seq{
      let sum = (0..cow)
      .map(|j| {
          // 此处会出现访问数组越界错误！
          let squ = _x[i*cow + j].powf(2.0);
          squ
      })
      .sum::<f32>();
      let sum_squ = ((sum / cow as f32)+ epsilon).sqrt();
      for j in 0..cow{
          _y[i*cow + j] = ((_w[j]) * (_x[i*cow + j])) / sum_squ;
      }
  }
   */
}


// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu<T>(y: &mut Tensor<T>, x: &Tensor<T>) 
    where T: Float + Default + FromPrimitive + ToPrimitive + std::ops::MulAssign
{
    debug_assert!(y.size() == x.size());
    let silu_x = x.data().iter().cloned().map(|x| x / (T::one() + (-x).exp()));
    unsafe { y.data_mut() }
        .iter_mut()
        .zip(silu_x)
        .for_each(|(y, x)| *y *= x);
    }

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb<T>(c: &mut Tensor<T>, beta: T, a: &Tensor<T>, b: &Tensor<T>, alpha: T) 
    where T: Float + Default + std::iter::Sum
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

    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();

    assert!(a_col == b_col);
    assert!(c_col == b_row);
    assert!(a_row == c_row);

    for l in 0..c_row {
        for i in 0..c_col {
            let sum = (0..a_col)
                .map(|j| _a[l * a_col + j] * _b[i * b_col + j])
                .sum::<T>();
            _c[l * c_col + i] = beta * _c[l * c_col + i] + alpha * sum;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot<T>(x: &Tensor<T>, y: &Tensor<T>) -> T
    where T: Float + Default + ToPrimitive + FromPrimitive
{
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += (x_[i].to_f64().unwrap()) * (y_[i].to_f64().unwrap());
    }
    T::from(sum).unwrap()
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample<T>(x: &Tensor<T>, top_p: f32, top_k: u32, temperature: f32) -> u32 
    where T: Float + Default + FromPrimitive
{
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0.{
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
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability <T: Float + Copy>{
        val: T,
        tok: u32,
    }
    impl<T: Float + Copy> Eq for Probability<T> {}
    impl<T: Float + Copy> PartialOrd for Probability<T> {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl<T: Float + Copy> Ord for Probability<T> {
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
                tok: i as _,
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

// Your implementation should at least pass the following tests:
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
