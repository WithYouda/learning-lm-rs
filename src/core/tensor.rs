use std::{mem::size_of, slice, sync::Arc, vec};

use crate::runtime::cpu;

/// 核心张量结构体，基于 Arc 共享数据所有权。
/// 支持零拷贝切片（slice）和形状重解释（reshape），
/// 是模型权重、中间激活值、KV 缓存等的统一载体。
pub struct Tensor<T> {
    /// 共享的底层数据，多个 Tensor 可指向同一份内存
    data: Arc<Box<[T]>>,
    /// 当前张量的形状（如 [batch, seq_len, hidden_size]）
    shape: Vec<usize>,
    /// 在底层 data 中的起始偏移（用于 slice 零拷贝）
    offset: usize,
    /// 当前张量包含的元素数量
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    /// 从给定数据和形状创建新张量，自动对大块内存应用大页提示以优化访存性能
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let mut data = data;
        let length = data.len();
        // 第四阶段的大页提示只对大块连续内存生效，小张量不值得触发 madvise。
        cpu::maybe_advise_hugepage(data.as_mut_ptr() as *mut u8, length * size_of::<T>());
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    /// 创建指定形状的零初始化张量
    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    /// 获取当前张量的只读数据切片（已考虑 offset 和 length）
    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    /// 获取当前张量的可变数据切片（unsafe：绕过 Arc 共享保护）
    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    /// 获取当前张量的形状
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// 获取当前张量的元素总数
    pub fn size(&self) -> usize {
        self.length
    }

    /// 将张量重新解释为新形状（不改变底层数据，要求元素总数一致）
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    /// 零拷贝切片：从 start 位置取出指定形状的子张量，共享底层数据
    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }
}

/// 测试和调试用的辅助方法（仅 f32 类型可用）
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();

        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shpae: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

/// 浮点数相对误差比较，rel 为允许的相对误差阈值
#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}
