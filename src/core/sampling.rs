use crate::core::tensor::Tensor;
use num_traits::{Float,FromPrimitive};
use std::cmp::Ordering;

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
    let mut logits: Vec<(u32, f32)> = x
        .data()
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let f: f32 = (*v).into();
            if f.is_finite() {
                Some((i as u32, f))
            } else {
                None
            }
        })
        .collect();

    if logits.is_empty() {
        return 0;
    }

    // repetition penalty
    if penalty != 1.0 && !history.is_empty() {
        for (tok, logit) in logits.iter_mut() {
            if history.contains(tok) {
                if *logit > 0.0 {
                    *logit /= penalty;
                } else {
                    *logit *= penalty;
                }
            }
        }
    }

    // greedy fallback
    if temperature <= 0.0 || top_p <= 0.0 {
        logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        return logits[0].0;
    }

    // temperature scaling
    for (_, logit) in logits.iter_mut() {
        *logit /= temperature;
    }

    // top-k pruning top-k 减枝
    logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let k = if top_k == 0 {
        logits.len()
    } else {
        (top_k as usize).min(logits.len())
    };
    logits.truncate(k.max(1));

    // softmax over candidates
    let max_logit = logits[0].1;
    let mut probs: Vec<(u32, f32)> = logits
        .into_iter()
        .map(|(tok, logit)| (tok, (logit - max_logit).exp()))
        .collect();
    let mut z: f32 = probs.iter().map(|(_, p)| *p).sum();
    if z <= 0.0 || !z.is_finite() {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        return probs[0].0;
    }
    for (_, p) in probs.iter_mut() {
        *p /= z;
    }

    // top-p pruning (probs already in descending-logit(降序) order) 
    if top_p < 1.0 {
        let mut cum = 0.0_f32;
        let mut keep = probs.len();
        for (idx, (_, p)) in probs.iter().enumerate() {
            cum += *p;
            if cum >= top_p {
                keep = idx + 1;
                break;
            }
        }
        probs.truncate(keep.max(1));
        z = probs.iter().map(|(_, p)| *p).sum();
        if z > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= z;
            }
        }
    }

    // multinomial sampling
    let r = rand::random::<f32>();
    let mut cdf = 0.0_f32;
    for (tok, p) in probs.iter() {
        cdf += *p;
        if r <= cdf {
            return *tok;
        }
    }
    probs.last().map(|(tok, _)| *tok).unwrap_or(0)
}