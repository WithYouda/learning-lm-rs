use crate::core::kvcache::KVCache;

/// 多轮会话状态的占位结构体，尚未完整实现。
#[allow(dead_code)]
pub struct ChatSession<T>{
    session_ids: u32,
    history_ids: Vec<u32>,
    cache: KVCache<T>,
}

