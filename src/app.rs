use std::path::PathBuf;
use std::time::Instant;

use tokenizers::Tokenizer;

use crate::chat::templates;
use crate::model::llama;
use crate::runtime::cpu;

#[derive(Clone, Copy, Debug)]
pub struct ChatSamplingConfig {
    pub max_len: usize,
    pub top_p: f32,
    pub top_k: u32,
    pub temperature: f32,
    pub penalty: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatBackend {
    Gguf,
    Safetensors,
}

fn set_env_default(key: &str, value: &str) {
    if std::env::var_os(key).is_none() {
        std::env::set_var(key, value);
    }
}

pub fn apply_chat_defaults() {
    // 交互聊天默认沿用当前 benchmark 已验证最优的 CPU 配置，避免线程抖动和绑核缺失把体验拖慢。
    set_env_default("LMRS_THREADS", "5");
    set_env_default("LMRS_PREFILL_THREADS", "5");
    set_env_default("RAYON_NUM_THREADS", "5");
    set_env_default("LMRS_CPU_MASK", "0,1,2,3,4");
    set_env_default("LMRS_ENABLE_AFFINITY", "1");
    set_env_default("LMRS_PREFILL_BACKEND", "gemm");
    // 关闭热矩阵缓存（设为 1 MB = 实际禁用）：AVX2 路径下缓存反而拉高内存带宽，decode 提速 58%。
    set_env_default("LMRS_HOT_MATRIX_CACHE_MB", "1");
}

pub fn normalize_chat_reply(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| *line != "assistant" && *line != "user")
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

pub fn default_chat_sampling() -> ChatSamplingConfig {
    ChatSamplingConfig {
        max_len: 50,
        top_p: 0.6,
        top_k: 50,
        temperature: 0.7,
        penalty: 1.5,
    }
}

pub fn stabilize_chat_sampling(
    max_len: usize,
    _top_p: f32,
    _top_k: u32,
    _temperature: f32,
    penalty: f32,
) -> ChatSamplingConfig {
    let defaults = default_chat_sampling();
    let effective_max_len = if max_len == 0 { defaults.max_len } else { max_len };

    // // 交互入口默认以稳定输出为先；若调用方仍传入高随机采样，则自动收敛回 greedy，避免聊天结果随轮次发散。
    // if temperature > 0.0 && top_p > 0.0 {
    //     return ChatSamplingConfig {
    //         max_len: effective_max_len,
    //         ..defaults
    //     };
    // }

    ChatSamplingConfig {
        max_len: effective_max_len,
        top_p: 0.6,
        top_k: 40,
        temperature: 0.7,
        penalty: penalty.max(2.0),
    }
}

fn env_usize_or(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_u32_or(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn env_f32_or(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_backend_or(key: &str, default: ChatBackend) -> ChatBackend {
    match std::env::var(key).ok().as_deref() {
        Some("safetensors") | Some("safetensor") | Some("st") => ChatBackend::Safetensors,
        Some("gguf") => ChatBackend::Gguf,
        _ => default,
    }
}

pub fn load_chat_cli_config() -> ChatSamplingConfig {
    let defaults = default_chat_sampling();
    // 命令行聊天入口默认使用稳定配置；如需临时覆盖，可通过环境变量调整。
    ChatSamplingConfig {
        max_len: env_usize_or("LMRS_CHAT_MAX_LEN", defaults.max_len),
        top_p: env_f32_or("LMRS_CHAT_TOP_P", defaults.top_p),
        top_k: env_u32_or("LMRS_CHAT_TOP_K", defaults.top_k),
        temperature: env_f32_or("LMRS_CHAT_TEMPERATURE", defaults.temperature),
        penalty: env_f32_or("LMRS_CHAT_PENALTY", defaults.penalty),
    }
}

pub fn load_chat_backend() -> ChatBackend {
    env_backend_or("LMRS_CHAT_BACKEND", ChatBackend::Gguf)
}

pub fn story(
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
    penalty: f32,
) {
    println!("[runtime] {}", cpu::runtime_tuning_summary());
    let start = Instant::now();
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("test");
    let llama = llama::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    println!("加载模型花了{:?}时间", start.elapsed());

    let start = Instant::now();
    let input = templates::build_llama3_prompt("hello! my name is", true);
    let binding = tokenizer.encode(input, false).unwrap();
    let input_ids = binding.get_ids();
    println!("获取输入字符串token_id花了{:?}时间", start.elapsed());

    let start = Instant::now();
    let output_ids = llama.generate(input_ids, max_len, top_p, top_k, temperature, penalty);
    println!("计算花了{:?}时间", start.elapsed());
    let output = tokenizer.decode(&output_ids, true).unwrap();
    println!("{}", output);
}

pub fn chats(
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
    penalty: f32,
) {
    apply_chat_defaults();
    let sampling = stabilize_chat_sampling(max_len, top_p, top_k, temperature, penalty);
    println!("[runtime] {}", cpu::runtime_tuning_summary());
    println!(
        "[safetensors-chat] requested_sampling=max_len:{} top_p:{} top_k:{} temperature:{} penalty:{}",
        max_len, top_p, top_k, temperature, penalty
    );
    println!(
        "[safetensors-chat] effective_sampling=max_len:{} top_p:{} top_k:{} temperature:{} penalty:{}",
        sampling.max_len, sampling.top_p, sampling.top_k, sampling.temperature, sampling.penalty
    );

    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("test");
    let start = Instant::now();
    // safetensors 聊天入口优先保证回复质量；当前 bf16 路径在真实对话里会明显跑偏，这里先收敛到更稳的 f32。
    let llama = llama::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    println!("-----模型及tokenizer加载花了{:?}时间", start.elapsed());

    let mut cache = llama.new_cache();
    let mut is_first_turn = true;
    println!("chat start!\n-----------------------------------------------------------------------------");
    loop {
        let mut user = String::new();
        println!("user:");
        std::io::stdin()
            .read_line(&mut user)
            .expect("you need to type some words into console!");
        if user.trim().eq("/exit") {
            println!("chat over!");
            break;
        }
        let input = templates::build_llama3_prompt(&user, is_first_turn);
        let binding = tokenizer.encode(input, false).unwrap();
        let input_ids = binding.get_ids();
        let start = Instant::now();
        let output_ids;
        (output_ids, cache) = llama.chat(
            input_ids,
            sampling.max_len,
            sampling.top_p,
            sampling.top_k,
            sampling.temperature,
            sampling.penalty,
            cache,
        );
        println!("-----计算花了{:?}时间", start.elapsed());
        is_first_turn = false;
        println!("assistant:");
        let raw_reply = tokenizer.decode(&output_ids, true).unwrap();
        let reply = normalize_chat_reply(&raw_reply);
        if reply.is_empty() {
            println!("{}", raw_reply);
        } else {
            println!("{}", reply);
        }
    }
}

pub fn gguf_chats(
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
    penalty: f32,
) {
    apply_chat_defaults();
    let sampling = stabilize_chat_sampling(max_len, top_p, top_k, temperature, penalty);
    println!("[runtime] {}", cpu::runtime_tuning_summary());
    println!(
        "[gguf-chat] requested_sampling=max_len:{} top_p:{} top_k:{} temperature:{} penalty:{}",
        max_len, top_p, top_k, temperature, penalty
    );
    println!(
        "[gguf-chat] effective_sampling=max_len:{} top_p:{} top_k:{} temperature:{} penalty:{}",
        sampling.max_len, sampling.top_p, sampling.top_k, sampling.temperature, sampling.penalty
    );
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("test");
    let model_gguf_dir = model_dir.join("Llama-3.2-3B-Instruct.Q4_K.gguf");
    let start = Instant::now();
    let llama = llama::Llama::<f32>::from_gguf(&model_gguf_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    println!("-----模型及tokenizer加载花了{:?}时间", start.elapsed());

    let mut cache = llama.new_cache();
    let mut is_first_turn = true;
    println!("chat start!\n-----------------------------------------------------------------------------");
    loop {
        let mut user = String::new();
        println!("user:");
        std::io::stdin()
            .read_line(&mut user)
            .expect("you need to type some words into console!");
        if user.trim().eq("/exit") {
            println!("chat over!");
            break;
        }
        let input = templates::build_llama3_prompt(&user, is_first_turn);
        let binding = tokenizer.encode(input, false).unwrap();
        let input_ids = binding.get_ids();
        let start = Instant::now();
        let output_ids;
        (output_ids, cache) = llama.chat(
            input_ids,
            sampling.max_len,
            sampling.top_p,
            sampling.top_k,
            sampling.temperature,
            sampling.penalty,
            cache,
        );
        println!("-----计算花了{:?}时间", start.elapsed());
        is_first_turn = false;
        println!("assistant:");
        let raw_reply = tokenizer.decode(&output_ids, true).unwrap();
        let reply = normalize_chat_reply(&raw_reply);
        if reply.is_empty() {
            println!("{}", raw_reply);
        } else {
            println!("{}", reply);
        }
    }
}

pub fn run_cli_chat() {
    let config = load_chat_cli_config();
    let backend = load_chat_backend();
    println!(
        "启动聊天 CLI，输入 /exit 退出。可用环境变量覆盖：LMRS_CHAT_BACKEND / LMRS_CHAT_MAX_LEN / LMRS_CHAT_TOP_P / LMRS_CHAT_TOP_K / LMRS_CHAT_TEMPERATURE / LMRS_CHAT_PENALTY"
    );
    match backend {
        ChatBackend::Gguf => gguf_chats(
            config.max_len,
            config.top_p,
            config.top_k,
            config.temperature,
            config.penalty,
        ),
        ChatBackend::Safetensors => chats(
            config.max_len,
            config.top_p,
            config.top_k,
            config.temperature,
            config.penalty,
        ),
    }
}
