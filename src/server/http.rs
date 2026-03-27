// HTTP 服务器实现
// 提供 REST API 和 SSE 流式推理，支持多用户并发

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::path::{Path, PathBuf};
use std::fs;

use tokenizers::Tokenizer;
use serde_json::{json, Value};

use crate::app::apply_chat_defaults;
use crate::chat::templates;
use crate::core::kvcache::KVCache;
use crate::core::sampling;
use crate::core::tensor::Tensor;
use crate::model::llama::Llama;

/// 嵌入前端网页HTML
const INDEX_HTML: &str = include_str!("../../web/index.html");

// ============================================================================
// 数据结构定义
// ============================================================================

/// 可用模型的描述信息
#[derive(Clone)]
struct ModelInfo {
    /// 显示名称（如 "Llama-3.2-3B-Instruct.Q4_K"）
    name: String,
    /// 模型文件或目录路径
    path: PathBuf,
    /// tokenizer.json 的路径
    tokenizer_path: PathBuf,
    /// 后端类型: "gguf" 或 "safetensors"
    backend: String,
}

/// 已加载到内存中的模型（模型权重+tokenizer）
struct LoadedModel {
    model: Arc<Llama<f32>>,
    tokenizer: Arc<Tokenizer>,
}

/// 单个对话会话的状态
struct Session {
    /// 当前使用的模型名
    model_name: String,
    /// 系统提示词
    system_prompt: String,
    /// KV缓存（推理期间会被暂时取走）
    cache: Option<KVCache<f32>>,
    /// 历史token id（用于重复惩罚）
    history_ids: Vec<u32>,
    /// 是否是第一轮对话
    is_first_turn: bool,
    /// 取消推理的原子标志
    cancel_flag: Arc<AtomicBool>,
    /// 是否正在生成中的原子标志
    generating: Arc<AtomicBool>,
}

/// 服务器全局共享状态
struct ServerState {
    /// 扫描到的可用模型列表
    available_models: Vec<ModelInfo>,
    /// 已加载到内存的模型缓存（按名称索引）
    loaded_models: Mutex<HashMap<String, LoadedModel>>,
    /// 所有活跃会话（按会话ID索引）
    sessions: Mutex<HashMap<u64, Session>>,
    /// 会话ID自增计数器
    next_session_id: AtomicU64,
}

/// 解析后的HTTP请求
struct HttpRequest {
    method: String,
    path: String,
    #[allow(dead_code)]
    headers: HashMap<String, String>,
    body: String,
}

// ============================================================================
// HTTP 解析与响应工具函数
// ============================================================================

/// 从TCP流中解析HTTP请求
fn parse_request(reader: &mut BufReader<&TcpStream>) -> Option<HttpRequest> {
    // 读取请求行
    let mut first_line = String::new();
    reader.read_line(&mut first_line).ok()?;
    let parts: Vec<&str> = first_line.trim().split(' ').collect();
    if parts.len() < 2 {
        return None;
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    // 读取请求头
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            headers.insert(k.trim().to_lowercase(), v.trim().to_string());
        }
    }

    // 读取请求体（根据Content-Length）
    let mut body = String::new();
    if let Some(len_str) = headers.get("content-length") {
        if let Ok(len) = len_str.parse::<usize>() {
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf).ok()?;
            body = String::from_utf8_lossy(&buf).to_string();
        }
    }

    Some(HttpRequest { method, path, headers, body })
}

/// 发送JSON格式的HTTP响应
fn send_json(stream: &mut TcpStream, status: u16, body: &Value) {
    let body_str = body.to_string();
    let status_text = match status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n\r\n{}",
        status, status_text, body_str.len(), body_str
    );
    let _ = stream.write_all(response.as_bytes());
}

/// 发送HTML格式的HTTP响应
fn send_html(stream: &mut TcpStream, html: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-cache\r\n\r\n{}",
        html.len(), html
    );
    let _ = stream.write_all(response.as_bytes());
}

/// 发送HTML响应头（用于HEAD请求，不包含响应体）
fn send_html_head(stream: &mut TcpStream, html: &str) {
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-cache\r\n\r\n",
        html.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.flush();
}

/// 发送SSE（Server-Sent Events）的响应头
fn send_sse_header(stream: &mut TcpStream) {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.flush();
}

/// 发送一条SSE事件数据，返回是否写入成功
fn send_sse_data(stream: &mut TcpStream, data: &str) -> bool {
    let event = format!("data: {}\n\n", data);
    stream.write_all(event.as_bytes()).is_ok() && stream.flush().is_ok()
}

/// 发送CORS预检请求的响应
fn send_cors_preflight(stream: &mut TcpStream) {
    let response = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nAccess-Control-Max-Age: 86400\r\n\r\n";
    let _ = stream.write_all(response.as_bytes());
}

// ============================================================================
// 服务器状态管理
// ============================================================================

impl ServerState {
    /// 创建服务器状态，扫描models目录下的可用模型
    fn new(models_dir: &Path) -> Self {
        let available_models = Self::scan_models(models_dir);
        println!("[server] 扫描到 {} 个可用模型:", available_models.len());
        for m in &available_models {
            println!("  - {} ({})", m.name, m.backend);
        }
        ServerState {
            available_models,
            loaded_models: Mutex::new(HashMap::new()),
            sessions: Mutex::new(HashMap::new()),
            next_session_id: AtomicU64::new(1),
        }
    }

    /// 扫描models目录，自动发现gguf和safetensors格式的模型
    fn scan_models(models_dir: &Path) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        if !models_dir.is_dir() {
            return models;
        }

        // 遍历models下的每个子目录
        if let Ok(entries) = fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let subdir = entry.path();
                if !subdir.is_dir() {
                    continue;
                }

                // 检查该目录下是否有tokenizer.json
                let tokenizer_path = subdir.join("tokenizer.json");
                if !tokenizer_path.exists() {
                    continue;
                }

                // 扫描GGUF文件
                if let Ok(files) = fs::read_dir(&subdir) {
                    for file in files.flatten() {
                        let file_path = file.path();
                        if let Some(ext) = file_path.extension() {
                            if ext == "gguf" {
                                let name = file_path.file_stem()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string();
                                models.push(ModelInfo {
                                    name,
                                    path: file_path,
                                    tokenizer_path: tokenizer_path.clone(),
                                    backend: "gguf".to_string(),
                                });
                            }
                        }
                    }
                }

                // 检查是否有safetensors模型（必须有config.json）
                let config_path = subdir.join("config.json");
                let st_path = subdir.join("model.safetensors");
                if config_path.exists() && st_path.exists() {
                    let dir_name = subdir.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    models.push(ModelInfo {
                        name: format!("{}-safetensors", dir_name),
                        path: subdir.clone(),
                        tokenizer_path,
                        backend: "safetensors".to_string(),
                    });
                }
            }
        }
        models
    }

    /// 确保指定模型已加载到内存，返回模型和tokenizer的Arc引用
    fn ensure_model_loaded(&self, model_name: &str) -> Result<(Arc<Llama<f32>>, Arc<Tokenizer>), String> {
        // 先检查是否已加载
        {
            let loaded = self.loaded_models.lock().unwrap();
            if let Some(lm) = loaded.get(model_name) {
                return Ok((Arc::clone(&lm.model), Arc::clone(&lm.tokenizer)));
            }
        }

        // 找到模型信息
        let info = self.available_models.iter()
            .find(|m| m.name == model_name)
            .ok_or_else(|| format!("模型 '{}' 不存在", model_name))?
            .clone();

        println!("[server] 开始加载模型: {} ({})", info.name, info.backend);
        let start = Instant::now();

        // 根据后端类型加载模型
        let model = match info.backend.as_str() {
            "gguf" => Llama::<f32>::from_gguf(&info.path),
            "safetensors" => Llama::<f32>::from_safetensors(&info.path),
            _ => return Err(format!("未知的后端类型: {}", info.backend)),
        };
        let tokenizer = Tokenizer::from_file(&info.tokenizer_path)
            .map_err(|e| format!("加载tokenizer失败: {}", e))?;

        println!("[server] 模型加载完成，耗时: {:?}", start.elapsed());

        let model = Arc::new(model);
        let tokenizer = Arc::new(tokenizer);

        // 缓存已加载的模型
        let mut loaded = self.loaded_models.lock().unwrap();
        loaded.insert(model_name.to_string(), LoadedModel {
            model: Arc::clone(&model),
            tokenizer: Arc::clone(&tokenizer),
        });

        Ok((model, tokenizer))
    }
}

// ============================================================================
// API 路由处理
// ============================================================================

/// 处理单个HTTP连接
fn handle_connection(state: Arc<ServerState>, stream: TcpStream) {
    let peer = stream.peer_addr().ok();
    let reader_stream = &stream;
    let mut reader = BufReader::new(reader_stream);

    if let Some(req) = parse_request(&mut reader) {
        let mut writer = stream.try_clone().unwrap();
        handle_request(&state, &mut writer, req, peer.map(|p| p.to_string()).unwrap_or_default());
    }
}

/// 根据请求路径分发到具体的处理函数
fn handle_request(state: &Arc<ServerState>, stream: &mut TcpStream, req: HttpRequest, _peer: String) {
    // 处理CORS预检
    if req.method == "OPTIONS" {
        send_cors_preflight(stream);
        return;
    }

    // 路由分发
    match (req.method.as_str(), req.path.as_str()) {
        // 首页 - 返回前端网页（支持 GET 和 HEAD）
        ("GET", "/") => {
            send_html(stream, INDEX_HTML);
        }
        ("HEAD", "/") => {
            send_html_head(stream, INDEX_HTML);
        }

        // 获取可用模型列表
        ("GET", "/api/models") => {
            handle_list_models(state, stream);
        }

        // 创建新会话
        ("POST", "/api/sessions") => {
            handle_create_session(state, stream, &req.body);
        }

        // 获取所有会话列表
        ("GET", "/api/sessions") => {
            handle_list_sessions(state, stream);
        }

        // 其它带路径参数的API
        _ => {
            // 解析路径: /api/sessions/{id}/...
            if let Some(rest) = req.path.strip_prefix("/api/sessions/") {
                let parts: Vec<&str> = rest.splitn(2, '/').collect();
                if let Ok(session_id) = parts[0].parse::<u64>() {
                    match (req.method.as_str(), parts.get(1).copied()) {
                        // POST /api/sessions/{id}/chat - 流式推理
                        ("POST", Some("chat")) => {
                            handle_chat(state, stream, session_id, &req.body);
                            return;
                        }
                        // POST /api/sessions/{id}/stop - 停止推理
                        ("POST", Some("stop")) => {
                            handle_stop(state, stream, session_id);
                            return;
                        }
                        // POST /api/sessions/{id}/reset - 重置会话
                        ("POST", Some("reset")) => {
                            handle_reset_session(state, stream, session_id);
                            return;
                        }
                        // DELETE /api/sessions/{id} - 删除会话
                        ("DELETE", None) => {
                            handle_delete_session(state, stream, session_id);
                            return;
                        }
                        _ => {}
                    }
                }
            }
            // 404
            send_json(stream, 404, &json!({"error": "接口不存在"}));
        }
    }
}

/// 获取可用模型列表
fn handle_list_models(state: &Arc<ServerState>, stream: &mut TcpStream) {
    let models: Vec<Value> = state.available_models.iter().map(|m| {
        json!({
            "name": m.name,
            "backend": m.backend,
        })
    }).collect();
    send_json(stream, 200, &json!({"models": models}));
}

/// 创建新会话
fn handle_create_session(state: &Arc<ServerState>, stream: &mut TcpStream, body: &str) {
    // 解析请求体
    let params: Value = serde_json::from_str(body).unwrap_or(json!({}));
    let model_name = params["model_name"].as_str().unwrap_or("").to_string();
    let system_prompt = params["system_prompt"].as_str().unwrap_or("").to_string();

    // 记录请求到日志，便于排查前端/后端交互问题
    println!("[server] create_session request: model='{}' system_prompt='{}'", model_name, system_prompt);

    // 验证模型是否存在
    if !state.available_models.iter().any(|m| m.name == model_name) {
        println!("[server] create_session failed: 模型 '{}' 不存在", model_name);
        send_json(stream, 400, &json!({"error": format!("模型 '{}' 不存在", model_name)}));
        return;
    }

    // 创建会话（此时不加载模型，等第一次推理时再加载）
    let session_id = state.next_session_id.fetch_add(1, Ordering::Relaxed);
    let session = Session {
        model_name: model_name.clone(),
        system_prompt,
        cache: None, // 延迟到首次推理时初始化
        history_ids: Vec::new(),
        is_first_turn: true,
        cancel_flag: Arc::new(AtomicBool::new(false)),
        generating: Arc::new(AtomicBool::new(false)),
    };

    let mut sessions = state.sessions.lock().unwrap();
    sessions.insert(session_id, session);

    println!("[server] session created: id={} model='{}'", session_id, model_name);

    send_json(stream, 201, &json!({
        "session_id": session_id,
        "model_name": model_name,
    }));
}

/// 获取所有会话列表
fn handle_list_sessions(state: &Arc<ServerState>, stream: &mut TcpStream) {
    let sessions = state.sessions.lock().unwrap();
    let list: Vec<Value> = sessions.iter().map(|(id, s)| {
        json!({
            "id": id,
            "model_name": s.model_name,
            "system_prompt": s.system_prompt,
            "is_first_turn": s.is_first_turn,
            "generating": s.generating.load(Ordering::Relaxed),
        })
    }).collect();
    send_json(stream, 200, &json!({"sessions": list}));
}

/// 删除会话
fn handle_delete_session(state: &Arc<ServerState>, stream: &mut TcpStream, session_id: u64) {
    let mut sessions = state.sessions.lock().unwrap();
    if sessions.remove(&session_id).is_some() {
        send_json(stream, 200, &json!({"ok": true}));
    } else {
        send_json(stream, 404, &json!({"error": "会话不存在"}));
    }
}

/// 重置会话（清空历史，重新开始对话）
fn handle_reset_session(state: &Arc<ServerState>, stream: &mut TcpStream, session_id: u64) {
    let mut sessions = state.sessions.lock().unwrap();
    if let Some(session) = sessions.get_mut(&session_id) {
        // 检查是否正在生成
        if session.generating.load(Ordering::Relaxed) {
            send_json(stream, 409, &json!({"error": "会话正在推理中，请先停止"}));
            return;
        }
        session.cache = None;
        session.history_ids.clear();
        session.is_first_turn = true;
        send_json(stream, 200, &json!({"ok": true}));
    } else {
        send_json(stream, 404, &json!({"error": "会话不存在"}));
    }
}

/// 停止正在进行的推理
fn handle_stop(state: &Arc<ServerState>, stream: &mut TcpStream, session_id: u64) {
    let sessions = state.sessions.lock().unwrap();
    if let Some(session) = sessions.get(&session_id) {
        // 设置取消标志，推理循环会在下一个token生成后检查并停止
        session.cancel_flag.store(true, Ordering::Relaxed);
        send_json(stream, 200, &json!({"ok": true}));
    } else {
        send_json(stream, 404, &json!({"error": "会话不存在"}));
    }
}

/// 流式推理处理（SSE）—— 核心功能
fn handle_chat(state: &Arc<ServerState>, stream: &mut TcpStream, session_id: u64, body: &str) {
    // 解析推理参数
    let params: Value = serde_json::from_str(body).unwrap_or(json!({}));
    let message = params["message"].as_str().unwrap_or("").to_string();
    let max_len = params["max_len"].as_u64().unwrap_or(256) as usize;
    let top_p = params["top_p"].as_f64().unwrap_or(0.9) as f32;
    let top_k = params["top_k"].as_u64().unwrap_or(40) as u32;
    let temperature = params["temperature"].as_f64().unwrap_or(0.7) as f32;
    let penalty = params["penalty"].as_f64().unwrap_or(1.1) as f32;

    // 设备与后端选项（尽力应用，OnceLock 初始化后不再生效）
    if let Some(backend) = params["prefill_backend"].as_str() {
        let val = match backend { "gemm" => "gemm", _ => "tiled" };
        std::env::set_var("LMRS_PREFILL_BACKEND", val);
    }
    if let Some(threads) = params["threads"].as_u64() {
        let t = threads.max(1).min(64).to_string();
        std::env::set_var("LMRS_THREADS", &t);
        std::env::set_var("LMRS_PREFILL_THREADS", &t);
        std::env::set_var("RAYON_NUM_THREADS", &t);
    }

    // 也允许在每轮对话中更新system_prompt（仅在非空时生效，避免误清空）
    let system_prompt_override = params["system_prompt"].as_str().map(|s| s.to_string());

    if message.is_empty() {
        send_json(stream, 400, &json!({"error": "消息不能为空"}));
        return;
    }

    // 从会话中提取推理所需的状态
        let (model_name, system_prompt, cache, history_ids, is_first_turn, cancel_flag, _generating_flag) = {
        let mut sessions = state.sessions.lock().unwrap();
        let session = match sessions.get_mut(&session_id) {
            Some(s) => s,
            None => {
                drop(sessions);
                send_json(stream, 404, &json!({"error": "会话不存在"}));
                return;
            }
        };

        // 检查是否已在生成中
        if session.generating.load(Ordering::Relaxed) {
            drop(sessions);
            send_json(stream, 409, &json!({"error": "该会话正在推理中"}));
            return;
        }

        // 如果请求中带了新的system prompt，更新到会话
        if let Some(ref sp) = system_prompt_override {
            if !sp.is_empty() {
                session.system_prompt = sp.clone();
            }
        }

        // 标记开始生成，重置取消标志
        session.generating.store(true, Ordering::Relaxed);
        session.cancel_flag.store(false, Ordering::Relaxed);

        // 取出cache（推理期间会话无法被其他请求使用）
        let cache = session.cache.take();
        let model_name = session.model_name.clone();
        let system_prompt = session.system_prompt.clone();
        let history_ids = session.history_ids.clone();
        let is_first_turn = session.is_first_turn;
        let cancel_flag = Arc::clone(&session.cancel_flag);
        let generating_flag = Arc::clone(&session.generating);

        (model_name, system_prompt, cache, history_ids, is_first_turn, cancel_flag, generating_flag)
    };

    // 确保模型已加载
    let (model, tokenizer) = match state.ensure_model_loaded(&model_name) {
        Ok(r) => r,
        Err(e) => {
            // 加载失败，恢复会话状态
            let mut sessions = state.sessions.lock().unwrap();
            if let Some(session) = sessions.get_mut(&session_id) {
                session.cache = cache;
                session.generating.store(false, Ordering::Relaxed);
            }
            send_json(stream, 500, &json!({"error": e}));
            return;
        }
    };

    // 初始化或复用KV缓存
    let mut cache = cache.unwrap_or_else(|| model.new_cache());

    // 构建prompt，支持系统提示词
    let prompt = templates::build_llama3_prompt_with_system(&message, &system_prompt, is_first_turn);
    let encoded = match tokenizer.encode(prompt, false) {
        Ok(e) => e,
        Err(e) => {
            let mut sessions = state.sessions.lock().unwrap();
            if let Some(session) = sessions.get_mut(&session_id) {
                session.cache = Some(cache);
                session.generating.store(false, Ordering::Relaxed);
            }
            send_json(stream, 500, &json!({"error": format!("编码失败: {}", e)}));
            return;
        }
    };
    let input_ids = encoded.get_ids();

    // 发送SSE响应头
    send_sse_header(stream);

    // 开始流式推理（逐token生成）
    let eos_ids = model.eos_token_ids();
    let mut all_ids = history_ids;
    all_ids.extend_from_slice(input_ids);

    let prompt_shape = vec![1, input_ids.len()];
    let one_shape = vec![1, 1];
    let mut input_tensor = Tensor::<u32>::new(input_ids.to_vec(), &prompt_shape);
    let mut decode_token_tensor = Tensor::new(vec![0u32], &one_shape);
    let mut generated_ids = Vec::new();
    let mut generated_tokens = 0usize;
    let mut cancelled = false;

    let gen_start = Instant::now();

    // 核心推理循环：每生成一个token就通过SSE推送给客户端
    let turn_end_token = loop {
        // 检查取消标志
        if cancel_flag.load(Ordering::Relaxed) {
            cancelled = true;
            break None;
        }

        // 前向传播 + 采样
        let logits = model.forward(&input_tensor, &mut cache);
        let id = sampling::random_sample(&logits, &all_ids, top_p, top_k, temperature, penalty);

        // 检查是否遇到结束token
        if eos_ids.contains(&id) {
            break Some(id);
        }

        all_ids.push(id);
        generated_ids.push(id);
        generated_tokens += 1;

        // 将所有已生成 token 整体解码（避免 BPE 多字节字符被截断产生乱码）
        let text_so_far = tokenizer.decode(&generated_ids, false).unwrap_or_default();
        let event = json!({
            "type": "token",
            "text": text_so_far,
            "token_id": id,
        });
        if !send_sse_data(stream, &event.to_string()) {
            // 客户端断开连接
            cancelled = true;
            break None;
        }

        // 检查是否达到最大长度
        if generated_tokens >= max_len {
            break eos_ids.first().copied();
        }

        // 准备下一轮decode的输入（单token）
        unsafe {
            decode_token_tensor.data_mut()[0] = id;
        }
        input_tensor = decode_token_tensor.slice(0, &one_shape);
    };

    // 如果有结束token，将其写入cache以保持多轮对话的一致性
    if let Some(end_id) = turn_end_token {
        unsafe {
            decode_token_tensor.data_mut()[0] = end_id;
        }
        let end_input = decode_token_tensor.slice(0, &one_shape);
        let _ = model.forward(&end_input, &mut cache);
    }

    let elapsed = gen_start.elapsed();

    // 发送完成事件
    let full_text = tokenizer.decode(&generated_ids, true).unwrap_or_default();
    let done_event = json!({
        "type": "done",
        "full_text": full_text,
        "tokens_generated": generated_tokens,
        "elapsed_ms": elapsed.as_millis(),
        "cancelled": cancelled,
        "tokens_per_sec": if elapsed.as_secs_f64() > 0.0 {
            generated_tokens as f64 / elapsed.as_secs_f64()
        } else { 0.0 },
    });
    let _ = send_sse_data(stream, &done_event.to_string());

    // 将状态写回会话
    let mut sessions = state.sessions.lock().unwrap();
    if let Some(session) = sessions.get_mut(&session_id) {
        session.cache = Some(cache);
        session.history_ids = all_ids;
        session.is_first_turn = false;
        session.generating.store(false, Ordering::Relaxed);
        session.cancel_flag.store(false, Ordering::Relaxed);
    }

    println!(
        "[server] 会话#{} 生成完成: {}个token, 耗时{:.2}s, {:.1} tok/s{}",
        session_id,
        generated_tokens,
        elapsed.as_secs_f64(),
        if elapsed.as_secs_f64() > 0.0 { generated_tokens as f64 / elapsed.as_secs_f64() } else { 0.0 },
        if cancelled { " (已取消)" } else { "" }
    );
}

// ============================================================================
// 服务器启动入口
// ============================================================================

/// 启动HTTP服务器
/// host: 绑定地址（如 "0.0.0.0"）
/// port: 监听端口（如 8080）
pub fn start_server(host: &str, port: u16) {
    // 应用默认的CPU调优配置
    apply_chat_defaults();

    // 扫描models目录
    let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let state = Arc::new(ServerState::new(&models_dir));

    let addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        panic!("无法绑定地址 {}: {}", addr, e);
    });

    println!("============================================================");
    println!("  LM-RS 推理服务器已启动");
    println!("  访问地址: http://{}:{}", host, port);
    println!("  可用模型: {} 个", state.available_models.len());
    println!("============================================================");

    // 主循环：为每个连接创建独立线程处理
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    handle_connection(state, stream);
                });
            }
            Err(e) => {
                eprintln!("[server] 接受连接失败: {}", e);
            }
        }
    }
}
