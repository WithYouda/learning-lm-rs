/// LM-RS 推理服务器入口
/// 启动HTTP服务器，提供Web界面和流式推理API
fn main() {
    // 默认监听地址和端口，可通过环境变量覆盖
    let host = std::env::var("LMRS_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("LMRS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    learning_lm_rust::server::http::start_server(&host, port);
}
