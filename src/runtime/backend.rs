use std::sync::OnceLock;

/// Prefill 阶段矩阵乘法后端的选择。
/// Tiled: 自实现的分块计算，Gemm: 调用高性能库（gemm crate）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMatmulBackend {
	Tiled,
	Gemm,
}

impl PrefillMatmulBackend {
	fn from_env() -> Self {
		let v = std::env::var("LMRS_PREFILL_BACKEND")
			.unwrap_or_else(|_| "tiled".to_string())
			.to_lowercase();
		match v.as_str() {
			"gemm" | "blas" | "openblas" | "blis" | "onednn" => PrefillMatmulBackend::Gemm,
			_ => PrefillMatmulBackend::Tiled,
		}
	}
}

static PREFILL_BACKEND: OnceLock<PrefillMatmulBackend> = OnceLock::new();

/// 获取当前 prefill 矩阵乘法后端（通过 LMRS_PREFILL_BACKEND 环境变量配置）
pub fn prefill_matmul_backend() -> PrefillMatmulBackend {
	*PREFILL_BACKEND.get_or_init(PrefillMatmulBackend::from_env)
}
