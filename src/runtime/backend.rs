use std::sync::OnceLock;

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

pub fn prefill_matmul_backend() -> PrefillMatmulBackend {
	*PREFILL_BACKEND.get_or_init(PrefillMatmulBackend::from_env)
}
