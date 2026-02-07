/// Error types for the Llama inference system
#[derive(Debug)]
pub enum LlamaError {
    /// Tensor shape mismatch error
    TensorShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        context: String,
    },
    /// Model loading error
    ModelLoadError(String),
    /// Tokenizer error
    TokenizerError(String),
    /// Inference error
    InferenceError(String),
    /// IO error
    IoError(std::io::Error),
    /// JSON parsing error
    JsonError(serde_json::Error),
    /// SafeTensors error
    SafeTensorsError(String),
}

impl std::fmt::Display for LlamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlamaError::TensorShapeMismatch { expected, actual, context } => {
                write!(
                    f,
                    "Tensor shape mismatch in {}: expected {:?}, got {:?}",
                    context, expected, actual
                )
            }
            LlamaError::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            LlamaError::TokenizerError(msg) => write!(f, "Tokenizer error: {}", msg),
            LlamaError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            LlamaError::IoError(e) => write!(f, "IO error: {}", e),
            LlamaError::JsonError(e) => write!(f, "JSON parsing error: {}", e),
            LlamaError::SafeTensorsError(msg) => write!(f, "SafeTensors error: {}", msg),
        }
    }
}

impl std::error::Error for LlamaError {}

impl From<std::io::Error> for LlamaError {
    fn from(err: std::io::Error) -> Self {
        LlamaError::IoError(err)
    }
}

impl From<serde_json::Error> for LlamaError {
    fn from(err: serde_json::Error) -> Self {
        LlamaError::JsonError(err)
    }
}

pub type Result<T> = std::result::Result<T, LlamaError>;
