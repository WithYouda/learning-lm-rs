/// Configuration for text generation sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Maximum number of tokens to generate
    pub max_len: usize,
    /// Top-p (nucleus) sampling parameter
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: u32,
    /// Temperature for sampling
    pub temperature: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            max_len: 100,
            top_p: 0.9,
            top_k: 50,
            temperature: 1.0,
        }
    }
}

impl SamplingConfig {
    /// Create a new sampling configuration
    pub fn new(max_len: usize, top_p: f32, top_k: u32, temperature: f32) -> Self {
        Self {
            max_len,
            top_p,
            top_k,
            temperature,
        }
    }

    /// Create a greedy sampling configuration (no randomness)
    pub fn greedy() -> Self {
        Self {
            max_len: 100,
            top_p: 0.0,
            top_k: 1,
            temperature: 0.0,
        }
    }

    /// Create a more creative sampling configuration
    pub fn creative() -> Self {
        Self {
            max_len: 200,
            top_p: 0.95,
            top_k: 100,
            temperature: 1.2,
        }
    }

    /// Validate the sampling configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_len == 0 {
            return Err("max_len must be greater than 0".to_string());
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err("top_p must be between 0.0 and 1.0".to_string());
        }
        if self.top_k == 0 {
            return Err("top_k must be greater than 0".to_string());
        }
        if self.temperature < 0.0 {
            return Err("temperature must be non-negative".to_string());
        }
        Ok(())
    }
}
