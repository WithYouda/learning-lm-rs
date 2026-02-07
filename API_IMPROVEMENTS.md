# API 改进建议 (API Improvement Recommendations)

## 目标 (Goals)

本文档提供了对现有API的改进建议，旨在提高代码的可用性、可维护性和安全性。

## 核心改进 (Core Improvements)

### 1. 错误处理 (Error Handling)

#### 现状 (Current State)
```rust
// 现有代码使用 unwrap()，可能导致panic
let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
```

#### 改进方案 (Improvement)
```rust
// 使用Result返回错误，让调用者决定如何处理
pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Result<Self, LlamaError> {
    let config_path = model_dir.as_ref().join("config.json");
    let config = File::open(&config_path)
        .map_err(|e| LlamaError::ModelLoadError(
            format!("Failed to open config at {:?}: {}", config_path, e)
        ))?;
    let config: LlamaConfigJson = serde_json::from_reader(config)?;
    // ...
    Ok(model)
}
```

### 2. 配置管理 (Configuration Management)

#### 现状 (Current State)
```rust
pub fn generate(
    &self,
    token_ids: &[u32],
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
) -> Vec<u32>
```

#### 改进方案 (Improvement)
```rust
pub fn generate(
    &self,
    token_ids: &[u32],
    config: &SamplingConfig,
) -> Vec<u32>

// 使用示例
let config = SamplingConfig::default();
let output = model.generate(&input_tokens, &config);

// 或使用预设配置
let config = SamplingConfig::creative();
let output = model.generate(&input_tokens, &config);
```

### 3. Builder 模式 (Builder Pattern)

#### 建议 (Recommendation)
```rust
pub struct LlamaBuilder {
    model_dir: PathBuf,
    device: Device,
    dtype: DType,
}

impl LlamaBuilder {
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            device: Device::Cpu,
            dtype: DType::F32,
        }
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn build(self) -> Result<Llama<f32>, LlamaError> {
        Llama::from_safetensors_with_config(self.model_dir, self.device, self.dtype)
    }
}

// 使用示例
let model = LlamaBuilder::new("models/story")
    .device(Device::Cuda(0))
    .dtype(DType::F16)
    .build()?;
```

### 4. 生成器模式 (Generator Pattern)

#### 建议 (Recommendation)
```rust
pub struct TokenGenerator<'a> {
    model: &'a Llama<f32>,
    cache: KVCache<f32>,
    config: SamplingConfig,
}

impl<'a> TokenGenerator<'a> {
    pub fn new(model: &'a Llama<f32>, config: SamplingConfig) -> Self {
        let cache = model.new_cache();
        Self { model, cache, config }
    }

    pub fn generate_next(&mut self, input: &[u32]) -> Option<u32> {
        // 实现单步生成
    }

    pub fn generate(&mut self, prompt: &[u32]) -> Vec<u32> {
        // 实现完整生成
    }
}

// 使用示例 - 流式生成
let mut generator = TokenGenerator::new(&model, config);
for token in generator.generate_streaming(&prompt) {
    print!("{}", tokenizer.decode(&[token]));
}
```

### 5. 类型安全的张量操作 (Type-safe Tensor Operations)

#### 现状 (Current State)
```rust
let mut tensor = Tensor::<f32>::default(&vec![seq_len, self.d]);
// 运行时检查形状
```

#### 改进方案 (Improvement)
```rust
// 选项1: 使用类型状态模式
pub struct Tensor1D<T> { /* ... */ }
pub struct Tensor2D<T> { /* ... */ }
pub struct Tensor3D<T> { /* ... */ }

// 选项2: 使用phantom types标记形状
pub struct Tensor<T, S: TensorShape> {
    data: Vec<T>,
    shape: S,
}

pub trait TensorShape {
    fn dims(&self) -> &[usize];
}

pub struct Shape2D {
    rows: usize,
    cols: usize,
}

// 选项3: 编译时形状检查 (需要const generics)
pub struct FixedTensor<T, const ROWS: usize, const COLS: usize> {
    data: [[T; COLS]; ROWS],
}
```

## 具体模块改进 (Module-specific Improvements)

### Operators 模块

#### 当前API
```rust
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32)
```

#### 改进建议
```rust
// 1. 更清晰的命名
pub fn matmul_add(
    output: &mut Tensor<f32>,
    lhs: &Tensor<f32>,
    rhs_transposed: &Tensor<f32>,
    alpha: f32,
    beta: f32,
) -> Result<(), LlamaError>

// 2. 或者使用builder模式
pub struct MatMulOp<'a> {
    lhs: &'a Tensor<f32>,
    rhs: &'a Tensor<f32>,
    alpha: f32,
    beta: f32,
    transpose_rhs: bool,
}

impl<'a> MatMulOp<'a> {
    pub fn new(lhs: &'a Tensor<f32>, rhs: &'a Tensor<f32>) -> Self {
        Self {
            lhs,
            rhs,
            alpha: 1.0,
            beta: 0.0,
            transpose_rhs: false,
        }
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    pub fn transpose_rhs(mut self) -> Self {
        self.transpose_rhs = true;
        self
    }

    pub fn compute(self, output: &mut Tensor<f32>) -> Result<(), LlamaError> {
        // 执行计算
    }
}

// 使用示例
MatMulOp::new(&a, &b)
    .transpose_rhs()
    .alpha(1.0)
    .beta(0.5)
    .compute(&mut c)?;
```

### Model 模块

#### 改进的生成API
```rust
pub struct GenerationOptions {
    pub sampling: SamplingConfig,
    pub stop_tokens: Vec<u32>,
    pub max_new_tokens: Option<usize>,
    pub min_new_tokens: Option<usize>,
    pub repetition_penalty: f32,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            sampling: SamplingConfig::default(),
            stop_tokens: vec![],
            max_new_tokens: None,
            min_new_tokens: None,
            repetition_penalty: 1.0,
        }
    }
}

pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub generation_time: std::time::Duration,
}

pub enum FinishReason {
    StopToken,
    MaxLength,
    Error(String),
}

impl Llama<f32> {
    pub fn generate_with_options(
        &self,
        prompt: &[u32],
        options: &GenerationOptions,
    ) -> Result<GenerationOutput, LlamaError> {
        let start_time = std::time::Instant::now();
        // ... 生成逻辑
        Ok(GenerationOutput {
            tokens: result,
            finish_reason: FinishReason::StopToken,
            generation_time: start_time.elapsed(),
        })
    }
}
```

## 向后兼容性 (Backward Compatibility)

为了保持向后兼容性，可以保留原有API并添加新的改进API：

```rust
// 保留原有API（标记为deprecated）
#[deprecated(since = "0.2.0", note = "Use generate_with_options instead")]
pub fn generate(
    &self,
    token_ids: &[u32],
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
) -> Vec<u32> {
    let config = SamplingConfig::new(max_len, top_p, top_k, temperature);
    let options = GenerationOptions {
        sampling: config,
        ..Default::default()
    };
    self.generate_with_options(token_ids, &options)
        .expect("Generation failed")
        .tokens
}

// 新的改进API
pub fn generate_with_options(
    &self,
    prompt: &[u32],
    options: &GenerationOptions,
) -> Result<GenerationOutput, LlamaError> {
    // 新实现
}
```

## 文档标准 (Documentation Standards)

每个公开函数都应该有rustdoc注释：

```rust
/// Generates text tokens given a prompt using the language model.
///
/// # Arguments
///
/// * `prompt` - Input token IDs to condition the generation on
/// * `options` - Configuration options for text generation
///
/// # Returns
///
/// Returns a `GenerationOutput` containing the generated tokens and metadata,
/// or a `LlamaError` if generation fails.
///
/// # Examples
///
/// ```
/// use learning_lm_rust::{Llama, GenerationOptions, SamplingConfig};
///
/// let model = Llama::from_safetensors("models/story")?;
/// let prompt = vec![1, 42, 100];  // token IDs
/// let options = GenerationOptions {
///     sampling: SamplingConfig::default(),
///     ..Default::default()
/// };
/// let output = model.generate_with_options(&prompt, &options)?;
/// println!("Generated {} tokens", output.tokens.len());
/// ```
///
/// # Errors
///
/// Returns `LlamaError::InferenceError` if the forward pass fails.
pub fn generate_with_options(
    &self,
    prompt: &[u32],
    options: &GenerationOptions,
) -> Result<GenerationOutput, LlamaError> {
    // ...
}
```

## 性能API (Performance APIs)

提供性能监控和分析工具：

```rust
pub struct PerformanceMetrics {
    pub total_time: std::time::Duration,
    pub tokens_per_second: f32,
    pub time_per_token: std::time::Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

pub struct Llama<f32> {
    // ... existing fields
    #[cfg(feature = "metrics")]
    metrics: Option<PerformanceMetrics>,
}

impl Llama<f32> {
    #[cfg(feature = "metrics")]
    pub fn get_metrics(&self) -> Option<&PerformanceMetrics> {
        self.metrics.as_ref()
    }

    #[cfg(feature = "metrics")]
    pub fn reset_metrics(&mut self) {
        self.metrics = None;
    }
}
```

## 总结 (Summary)

这些改进将使API：
1. **更安全** - 通过Result类型处理错误
2. **更易用** - 通过配置对象和Builder模式简化调用
3. **更灵活** - 支持更多自定义选项
4. **更高效** - 提供性能监控工具
5. **更专业** - 符合Rust生态系统的最佳实践

实施这些改进时，建议：
- 保持向后兼容性
- 逐步迁移，先添加新API
- 提供迁移指南
- 添加充分的测试和文档
