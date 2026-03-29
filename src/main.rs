
#[cfg(test)]
mod test{
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::ffi::OsString;
    use std::process::Command;
    use std::time::Instant;
    use std::path::PathBuf;
    #[cfg(target_os = "linux")]
    use std::os::unix::process::CommandExt;
    use tokenizers::Tokenizer;

    use learning_lm_rust::chat::templates;
    use learning_lm_rust::core::operators::quant::generic;
    use learning_lm_rust::core::sampling;
    use learning_lm_rust::core::tensor::Tensor;
    use learning_lm_rust::model::llama;
    use learning_lm_rust::runtime;

    #[derive(Clone, Debug)]
    struct BenchSample {
        prefill_tok_s: f64,
        decode_tok_s: f64,
        ttft_s: f64,
        kv_cache_hit_rate_est: f64,
        quant_row_cache_hits: usize,
        quant_row_cache_misses: usize,
        quant_row_cache_size: usize,
        quant_row_cache_hit_rate: f64,
    }

    #[derive(Clone, Debug)]
    struct CompareBenchSample {
        ours: BenchSample,
        cpp_prefill_tok_s: f64,
        cpp_decode_tok_s: f64,
        prefill_ratio: f64,
        decode_ratio: f64,
    }

    struct BenchContext {
        model: llama::Llama<f32>,
        input_ids: Vec<u32>,
        prefill_shape: Vec<usize>,
        one_shape: Vec<usize>,
    }

    fn set_env_default(key: &str, value: impl Into<OsString>) {
        if std::env::var_os(key).is_none() {
            let value: OsString = value.into();
            std::env::set_var(key, &value);
        }
    }

    fn default_llamacpp_model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("test")
            .join("Llama-3.2-1B-Instruct-Q4_K_L.gguf")
    }

    fn apply_benchmark_defaults() {
        // 固化当前阶段推荐配置，降低本地回归时的线程与调度抖动。
        set_env_default("LMRS_THREADS", "5");
        set_env_default("LMRS_PREFILL_THREADS", "5");
        set_env_default("RAYON_NUM_THREADS", "5");
        set_env_default("LMRS_CPU_MASK", "0,1,2,3,4");
        set_env_default("LMRS_ENABLE_AFFINITY", "1");
        set_env_default("LMRS_PREFILL_BACKEND", "gemm");

        // 禁用 hot matrix cache（decode 时密集 f32 带宽远大于 AVX2 量化点积带宽，
        // 实测：启用 2GB cache 时 decode ≈ 9.3 tok/s，禁用后 ≈ 14.7 tok/s）。
        set_env_default("LMRS_HOT_MATRIX_CACHE_MB", "1");

        // compare benchmark 默认直接对接当前仓库内的 llama.cpp 命令与量化模型。
        set_env_default("LLAMA_CPP_CLI", "llama-cli");
        set_env_default("LLAMA_CPP_MODEL", default_llamacpp_model_path().into_os_string());
        // compare benchmark 默认强制 CPU-only，避免误把 GPU offload 结果当作 CPU 基线。
        set_env_default("LMRS_LLAMA_CPP_NGL", "0");
    }

    fn bench_rounds() -> usize {
        std::env::var("LMRS_BENCH_ROUNDS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(5)
    }

    fn bench_decode_steps() -> usize {
        std::env::var("LMRS_BENCH_DECODE_STEPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(64)
    }

    fn bench_llamacpp_ngl() -> String {
        std::env::var("LMRS_LLAMA_CPP_NGL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "0".to_string())
    }

    fn bench_steady_state_warmup_enabled() -> bool {
        std::env::var("LMRS_BENCH_STEADY_STATE_WARMUP")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
    }

    fn mean_std(values: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|v| {
                let diff = *v - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;
        (mean, variance.sqrt())
    }

    fn bench_csv_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("bench")
            .join("metrics.csv")
    }

    fn llamacpp_debug_output_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("bench")
            .join("llamacpp_last_output.txt")
    }

    fn append_bench_csv(kind: &str, round: usize, sample: &BenchSample) {
        let csv_path = bench_csv_path();
        if let Some(parent) = csv_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let need_header = !csv_path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_path)
            .expect("open bench csv failed");
        if need_header {
            writeln!(
                file,
                "kind,round,prefill_tok_s,decode_tok_s,ttft_s,kv_cache_hit_rate_est,quant_row_cache_hits,quant_row_cache_misses,quant_row_cache_hit_rate,quant_row_cache_size"
            )
            .unwrap();
        }
        writeln!(
            file,
            "{},{},{:.6},{:.6},{:.6},{:.6},{},{},{:.6},{}",
            kind,
            round,
            sample.prefill_tok_s,
            sample.decode_tok_s,
            sample.ttft_s,
            sample.kv_cache_hit_rate_est,
            sample.quant_row_cache_hits,
            sample.quant_row_cache_misses,
            sample.quant_row_cache_hit_rate,
            sample.quant_row_cache_size,
        )
        .unwrap();
    }

    fn print_metric_summary(label: &str, values: &[f64], unit: &str) {
        let (mean, std) = mean_std(values);
        println!("[bench-summary] {} mean: {:.4} {} std: {:.4}", label, mean, unit, std);
    }

    fn summarize_compare_samples(label: &str, samples: &[CompareBenchSample]) {
        let prefill_ratio_values = samples.iter().map(|s| s.prefill_ratio).collect::<Vec<_>>();
        let decode_ratio_values = samples.iter().map(|s| s.decode_ratio).collect::<Vec<_>>();
        let ours_prefill_values = samples.iter().map(|s| s.ours.prefill_tok_s).collect::<Vec<_>>();
        let ours_decode_values = samples.iter().map(|s| s.ours.decode_tok_s).collect::<Vec<_>>();
        let cpp_prefill_values = samples.iter().map(|s| s.cpp_prefill_tok_s).collect::<Vec<_>>();
        let cpp_decode_values = samples.iter().map(|s| s.cpp_decode_tok_s).collect::<Vec<_>>();

        print_metric_summary(&format!("{}_prefill_ratio_ours_over_llamacpp", label), &prefill_ratio_values, "ratio");
        print_metric_summary(&format!("{}_decode_ratio_ours_over_llamacpp", label), &decode_ratio_values, "ratio");
        print_metric_summary(&format!("{}_ours_prefill_tok/s", label), &ours_prefill_values, "tok/s");
        print_metric_summary(&format!("{}_ours_decode_tok/s", label), &ours_decode_values, "tok/s");
        print_metric_summary(&format!("{}_llamacpp_prefill_tok/s", label), &cpp_prefill_values, "tok/s");
        print_metric_summary(&format!("{}_llamacpp_decode_tok/s", label), &cpp_decode_values, "tok/s");
    }

    fn load_bench_context() -> BenchContext {
        apply_benchmark_defaults();
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("test");
        let model_gguf_dir = model_dir.join("Llama-3.2-1B-Instruct-Q4_K_L.gguf");
        let model = llama::Llama::<f32>::from_gguf(&model_gguf_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

        let prompt = templates::build_llama3_prompt("请用一句话介绍你自己", true);
        let enc = tokenizer.encode(prompt, false).unwrap();
        let input_ids = enc.get_ids().to_vec();
        let prefill_shape = vec![1, input_ids.len()];

        BenchContext {
            model,
            input_ids,
            prefill_shape,
            one_shape: vec![1, 1],
        }
    }

    fn warmup_ours_bench_context(ctx: &BenchContext) {
        let mut cache = ctx.model.new_cache();
        let prefill_in = Tensor::new(ctx.input_ids.clone(), &ctx.prefill_shape);
        let _ = ctx.model.forward(&prefill_in, &mut cache);
    }

    fn run_ours_bench_once(ctx: &BenchContext, decode_steps: usize) -> BenchSample {
        if bench_steady_state_warmup_enabled() {
            // 交错 benchmark 的根因之一是 llama.cpp 在相邻轮次之间打散页缓存/工作集。
            // 这里在计时前做一次不计时 prefill 预热，把口径收敛到更接近真实服务态的 steady-state。
            warmup_ours_bench_context(ctx);
        }

        generic::quant_row_cache_clear();

        let mut cache = ctx.model.new_cache();

        // 1) Prefill 吞吐：一次性喂入 prompt。
        let prefill_start = Instant::now();
        let prefill_in = Tensor::new(ctx.input_ids.clone(), &ctx.prefill_shape);
        let logits = ctx.model.forward(&prefill_in, &mut cache);
        let prefill_elapsed = prefill_start.elapsed().as_secs_f64();
        let prefill_tok_s = ctx.input_ids.len() as f64 / prefill_elapsed.max(1e-9);

        // 2) TTFT：从 prefill 结束到拿到首个 token。
        let ttft_start = Instant::now();
        let first_id = sampling::random_sample(&logits, &ctx.input_ids, 0.9, 40, 0.7, 1.05);
        let ttft = ttft_start.elapsed().as_secs_f64();

        // 3) Decode 吞吐：逐 token 生成，并复用单 token 输入张量避免测试自身分配噪声。
        let mut history = ctx.input_ids.clone();
        history.push(first_id);
        let mut decode_input = Tensor::new(vec![first_id], &ctx.one_shape);

        let decode_start = Instant::now();
        for _ in 0..decode_steps {
            let l = ctx.model.forward(&decode_input, &mut cache);
            let id = sampling::random_sample(&l, &history, 0.9, 40, 0.7, 1.05);
            history.push(id);
            unsafe {
                decode_input.data_mut()[0] = id;
            }
        }
        let decode_elapsed = decode_start.elapsed().as_secs_f64();
        let decode_tok_s = decode_steps as f64 / decode_elapsed.max(1e-9);

        // 4) KV cache 命中率（理论统计）：每步 decode 新增 1 token，可复用其余历史 token。
        let prompt_len = ctx.input_ids.len();
        let mut hits = 0usize;
        let mut total = 0usize;
        for t in 0..decode_steps {
            let total_seq = prompt_len + 1 + t;
            hits += total_seq.saturating_sub(1);
            total += total_seq;
        }
        let hit_rate = hits as f64 / (total as f64).max(1.0);

        let (cache_hits, cache_misses, cache_size) = generic::quant_row_cache_stats();
        let cache_total = cache_hits + cache_misses;
        let cache_hit_rate = if cache_total > 0 {
            cache_hits as f64 / cache_total as f64
        } else {
            0.0
        };

        BenchSample {
            prefill_tok_s,
            decode_tok_s,
            ttft_s: ttft,
            kv_cache_hit_rate_est: hit_rate,
            quant_row_cache_hits: cache_hits as usize,
            quant_row_cache_misses: cache_misses as usize,
            quant_row_cache_size: cache_size as usize,
            quant_row_cache_hit_rate: cache_hit_rate,
        }
    }

    fn parse_rate_after_label(line: &str, label: &str) -> Option<f64> {
        let lower = line.to_ascii_lowercase();
        let pos = lower.find(label)? + label.len();
        let suffix = &line[pos..];

        let mut start = None;
        let mut end = None;
        for (i, ch) in suffix.char_indices() {
            if start.is_none() {
                if ch.is_ascii_digit() || ch == '.' {
                    start = Some(i);
                }
            } else if !(ch.is_ascii_digit() || ch == '.') {
                end = Some(i);
                break;
            }
        }

        let s = start?;
        let e = end.unwrap_or(suffix.len());
        suffix[s..e].trim().parse::<f64>().ok()
    }

    fn parse_rate_before_suffix(line: &str, suffix: &str) -> Option<f64> {
        let lower = line.to_ascii_lowercase();
        let pos = lower.find(suffix)?;
        let prefix = &line[..pos];

        let mut start = prefix.len();
        for (i, ch) in prefix.char_indices().rev() {
            if ch.is_ascii_digit() || ch == '.' {
                start = i;
            } else if start != prefix.len() {
                break;
            }
        }

        if start == prefix.len() {
            return None;
        }
        prefix[start..].trim().parse::<f64>().ok()
    }

    fn parse_tok_s_from_line(line: &str) -> Option<f64> {
        // 统一提取速度单位前面的最后一个数字，避免把 token 数或耗时误读成 tok/s。
        parse_rate_before_suffix(line, "tokens per second")
            .or_else(|| parse_rate_before_suffix(line, "tok/s"))
            .or_else(|| parse_rate_before_suffix(line, "t/s"))
    }

    fn run_llamacpp_bench(prompt: &str, decode_steps: usize, threads: usize) -> Option<(f64, f64)> {
        apply_benchmark_defaults();
        let cli = std::env::var("LLAMA_CPP_CLI").ok()?;
        let model = std::env::var("LLAMA_CPP_MODEL").ok()?;
        let ngl = bench_llamacpp_ngl();

        // 对照 benchmark 允许短暂重试：llama-cli 在高负载下可能偶发被 OOM killer 杀掉，
        // 这不是解析逻辑问题，重试一次通常可恢复。
        const MAX_ATTEMPTS: usize = 3;
        let mut last_diag = String::new();

        for attempt in 1..=MAX_ATTEMPTS {
            let mut cmd = Command::new(&cli);
            cmd.args([
                    "-m",
                    model.as_str(),
                    "-p",
                    prompt,
                    "-n",
                    &decode_steps.to_string(),
                    "-c",
                    "1024",
                    "-t",
                    &threads.to_string(),
                    "--threads-batch",
                    &threads.to_string(),
                    "--temp",
                    "0.7",
                    "--top-p",
                    "0.9",
                    "--top-k",
                    "40",
                    "--repeat-penalty",
                    "1.05",
                    "-ngl",
                    ngl.as_str(),
                    "-st",
                    "--simple-io",
                    "--log-disable",
                    "--show-timings",
                    "--no-display-prompt",
                ]);

            #[cfg(target_os = "linux")]
            unsafe {
                cmd.pre_exec(|| {
                    let cpu_count = libc::sysconf(libc::_SC_NPROCESSORS_CONF);
                    if cpu_count <= 0 {
                        return Ok(());
                    }
                    let mut set = std::mem::zeroed::<libc::cpu_set_t>();
                    libc::CPU_ZERO(&mut set);
                    for cpu in 0..cpu_count as usize {
                        libc::CPU_SET(cpu, &mut set);
                    }
                    let ret = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
                    if ret == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                });
            }

            let output = cmd.output().ok()?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let merged = format!("{}\n{}", stdout, stderr);

            let mut prompt_tok_s = None;
            let mut decode_tok_s = None;
            for line in merged.lines() {
                let lower = line.to_ascii_lowercase();
                let has_summary_units = lower.contains("tok/s") || lower.contains("t/s");

                if lower.contains("prompt:") && has_summary_units {
                    prompt_tok_s = parse_rate_after_label(line, "prompt:").or(prompt_tok_s);
                }
                if lower.contains("generation:") && has_summary_units {
                    decode_tok_s = parse_rate_after_label(line, "generation:").or(decode_tok_s);
                }

                if lower.contains("prompt:") && (lower.contains("tok/s") || lower.contains("t/s")) {
                    continue;
                }
                if lower.contains("prompt eval")
                    && (lower.contains("tok/s") || lower.contains("t/s") || lower.contains("tokens per second"))
                {
                    prompt_tok_s = parse_tok_s_from_line(line).or(prompt_tok_s);
                    continue;
                }
                if lower.contains(" eval")
                    && (lower.contains("tok/s") || lower.contains("t/s") || lower.contains("tokens per second"))
                    && !lower.contains("prompt eval")
                {
                    decode_tok_s = parse_tok_s_from_line(line).or(decode_tok_s);
                    continue;
                }
                if lower.contains("decode")
                    && (lower.contains("tok/s") || lower.contains("t/s") || lower.contains("tokens per second"))
                {
                    decode_tok_s = parse_tok_s_from_line(line).or(decode_tok_s);
                }
            }

            if output.status.success() {
                if let (Some(p), Some(d)) = (prompt_tok_s, decode_tok_s) {
                    return Some((p, d));
                }
            }

            last_diag = format!(
                "attempt: {}\nstatus: {}\ncli: {}\nmodel: {}\ndecode_steps: {}\nthreads: {}\nngl: {}\nprompt_len_bytes: {}\n--- stdout ---\n{}\n--- stderr ---\n{}\n",
                attempt,
                output.status,
                cli,
                model,
                decode_steps,
                threads,
                ngl,
                prompt.len(),
                stdout,
                stderr,
            );

            if attempt < MAX_ATTEMPTS {
                std::thread::sleep(std::time::Duration::from_millis(120));
            }
        }

        let debug_path = llamacpp_debug_output_path();
        if let Some(parent) = debug_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&debug_path, last_diag);
        eprintln!("[compare] llama.cpp benchmark failed after retries, debug output written to {}", debug_path.display());

        None
    }

    #[test]
    fn test_parse_llamacpp_rates() {
        let summary = "[ Prompt: 131.4 t/s | Generation: 37.7 t/s ]";
        assert_eq!(parse_rate_after_label(summary, "prompt:"), Some(131.4));
        assert_eq!(parse_rate_after_label(summary, "generation:"), Some(37.7));

        let prompt_eval = "llama_perf_context_print:        prompt eval time =   314.08 ms /    42 tokens (  133.72 tokens per second,     7.48 ms per token)";
        let decode_eval = "llama_perf_context_print:               eval time =  1698.04 ms /    64 runs   (   37.69 tokens per second,    26.53 ms per token)";
        assert_eq!(parse_tok_s_from_line(prompt_eval), Some(133.72));
        assert_eq!(parse_tok_s_from_line(decode_eval), Some(37.69));
    }

    #[test]
    #[ignore = "slow GGUF integration smoke test"]
    fn test_gguf_short_generate() {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("test");
        let model_gguf_dir = model_dir.join("Llama-3.2-1B-Instruct-Q4_K_L.gguf");

        let start = Instant::now();
        let model = llama::Llama::<f32>::from_gguf(&model_gguf_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        println!("GGUF load elapsed: {:?}", start.elapsed());

        let prompt = templates::build_llama3_prompt("hello", true);
        let enc = tokenizer.encode(prompt, false).unwrap();
        // Deterministic greedy decode for easier debugging.
        let out_ids = model.generate(enc.get_ids(), 6, 0.0, 0, 0.0, 1.0);
        let text = tokenizer.decode(&out_ids, true).unwrap();
        println!("GGUF short generate output: {}", text);
    }

    #[test]
    #[ignore = "slow safetensors integration smoke test"]
    fn test_safetensors_short_generate() {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("test");

        let start = Instant::now();
        let model = llama::Llama::<f32>::from_safetensors(&model_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        println!("Safetensors load elapsed: {:?}", start.elapsed());

        let prompt = templates::build_llama3_prompt("hello", true);
        let enc = tokenizer.encode(prompt, false).unwrap();
        let out_ids = model.generate(enc.get_ids(), 6, 0.0, 0, 0.0, 1.0);
        let text = tokenizer.decode(&out_ids, true).unwrap();
        println!("Safetensors short generate output: {}", text);
    }


    #[test]

    //#[ignore = "benchmark only; run with --release -- --ignored --nocapture"]
    fn bench_prefill_decode_metrics() {
        apply_benchmark_defaults();
        let rounds = bench_rounds();
        let decode_steps = bench_decode_steps();
        let bench_ctx = load_bench_context();
        println!("[bench] runtime: {}", runtime::cpu::runtime_tuning_summary());
        println!("[bench] perf_counter_available: {}", runtime::cpu::perf_counter_available());
        println!("[bench] defaults: threads={} cpu_mask={} affinity={} prefill_backend={}",
            std::env::var("LMRS_THREADS").unwrap_or_else(|_| "unset".to_string()),
            std::env::var("LMRS_CPU_MASK").unwrap_or_else(|_| "unset".to_string()),
            std::env::var("LMRS_ENABLE_AFFINITY").unwrap_or_else(|_| "unset".to_string()),
            std::env::var("LMRS_PREFILL_BACKEND").unwrap_or_else(|_| "unset".to_string()),
        );
        println!("[bench] rounds: {}", rounds);
        println!("[bench] decode_steps: {}", decode_steps);
        println!("[bench] steady_state_warmup: {}", bench_steady_state_warmup_enabled());

        let mut samples = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let sample = run_ours_bench_once(&bench_ctx, decode_steps);
            append_bench_csv("ours_local", round + 1, &sample);
            println!("[bench-round] round: {} prefill_tok/s: {:.2} decode_tok/s: {:.2} ttft_s: {:.4} quant_row_cache_hit_rate: {:.4}", round + 1, sample.prefill_tok_s, sample.decode_tok_s, sample.ttft_s, sample.quant_row_cache_hit_rate);
            samples.push(sample);
        }

        let prefill_values = samples.iter().map(|s| s.prefill_tok_s).collect::<Vec<_>>();
        let decode_values = samples.iter().map(|s| s.decode_tok_s).collect::<Vec<_>>();
        let ttft_values = samples.iter().map(|s| s.ttft_s).collect::<Vec<_>>();
        let kv_values = samples.iter().map(|s| s.kv_cache_hit_rate_est).collect::<Vec<_>>();
        let quant_cache_values = samples.iter().map(|s| s.quant_row_cache_hit_rate).collect::<Vec<_>>();

        print_metric_summary("prefill_tok/s", &prefill_values, "tok/s");
        print_metric_summary("decode_tok/s", &decode_values, "tok/s");
        print_metric_summary("ttft_s", &ttft_values, "s");
        print_metric_summary("kv_cache_hit_rate_est", &kv_values, "ratio");
        print_metric_summary("quant_row_cache_hit_rate", &quant_cache_values, "ratio");
        println!("[bench] csv: {}", bench_csv_path().display());
    }

    #[test]
    //#[ignore = "benchmark only; requires LLAMA_CPP_CLI and LLAMA_CPP_MODEL"]
    fn bench_compare_with_llamacpp_metrics() {
        apply_benchmark_defaults();
        let rounds = bench_rounds();
        let decode_steps = bench_decode_steps();
        let threads = runtime::cpu::init_runtime_tuning().threads;
        let bench_ctx = load_bench_context();
        let prompt = templates::build_llama3_prompt("请用一句话介绍你自己", true);
        let Some((_, _)) = run_llamacpp_bench(&prompt, decode_steps, threads) else {
            panic!("llama.cpp 对照 benchmark 运行失败，请检查 llama-cli 是否可执行以及模型路径是否可读");
        };

        println!("[compare] runtime: {}", runtime::cpu::runtime_tuning_summary());
        println!("[compare] llamacpp_cli: {}", std::env::var("LLAMA_CPP_CLI").unwrap_or_else(|_| "unset".to_string()));
        println!("[compare] llamacpp_model: {}", std::env::var("LLAMA_CPP_MODEL").unwrap_or_else(|_| "unset".to_string()));
        println!("[compare] llamacpp_ngl: {}", bench_llamacpp_ngl());
        println!("[compare] rounds: {}", rounds);
        println!("[compare] decode_steps: {}", decode_steps);
        println!("[compare] mode: interleaved");
        println!("[compare] steady_state_warmup: {}", bench_steady_state_warmup_enabled());

        let mut samples = Vec::with_capacity(rounds);

        for round in 0..rounds {
            let ours = run_ours_bench_once(&bench_ctx, decode_steps);
            let (cpp_prefill_tok_s, cpp_decode_tok_s) = run_llamacpp_bench(&prompt, decode_steps, threads)
                .expect("llama.cpp bench should stay available during repeated rounds");
            let prefill_ratio = ours.prefill_tok_s / cpp_prefill_tok_s.max(1e-9);
            let decode_ratio = ours.decode_tok_s / cpp_decode_tok_s.max(1e-9);
            append_bench_csv("ours_vs_llamacpp", round + 1, &ours);

            println!("[compare-round] round: {} ours_prefill_tok/s: {:.2} llamacpp_prefill_tok/s: {:.2} prefill_ratio: {:.4}", round + 1, ours.prefill_tok_s, cpp_prefill_tok_s, prefill_ratio);
            println!("[compare-round] round: {} ours_decode_tok/s: {:.2} llamacpp_decode_tok/s: {:.2} decode_ratio: {:.4}", round + 1, ours.decode_tok_s, cpp_decode_tok_s, decode_ratio);

            samples.push(CompareBenchSample {
                ours,
                cpp_prefill_tok_s,
                cpp_decode_tok_s,
                prefill_ratio,
                decode_ratio,
            });
        }

        summarize_compare_samples("interleaved", &samples);
    }

    #[test]
    //#[ignore = "benchmark only; requires LLAMA_CPP_CLI and LLAMA_CPP_MODEL"]
    fn bench_compare_with_llamacpp_metrics_continuous() {
        apply_benchmark_defaults();
        let rounds = bench_rounds();
        let decode_steps = bench_decode_steps();
        let threads = runtime::cpu::init_runtime_tuning().threads;
        let bench_ctx = load_bench_context();
        let prompt = templates::build_llama3_prompt("请用一句话介绍你自己", true);
        let Some((_, _)) = run_llamacpp_bench(&prompt, decode_steps, threads) else {
            panic!("llama.cpp 对照 benchmark 运行失败，请检查 llama-cli 是否可执行以及模型路径是否可读");
        };

        println!("[compare-continuous] runtime: {}", runtime::cpu::runtime_tuning_summary());
        println!("[compare-continuous] llamacpp_cli: {}", std::env::var("LLAMA_CPP_CLI").unwrap_or_else(|_| "unset".to_string()));
        println!("[compare-continuous] llamacpp_model: {}", std::env::var("LLAMA_CPP_MODEL").unwrap_or_else(|_| "unset".to_string()));
        println!("[compare-continuous] llamacpp_ngl: {}", bench_llamacpp_ngl());
        println!("[compare-continuous] rounds: {}", rounds);
        println!("[compare-continuous] decode_steps: {}", decode_steps);
        println!("[compare-continuous] mode: continuous");
        println!("[compare-continuous] steady_state_warmup: {}", bench_steady_state_warmup_enabled());

        let mut ours_samples = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let ours = run_ours_bench_once(&bench_ctx, decode_steps);
            append_bench_csv("ours_vs_llamacpp_continuous", round + 1, &ours);
            println!("[compare-continuous-ours] round: {} prefill_tok/s: {:.2} decode_tok/s: {:.2}", round + 1, ours.prefill_tok_s, ours.decode_tok_s);
            ours_samples.push(ours);
        }

        let mut samples = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (cpp_prefill_tok_s, cpp_decode_tok_s) = run_llamacpp_bench(&prompt, decode_steps, threads)
                .expect("llama.cpp bench should stay available during repeated rounds");
            let ours = ours_samples[round].clone();
            let prefill_ratio = ours.prefill_tok_s / cpp_prefill_tok_s.max(1e-9);
            let decode_ratio = ours.decode_tok_s / cpp_decode_tok_s.max(1e-9);

            println!("[compare-continuous-round] round: {} ours_prefill_tok/s: {:.2} llamacpp_prefill_tok/s: {:.2} prefill_ratio: {:.4}", round + 1, ours.prefill_tok_s, cpp_prefill_tok_s, prefill_ratio);
            println!("[compare-continuous-round] round: {} ours_decode_tok/s: {:.2} llamacpp_decode_tok/s: {:.2} decode_ratio: {:.4}", round + 1, ours.decode_tok_s, cpp_decode_tok_s, decode_ratio);

            samples.push(CompareBenchSample {
                ours,
                cpp_prefill_tok_s,
                cpp_decode_tok_s,
                prefill_ratio,
                decode_ratio,
            });
        }

        summarize_compare_samples("continuous", &samples);
    }
}


fn main() {
    println!("主二进制保留给 benchmark/test 入口。真实聊天请运行: cargo run --release --bin cli");
} 

