use std::fs;
use std::hint::black_box;
use std::mem::{size_of, zeroed};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct RuntimeTuningReport {
	pub threads: usize,
	pub prefill_threads: usize,
	pub cpu_mask: Option<String>,
	pub affinity_enabled: bool,
	pub hugepage_hint_enabled: bool,
	pub hugepage_threshold_bytes: usize,
	pub transparent_hugepage: Option<String>,
	pub perf_event_paranoid: Option<i32>,
}

#[derive(Clone, Debug)]
struct RuntimeTuningConfig {
	threads: usize,
	prefill_threads: usize,
	cpu_mask: Vec<usize>,
	affinity_enabled: bool,
	hugepage_hint_enabled: bool,
	hugepage_threshold_bytes: usize,
}

static RUNTIME_TUNING: OnceLock<RuntimeTuningReport> = OnceLock::new();
static HUGEPAGE_HINT_ENABLED: OnceLock<bool> = OnceLock::new();
static HUGEPAGE_THRESHOLD_BYTES: OnceLock<usize> = OnceLock::new();
static PREFILL_WILLNEED_ENABLED: OnceLock<bool> = OnceLock::new();
static PREFILL_PRETOUCH_ENABLED: OnceLock<bool> = OnceLock::new();
static PREFILL_PRETOUCH_MAX_BYTES: OnceLock<usize> = OnceLock::new();

fn env_flag(name: &str, default: bool) -> bool {
	std::env::var(name)
		.ok()
		.map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
		.unwrap_or(default)
}

fn parse_cpu_mask(mask: &str) -> Vec<usize> {
	let mut cpus = Vec::new();
	for part in mask.split(',') {
		let item = part.trim();
		if item.is_empty() {
			continue;
		}
		if let Some((start, end)) = item.split_once('-') {
			let Ok(start_cpu) = start.trim().parse::<usize>() else {
				continue;
			};
			let Ok(end_cpu) = end.trim().parse::<usize>() else {
				continue;
			};
			if start_cpu <= end_cpu {
				cpus.extend(start_cpu..=end_cpu);
			}
		} else if let Ok(cpu) = item.parse::<usize>() {
			cpus.push(cpu);
		}
	}
	cpus.sort_unstable();
	cpus.dedup();
	cpus
}

fn parse_threads() -> usize {
	std::env::var("LMRS_THREADS")
		.ok()
		.and_then(|v| v.parse::<usize>().ok())
		.filter(|&v| v > 0)
		.or_else(|| {
			std::env::var("RAYON_NUM_THREADS")
				.ok()
				.and_then(|v| v.parse::<usize>().ok())
				.filter(|&v| v > 0)
		})
		.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

fn parse_prefill_threads(default: usize) -> usize {
	std::env::var("LMRS_PREFILL_THREADS")
		.ok()
		.and_then(|v| v.parse::<usize>().ok())
		.filter(|&v| v > 0)
		.unwrap_or(default)
}

fn current_thp_mode() -> Option<String> {
	let raw = fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled").ok()?;
	for token in raw.split_whitespace() {
		if token.starts_with('[') && token.ends_with(']') {
			return Some(token.trim_matches(&['[', ']'][..]).to_string());
		}
	}
	Some(raw.trim().to_string())
}

fn current_perf_event_paranoid() -> Option<i32> {
	fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
		.ok()?
		.trim()
		.parse::<i32>()
		.ok()
}

fn runtime_config_from_env() -> RuntimeTuningConfig {
	let threads = parse_threads();
	let cpu_mask = std::env::var("LMRS_CPU_MASK")
		.ok()
		.map(|v| parse_cpu_mask(&v))
		.unwrap_or_default();
	let affinity_enabled = !cpu_mask.is_empty() && env_flag("LMRS_ENABLE_AFFINITY", true);
	let hugepage_threshold_bytes = std::env::var("LMRS_HUGEPAGE_MIN_BYTES")
		.ok()
		.and_then(|v| v.parse::<usize>().ok())
		.unwrap_or(2 * 1024 * 1024);

	RuntimeTuningConfig {
		threads,
		prefill_threads: parse_prefill_threads(threads),
		cpu_mask,
		affinity_enabled,
		hugepage_hint_enabled: env_flag("LMRS_ENABLE_HUGEPAGE", true),
		hugepage_threshold_bytes,
	}
}

#[cfg(target_os = "linux")]
fn bind_current_thread(cpu: usize) -> std::io::Result<()> {
	unsafe {
		let mut set = zeroed::<libc::cpu_set_t>();
		libc::CPU_ZERO(&mut set);
		libc::CPU_SET(cpu, &mut set);
		let ret = libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
		if ret == 0 {
			Ok(())
		} else {
			Err(std::io::Error::last_os_error())
		}
	}
}

#[cfg(not(target_os = "linux"))]
fn bind_current_thread(_cpu: usize) -> std::io::Result<()> {
	Ok(())
}

fn bind_thread_for_index(cpus: &[usize], index: usize) {
	if cpus.is_empty() {
		return;
	}
	let cpu = cpus[index % cpus.len()];
	if let Err(err) = bind_current_thread(cpu) {
		eprintln!("[runtime] 线程绑核失败: cpu={} err={}", cpu, err);
	}
}

pub fn init_runtime_tuning() -> &'static RuntimeTuningReport {
	RUNTIME_TUNING.get_or_init(|| {
		let config = runtime_config_from_env();
		let cpu_mask_text = if config.cpu_mask.is_empty() {
			None
		} else {
			Some(
				config
					.cpu_mask
					.iter()
					.map(|cpu| cpu.to_string())
					.collect::<Vec<_>>()
					.join(","),
			)
		};

		HUGEPAGE_HINT_ENABLED.get_or_init(|| config.hugepage_hint_enabled);
		HUGEPAGE_THRESHOLD_BYTES.get_or_init(|| config.hugepage_threshold_bytes);

		let build_result = rayon::ThreadPoolBuilder::new()
			.num_threads(config.threads.max(config.prefill_threads))
			.start_handler({
				let cpus = config.cpu_mask.clone();
				let affinity_enabled = config.affinity_enabled;
				move |index| {
					if affinity_enabled {
						bind_thread_for_index(&cpus, index);
					}
				}
			})
			.build_global();

		if let Err(err) = build_result {
			eprintln!("[runtime] rayon 全局线程池保持默认配置: {}", err);
		}

		if config.affinity_enabled {
			bind_thread_for_index(&config.cpu_mask, 0);
		}

		RuntimeTuningReport {
			threads: config.threads,
			prefill_threads: config.prefill_threads,
			cpu_mask: cpu_mask_text,
			affinity_enabled: config.affinity_enabled,
			hugepage_hint_enabled: config.hugepage_hint_enabled,
			hugepage_threshold_bytes: config.hugepage_threshold_bytes,
			transparent_hugepage: current_thp_mode(),
			perf_event_paranoid: current_perf_event_paranoid(),
		}
	})
}

pub fn runtime_tuning_summary() -> String {
	let report = init_runtime_tuning();
	format!(
		"threads={} prefill_threads={} affinity={} cpu_mask={} hugepage_hint={} hugepage_min_bytes={} thp={} perf_event_paranoid={}",
		report.threads,
		report.prefill_threads,
		report.affinity_enabled,
		report.cpu_mask.as_deref().unwrap_or("auto"),
		report.hugepage_hint_enabled,
		report.hugepage_threshold_bytes,
		report.transparent_hugepage.as_deref().unwrap_or("unknown"),
		report
			.perf_event_paranoid
			.map(|v| v.to_string())
			.unwrap_or_else(|| "unknown".to_string())
	)
}

pub fn prefill_threads() -> usize {
	init_runtime_tuning().prefill_threads
}

pub fn perf_counter_available() -> bool {
	current_perf_event_paranoid().map(|v| v <= 2).unwrap_or(false)
}

pub fn maybe_advise_hugepage(ptr: *mut u8, bytes: usize) {
	if bytes == 0 {
		return;
	}
	let enabled = *HUGEPAGE_HINT_ENABLED.get_or_init(|| runtime_config_from_env().hugepage_hint_enabled);
	let threshold = *HUGEPAGE_THRESHOLD_BYTES.get_or_init(|| runtime_config_from_env().hugepage_threshold_bytes);
	if !enabled || bytes < threshold {
		return;
	}

	#[cfg(target_os = "linux")]
	unsafe {
		let page_size = libc::sysconf(libc::_SC_PAGESIZE);
		if page_size <= 0 {
			return;
		}
		let page_size = page_size as usize;
		let start = ptr as usize;
		let aligned_start = start & !(page_size - 1);
		let aligned_end = (start + bytes + page_size - 1) & !(page_size - 1);
		if aligned_end > aligned_start {
			let advise_ptr = aligned_start as *mut libc::c_void;
			let advise_len = aligned_end - aligned_start;
			let _ = libc::madvise(advise_ptr, advise_len, libc::MADV_HUGEPAGE);
		}
	}
}

pub fn maybe_prepare_prefill_weight_pages(ptr: *const u8, bytes: usize) {
	if bytes == 0 {
		return;
	}

	#[cfg(target_os = "linux")]
	unsafe {
		let page_size = libc::sysconf(libc::_SC_PAGESIZE);
		if page_size <= 0 {
			return;
		}
		let page_size = page_size as usize;
		let start = ptr as usize;
		let aligned_start = start & !(page_size - 1);
		let aligned_end = (start + bytes + page_size - 1) & !(page_size - 1);
		if aligned_end <= aligned_start {
			return;
		}

		let advise_ptr = aligned_start as *mut libc::c_void;
		let advise_len = aligned_end - aligned_start;
		// 阶段 F 已证明默认开启会伤害交错 benchmark，
		// 这里只保留为显式实验开关。
		let willneed = *PREFILL_WILLNEED_ENABLED.get_or_init(|| env_flag("LMRS_ENABLE_PREFILL_WILLNEED", false));
		if willneed {
			let _ = libc::madvise(advise_ptr, advise_len, libc::MADV_WILLNEED);
			let _ = libc::madvise(advise_ptr, advise_len, libc::MADV_SEQUENTIAL);
		}

		// 顺序预触页同样只保留给显式实验，不再默认进入主路径。
		let pretouch = *PREFILL_PRETOUCH_ENABLED.get_or_init(|| env_flag("LMRS_ENABLE_PREFILL_PRETOUCH", false));
		if !pretouch {
			return;
		}

		let max_bytes = *PREFILL_PRETOUCH_MAX_BYTES.get_or_init(|| {
			std::env::var("LMRS_PREFILL_PRETOUCH_MB")
				.ok()
				.and_then(|v| v.parse::<usize>().ok())
				.filter(|v| *v > 0)
				.unwrap_or(768)
				* 1024
				* 1024
		});
		let touch_len = advise_len.min(max_bytes);
		if touch_len == 0 {
			return;
		}

		let base = aligned_start as *const u8;
		let mut checksum = 0u8;
		let mut offset = 0usize;
		while offset < touch_len {
			checksum ^= std::ptr::read_volatile(base.add(offset));
			offset = offset.saturating_add(page_size);
		}
		checksum ^= std::ptr::read_volatile(base.add(touch_len - 1));
		black_box(checksum);
	}
}
