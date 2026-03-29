use std::fs;
use std::hint::black_box;
use std::mem::{size_of, zeroed};
use std::sync::OnceLock;

/// 运行时调优报告，记录当前生效的线程、绑核、大页等配置
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

/// 运行时调优的内部配置（从环境变量解析）
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

/// 读取布尔型环境变量，支持 1/true/yes/on 等写法，未设置时返回 default
fn env_flag(name: &str, default: bool) -> bool {
	std::env::var(name)
		.ok()
		.map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
		.unwrap_or(default)
}

/// 解析 CPU 绑核掩码字符串（如 "0,1,2-4"），返回去重排序后的 CPU 编号列表
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

/// 解析线程数：优先 LMRS_THREADS > RAYON_NUM_THREADS > 系统可用核心数
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

/// 解析 prefill 专用线程数（LMRS_PREFILL_THREADS），未设置时使用 default
fn parse_prefill_threads(default: usize) -> usize {
	std::env::var("LMRS_PREFILL_THREADS")
		.ok()
		.and_then(|v| v.parse::<usize>().ok())
		.filter(|&v| v > 0)
		.unwrap_or(default)
}

/// 读取 Linux 透明大页（THP）当前模式（always/madvise/never）
fn current_thp_mode() -> Option<String> {
	let raw = fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled").ok()?;
	for token in raw.split_whitespace() {
		if token.starts_with('[') && token.ends_with(']') {
			return Some(token.trim_matches(&['[', ']'][..]).to_string());
		}
	}
	Some(raw.trim().to_string())
}

/// 读取 /proc/sys/kernel/perf_event_paranoid 值，判断性能计数器可用性
fn current_perf_event_paranoid() -> Option<i32> {
	fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
		.ok()?
		.trim()
		.parse::<i32>()
		.ok()
}

/// 从环境变量汇总解析出完整的运行时调优配置
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

/// 将当前线程绑定到指定 CPU 核心（Linux 下调用 sched_setaffinity）
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

/// 非 Linux 平台的绑核空实现
#[cfg(not(target_os = "linux"))]
fn bind_current_thread(_cpu: usize) -> std::io::Result<()> {
	Ok(())
}

/// 按线程索引循环分配 CPU 核心并绑定（index % cpus.len()）
fn bind_thread_for_index(cpus: &[usize], index: usize) {
	if cpus.is_empty() {
		return;
	}
	let cpu = cpus[index % cpus.len()];
	if let Err(err) = bind_current_thread(cpu) {
		eprintln!("[runtime] 线程绑核失败: cpu={} err={}", cpu, err);
	}
}

/// 初始化运行时线程池与绑核策略，确保仅执行一次。
/// 解析 LMRS_THREADS、LMRS_CPU_MASK、LMRS_ENABLE_AFFINITY 等环境变量，
/// 初始化自定义全局线程池并将每个工作线程绑定到指定 CPU 核心。
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

		let total_threads = config.threads.max(config.prefill_threads);
		{
			let cpus = config.cpu_mask.clone();
			let affinity_enabled = config.affinity_enabled;
			super::threadpool::init(
				total_threads,
				if affinity_enabled {
					Some(std::sync::Arc::new(move |worker_idx| {
						// worker_idx 从 0 开始，主线程为 index 0，
						// 工作线程从 index 1 起
						bind_thread_for_index(&cpus, worker_idx + 1);
					}))
				} else {
					None
				},
			);
		}
		// 限制 rayon 全局池为 1 线程，避免 gemm 传递依赖创建默认线程池
		// 与我们的线程池竞争 CPU 资源
		let _ = rayon::ThreadPoolBuilder::new()
			.num_threads(1)
			.build_global();

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

/// 获取 prefill 阶段使用的线程数
pub fn prefill_threads() -> usize {
	init_runtime_tuning().prefill_threads
}

/// 检查 perf_event_paranoid 是否允许用户级性能计数器
pub fn perf_counter_available() -> bool {
	current_perf_event_paranoid().map(|v| v <= 2).unwrap_or(false)
}

/// 对大内存分配尝试应用透明大页 (THP) 建议，以降低 TLB 未命中率
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

/// Prefill 前尝试预触页/madvise 权重内存，减少 page fault。
/// 默认关闭，需显式设置 LMRS_ENABLE_PREFILL_WILLNEED/PRETOUCH 开启。
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
