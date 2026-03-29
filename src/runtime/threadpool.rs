//! 持久化线程池，替代 Rayon per-op 调度。
//!
//! 设计要点：
//! - 工作线程常驻，通过 spin-poll + condvar 混合策略接收新任务
//! - `parallel_for`：原子计数器分发工作块，调用线程同时参与计算
//! - `join`：二路并行（prefill 专用），通过 `std::thread::scope` 实现
//! - 针对单 token decode 场景优化：spin-poll 最小化线程唤醒延迟

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// spin-poll 轮数上限，超过后回退到 condvar 等待
const SPIN_ITERS: u32 = 128_000;

static POOL: OnceLock<ThreadPool> = OnceLock::new();

/// 初始化全局线程池。
/// * `num_threads` — 总线程数（含调用线程），工作线程实际数量为 num_threads - 1
/// * `start_handler` — 可选的线程启动回调，参数为 worker index（从 0 开始），
///   用于绑定 CPU 亲和性等
pub fn init(num_threads: usize, start_handler: Option<Arc<dyn Fn(usize) + Send + Sync>>) {
    POOL.get_or_init(|| ThreadPool::new(num_threads, start_handler));
}

/// 获取全局线程池引用（init 未调用时 panic）
pub fn pool() -> &'static ThreadPool {
    POOL.get().expect("threadpool: 请先调用 init()")
}

/// 返回线程池总线程数（含调用线程）
pub fn num_threads() -> usize {
    POOL.get().map_or(1, |p| p.inner.num_workers + 1)
}

/// 持久化线程池主体
pub struct ThreadPool {
    inner: Arc<Inner>,
    _handles: Vec<std::thread::JoinHandle<()>>,
}

/// 类型擦除的工作函数描述：call(data_ptr, chunk_index)
struct WorkFn {
    call: unsafe fn(*const (), usize),
    ptr: *const (),
}

/// 线程池共享状态
struct Inner {
    num_workers: usize,

    // ---- parallel_for 专用 ----
    /// 串行化并发 parallel_for 调用（如 join 两侧同时调 parallel_for）
    par_lock: Mutex<()>,
    /// 工作线程 spin-poll 监听此代数变化
    par_gen: AtomicU64,
    /// 当前工作函数指针（仅在 par_lock 持有期间有效）
    par_fn: UnsafeCell<Option<WorkFn>>,
    /// 总块数
    par_total: AtomicUsize,
    /// 下一个待领取的块索引
    par_next: AtomicUsize,
    /// 已完成的块数
    par_done: AtomicUsize,

    // ---- 完成信号 ----
    done_lock: Mutex<()>,
    done_cv: Condvar,

    // ---- 工作线程休眠/唤醒 ----
    wake_lock: Mutex<()>,
    wake_cv: Condvar,
    /// 当前在 condvar 上休眠的工作线程计数
    sleeping: AtomicUsize,

    shutdown: AtomicBool,
}

// SAFETY:
// - par_fn 的写入受 par_lock 保护；
// - 读取发生在 par_gen Acquire 之后，保证可见性；
// - parallel_for 阻塞至所有 chunk 完成才清理 par_fn，不存在悬垂指针。
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// 类型擦除调用桩：将 ptr 转回 &F 并以 idx 调用
unsafe fn call_typed<F: Fn(usize)>(ptr: *const (), idx: usize) {
    let f = &*(ptr as *const F);
    f(idx);
}

impl ThreadPool {
    fn new(
        num_threads: usize,
        start_handler: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    ) -> Self {
        let num_workers = if num_threads > 1 { num_threads - 1 } else { 0 };
        let inner = Arc::new(Inner {
            num_workers,
            par_lock: Mutex::new(()),
            par_gen: AtomicU64::new(0),
            par_fn: UnsafeCell::new(None),
            par_total: AtomicUsize::new(0),
            par_next: AtomicUsize::new(0),
            par_done: AtomicUsize::new(0),
            done_lock: Mutex::new(()),
            done_cv: Condvar::new(),
            wake_lock: Mutex::new(()),
            wake_cv: Condvar::new(),
            sleeping: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        });

        let mut handles = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let inner_c = inner.clone();
            let handler = start_handler.clone();
            handles.push(std::thread::spawn(move || {
                // 执行用户回调（绑核等）
                if let Some(h) = handler {
                    h(i);
                }
                worker_main(&inner_c);
            }));
        }

        ThreadPool {
            inner,
            _handles: handles,
        }
    }

    /// 将 `total_chunks` 个任务并行分发到所有工作线程。
    /// 调用线程同时参与计算；函数返回时所有 chunk 均已完成。
    ///
    /// `f(i)` 中 `i` 为 chunk 索引 `0..total_chunks`。
    pub fn parallel_for<F: Fn(usize) + Sync>(&self, total_chunks: usize, f: F) {
        if total_chunks == 0 {
            return;
        }
        if self.inner.num_workers == 0 || total_chunks == 1 {
            for i in 0..total_chunks {
                f(i);
            }
            return;
        }

        // 串行化并发调用（join 两侧可能同时进入）
        let _guard = self.inner.par_lock.lock().unwrap();

        // 设置原子参数（Release 语义确保 worker 看到一致的状态）
        self.inner.par_total.store(total_chunks, Ordering::Relaxed);
        self.inner.par_next.store(0, Ordering::Relaxed);
        self.inner.par_done.store(0, Ordering::Relaxed);

        let work = WorkFn {
            call: call_typed::<F>,
            ptr: &f as *const F as *const (),
        };
        // SAFETY: parallel_for 在此栈帧返回前不会释放
        unsafe {
            *self.inner.par_fn.get() = Some(work);
        }

        // 唤醒工作线程：先递增代数（Release 保证 par_fn 对 worker 可见），
        // 仅在有 worker 休眠时通过 condvar 唤醒
        self.inner.par_gen.fetch_add(1, Ordering::Release);
        if self.inner.sleeping.load(Ordering::Acquire) > 0 {
            let _wake = self.inner.wake_lock.lock().unwrap();
            self.inner.wake_cv.notify_all();
        }

        // 调用线程参与处理
        process_chunks(&self.inner);

        // 等待全部完成
        wait_all_done(&self.inner, total_chunks);

        // 清理工作函数指针
        unsafe {
            *self.inner.par_fn.get() = None;
        }
    }
}

/// 原始指针的 Send+Sync 包装器。
/// SAFETY: 调用方保证并行访问不重叠且指针在 parallel_for 完成前有效。
struct SyncPtr<T>(*mut T);
impl<T> Copy for SyncPtr<T> {}
impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self { *self }
}
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// 按 chunk_size 将 data 分块，对每个块并行调用 `f(chunk_index, chunk_slice)`。
/// 等价于 `rayon::par_chunks_mut`。
///
/// # Safety
/// 各块互不重叠，可安全并行写入。
pub fn parallel_chunks_mut<T: Send, F>(data: &mut [T], chunk_size: usize, f: F)
where
    F: Fn(usize, &mut [T]) + Sync,
{
    if data.is_empty() || chunk_size == 0 {
        return;
    }
    let total_chunks = (data.len() + chunk_size - 1) / chunk_size;
    let ptr = SyncPtr(data.as_mut_ptr());
    let len = data.len();

    // SAFETY: 每个 chunk 映射到互不重叠的连续子区间
    pool().parallel_for(total_chunks, |i| {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(len);
        let chunk = unsafe { std::slice::from_raw_parts_mut(ptr.get().add(start), end - start) };
        f(i, chunk);
    });
}

/// 对切片每个元素并行调用 `f(index, &mut element)`。
/// 等价于 `rayon::par_iter_mut().enumerate().for_each(...)`。
/// 使用 16×线程数分块以最大化调用线程利用率：调用线程处理完自己的
/// chunk 后继续领取剩余块，而不是在 wait_all_done 中空转。
pub fn parallel_iter_mut<T: Send, F>(data: &mut [T], f: F)
where
    F: Fn(usize, &mut T) + Sync,
{
    let len = data.len();
    if len == 0 {
        return;
    }
    let nthreads = num_threads().max(1);
    let num_chunks = (nthreads * 16).min(len);
    let chunk_size = (len + num_chunks - 1) / num_chunks;
    let ptr = SyncPtr(data.as_mut_ptr());
    let actual_chunks = (len + chunk_size - 1) / chunk_size;
    pool().parallel_for(actual_chunks, |chunk_idx| {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(len);
        for i in start..end {
            // SAFETY: 各 chunk_idx 对应不重叠的元素范围
            let elem = unsafe { &mut *ptr.get().add(i) };
            f(i, elem);
        }
    });
}

/// 二路并行执行（prefill 阶段 batch2/batch3 使用）。
/// 在调用线程执行 `a`，同时在另一线程执行 `b`，等待两者完成后返回结果。
pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    if POOL.get().map_or(true, |p| p.inner.num_workers == 0) {
        return (a(), b());
    }
    std::thread::scope(|s| {
        let handle = s.spawn(|| b());
        let ra = a();
        let rb = handle.join().unwrap();
        (ra, rb)
    })
}

// ---- 内部函数 ----

/// 工作线程主循环：spin-poll 等待新任务，超时后回退到 condvar 休眠
fn worker_main(inner: &Inner) {
    let mut last_gen = 0u64;
    loop {
        // ---- 阶段 1：spin-poll 检测代数变化 ----
        let mut found = false;
        for _ in 0..SPIN_ITERS {
            if inner.shutdown.load(Ordering::Relaxed) {
                return;
            }
            let gen = inner.par_gen.load(Ordering::Acquire);
            if gen != last_gen {
                last_gen = gen;
                found = true;
                break;
            }
            std::hint::spin_loop();
        }

        // ---- 阶段 2：condvar 休眠 ----
        if !found {
            inner.sleeping.fetch_add(1, Ordering::Release);
            let guard = inner.wake_lock.lock().unwrap();
            // 双重检查：锁获取期间代数可能已变
            let gen = inner.par_gen.load(Ordering::Acquire);
            if gen == last_gen {
                if inner.shutdown.load(Ordering::Relaxed) {
                    inner.sleeping.fetch_sub(1, Ordering::Release);
                    return;
                }
                // 无限等待直到被唤醒
                let _g = inner.wake_cv.wait(guard).unwrap();
                inner.sleeping.fetch_sub(1, Ordering::Release);
                let gen = inner.par_gen.load(Ordering::Acquire);
                if gen == last_gen {
                    continue; // 虚假唤醒
                }
                last_gen = gen;
            } else {
                inner.sleeping.fetch_sub(1, Ordering::Release);
                last_gen = gen;
                drop(guard);
            }
        }

        if inner.shutdown.load(Ordering::Relaxed) {
            return;
        }

        // 处理工作块
        process_chunks(inner);
    }
}

/// 领取并执行工作块，直到所有块被消耗完毕
fn process_chunks(inner: &Inner) {
    loop {
        let idx = inner.par_next.fetch_add(1, Ordering::Relaxed);
        let total = inner.par_total.load(Ordering::Relaxed);
        if idx >= total {
            break;
        }
        // SAFETY: par_fn 在 parallel_for 返回前保持有效（栈上闭包 + par_lock）
        let work = unsafe { &*inner.par_fn.get() };
        if let Some(ref w) = work {
            unsafe {
                (w.call)(w.ptr, idx);
            }
        }
        let prev_done = inner.par_done.fetch_add(1, Ordering::Release);
        if prev_done + 1 >= total {
            // 最后一个 chunk 完成 → 通知等待者
            let _lock = inner.done_lock.lock().unwrap();
            inner.done_cv.notify_all();
        }
    }
}

/// 等待所有 chunk 完成：先短暂 spin，再 condvar
fn wait_all_done(inner: &Inner, total: usize) {
    // 短暂 spin（覆盖大部分同步窗口）
    for _ in 0..4096 {
        if inner.par_done.load(Ordering::Acquire) >= total {
            return;
        }
        std::hint::spin_loop();
    }
    // 回退到 condvar
    let mut guard = inner.done_lock.lock().unwrap();
    while inner.par_done.load(Ordering::Acquire) < total {
        guard = inner.done_cv.wait(guard).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        // 递增代数 + condvar 唤醒，确保所有 worker 检测到 shutdown
        self.inner.par_gen.fetch_add(1, Ordering::Release);
        let _lock = self.inner.wake_lock.lock().unwrap();
        self.inner.wake_cv.notify_all();
    }
}
