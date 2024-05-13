//! Implements the [`Backend`] and [`BackendRef`] structs for managing llama.cpp
//! backends

use std::ptr;

use std::sync::Mutex;
use tracing::error;

use llama_cpp_sys::{
    ggml_numa_strategy, llama_backend_free, llama_backend_init, llama_log_set, llama_numa_init,
};

use crate::detail;

/// The current instance of [`Backend`], if it exists. Also stored is a reference count used for
/// initialisation and freeing.
static BACKEND: Mutex<Option<(Backend, usize)>> = Mutex::new(None);

/// Empty struct used to initialise and free the [llama.cpp][llama.cpp] backend when it is created
/// dropped respectively.
///
/// [llama.cpp]: https://github.com/ggerganov/llama.cpp/
struct Backend;

impl Backend {
    fn init() -> Self {
        Self::init_with_numa(NumaStrategy::Distribute)
    }

    /// Initialises the [llama.cpp][llama.cpp] backend and sets its logger.
    ///
    /// There should only ever be one instance of this struct at any given time.
    ///
    /// [llama.cpp]: https://github.com/ggerganov/llama.cpp/
    fn init_with_numa(numa: NumaStrategy) -> Self {
        unsafe {
            // SAFETY: This is only called when no models or sessions exist.
            llama_backend_init();

            // TODO look into numa strategies, this should probably be part of the API
            llama_numa_init(numa.into());

            // SAFETY: performs a simple assignment to static variables. Should only execute once
            // before any logs are made.
            llama_log_set(Some(detail::llama_log_callback), ptr::null_mut());
        }
        Self
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This is only called when no models or sessions exist.
            llama_backend_free();
        }
    }
}

/// A "reference" to [`BACKEND`].
///
/// Initialises [`BACKEND`] if there is no [`Backend`] inside. If there are no other references,
/// this drops [`Backend`] upon getting itself dropped.
pub(crate) struct BackendRef {}

impl BackendRef {
    /// Creates a new reference, initialising [`BACKEND`] if necessary.
    pub(crate) fn new() -> Self {
        let mut lock = BACKEND.lock().unwrap();
        if let Some((_, count)) = lock.as_mut() {
            *count += 1;
        } else {
            let _ = lock.insert((Backend::init(), 1));
        }

        Self {}
    }
}

impl Drop for BackendRef {
    fn drop(&mut self) {
        let mut lock = BACKEND.lock().unwrap();
        if let Some((_, count)) = lock.as_mut() {
            *count -= 1;

            if *count == 0 {
                lock.take();
            }
        } else {
            error!("Backend as already been freed, this should never happen")
        }
    }
}

impl Clone for BackendRef {
    fn clone(&self) -> Self {
        Self::new()
    }
}

/// A policy to split the model across multiple GPUs
#[non_exhaustive]
pub enum NumaStrategy {
    Disable,
    Distribute,
    Isolate,
    Numactl,
    Mirror,
    Count,
}

impl From<NumaStrategy> for ggml_numa_strategy {
    fn from(value: NumaStrategy) -> Self {
        match value {
            NumaStrategy::Disable => ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED,
            NumaStrategy::Distribute => ggml_numa_strategy::GGML_NUMA_STRATEGY_DISTRIBUTE,
            NumaStrategy::Isolate => ggml_numa_strategy::GGML_NUMA_STRATEGY_ISOLATE,
            NumaStrategy::Numactl => ggml_numa_strategy::GGML_NUMA_STRATEGY_NUMACTL,
            NumaStrategy::Mirror => ggml_numa_strategy::GGML_NUMA_STRATEGY_MIRROR,
            NumaStrategy::Count => ggml_numa_strategy::GGML_NUMA_STRATEGY_COUNT,
        }
    }
}

impl From<ggml_numa_strategy> for NumaStrategy {
    fn from(value: ggml_numa_strategy) -> Self {
        #![allow(non_upper_case_globals)]
        match value {
            ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED => NumaStrategy::Disable,
            ggml_numa_strategy::GGML_NUMA_STRATEGY_DISTRIBUTE => NumaStrategy::Distribute,
            ggml_numa_strategy::GGML_NUMA_STRATEGY_ISOLATE => NumaStrategy::Isolate,
            ggml_numa_strategy::GGML_NUMA_STRATEGY_NUMACTL => NumaStrategy::Numactl,
            ggml_numa_strategy::GGML_NUMA_STRATEGY_MIRROR => NumaStrategy::Mirror,
            ggml_numa_strategy::GGML_NUMA_STRATEGY_COUNT => NumaStrategy::Count,
            _ => unimplemented!(),
        }
    }
}
