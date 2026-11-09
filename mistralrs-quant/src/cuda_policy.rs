use crate::cuda_device::CudaDeviceInfo;

// Measured on sm_120: MMQ's tensor-core tiles cost the same from 1 to 8 tokens while MMVQ's per-token
// dot loops grow linearly, so MMQ wins from 3 tokens (3.5x on a fused GLU at 8).
const BLACKWELL_MMVQ_MAX_BATCH: usize = 2;
const BLACKWELL_MAJOR: i32 = 12;

pub(crate) fn mmvq_max_batch(device: &CudaDeviceInfo, kernel_max: usize) -> usize {
    if device.major >= BLACKWELL_MAJOR {
        BLACKWELL_MMVQ_MAX_BATCH.min(kernel_max)
    } else {
        kernel_max
    }
}

pub fn prefers_weight_prefetch(device: &CudaDeviceInfo) -> bool {
    device.integrated && device.has_cp_async()
}

pub(crate) fn fuse_underfilled_grids(
    device: &CudaDeviceInfo,
    blocks: &[usize],
    threads_per_block: usize,
) -> bool {
    let resident_blocks = device.sms * (device.threads_per_sm / threads_per_block);
    // Separate full waves amortize launch overhead; combining underfilled grids shares that cost.
    blocks.iter().all(|&grid| grid <= resident_blocks)
}
