try:
    import gpu_affinity as ga

    SUPPORT_GPU_AFFINITY = True
except ImportError:
    SUPPORT_GPU_AFFINITY = False

from hyperpod_nemo_adapter.utils.log_utils import Logger

logger = Logger().get_logger()


def initialize_gpu_affinity(gpu_id, nproc_per_node):
    if not SUPPORT_GPU_AFFINITY:
        return

    try:
        affinity = ga.set_affinity(gpu_id, nproc_per_node)
        logger.debug(f"[GPU:{gpu_id}] affinity: {affinity}")
    except Exception as e:
        logger.warning(f"set_affinity fail. error: {e}")
