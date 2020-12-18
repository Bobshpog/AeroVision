from src.util.memoization.memoization.constant.flag import CachingAlgorithmFlag
import src.util.memoization.memoization.caching.fifo_cache as fifo_cache
import src.util.memoization.memoization.caching.lru_cache as lru_cache
import src.util.memoization.memoization.caching.lfu_cache as lfu_cache


def get_cache_toolkit(algorithm=CachingAlgorithmFlag.FIFO):
    algorithm_mapping = {
        CachingAlgorithmFlag.FIFO: fifo_cache,
        CachingAlgorithmFlag.LRU: lru_cache,
        CachingAlgorithmFlag.LFU: lfu_cache,
    }
    try:
        return algorithm_mapping[algorithm]
    except KeyError:
        raise KeyError('Unrecognized caching algorithm flag')
