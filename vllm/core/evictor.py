import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

def _log(msg: str):
    try:
        with open("evict.temp", "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate
    the correct Evictor subclass.
    """
    LRU = enum.auto()
    FIFO = enum.auto()
    LFU = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        physical block id along with its content hash.
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describing a cached block so that
    the evictor can decide which one to evict.

    We key by physical block id in the evictor’s free_table because multiple
    blocks can share the same content_hash, but physical IDs are unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed

        # For LFU
        self.access_count: int = 0

        # For FIFO (monotonic insertion order)
        self.insert_order: int = 0


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted.
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        # Heap entries: (last_accessed, -num_hashed_tokens, block_id, content_hash)
        self.priority_queue: List[Tuple[float, int, int, int]] = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:        
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We keep stale entries in the heap and filter them out at eviction.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue
            )
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                # debug line
                _log(f"[{self.__class__.__name__}] Evicting block {block_id}")
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        self.free_table[block_id] = meta
        heapq.heappush(
            self.priority_queue,
            (meta.last_accessed, -meta.num_hashed_tokens, block_id,
             meta.content_hash),
        )
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        # For LRU we only update the stored last_accessed; the heap is lazily
        # cleaned during evict() / _cleanup().
        if block_id in self.free_table:
            self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if self.free_table and len(self.priority_queue) > (
            LRUEvictor.CLEANUP_THRESHOLD * len(self.free_table)
        ):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []
        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash)
            )
        heapq.heapify(new_priority_queue)
        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor"
            )
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class FIFOEvictor(Evictor):
    """Evicts blocks in first-in-first-out order, ignoring recency of access."""

    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        # Heap entries: (insert_order, -num_hashed_tokens, block_id, content_hash)
        self.priority_queue: List[Tuple[int, int, int, int]] = []
        self._counter: int = 0  # monotonic insertion counter

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            insert_order, _, block_id, content_hash = heapq.heappop(
                self.priority_queue
            )
            if block_id in self.free_table:
                # insert_order doesn't change, so no staleness check needed
                self.free_table.pop(block_id)
                # debug line
                _log(f"[{self.__class__.__name__}] Evicting block {block_id}")
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        self._counter += 1
        meta.insert_order = self._counter
        self.free_table[block_id] = meta
        heapq.heappush(
            self.priority_queue,
            (meta.insert_order, -meta.num_hashed_tokens, block_id,
             meta.content_hash),
        )
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        # FIFO ignores recency for eviction, but we can keep the timestamp
        # for debugging.
        if block_id in self.free_table:
            self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if self.free_table and len(self.priority_queue) > (
            FIFOEvictor.CLEANUP_THRESHOLD * len(self.free_table)
        ):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[int, int, int, int]] = []
        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.insert_order, -block.num_hashed_tokens, block_id,
                 block.content_hash)
            )
        heapq.heapify(new_priority_queue)
        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor"
            )
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class LFUEvictor(Evictor):
    """Evicts blocks in least-frequently-used order.
    - Primary key: access_count (smaller -> evict first)
    - Tiebreak: last_accessed (older -> evict first)
    """

    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        # Heap entries: (access_count, last_accessed, block_id, content_hash)
        self.priority_queue: List[Tuple[int, float, int, int]] = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            access_count, last_accessed, block_id, content_hash = heapq.heappop(
                self.priority_queue
            )
            if (block_id in self.free_table and
                    self.free_table[block_id].access_count == access_count and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                # debug line
                _log(f"[{self.__class__.__name__}] Evicting block {block_id}")
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        meta.access_count = 0
        self.free_table[block_id] = meta
        heapq.heappush(
            self.priority_queue,
            (meta.access_count, meta.last_accessed, block_id,
             meta.content_hash),
        )
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        # Each update is treated as a "use": increment frequency and record
        # latest access time for tiebreaking.
        if block_id in self.free_table:
            meta = self.free_table[block_id]
            meta.access_count += 1
            meta.last_accessed = last_accessed
            # We don't push a new heap entry here; we rely on lazy cleanup
            # like LRU. _cleanup() will rebuild a fresh heap from metadata.

    def _cleanup_if_necessary(self):
        if self.free_table and len(self.priority_queue) > (
            LFUEvictor.CLEANUP_THRESHOLD * len(self.free_table)
        ):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[int, float, int, int]] = []
        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.access_count, block.last_accessed, block_id,
                 block.content_hash)
            )
        heapq.heapify(new_priority_queue)
        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor"
            )
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    if eviction_policy == EvictionPolicy.FIFO:
        return FIFOEvictor()
    if eviction_policy == EvictionPolicy.LFU:
        return LFUEvictor()
    raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")