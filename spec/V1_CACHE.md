# V1 routed-expert cache

## Goal

Prove the K3-style storage primitive before implementing Kimi-VL math: a router can hand
an entire top-k to the cache, the cache can overlap independent expert reads, and every
returned gate/up/down pointer still names the exact checkpoint bytes while the arena
stays under a caller-declared memory budget.

## State machine

Each slot is exactly one of:

- `EMPTY`: free for reservation.
- `INFLIGHT`: exclusively reserved by a prefetch batch; no lookup may serve it.
- `VALID(key)`: contains a fully-read `(layer, expert)` record and may be returned as hit.

The key is flattened as `layer * n_experts + expert`.

## `getmany(layer, ids, n)`

1. **Reserve serially**: skip hits/duplicates, choose distinct LRU victims, invalidate old
   reverse mappings, mark each chosen slot `INFLIGHT`.
2. **Sort by physical offset**: missing records are sorted by `file_offset`.
3. **Read concurrently**: only positioned reads touch the reserved buffers; cache metadata
   is not mutated by worker threads.
4. **Publish serially**: only complete reads become `VALID`; a failed read returns its slot
   to `EMPTY` and cannot become a false hit.

If the cache has fewer free/victim slots than the requested top-k, `getmany` prefetches as
many as it can. Later `get()` calls synchronously load the remaining experts. This trades
speed for correctness without exceeding the budget.

## Hard budget

`slot_bytes = align_up(max(record.read_bytes), 4096)`

`n_slots = floor(cache_budget / slot_bytes)`

`arena_bytes = n_slots * slot_bytes <= cache_budget`

V1 deliberately treats metadata as process overhead rather than part of the expert arena;
a whole-runtime memory planner will account for trunk/state/cache/vision together in a
later milestone.

For real Kimi-VL BF16 experts, one slot is roughly 16.5 MiB. After MXFP4 conversion the
same logical expert is expected to be roughly 4.38 MiB.

## Metrics

V1 reports request hit/miss counts, evictions, prefetch batches/reads, positioned-read
count, bytes read, aggregate batch wall-time throughput, and read failures.

## Tests completed

- Packer round-trip: exact BF16 bytes + 4096-byte alignment.
- Over-capacity cache stress: top-6 requests against 3 slots for 25 repetitions.
- Exact FNV-1a payload checksum after every cache lookup.
- AddressSanitizer + UndefinedBehaviorSanitizer run.
- Direct-I/O probe succeeded on the development Linux filesystem (`direct_io=yes`).
