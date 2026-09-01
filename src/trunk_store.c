#define _FILE_OFFSET_BITS 64
#ifndef _WIN32
#define _GNU_SOURCE
#endif
#include "kvl/trunk_store.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>

static int open_data(const char *path, int direct) {
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED |
                  (direct ? FILE_FLAG_NO_BUFFERING : 0);
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    int fd = _open_osfhandle((intptr_t)h, _O_RDONLY | _O_BINARY);
    if (fd < 0) CloseHandle(h);
    return fd;
}

static int64_t pread_all(int fd, void *buf, size_t n, uint64_t off) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    size_t done = 0;
    while (done < n) {
        DWORD chunk = (DWORD)((n - done) > 0x40000000u ? 0x40000000u : (n - done));
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        uint64_t at = off + done;
        ov.Offset = (DWORD)at;
        ov.OffsetHigh = (DWORD)(at >> 32);
        ov.hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
        if (!ov.hEvent) return -1;

        DWORD got = 0;
        BOOL ok = ReadFile(h, (unsigned char *)buf + done, chunk, &got, &ov);
        if (!ok) {
            DWORD err = GetLastError();
            if (err != ERROR_IO_PENDING || !GetOverlappedResult(h, &ov, &got, TRUE)) {
                CloseHandle(ov.hEvent);
                return -1;
            }
        }
        CloseHandle(ov.hEvent);
        if (!got) break;
        done += got;
    }
    return (int64_t)done;
}

static int aligned_alloc_buf(void **out, size_t align, size_t n) {
    *out = _aligned_malloc(n, align);
    return *out ? 0 : -1;
}
static void aligned_free_buf(void *p) { _aligned_free(p); }
#else
#include <unistd.h>
static int open_data(const char *path, int direct) {
#ifdef O_DIRECT
    if (direct) return open(path, O_RDONLY | O_DIRECT);
#else
    if (direct) { errno = EINVAL; return -1; }
#endif
    return open(path, O_RDONLY);
}
static int64_t pread_all(int fd, void *buf, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t got = pread(fd, (unsigned char*)buf + done, n - done, (off_t)(off + done));
        if (got < 0) { if (errno == EINTR) continue; return -1; }
        if (!got) break;
        done += (size_t)got;
    }
    return (int64_t)done;
}
static int aligned_alloc_buf(void **out, size_t align, size_t n) {
    return posix_memalign(out, align, n);
}
static void aligned_free_buf(void *p) { free(p); }
#endif

static uint64_t lookup_hash(uint32_t layer, uint32_t kind) {
    uint64_t x = ((uint64_t)layer << 32) | (uint64_t)kind;
    x ^= x >> 33;
    x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x *= UINT64_C(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    return x;
}

static int build_record_lookup(KvlTrunkStore *s) {
    if (!s || !s->records || s->hdr.n_records == 0) return -1;
    const size_t n = (size_t)s->hdr.n_records;
    if (n > SIZE_MAX / 2u) return -1;
    const size_t need = n * 2u;
    size_t cap = 8u;
    while (cap < need) {
        if (cap > SIZE_MAX / 2u) return -1;
        cap <<= 1u;
    }
    if (cap > SIZE_MAX / sizeof *s->lookup_slots) return -1;

    s->lookup_slots = (int32_t *)malloc(cap * sizeof *s->lookup_slots);
    if (!s->lookup_slots) return -1;
    s->lookup_cap = cap;
    for (size_t i = 0; i < cap; ++i) s->lookup_slots[i] = -1;

    const size_t mask = cap - 1u;
    for (uint32_t i = 0; i < s->hdr.n_records; ++i) {
        const KvlTrunkRecord *r = &s->records[i];
        size_t at = (size_t)lookup_hash(r->layer, r->kind) & mask;
        int inserted = 0;
        for (size_t probe = 0; probe < cap; ++probe) {
            const int32_t prev = s->lookup_slots[at];
            if (prev < 0) {
                s->lookup_slots[at] = (int32_t)i;
                inserted = 1;
                break;
            }
            const KvlTrunkRecord *p = &s->records[prev];
            if (p->layer == r->layer && p->kind == r->kind) {
                fprintf(stderr, "kvl_trunk: duplicate record layer=%u kind=%u\n",
                        r->layer, r->kind);
                return -1;
            }
            at = (at + 1u) & mask;
        }
        if (!inserted) return -1;
    }
    return 0;
}

static int read_index(KvlTrunkStore *s, const char *idx_path) {
    FILE *f = fopen(idx_path, "rb");
    if (!f) return -1;
    if (fread(&s->hdr, 1, sizeof s->hdr, f) != sizeof s->hdr) { fclose(f); return -1; }
    if (memcmp(s->hdr.magic, KVL_TRUNK_MAGIC, 8) != 0 ||
        s->hdr.version != KVL_TRUNK_VERSION || s->hdr.align != KVL_TRUNK_ALIGN) {
        fclose(f); return -1;
    }
    if (s->hdr.n_records == 0 || fseek(f, (long)s->hdr.records_offset, SEEK_SET) != 0) {
        fclose(f); return -1;
    }
    s->records = (KvlTrunkRecord*)calloc(s->hdr.n_records, sizeof *s->records);
    if (!s->records) { fclose(f); return -1; }
    if (fread(s->records, sizeof *s->records, s->hdr.n_records, f) != s->hdr.n_records) {
        fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

static int read_cache_budget(size_t *out, int *configured) {
    if (!out || !configured) return -1;
    *out = 0;
    *configured = 0;
    const char *raw = getenv("KVL_TRUNK_CACHE_MIB");
    if (!raw || !raw[0]) return 0;
    *configured = 1;
    errno = 0;
    char *end = NULL;
    const unsigned long long mib = strtoull(raw, &end, 10);
    if (errno || !end || *end != '\0' || mib > (unsigned long long)(SIZE_MAX / 1048576u)) {
        fprintf(stderr, "kvl_trunk_cache: invalid KVL_TRUNK_CACHE_MIB=%s\n", raw);
        return -1;
    }
    *out = (size_t)mib * 1048576u;
    return 0;
}

int kvl_trunk_store_open(KvlTrunkStore *s, const char *bin_path, const char *idx_path,
                         int prefer_direct_io) {
    if (!s) return -1;
    memset(s, 0, sizeof *s); s->fd = -1;
    if (read_index(s, idx_path) != 0 || build_record_lookup(s) != 0) {
        kvl_trunk_store_close(s);
        return -1;
    }
    if (read_cache_budget(&s->cache_budget_bytes, &s->cache_configured) != 0) {
        kvl_trunk_store_close(s);
        return -1;
    }
    if (s->cache_budget_bytes) {
        s->cache_base = (void **)calloc(s->hdr.n_records, sizeof *s->cache_base);
        if (!s->cache_base) { kvl_trunk_store_close(s); return -1; }
    }
    s->fd = open_data(bin_path, prefer_direct_io);
    if (s->fd >= 0) s->direct_io = prefer_direct_io ? 1 : 0;
    else if (prefer_direct_io) { s->fd = open_data(bin_path, 0); s->direct_io = 0; }
    if (s->fd < 0) { kvl_trunk_store_close(s); return -1; }
    return 0;
}

void kvl_trunk_cache_report(const KvlTrunkStore *s) {
    if (!s || !s->cache_configured) return;
    fprintf(stderr,
            "kvl_trunk_cache: resident=%.2f/%.2f MiB loads=%llu hits=%llu "
            "inserts=%llu reads=%.2f MiB\n",
            s->cache_bytes / 1048576.0, s->cache_budget_bytes / 1048576.0,
            (unsigned long long)s->load_calls,
            (unsigned long long)s->cache_hits,
            (unsigned long long)s->cache_inserts,
            s->bytes_read / 1048576.0);
}

void kvl_trunk_store_close(KvlTrunkStore *s) {
    if (!s) return;
    kvl_trunk_cache_report(s);
    if (s->fd >= 0) {
#ifdef _WIN32
        _close(s->fd);
#else
        close(s->fd);
#endif
    }
    if (s->cache_base) {
        for (uint32_t i = 0; i < s->hdr.n_records; ++i)
            aligned_free_buf(s->cache_base[i]);
    }
    free(s->cache_base);
    free(s->lookup_slots);
    free(s->records);
    memset(s, 0, sizeof *s); s->fd = -1;
}

const KvlTrunkRecord *kvl_trunk_find(const KvlTrunkStore *s, uint32_t layer, uint32_t kind) {
    if (!s) return NULL;
    if (s->lookup_slots && s->lookup_cap) {
        const size_t mask = s->lookup_cap - 1u;
        size_t at = (size_t)lookup_hash(layer, kind) & mask;
        for (size_t probe = 0; probe < s->lookup_cap; ++probe) {
            const int32_t idx = s->lookup_slots[at];
            if (idx < 0) return NULL;
            const KvlTrunkRecord *r = &s->records[idx];
            if (r->layer == layer && r->kind == kind) return r;
            at = (at + 1u) & mask;
        }
        return NULL;
    }

    /* Preserve lookup behavior for manually initialised stores used by small
     * probes, while all stores opened through kvl_trunk_store_open use the
     * indexed path above. */
    for (uint32_t i = 0; i < s->hdr.n_records; ++i) {
        const KvlTrunkRecord *r = &s->records[i];
        if (r->layer == layer && r->kind == kind) return r;
    }
    return NULL;
}

int kvl_trunk_load(KvlTrunkStore *s, uint32_t layer, uint32_t kind, KvlTrunkTensor *out) {
    if (!s || s->fd < 0 || !out) return -1;
    memset(out, 0, sizeof *out);
    const KvlTrunkRecord *r = kvl_trunk_find(s, layer, kind);
    if (!r || !r->read_bytes || (r->file_offset % KVL_TRUNK_ALIGN) ||
        (r->read_bytes % KVL_TRUNK_ALIGN)) return -1;
    s->load_calls++;

    const size_t idx = (size_t)(r - s->records);
    if (s->cache_base && idx < s->hdr.n_records && s->cache_base[idx]) {
        s->cache_hits++;
        out->record = r;
        out->base = s->cache_base[idx];
        out->data = s->cache_base[idx];
        out->owned = 0;
        return 0;
    }

    void *buf = NULL;
    if (aligned_alloc_buf(&buf, KVL_TRUNK_ALIGN, (size_t)r->read_bytes) != 0) return -1;
    const int64_t got = pread_all(s->fd, buf, (size_t)r->read_bytes, r->file_offset);
    if (got != (int64_t)r->read_bytes) { aligned_free_buf(buf); return -1; }
    s->bytes_read += r->read_bytes;

    const int cache_eligible = r->layer != KVL_TRUNK_GLOBAL_LAYER && s->cache_base &&
        r->read_bytes <= (uint64_t)(s->cache_budget_bytes - s->cache_bytes);
    if (cache_eligible) {
        s->cache_base[idx] = buf;
        s->cache_bytes += (size_t)r->read_bytes;
        s->cache_inserts++;
        out->owned = 0;
    } else {
        out->owned = 1;
    }
    out->record = r;
    out->base = buf;
    out->data = buf;
    return 0;
}

void kvl_trunk_tensor_free(KvlTrunkTensor *t) {
    if (!t) return;
    if (t->owned && t->base) aligned_free_buf(t->base);
    memset(t, 0, sizeof *t);
}
