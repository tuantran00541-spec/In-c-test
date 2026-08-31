#define _FILE_OFFSET_BITS 64
#ifndef _WIN32
#define _GNU_SOURCE
#endif
#include "kvl/expert_store.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>

/* Windows positioned reads use an overlapped handle even though the public API is
 * synchronous. Each read owns an event and waits for completion, which makes concurrent
 * cache prefetches safe without sharing or mutating a file pointer. FILE_FLAG_NO_BUFFERING
 * is added independently when direct I/O is requested. */
static int kvl_open_data(const char *path, int direct) {
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED |
                  (direct ? FILE_FLAG_NO_BUFFERING : 0);
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    int fd = _open_osfhandle((intptr_t)h, _O_RDONLY | _O_BINARY);
    if (fd < 0) CloseHandle(h);
    return fd;
}

static int64_t kvl_pread(int fd, void *buf, size_t n, uint64_t off) {
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

static int kvl_aligned_alloc(void **out, size_t align, size_t n) {
    *out = _aligned_malloc(n, align);
    return *out ? 0 : -1;
}
void kvl_expert_free_buffer(void *p) { _aligned_free(p); }
#else
#include <unistd.h>
static int kvl_open_data(const char *path, int direct) {
#ifdef O_DIRECT
    if (direct) return open(path, O_RDONLY | O_DIRECT);
#else
    if (direct) { errno = EINVAL; return -1; }
#endif
    return open(path, O_RDONLY);
}
static int64_t kvl_pread(int fd, void *buf, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t got = pread(fd, (unsigned char*)buf + done, n - done, (off_t)(off + done));
        if (got < 0) { if (errno == EINTR) continue; return -1; }
        if (!got) break;
        done += (size_t)got;
    }
    return (int64_t)done;
}
static int kvl_aligned_alloc(void **out, size_t align, size_t n) {
    return posix_memalign(out, align, n);
}
void kvl_expert_free_buffer(void *p) { free(p); }
#endif

static char *mask_sidecar_path(const char *idx_path) {
    if (!idx_path) return NULL;
    const size_t n = strlen(idx_path);
    const int replace_idx = n >= 4 && strcmp(idx_path + n - 4, ".idx") == 0;
    const size_t base = replace_idx ? n - 4 : n;
    char *out = (char *)malloc(base + 6);
    if (!out) return NULL;
    memcpy(out, idx_path, base);
    memcpy(out + base, ".mask", 6);
    return out;
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static int read_mask_file(const char *path, const KvlExpertIndexHeader *hdr,
                          unsigned char *disabled, size_t map_n, size_t *out_count) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "kvl: cannot open expert mask %s\n", path);
        return -1;
    }
    memset(disabled, 0, map_n);
    size_t count = 0;
    char line[512];
    int lineno = 0;
    while (fgets(line, sizeof line, f)) {
        ++lineno;
        if (!strchr(line, '\n') && !feof(f)) {
            fprintf(stderr, "kvl: expert mask line too long %s:%d\n", path, lineno);
            fclose(f);
            return -1;
        }
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (!*p) continue;
        int layer = -1, expert = -1;
        char extra = '\0';
        const int parsed = sscanf(p, "%d %d %c", &layer, &expert, &extra);
        if (parsed != 2 || layer < 0 || expert < 0 ||
            layer >= (int)hdr->n_layers || expert >= (int)hdr->n_experts) {
            fprintf(stderr, "kvl: bad expert mask line %s:%d\n", path, lineno);
            fclose(f);
            return -1;
        }
        const size_t at = (size_t)layer * hdr->n_experts + (size_t)expert;
        if (disabled[at]) {
            fprintf(stderr, "kvl: duplicate expert mask entry %s:%d L%d/E%d\n",
                    path, lineno, layer, expert);
            fclose(f);
            return -1;
        }
        disabled[at] = 1;
        ++count;
    }
    if (ferror(f)) { fclose(f); return -1; }
    fclose(f);
    *out_count = count;
    return 0;
}

static int bind_sparse_q8_mask(KvlExpertStore *s, const char *idx_path) {
    if (!s || s->hdr.dtype != KVL_DTYPE_Q8_ROW) return 0;
    const size_t map_n = (size_t)s->hdr.n_layers * s->hdr.n_experts;
    const uint32_t first_moe_layer = s->hdr.n_layers > 1 ? 1u : 0u;
    size_t routed_slots = 0, routed_present = 0;
    for (uint32_t layer = first_moe_layer; layer < s->hdr.n_layers; ++layer) {
        for (uint32_t expert = 0; expert < s->hdr.n_experts; ++expert) {
            ++routed_slots;
            if (s->record_of[(size_t)layer * s->hdr.n_experts + expert] >= 0)
                ++routed_present;
        }
    }
    const int sparse = routed_present < routed_slots;
    char *sidecar = mask_sidecar_path(idx_path);
    if (!sidecar) return -1;
    const int sidecar_exists = file_exists(sidecar);

    if (!sparse) {
        if (sidecar_exists) {
            fprintf(stderr,
                    "kvl: refusing mask sidecar for a full Q8 expert store: %s\n",
                    sidecar);
            free(sidecar);
            return -1;
        }
        free(sidecar);
        return 0;
    }

    if (!sidecar_exists) {
        fprintf(stderr,
                "kvl: sparse Q8 expert store requires bound mask sidecar %s "
                "(%zu/%zu routed records present)\n",
                sidecar, routed_present, routed_slots);
        free(sidecar);
        return -1;
    }

    unsigned char *bound = (unsigned char *)calloc(map_n, 1);
    unsigned char *requested = (unsigned char *)calloc(map_n, 1);
    if (!bound || !requested) {
        free(bound); free(requested); free(sidecar);
        return -1;
    }
    size_t bound_count = 0;
    if (read_mask_file(sidecar, &s->hdr, bound, map_n, &bound_count) != 0) {
        free(bound); free(requested); free(sidecar);
        return -1;
    }

    const size_t missing = routed_slots - routed_present;
    int mismatch = bound_count != missing;
    for (uint32_t layer = first_moe_layer; !mismatch && layer < s->hdr.n_layers; ++layer) {
        for (uint32_t expert = 0; expert < s->hdr.n_experts; ++expert) {
            const size_t at = (size_t)layer * s->hdr.n_experts + expert;
            const int absent = s->record_of[at] < 0;
            if ((bound[at] != 0) != absent) {
                fprintf(stderr,
                        "kvl: sparse expert mask/store mismatch L%u/E%u mask=%d present=%d\n",
                        layer, expert, bound[at] ? 1 : 0, absent ? 0 : 1);
                mismatch = 1;
                break;
            }
        }
    }
    for (uint32_t layer = 0; !mismatch && layer < first_moe_layer; ++layer) {
        for (uint32_t expert = 0; expert < s->hdr.n_experts; ++expert) {
            if (bound[(size_t)layer * s->hdr.n_experts + expert]) {
                fprintf(stderr,
                        "kvl: sparse expert mask contains non-MoE entry L%u/E%u\n",
                        layer, expert);
                mismatch = 1;
                break;
            }
        }
    }
    if (mismatch) {
        free(bound); free(requested); free(sidecar);
        return -1;
    }

    const char *env_mask = getenv("KVL_MOE_MASK");
    if (env_mask && env_mask[0]) {
        size_t requested_count = 0;
        if (read_mask_file(env_mask, &s->hdr, requested, map_n, &requested_count) != 0 ||
            requested_count != bound_count || memcmp(requested, bound, map_n) != 0) {
            fprintf(stderr,
                    "kvl: KVL_MOE_MASK does not match sparse-store sidecar %s\n",
                    sidecar);
            free(bound); free(requested); free(sidecar);
            return -1;
        }
    } else {
#ifdef _WIN32
        if (_putenv_s("KVL_MOE_MASK", sidecar) != 0) {
#else
        if (setenv("KVL_MOE_MASK", sidecar, 1) != 0) {
#endif
            fprintf(stderr, "kvl: failed to bind sparse expert mask %s\n", sidecar);
            free(bound); free(requested); free(sidecar);
            return -1;
        }
    }

    fprintf(stderr,
            "kvl: sparse expert store bound mask=%s disabled=%zu routed_records=%zu/%zu\n",
            sidecar, bound_count, routed_present, routed_slots);
    free(bound); free(requested); free(sidecar);
    return 0;
}

static int validate_gguf_source(const KvlExpertIndexHeader *hdr,
                                const KvlExpertRecord *r,
                                const KvlGgufQ8Source *q) {
    if (!hdr || !r || !q) return -1;
    const uint64_t align = KVL_EXPERT_ALIGN;
    const uint64_t src_off[3] = {q->gate_file_offset, q->up_file_offset, q->down_file_offset};
    const uint64_t src_bytes[3] = {q->gate_read_bytes, q->up_read_bytes, q->down_read_bytes};
    const uint64_t dst_off[3] = {q->gate_dst_offset, q->up_dst_offset, q->down_dst_offset};
    const uint64_t payload_off[3] = {r->gate_off, r->up_off, r->down_off};
    const uint64_t payload_bytes[3] = {r->gate_bytes, r->up_bytes, r->down_bytes};
    uint64_t total = 0;
    for (int i = 0; i < 3; ++i) {
        if (!src_bytes[i] || (src_off[i] % align) || (src_bytes[i] % align) ||
            (dst_off[i] % align) || dst_off[i] + src_bytes[i] > r->read_bytes ||
            src_off[i] + src_bytes[i] > hdr->data_file_bytes ||
            payload_off[i] < dst_off[i] ||
            payload_off[i] + payload_bytes[i] > dst_off[i] + src_bytes[i])
            return -1;
        total += src_bytes[i];
    }
    return total == r->read_bytes ? 0 : -1;
}

static int read_index(KvlExpertStore *s, const char *idx_path) {
    FILE *f = fopen(idx_path, "rb");
    if (!f) return -1;
    if (fread(&s->hdr, 1, sizeof s->hdr, f) != sizeof s->hdr) { fclose(f); return -1; }
    if (memcmp(s->hdr.magic, KVL_EXPERT_MAGIC, 8) != 0 || s->hdr.version != KVL_EXPERT_VERSION) {
        fclose(f); return -1;
    }
    if (s->hdr.align != KVL_EXPERT_ALIGN || s->hdr.n_records == 0) { fclose(f); return -1; }
    if (fseek(f, (long)s->hdr.records_offset, SEEK_SET) != 0) { fclose(f); return -1; }
    s->records = (KvlExpertRecord*)calloc(s->hdr.n_records, sizeof *s->records);
    if (!s->records) { fclose(f); return -1; }
    if (fread(s->records, sizeof *s->records, s->hdr.n_records, f) != s->hdr.n_records) {
        fclose(f); return -1;
    }
    if (s->hdr.dtype == KVL_DTYPE_GGUF_Q8_0) {
        s->gguf_q8_sources = (KvlGgufQ8Source *)calloc(s->hdr.n_records, sizeof *s->gguf_q8_sources);
        if (!s->gguf_q8_sources ||
            fread(s->gguf_q8_sources, sizeof *s->gguf_q8_sources, s->hdr.n_records, f) != s->hdr.n_records) {
            fclose(f); return -1;
        }
        for (uint32_t i = 0; i < s->hdr.n_records; ++i) {
            if (validate_gguf_source(&s->hdr, &s->records[i], &s->gguf_q8_sources[i]) != 0) {
                fprintf(stderr, "kvl: invalid GGUF Q8_0 source record %u\n", i);
                fclose(f); return -1;
            }
        }
    }
    fclose(f);
    size_t map_n = (size_t)s->hdr.n_layers * s->hdr.n_experts;
    s->record_of = (int32_t*)malloc(map_n * sizeof *s->record_of);
    if (!s->record_of) return -1;
    for (size_t i = 0; i < map_n; ++i) s->record_of[i] = -1;
    for (uint32_t i = 0; i < s->hdr.n_records; ++i) {
        KvlExpertRecord *r = &s->records[i];
        if (r->layer >= s->hdr.n_layers || r->expert >= s->hdr.n_experts) return -1;
        const size_t at = (size_t)r->layer * s->hdr.n_experts + r->expert;
        if (s->record_of[at] >= 0) return -1;
        s->record_of[at] = (int32_t)i;
    }
    return 0;
}

int kvl_expert_store_open(KvlExpertStore *s, const char *bin_path, const char *idx_path,
                          int prefer_direct_io) {
    memset(s, 0, sizeof *s); s->fd = -1;
    if (read_index(s, idx_path) != 0) { kvl_expert_store_close(s); return -1; }
    if (bind_sparse_q8_mask(s, idx_path) != 0) { kvl_expert_store_close(s); return -1; }
    s->fd = kvl_open_data(bin_path, prefer_direct_io);
    if (s->fd >= 0) {
        s->direct_io = prefer_direct_io ? 1 : 0;
    } else if (prefer_direct_io) {
        s->fd = kvl_open_data(bin_path, 0);
        s->direct_io = 0;
    }
    if (s->fd < 0) { kvl_expert_store_close(s); return -1; }
    return 0;
}

void kvl_expert_store_close(KvlExpertStore *s) {
    if (!s) return;
    if (s->fd >= 0) {
#ifdef _WIN32
        _close(s->fd);
#else
        close(s->fd);
#endif
    }
    free(s->records); free(s->gguf_q8_sources); free(s->record_of);
    memset(s, 0, sizeof *s); s->fd = -1;
}

const KvlExpertRecord *kvl_expert_find(const KvlExpertStore *s, int layer, int expert) {
    if (!s || layer < 0 || expert < 0 || layer >= (int)s->hdr.n_layers || expert >= (int)s->hdr.n_experts)
        return NULL;
    int32_t idx = s->record_of[(size_t)layer * s->hdr.n_experts + expert];
    return idx < 0 ? NULL : &s->records[idx];
}

int kvl_expert_alloc_buffer(const KvlExpertRecord *r, void **out) {
    if (!r || !out || (r->read_bytes % KVL_EXPERT_ALIGN)) return -1;
    return kvl_aligned_alloc(out, KVL_EXPERT_ALIGN, (size_t)r->read_bytes);
}

int64_t kvl_expert_load(const KvlExpertStore *s, const KvlExpertRecord *r, void *aligned_buf) {
    if (!s || s->fd < 0 || !r || !aligned_buf) return -1;
    if (s->hdr.dtype != KVL_DTYPE_GGUF_Q8_0)
        return kvl_pread(s->fd, aligned_buf, (size_t)r->read_bytes, r->file_offset);

    if (!s->gguf_q8_sources || r < s->records || r >= s->records + s->hdr.n_records)
        return -1;
    const size_t idx = (size_t)(r - s->records);
    const KvlGgufQ8Source *q = &s->gguf_q8_sources[idx];
    unsigned char *base = (unsigned char *)aligned_buf;
    int64_t total = 0;

    const int64_t g = kvl_pread(s->fd, base + q->gate_dst_offset,
                                (size_t)q->gate_read_bytes, q->gate_file_offset);
    if (g != (int64_t)q->gate_read_bytes) return -1;
    total += g;
    const int64_t u = kvl_pread(s->fd, base + q->up_dst_offset,
                                (size_t)q->up_read_bytes, q->up_file_offset);
    if (u != (int64_t)q->up_read_bytes) return -1;
    total += u;
    const int64_t d = kvl_pread(s->fd, base + q->down_dst_offset,
                                (size_t)q->down_read_bytes, q->down_file_offset);
    if (d != (int64_t)q->down_read_bytes) return -1;
    total += d;
    return total;
}

uint32_t kvl_expert_load_read_ops(const KvlExpertStore *s, const KvlExpertRecord *r) {
    (void)r;
    return s && s->hdr.dtype == KVL_DTYPE_GGUF_Q8_0 ? 3u : 1u;
}