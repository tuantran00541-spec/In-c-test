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
    fclose(f);
    size_t map_n = (size_t)s->hdr.n_layers * s->hdr.n_experts;
    s->record_of = (int32_t*)malloc(map_n * sizeof *s->record_of);
    if (!s->record_of) return -1;
    for (size_t i = 0; i < map_n; ++i) s->record_of[i] = -1;
    for (uint32_t i = 0; i < s->hdr.n_records; ++i) {
        KvlExpertRecord *r = &s->records[i];
        if (r->layer >= s->hdr.n_layers || r->expert >= s->hdr.n_experts) return -1;
        s->record_of[(size_t)r->layer * s->hdr.n_experts + r->expert] = (int32_t)i;
    }
    return 0;
}

int kvl_expert_store_open(KvlExpertStore *s, const char *bin_path, const char *idx_path,
                          int prefer_direct_io) {
    memset(s, 0, sizeof *s); s->fd = -1;
    if (read_index(s, idx_path) != 0) { kvl_expert_store_close(s); return -1; }
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
    free(s->records); free(s->record_of);
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
    return kvl_pread(s->fd, aligned_buf, (size_t)r->read_bytes, r->file_offset);
}
