#define _FILE_OFFSET_BITS 64
#ifndef _WIN32
#define _GNU_SOURCE
#endif
#include "kvl/trunk_store.h"

#include <errno.h>
#include <fcntl.h>
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

int kvl_trunk_store_open(KvlTrunkStore *s, const char *bin_path, const char *idx_path,
                         int prefer_direct_io) {
    if (!s) return -1;
    memset(s, 0, sizeof *s); s->fd = -1;
    if (read_index(s, idx_path) != 0) { kvl_trunk_store_close(s); return -1; }
    s->fd = open_data(bin_path, prefer_direct_io);
    if (s->fd >= 0) s->direct_io = prefer_direct_io ? 1 : 0;
    else if (prefer_direct_io) { s->fd = open_data(bin_path, 0); s->direct_io = 0; }
    if (s->fd < 0) { kvl_trunk_store_close(s); return -1; }
    return 0;
}

void kvl_trunk_store_close(KvlTrunkStore *s) {
    if (!s) return;
    if (s->fd >= 0) {
#ifdef _WIN32
        _close(s->fd);
#else
        close(s->fd);
#endif
    }
    free(s->records);
    memset(s, 0, sizeof *s); s->fd = -1;
}

const KvlTrunkRecord *kvl_trunk_find(const KvlTrunkStore *s, uint32_t layer, uint32_t kind) {
    if (!s) return NULL;
    for (uint32_t i = 0; i < s->hdr.n_records; ++i) {
        const KvlTrunkRecord *r = &s->records[i];
        if (r->layer == layer && r->kind == kind) return r;
    }
    return NULL;
}

int kvl_trunk_load(const KvlTrunkStore *s, uint32_t layer, uint32_t kind, KvlTrunkTensor *out) {
    if (!s || s->fd < 0 || !out) return -1;
    memset(out, 0, sizeof *out);
    const KvlTrunkRecord *r = kvl_trunk_find(s, layer, kind);
    if (!r || !r->read_bytes || (r->file_offset % KVL_TRUNK_ALIGN) ||
        (r->read_bytes % KVL_TRUNK_ALIGN)) return -1;
    void *buf = NULL;
    if (aligned_alloc_buf(&buf, KVL_TRUNK_ALIGN, (size_t)r->read_bytes) != 0) return -1;
    int64_t got = pread_all(s->fd, buf, (size_t)r->read_bytes, r->file_offset);
    if (got != (int64_t)r->read_bytes) { aligned_free_buf(buf); return -1; }
    out->record = r; out->base = buf; out->data = buf;
    return 0;
}

void kvl_trunk_tensor_free(KvlTrunkTensor *t) {
    if (!t) return;
    if (t->base) aligned_free_buf(t->base);
    memset(t, 0, sizeof *t);
}
