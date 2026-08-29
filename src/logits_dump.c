#include "kvl/ops.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static FILE *g_logits_dump;
static int g_logits_dump_state;
static uint32_t g_logits_dump_record;
static long g_logits_dump_limit = 1;

static void logits_dump_close(void) {
    if (g_logits_dump) fclose(g_logits_dump);
    g_logits_dump = NULL;
}

static int logits_dump_init(int vocab) {
    if (g_logits_dump_state != 0) return g_logits_dump_state > 0;
    g_logits_dump_state = -1;
    const char *path = getenv("KVL_LOGITS_DUMP");
    if (!path || !path[0]) return 0;

    const char *limit_s = getenv("KVL_LOGITS_DUMP_LIMIT");
    if (limit_s && limit_s[0]) {
        char *end = NULL;
        errno = 0;
        long limit = strtol(limit_s, &end, 10);
        if (errno || !end || *end || limit < 0) {
            fprintf(stderr, "kvl: invalid KVL_LOGITS_DUMP_LIMIT=%s\n", limit_s);
            return 0;
        }
        g_logits_dump_limit = limit; /* 0 means unlimited. */
    }

    g_logits_dump = fopen(path, "wb");
    if (!g_logits_dump) {
        fprintf(stderr, "kvl: cannot open KVL_LOGITS_DUMP=%s\n", path);
        return 0;
    }
    static const unsigned char magic[8] = {'K','V','L','L','O','G','1',0};
    const uint32_t version = 1;
    const uint32_t n = (uint32_t)vocab;
    if (fwrite(magic, 1, sizeof magic, g_logits_dump) != sizeof magic ||
        fwrite(&version, sizeof version, 1, g_logits_dump) != 1 ||
        fwrite(&n, sizeof n, 1, g_logits_dump) != 1) {
        fprintf(stderr, "kvl: failed to write logits dump header\n");
        logits_dump_close();
        return 0;
    }
    if (atexit(logits_dump_close) != 0) {
        fprintf(stderr, "kvl: failed to register logits dump cleanup\n");
        logits_dump_close();
        return 0;
    }
    g_logits_dump_state = 1;
    fprintf(stderr, "kvl: logits dump enabled path=%s limit=%ld\n", path, g_logits_dump_limit);
    return 1;
}

static void logits_dump_record(const float *logits, int vocab) {
    if (!logits || vocab <= 0 || !logits_dump_init(vocab)) return;
    if (g_logits_dump_limit > 0 && (long)g_logits_dump_record >= g_logits_dump_limit) return;
    const uint32_t index = g_logits_dump_record;
    const uint32_t n = (uint32_t)vocab;
    if (fwrite(&index, sizeof index, 1, g_logits_dump) != 1 ||
        fwrite(&n, sizeof n, 1, g_logits_dump) != 1 ||
        fwrite(logits, sizeof(float), (size_t)vocab, g_logits_dump) != (size_t)vocab) {
        fprintf(stderr, "kvl: failed to write logits dump record=%u\n", index);
        logits_dump_close();
        g_logits_dump_state = -1;
        return;
    }
    fflush(g_logits_dump);
    ++g_logits_dump_record;
}

void kvl_matvec_bf16_dump(float *y, const float *x, const uint16_t *w,
                          int in, int out) {
    kvl_matvec_bf16(y, x, w, in, out);
    logits_dump_record(y, out);
}
