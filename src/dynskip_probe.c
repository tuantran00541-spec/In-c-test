#include "kvl/moe_dynamic_skip.h"

#include <stdio.h>
#include <string.h>

static int fail(const char *msg) {
    fprintf(stderr, "KIMI_DYNSKIP_UNIT_FAIL %s\n", msg);
    return 1;
}

int main(int argc, char **argv) {
    const int prompt[] = {
        163594, 101, 163601, 102, 163586,
        163587, 103, 163601,
        163602, 104, 163603, 163605, 163604,
        201, 202, 163586,
        163588, 105, 163601,
    };
    const unsigned char expected[] = {
        KVL_MOE_FAMILY_CONTROL, KVL_MOE_FAMILY_CONTROL,
        KVL_MOE_FAMILY_CONTROL, KVL_MOE_FAMILY_CONTROL,
        KVL_MOE_FAMILY_CONTROL,
        KVL_MOE_FAMILY_CONTROL, KVL_MOE_FAMILY_CONTROL,
        KVL_MOE_FAMILY_CONTROL,
        KVL_MOE_FAMILY_CONTROL, KVL_MOE_FAMILY_CONTROL,
        KVL_MOE_FAMILY_CONTROL, KVL_MOE_FAMILY_MEDIA,
        KVL_MOE_FAMILY_CONTROL,
        KVL_MOE_FAMILY_CONTENT, KVL_MOE_FAMILY_CONTENT,
        KVL_MOE_FAMILY_CONTROL,
        KVL_MOE_FAMILY_CONTROL, KVL_MOE_FAMILY_CONTROL,
        KVL_MOE_FAMILY_CONTROL,
    };
    unsigned char got[sizeof prompt / sizeof prompt[0]];
    if (kvl_moe_dynskip_classify_prompt(prompt,
            (int)(sizeof prompt / sizeof prompt[0]), got) != 0)
        return fail("classify returned error");
    if (memcmp(got, expected, sizeof expected) != 0)
        return fail("chat-template classification mismatch");

    kvl_moe_dynskip_reset_policy();
    if (kvl_moe_dynskip_set_policy(KVL_MOE_FAMILY_MEDIA, 20, 0.14f, 5) != 0)
        return fail("cannot set media policy");
    if (kvl_moe_dynskip_set_policy(KVL_MOE_FAMILY_CONTROL, 20, 0.5f, 1) == 0)
        return fail("control policy must be rejected");

    const float weights[] = {0.50f, 0.40f, 0.30f, 0.20f, 0.10f, 0.05f};
    unsigned char keep[6];
    int skipped = -1;
    if (kvl_moe_dynskip_apply_policy(KVL_MOE_FAMILY_MEDIA, 20, 6,
                                     weights, keep, &skipped) != 0)
        return fail("media decision returned error");
    if (skipped != 1) return fail("min_keep=5 must skip exactly one route");
    if (!keep[0] || !keep[1] || !keep[2] || !keep[3] || !keep[4] || keep[5])
        return fail("wrong route restored by min_keep");

    if (kvl_moe_dynskip_apply_policy(KVL_MOE_FAMILY_CONTROL, 20, 6,
                                     weights, keep, &skipped) != 0)
        return fail("control decision returned error");
    if (skipped != 0) return fail("control token was not protected");
    for (int i = 0; i < 6; ++i) if (!keep[i]) return fail("control keep mask changed");

    if (kvl_moe_dynskip_set_policy(KVL_MOE_FAMILY_CONTENT, 9, 0.90f, 1) != 0)
        return fail("cannot set content policy");
    if (kvl_moe_dynskip_apply_policy(KVL_MOE_FAMILY_CONTENT, 9, 6,
                                     weights, keep, &skipped) != 0)
        return fail("content decision returned error");
    if (skipped != 5) return fail("min_keep=1 must retain one route");
    if (!keep[0]) return fail("highest normalized route was not retained");

    if (argc == 2) {
        if (kvl_moe_dynskip_load_policy(argv[1]) != 0)
            return fail("policy file parser rejected valid fixture");
        if (kvl_moe_dynskip_apply_policy(KVL_MOE_FAMILY_MEDIA, 26, 6,
                                         weights, keep, &skipped) != 0)
            return fail("parsed media policy decision returned error");
        if (skipped != 1)
            return fail("parsed policy did not preserve min_keep=5");
    } else if (argc != 1) {
        return fail("usage: kvl_dynskip_probe [policy-file]");
    }

    puts("KIMI_DYNSKIP_UNIT_PASS");
    return 0;
}
