#!/usr/bin/env python3
"""CI-only transform that swaps V9 compressed MLA state for expanded K/V state.

This is a correctness diagnostic, not a production memory-format decision. It keeps the
validated batch prefill outputs, materializes per-layer prompt K/V once, then uses the older
V6a expanded decode kernel for generated tokens. For the 136-token user prompt the extra
state is modest and cleanly isolates the compressed-latent decode algebra.
"""
from pathlib import Path

# 1) Add a batch state-population API for the existing expanded K/V state.
h = Path('include/kvl/mla_state.h')
s = h.read_text()
anchor = '''size_t kvl_mla_state_bytes(const KvlMlaState *state);\n'''
insert = anchor + '''\n/* Populate expanded K/V for an already-normalized causal prompt without recomputing\n * attention output. State must be empty and large enough for seq_len. */\nint kvl_mla_state_prefill_bf16(const float *x, int seq_len,\n                               const KvlMlaConfig *cfg, const KvlMlaBF16 *w,\n                               KvlMlaState *state);\n'''
if 'kvl_mla_state_prefill_bf16' not in s:
    assert anchor in s
    h.write_text(s.replace(anchor, insert, 1))

c = Path('src/mla_state.c')
s = c.read_text()
anchor = '''int kvl_mla_decode_bf16(float *out,\n'''
impl = r'''int kvl_mla_state_prefill_bf16(const float *x,
                               int seq_len,
                               const KvlMlaConfig *cfg,
                               const KvlMlaBF16 *w,
                               KvlMlaState *state) {
    if (!x || !cfg || !w || !state || !w->kv_a_proj || !w->kv_a_norm ||
        !w->kv_b_proj || seq_len <= 0 || state->len != 0 ||
        seq_len > state->capacity || state->num_heads != cfg->num_heads ||
        state->qk_nope_dim != cfg->qk_nope_dim ||
        state->qk_rope_dim != cfg->qk_rope_dim ||
        state->v_head_dim != cfg->v_head_dim)
        return -1;

    const int H=cfg->hidden_size, NH=cfg->num_heads, DN=cfg->qk_nope_dim;
    const int DR=cfg->qk_rope_dim, DV=cfg->v_head_dim, R=cfg->kv_lora_rank;
    const int QD=DN+DR, KVO=R+DR, KVB=NH*(DN+DV);
    float *katmp=(float*)malloc((size_t)KVO*sizeof(float));
    float *latent=(float*)malloc((size_t)R*sizeof(float));
    float *kvtmp=(float*)malloc((size_t)KVB*sizeof(float));
    float *rope=(float*)malloc((size_t)DR*sizeof(float));
    if(!katmp||!latent||!kvtmp||!rope){free(katmp);free(latent);free(kvtmp);free(rope);return -1;}

    for(int t=0;t<seq_len;++t){
        const float *xt=x+(size_t)t*H;
        kvl_matvec_bf16(katmp,xt,w->kv_a_proj,H,KVO);
        kvl_rmsnorm_bf16(latent,katmp,w->kv_a_norm,R,cfg->rms_eps);
        kvl_matvec_bf16(kvtmp,latent,w->kv_b_proj,R,KVB);
        rope_interleaved_v6(rope,katmp+R,DR,t,cfg->rope_theta);
        for(int head=0;head<NH;++head){
            const float *kvh=kvtmp+(size_t)head*(DN+DV);
            float *kd=state->keys+((size_t)t*NH+head)*QD;
            float *vd=state->values+((size_t)t*NH+head)*DV;
            memcpy(kd,kvh,(size_t)DN*sizeof(float));
            memcpy(kd+DN,rope,(size_t)DR*sizeof(float));
            memcpy(vd,kvh+DN,(size_t)DV*sizeof(float));
        }
    }
    state->len=seq_len;
    free(katmp);free(latent);free(kvtmp);free(rope);
    return 0;
}

'''
if 'int kvl_mla_state_prefill_bf16(' not in s:
    assert anchor in s
    c.write_text(s.replace(anchor, impl + anchor, 1))

# 2) Swap generator state type/API. generate_vl.c includes generate.c, so this covers both.
g = Path('src/generate.c')
s = g.read_text()
if '#include "kvl/mla_state.h"' not in s:
    s = s.replace('#include "kvl/mla_compressed_state.h"\n',
                  '#include "kvl/mla_compressed_state.h"\n#include "kvl/mla_state.h"\n', 1)
repls = {
    'KvlMlaCompressedState *state': 'KvlMlaState *state',
    'KvlMlaCompressedState *states': 'KvlMlaState *states',
    'KvlMlaCompressedState states[LN]': 'KvlMlaState states[LN]',
    'kvl_mla_decode_compressed_bf16': 'kvl_mla_decode_bf16',
    'kvl_mla_compressed_state_prefill_bf16': 'kvl_mla_state_prefill_bf16',
    'kvl_mla_compressed_state_init': 'kvl_mla_state_init',
    'kvl_mla_compressed_state_bytes': 'kvl_mla_state_bytes',
    'kvl_mla_compressed_state_free': 'kvl_mla_state_free',
}
for old,new in repls.items():
    s=s.replace(old,new)
g.write_text(s)

v = Path('src/generate_vl.c')
s = v.read_text()
repls_v = {
    'KvlMlaCompressedState *states': 'KvlMlaState *states',
    'KvlMlaCompressedState states[LN]': 'KvlMlaState states[LN]',
    'kvl_mla_compressed_state_prefill_bf16': 'kvl_mla_state_prefill_bf16',
    'kvl_mla_compressed_state_init': 'kvl_mla_state_init',
    'kvl_mla_compressed_state_bytes': 'kvl_mla_state_bytes',
    'kvl_mla_compressed_state_free': 'kvl_mla_state_free',
}
for old,new in repls_v.items():
    s=s.replace(old,new)
v.write_text(s)
print('patched V9 to expanded K/V state for decode diagnostic')
