#include "kvl/mla_compressed_state.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static void rope_interleaved_v6c(float *dst, const float *raw, int dim,
                                 int position, float theta) {
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const double exponent = (double)(2 * i) / (double)dim;
        const double inv_freq = pow((double)theta, -exponent);
        const double angle = (double)position * inv_freq;
        const float c = (float)cos(angle);
        const float s = (float)sin(angle);
        const float a = raw[2 * i];
        const float b = raw[2 * i + 1];
        dst[i] = a * c - b * s;
        dst[half + i] = b * c + a * s;
    }
}

int kvl_mla_compressed_state_init(KvlMlaCompressedState *state,
                                  const KvlMlaConfig *cfg,
                                  int capacity) {
    if (!state || !cfg || capacity <= 0 || cfg->kv_lora_rank <= 0 ||
        cfg->qk_rope_dim <= 0 || (cfg->qk_rope_dim & 1))
        return -1;
    memset(state,0,sizeof(*state));
    state->latent=(float*)calloc((size_t)capacity*cfg->kv_lora_rank,sizeof(float));
    state->rope=(float*)calloc((size_t)capacity*cfg->qk_rope_dim,sizeof(float));
    if(!state->latent||!state->rope){kvl_mla_compressed_state_free(state);return -1;}
    state->capacity=capacity;
    state->kv_lora_rank=cfg->kv_lora_rank;
    state->qk_rope_dim=cfg->qk_rope_dim;
    return 0;
}

void kvl_mla_compressed_state_reset(KvlMlaCompressedState *state){if(state)state->len=0;}
int kvl_mla_compressed_state_truncate(KvlMlaCompressedState *state,int new_len){
    if(!state||new_len<0||new_len>state->len)return -1;
    state->len=new_len;
    return 0;
}
void kvl_mla_compressed_state_free(KvlMlaCompressedState *state){if(!state)return;free(state->latent);free(state->rope);memset(state,0,sizeof(*state));}
size_t kvl_mla_compressed_state_bytes(const KvlMlaCompressedState *state){
    if(!state||state->capacity<=0)return 0;
    return sizeof(*state)+(size_t)state->capacity*((size_t)state->kv_lora_rank+(size_t)state->qk_rope_dim)*sizeof(float);
}

int kvl_mla_compressed_state_prefill_bf16(const float *x,
                                          int seq_len,
                                          const KvlMlaConfig *cfg,
                                          const KvlMlaBF16 *w,
                                          KvlMlaCompressedState *state) {
    if (!x || !cfg || !w || !state || !w->kv_a_proj || !w->kv_a_norm ||
        seq_len <= 0 || state->len != 0 || seq_len > state->capacity ||
        state->kv_lora_rank != cfg->kv_lora_rank ||
        state->qk_rope_dim != cfg->qk_rope_dim)
        return -1;

    const int H = cfg->hidden_size;
    const int R = cfg->kv_lora_rank;
    const int DR = cfg->qk_rope_dim;
    const int KVO = R + DR;
    if (H <= 0 || R <= 0 || DR <= 0 || (DR & 1) || cfg->rope_theta <= 0.0f)
        return -1;

    float *katmp = (float *)malloc((size_t)KVO * sizeof(float));
    if (!katmp) return -1;

    for (int t = 0; t < seq_len; ++t) {
        const float *xt = x + (size_t)t * H;
        float *latent = state->latent + (size_t)t * R;
        float *rope = state->rope + (size_t)t * DR;
        kvl_matvec_bf16(katmp, xt, w->kv_a_proj, H, KVO);
        kvl_rmsnorm_bf16(latent, katmp, w->kv_a_norm, R, cfg->rms_eps);
        rope_interleaved_v6c(rope, katmp + R, DR, t, cfg->rope_theta);
    }
    state->len = seq_len;
    free(katmp);
    return 0;
}

int kvl_mla_decode_compressed_bf16(float *out,
                                   const float *x,
                                   int position,
                                   const KvlMlaConfig *cfg,
                                   const KvlMlaBF16 *w,
                                   KvlMlaCompressedState *state){
    if(!out||!x||!cfg||!w||!state||!w->q_proj||!w->kv_a_proj||!w->kv_a_norm||
       !w->kv_b_proj||!w->o_proj||position!=state->len||position<0||position>=state->capacity||
       state->kv_lora_rank!=cfg->kv_lora_rank||state->qk_rope_dim!=cfg->qk_rope_dim)
        return -1;
    const int H=cfg->hidden_size,NH=cfg->num_heads,DN=cfg->qk_nope_dim,DR=cfg->qk_rope_dim;
    const int DV=cfg->v_head_dim,R=cfg->kv_lora_rank,QD=DN+DR,QO=NH*QD,KVO=R+DR;
    if(H<=0||NH<=0||DN<=0||DR<=0||(DR&1)||DV<=0||R<=0||cfg->rope_theta<=0.0f)return -1;

    float *qtmp=(float*)malloc((size_t)QO*sizeof(float));
    float *katmp=(float*)malloc((size_t)KVO*sizeof(float));
    float *qrope=(float*)malloc((size_t)DR*sizeof(float));
    float *q_lat=(float*)malloc((size_t)R*sizeof(float));
    float *mix=(float*)malloc((size_t)R*sizeof(float));
    float *scores=(float*)malloc((size_t)(position+1)*sizeof(float));
    float *head_out=(float*)malloc((size_t)NH*DV*sizeof(float));
    if(!qtmp||!katmp||!qrope||!q_lat||!mix||!scores||!head_out){
        free(qtmp);free(katmp);free(qrope);free(q_lat);free(mix);free(scores);free(head_out);return -1;
    }

    kvl_matvec_bf16(qtmp,x,w->q_proj,H,QO);
    kvl_matvec_bf16(katmp,x,w->kv_a_proj,H,KVO);
    float *cur_lat=state->latent+(size_t)position*R;
    float *cur_rope=state->rope+(size_t)position*DR;
    kvl_rmsnorm_bf16(cur_lat,katmp,w->kv_a_norm,R,cfg->rms_eps);
    rope_interleaved_v6c(cur_rope,katmp+R,DR,position,cfg->rope_theta);

    const float scale=1.0f/sqrtf((float)QD);
    for(int h=0;h<NH;++h){
        const float *qh=qtmp+(size_t)h*QD;
        rope_interleaved_v6c(qrope,qh+DN,DR,position,cfg->rope_theta);

        /* Absorb W_k into the current query: q_lat = W_k^T q_nope. */
        for(int r=0;r<R;++r){
            double acc=0.0;
            for(int d=0;d<DN;++d){
                const int row=h*(DN+DV)+d;
                acc+=(double)kvl_bf16_to_f32(w->kv_b_proj[(size_t)row*R+r])*(double)qh[d];
            }
            q_lat[r]=(float)acc;
        }

        float max_score=-INFINITY;
        for(int j=0;j<=position;++j){
            const float *lj=state->latent+(size_t)j*R;
            const float *rj=state->rope+(size_t)j*DR;
            double dot=0.0;
            for(int r=0;r<R;++r)dot+=(double)q_lat[r]*(double)lj[r];
            for(int d=0;d<DR;++d)dot+=(double)qrope[d]*(double)rj[d];
            scores[j]=(float)dot*scale;
            if(scores[j]>max_score)max_score=scores[j];
        }
        double denom=0.0;
        for(int j=0;j<=position;++j){scores[j]=expf(scores[j]-max_score);denom+=scores[j];}
        for(int r=0;r<R;++r){
            double acc=0.0;
            for(int j=0;j<=position;++j)acc+=((double)scores[j]/denom)*(double)state->latent[(size_t)j*R+r];
            mix[r]=(float)acc;
        }
        float *ho=head_out+(size_t)h*DV;
        for(int d=0;d<DV;++d){
            const int row=h*(DN+DV)+DN+d;
            double acc=0.0;
            for(int r=0;r<R;++r)acc+=(double)kvl_bf16_to_f32(w->kv_b_proj[(size_t)row*R+r])*(double)mix[r];
            ho[d]=(float)acc;
        }
    }
    kvl_matvec_bf16(out,head_out,w->o_proj,NH*DV,H);
    state->len=position+1;
    free(qtmp);free(katmp);free(qrope);free(q_lat);free(mix);free(scores);free(head_out);
    return 0;
}

int kvl_mla_decode_compressed_block_bf16(float *out,
                                         const float *x,
                                         int count,
                                         int start_position,
                                         const KvlMlaConfig *cfg,
                                         const KvlMlaBF16 *w,
                                         KvlMlaCompressedState *state) {
    if (!out || !x || !cfg || !w || !state || count <= 0 ||
        start_position != state->len || start_position < 0 ||
        count > state->capacity - start_position)
        return -1;
    const int H = cfg->hidden_size;
    if (H <= 0) return -1;
    for (int t = 0; t < count; ++t) {
        if (kvl_mla_decode_compressed_bf16(out + (size_t)t * H,
                                           x + (size_t)t * H,
                                           start_position + t,
                                           cfg, w, state) != 0)
            return -1;
    }
    return 0;
}
