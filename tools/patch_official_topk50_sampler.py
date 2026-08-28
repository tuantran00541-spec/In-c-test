#!/usr/bin/env python3
"""CI-only source transform: match the released generation config's top_k=50 sampling.

Kimi-VL generation_config enables sampling at temperature 0.2 and leaves top_k unspecified;
Transformers 4.50.x therefore uses GenerationConfig's default top_k=50. The current C runtime
samples the full vocabulary whenever temperature>0. This transform changes only that sampler
for causal A/B testing; it does not touch greedy decoding or the model forward path.
"""
from pathlib import Path

p = Path("src/generate.c")
s = p.read_text()
old = '''static int sample_token(const float *logits, float temperature, uint64_t *rng) {
    int argmax = 0;
    for (int i = 1; i < V; ++i) if (logits[i] > logits[argmax]) argmax = i;
    if (temperature <= 0.0f) return argmax;

    float maxv = logits[argmax];
    double sum = 0.0;
    for (int i = 0; i < V; ++i) sum += exp(((double)logits[i] - maxv) / temperature);
    double target = rng_unit(rng) * sum;
    double acc = 0.0;
    for (int i = 0; i < V; ++i) {
        acc += exp(((double)logits[i] - maxv) / temperature);
        if (acc >= target) return i;
    }
    return argmax;
}
'''
new = '''static int sample_token(const float *logits, float temperature, uint64_t *rng) {
    int argmax = 0;
    for (int i = 1; i < V; ++i) if (logits[i] > logits[argmax]) argmax = i;
    if (temperature <= 0.0f) return argmax;

    enum { SAMPLE_TOPK = 50 };
    int top_id[SAMPLE_TOPK];
    float top_logit[SAMPLE_TOPK];
    for (int j = 0; j < SAMPLE_TOPK; ++j) {
        top_id[j] = -1;
        top_logit[j] = -INFINITY;
    }
    for (int i = 0; i < V; ++i) {
        int pos = SAMPLE_TOPK;
        for (int j = 0; j < SAMPLE_TOPK; ++j) {
            if (logits[i] > top_logit[j] ||
                (logits[i] == top_logit[j] && (top_id[j] < 0 || i < top_id[j]))) {
                pos = j;
                break;
            }
        }
        if (pos == SAMPLE_TOPK) continue;
        for (int j = SAMPLE_TOPK - 1; j > pos; --j) {
            top_logit[j] = top_logit[j - 1];
            top_id[j] = top_id[j - 1];
        }
        top_logit[pos] = logits[i];
        top_id[pos] = i;
    }

    const float maxv = top_logit[0];
    double sum = 0.0;
    for (int j = 0; j < SAMPLE_TOPK; ++j)
        sum += exp(((double)top_logit[j] - maxv) / temperature);
    const double target = rng_unit(rng) * sum;
    double acc = 0.0;
    for (int j = 0; j < SAMPLE_TOPK; ++j) {
        acc += exp(((double)top_logit[j] - maxv) / temperature);
        if (acc >= target) return top_id[j];
    }
    return argmax;
}
'''
if old not in s:
    raise SystemExit("sample_token source pattern not found")
p.write_text(s.replace(old, new, 1))
print("patched C sampler to temperature + top_k=50")
