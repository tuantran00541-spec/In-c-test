add_library(kvl_shared_q8_runtime STATIC
  "${CMAKE_CURRENT_LIST_DIR}/shared_q8_runtime.c"
)
target_include_directories(kvl_shared_q8_runtime PRIVATE
  "${KVL_ROOT}/include" "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_shared_q8_runtime PUBLIC kvl_storage)

file(READ "${KVL_ROOT}/src/generate.c" KVL_SHARED_Q8_GENERATE)
string(REPLACE
  "#include \"kvl/trunk_store.h\"\n"
  "#include \"kvl/trunk_store.h\"\n#include \"shared_q8_runtime.h\"\n"
  KVL_SHARED_Q8_GENERATE "${KVL_SHARED_Q8_GENERATE}")

set(KVL_SHARED_Q8_DECODE_OLD [=[    KvlTrunkTensor rt = {0}, rb = {0}, sg = {0}, su = {0}, sd = {0};
    int rc = -1;
    if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_WEIGHT, &rt) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_BIAS, &rb) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_GATE, &sg) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_UP, &su) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_DOWN, &sd)) goto done_moe;

    expand(router, (const uint16_t *)rt.data, (size_t)E * H);
    expand(bias, (const uint16_t *)rb.data, E);
    KvlRouterConfig r = {H, E, TOPK, 1, 1, 1, 2.446f};
    KvlMlpBF16 shared = {
        (const uint16_t *)sg.data, (const uint16_t *)su.data,
        (const uint16_t *)sd.data, SHARED_I
    };
    rc = kvl_moe_token_bf16(cache, layer, &r, n, router, bias, EXP_I,
                            &shared, y, ids, weights, scratch);

done_moe:
    kvl_trunk_tensor_free(&rt);
    kvl_trunk_tensor_free(&rb);
    kvl_trunk_tensor_free(&sg);
    kvl_trunk_tensor_free(&su);
    kvl_trunk_tensor_free(&sd);
    return rc;]=])
set(KVL_SHARED_Q8_DECODE_NEW [=[    KvlTrunkTensor rt = {0}, rb = {0};
    int rc = -1;
    if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_WEIGHT, &rt) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_BIAS, &rb)) goto done_moe;

    expand(router, (const uint16_t *)rt.data, (size_t)E * H);
    expand(bias, (const uint16_t *)rb.data, E);
    KvlRouterConfig r = {H, E, TOPK, 1, 1, 1, 2.446f};
    rc = kvl_moe_token_q8_shared_sidecar_auto(
        cache, layer, &r, n, router, bias, EXP_I, NULL,
        y, ids, weights, scratch);

done_moe:
    kvl_trunk_tensor_free(&rt);
    kvl_trunk_tensor_free(&rb);
    return rc;]=])
string(REPLACE "${KVL_SHARED_Q8_DECODE_OLD}" "${KVL_SHARED_Q8_DECODE_NEW}"
       KVL_SHARED_Q8_GENERATE "${KVL_SHARED_Q8_GENERATE}")

set(KVL_SHARED_Q8_PREFILL_OLD [=[        } else {
            KvlTrunkTensor rt = {0}, rb = {0}, sg = {0}, su = {0}, sd = {0};
            if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_WEIGHT, &rt) ||
                load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_BIAS, &rb) ||
                load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_GATE, &sg) ||
                load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_UP, &su) ||
                load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_DOWN, &sd)) {
                kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
                kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
                kvl_trunk_tensor_free(&sd);
                goto done;
            }
            expand(router, (const uint16_t *)rt.data, (size_t)E * H);
            expand(bias, (const uint16_t *)rb.data, E);
            KvlMlpBF16 shared = {
                (const uint16_t *)sg.data, (const uint16_t *)su.data,
                (const uint16_t *)sd.data, SHARED_I
            };
            for (int t = 0; t < seq_len; ++t) {
                const size_t base = (size_t)t * H;
                if (kvl_moe_token_bf16(cache, layer, &router_cfg, work_a + base,
                                       router, bias, EXP_I, &shared, x + base,
                                       ids, weights, scratch) != 0) {
                    kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
                    kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
                    kvl_trunk_tensor_free(&sd);
                    goto done;
                }
            }
            kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
            kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
            kvl_trunk_tensor_free(&sd);
        }]=])
set(KVL_SHARED_Q8_PREFILL_NEW [=[        } else {
            KvlTrunkTensor rt = {0}, rb = {0};
            if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_WEIGHT, &rt) ||
                load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_BIAS, &rb)) {
                kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
                goto done;
            }
            expand(router, (const uint16_t *)rt.data, (size_t)E * H);
            expand(bias, (const uint16_t *)rb.data, E);
            for (int t = 0; t < seq_len; ++t) {
                const size_t base = (size_t)t * H;
                if (kvl_moe_token_q8_shared_sidecar_auto(
                        cache, layer, &router_cfg, work_a + base,
                        router, bias, EXP_I, NULL, x + base,
                        ids, weights, scratch) != 0) {
                    kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
                    goto done;
                }
            }
            kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
        }]=])
string(REPLACE "${KVL_SHARED_Q8_PREFILL_OLD}" "${KVL_SHARED_Q8_PREFILL_NEW}"
       KVL_SHARED_Q8_GENERATE "${KVL_SHARED_Q8_GENERATE}")

string(FIND "${KVL_SHARED_Q8_GENERATE}" "KVL_TENSOR_SHARED_GATE" KVL_SHARED_Q8_LEFTOVER)
if(NOT KVL_SHARED_Q8_LEFTOVER LESS 0)
  message(FATAL_ERROR "shared-Q8 generator still references BF16 shared trunk tensors")
endif()
string(FIND "${KVL_SHARED_Q8_GENERATE}" "kvl_moe_token_q8_shared_sidecar_auto" KVL_SHARED_Q8_CALL)
if(KVL_SHARED_Q8_CALL LESS 0)
  message(FATAL_ERROR "shared-Q8 generator replacement failed")
endif()

set(KVL_SHARED_Q8_SOURCE "${CMAKE_BINARY_DIR}/generate_shared_q8.c")
file(WRITE "${KVL_SHARED_Q8_SOURCE}" "${KVL_SHARED_Q8_GENERATE}")
set_source_files_properties("${KVL_SHARED_Q8_SOURCE}" PROPERTIES
  GENERATED TRUE
  COMPILE_DEFINITIONS "kvl_matvec_bf16=kvl_matvec_bf16_dump"
)
add_executable(kvl_generate_shared_q8
  "${KVL_SHARED_Q8_SOURCE}"
  "${KVL_ROOT}/src/logits_dump.c"
)
target_include_directories(kvl_generate_shared_q8 PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(kvl_generate_shared_q8 PRIVATE kvl_storage kvl_shared_q8_runtime)
