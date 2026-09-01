cmake_minimum_required(VERSION 3.16)
project(kimi_perf_research C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

get_filename_component(KVL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# This lab intentionally compares candidate schedulers with all other packed-Q8
# Windows fast paths enabled identically.
set(KVL_USE_AVX2 ON CACHE BOOL "" FORCE)
set(KVL_MSVC_PACKED_Q8_PARALLEL_KERNELS ON CACHE BOOL "" FORCE)
add_subdirectory("${KVL_ROOT}" "${CMAKE_BINARY_DIR}/core")

add_library(kvl_q8_expert_parallel STATIC
  "${CMAKE_CURRENT_LIST_DIR}/q8_expert_parallel.c"
)
target_include_directories(kvl_q8_expert_parallel PRIVATE "${KVL_ROOT}/include")
target_link_libraries(kvl_q8_expert_parallel PUBLIC kvl_storage)
target_compile_definitions(kvl_q8_expert_parallel PRIVATE KVL_USE_AVX2=1)
if(MSVC)
  target_compile_options(kvl_q8_expert_parallel PRIVATE /arch:AVX2 /openmp)
else()
  target_compile_options(kvl_q8_expert_parallel PRIVATE -mavx2)
  find_package(OpenMP)
  if(OpenMP_C_FOUND)
    target_link_libraries(kvl_q8_expert_parallel PUBLIC OpenMP::OpenMP_C)
  endif()
endif()

# Same generator source and packed runtime as production. Only the MoE dispatch
# symbol is changed, so routing/input/output/token loop remain directly comparable.
add_executable(kvl_generate_expert_parallel
  "${KVL_ROOT}/src/generate.c"
  "${KVL_ROOT}/src/logits_dump.c"
)
target_link_libraries(kvl_generate_expert_parallel PRIVATE
  kvl_storage kvl_q8_expert_parallel
)
target_compile_definitions(kvl_generate_expert_parallel PRIVATE
  kvl_moe_token_bf16=kvl_moe_token_q8_expert_parallel_auto
)

add_executable(kvl_q8_expert_parallel_probe
  "${CMAKE_CURRENT_LIST_DIR}/q8_expert_parallel_probe.c"
)
target_include_directories(kvl_q8_expert_parallel_probe PRIVATE "${KVL_ROOT}/include")
target_link_libraries(kvl_q8_expert_parallel_probe PRIVATE
  kvl_storage kvl_q8_expert_parallel
)

# Fused MLA prefill: same established attention algorithm, but latent/RoPE values
# from the first kv_a projection pass are written directly into compressed state.
add_library(kvl_mla_prefill_fused STATIC
  "${CMAKE_CURRENT_LIST_DIR}/mla_prefill_fused.c"
)
target_include_directories(kvl_mla_prefill_fused PRIVATE
  "${KVL_ROOT}/include" "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_mla_prefill_fused PUBLIC kvl_storage)
target_compile_definitions(kvl_mla_prefill_fused PRIVATE KVL_USE_AVX2=1)
if(MSVC)
  target_compile_options(kvl_mla_prefill_fused PRIVATE /arch:AVX2 /openmp)
else()
  target_compile_options(kvl_mla_prefill_fused PRIVATE -mavx2)
  if(OpenMP_C_FOUND)
    target_link_libraries(kvl_mla_prefill_fused PUBLIC OpenMP::OpenMP_C)
  endif()
endif()

add_executable(kvl_mla_fused_prefill_probe
  "${CMAKE_CURRENT_LIST_DIR}/mla_prefill_fused_probe.c"
)
target_include_directories(kvl_mla_fused_prefill_probe PRIVATE
  "${KVL_ROOT}/include" "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_mla_fused_prefill_probe PRIVATE
  kvl_storage kvl_mla_prefill_fused
)

# Build a generator clone that replaces only the duplicated
# prefill+compressed-state pair with the fused research call.
file(READ "${KVL_ROOT}/src/generate.c" KVL_FUSED_GENERATE)
string(REPLACE
  "#include \"kvl/trunk_store.h\"\n"
  "#include \"kvl/trunk_store.h\"\n#include \"mla_prefill_fused.h\"\n"
  KVL_FUSED_GENERATE "${KVL_FUSED_GENERATE}")
set(KVL_FUSED_NEEDLE
"        if (kvl_mla_prefill_bf16(work_b, work_a, seq_len, &cfg, &aw) != 0 ||\n            kvl_mla_compressed_state_prefill_bf16(work_a, seq_len, &cfg, &aw,\n                                                   &states[layer]) != 0) {")
set(KVL_FUSED_REPLACEMENT
"        if (kvl_mla_prefill_compressed_fused_bf16(work_b, work_a, seq_len,\n                                                   &cfg, &aw, &states[layer]) != 0) {")
string(REPLACE "${KVL_FUSED_NEEDLE}" "${KVL_FUSED_REPLACEMENT}"
       KVL_FUSED_GENERATE "${KVL_FUSED_GENERATE}")
string(FIND "${KVL_FUSED_GENERATE}" "kvl_mla_prefill_compressed_fused_bf16" KVL_FUSED_POS)
if(KVL_FUSED_POS LESS 0)
  message(FATAL_ERROR "generate.c fused-prefill insertion point changed")
endif()
set(KVL_FUSED_SOURCE "${CMAKE_BINARY_DIR}/generate_fused_prefill.c")
file(WRITE "${KVL_FUSED_SOURCE}" "${KVL_FUSED_GENERATE}")
set_source_files_properties("${KVL_FUSED_SOURCE}" PROPERTIES
  GENERATED TRUE
  COMPILE_DEFINITIONS "kvl_matvec_bf16=kvl_matvec_bf16_dump;kvl_moe_token_bf16=kvl_moe_token_auto"
)
add_executable(kvl_generate_fused_prefill
  "${KVL_FUSED_SOURCE}"
  "${KVL_ROOT}/src/logits_dump.c"
)
target_include_directories(kvl_generate_fused_prefill PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(kvl_generate_fused_prefill PRIVATE
  kvl_storage kvl_mla_prefill_fused
)

# Token-parallel fused prefill: keep each AVX2 GEMV serial and parallelise the
# independent prompt-token dimension instead. This avoids opening an OpenMP
# region for every small GEMV while retaining each row's arithmetic order.
add_library(kvl_mla_prefill_token_parallel STATIC
  "${CMAKE_CURRENT_LIST_DIR}/mla_prefill_token_parallel.c"
)
target_include_directories(kvl_mla_prefill_token_parallel PRIVATE
  "${KVL_ROOT}/include" "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_mla_prefill_token_parallel PUBLIC kvl_storage)
target_compile_definitions(kvl_mla_prefill_token_parallel PRIVATE KVL_USE_AVX2=1)
if(MSVC)
  target_compile_options(kvl_mla_prefill_token_parallel PRIVATE /arch:AVX2 /openmp)
else()
  target_compile_options(kvl_mla_prefill_token_parallel PRIVATE -mavx2)
  if(OpenMP_C_FOUND)
    target_link_libraries(kvl_mla_prefill_token_parallel PUBLIC OpenMP::OpenMP_C)
  endif()
endif()

add_executable(kvl_mla_token_parallel_prefill_probe
  "${CMAKE_CURRENT_LIST_DIR}/mla_prefill_token_parallel_probe.c"
)
target_include_directories(kvl_mla_token_parallel_prefill_probe PRIVATE
  "${KVL_ROOT}/include" "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_mla_token_parallel_prefill_probe PRIVATE
  kvl_storage kvl_mla_prefill_token_parallel
)

file(READ "${KVL_ROOT}/src/generate.c" KVL_TOKEN_PARALLEL_GENERATE)
string(REPLACE
  "#include \"kvl/trunk_store.h\"\n"
  "#include \"kvl/trunk_store.h\"\n#include \"mla_prefill_token_parallel.h\"\n"
  KVL_TOKEN_PARALLEL_GENERATE "${KVL_TOKEN_PARALLEL_GENERATE}")
set(KVL_TOKEN_PARALLEL_NEEDLE
"        if (kvl_mla_prefill_bf16(work_b, work_a, seq_len, &cfg, &aw) != 0 ||\n            kvl_mla_compressed_state_prefill_bf16(work_a, seq_len, &cfg, &aw,\n                                                   &states[layer]) != 0) {")
set(KVL_TOKEN_PARALLEL_REPLACEMENT
"        if (kvl_mla_prefill_compressed_token_parallel_bf16(\n                work_b, work_a, seq_len, &cfg, &aw, &states[layer]) != 0) {")
string(REPLACE "${KVL_TOKEN_PARALLEL_NEEDLE}" "${KVL_TOKEN_PARALLEL_REPLACEMENT}"
       KVL_TOKEN_PARALLEL_GENERATE "${KVL_TOKEN_PARALLEL_GENERATE}")
string(FIND "${KVL_TOKEN_PARALLEL_GENERATE}"
       "kvl_mla_prefill_compressed_token_parallel_bf16" KVL_TOKEN_PARALLEL_POS)
if(KVL_TOKEN_PARALLEL_POS LESS 0)
  message(FATAL_ERROR "generate.c token-parallel prefill insertion point changed")
endif()
set(KVL_TOKEN_PARALLEL_SOURCE "${CMAKE_BINARY_DIR}/generate_token_parallel_prefill.c")
file(WRITE "${KVL_TOKEN_PARALLEL_SOURCE}" "${KVL_TOKEN_PARALLEL_GENERATE}")
set_source_files_properties("${KVL_TOKEN_PARALLEL_SOURCE}" PROPERTIES
  GENERATED TRUE
  COMPILE_DEFINITIONS "kvl_matvec_bf16=kvl_matvec_bf16_dump;kvl_moe_token_bf16=kvl_moe_token_auto"
)
add_executable(kvl_generate_token_parallel_prefill
  "${KVL_TOKEN_PARALLEL_SOURCE}"
  "${KVL_ROOT}/src/logits_dump.c"
)
target_include_directories(kvl_generate_token_parallel_prefill PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_generate_token_parallel_prefill PRIVATE
  kvl_storage kvl_mla_prefill_token_parallel
)

# Decoder workspace candidate: baseline compressed MLA allocates seven temporary
# buffers per layer/token. The research function reuses one ~tens-of-KiB process
# arena while retaining the same arithmetic and state update order.
add_library(kvl_mla_decode_reuse STATIC
  "${CMAKE_CURRENT_LIST_DIR}/mla_decode_reuse.c"
)
target_include_directories(kvl_mla_decode_reuse PRIVATE
  "${KVL_ROOT}/include" "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_mla_decode_reuse PUBLIC kvl_storage)

add_executable(kvl_mla_decode_reuse_probe
  "${CMAKE_CURRENT_LIST_DIR}/mla_decode_reuse_probe.c"
)
target_include_directories(kvl_mla_decode_reuse_probe PRIVATE
  "${KVL_ROOT}/include" "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_mla_decode_reuse_probe PRIVATE
  kvl_storage kvl_mla_decode_reuse
)

add_executable(kvl_generate_decode_reuse
  "${KVL_ROOT}/src/generate.c"
  "${KVL_ROOT}/src/logits_dump.c"
)
target_link_libraries(kvl_generate_decode_reuse PRIVATE
  kvl_storage kvl_mla_decode_reuse
)
target_compile_definitions(kvl_generate_decode_reuse PRIVATE
  kvl_moe_token_bf16=kvl_moe_token_auto
  kvl_mla_decode_compressed_bf16=kvl_mla_decode_compressed_reuse_bf16
)

# Generate a measurement-only clone of generate.c at configure time. It reports
# cache/trunk counters after prefill, resets counters without dropping resident
# data, then reports decode separately. Model math and token selection are not
# changed.
file(READ "${KVL_ROOT}/src/generate.c" KVL_PROFILE_GENERATE)
set(KVL_PROFILE_NEEDLE
"    int generated = 0;\n    int position = prompt_n;")
set(KVL_PROFILE_REPLACEMENT
"    fprintf(stderr, \"kvl_profile: phase=prefill\\n\");\n    kvl_expert_cache_report(&cache);\n    kvl_trunk_cache_report(&ts);\n    kvl_expert_cache_reset_stats(&cache);\n    ts.load_calls = 0;\n    ts.cache_hits = 0;\n    ts.cache_inserts = 0;\n    ts.bytes_read = 0;\n    fprintf(stderr, \"kvl_profile: phase=decode\\n\");\n\n    int generated = 0;\n    int position = prompt_n;")
string(REPLACE "${KVL_PROFILE_NEEDLE}" "${KVL_PROFILE_REPLACEMENT}"
       KVL_PROFILE_GENERATE "${KVL_PROFILE_GENERATE}")
string(FIND "${KVL_PROFILE_GENERATE}" "kvl_profile: phase=prefill" KVL_PROFILE_POS)
if(KVL_PROFILE_POS LESS 0)
  message(FATAL_ERROR "generate.c profile insertion point changed")
endif()

set(KVL_PROFILE_FINAL_NEEDLE
"    kvl_expert_cache_report(&cache);\n\n    for (int layer = 0; layer < LN; ++layer)")
set(KVL_PROFILE_FINAL_REPLACEMENT
"    fprintf(stderr, \"kvl_profile: phase=decode-final\\n\");\n    kvl_expert_cache_report(&cache);\n    kvl_trunk_cache_report(&ts);\n\n    for (int layer = 0; layer < LN; ++layer)")
string(REPLACE "${KVL_PROFILE_FINAL_NEEDLE}" "${KVL_PROFILE_FINAL_REPLACEMENT}"
       KVL_PROFILE_GENERATE "${KVL_PROFILE_GENERATE}")
string(FIND "${KVL_PROFILE_GENERATE}" "kvl_profile: phase=decode-final" KVL_PROFILE_FINAL_POS)
if(KVL_PROFILE_FINAL_POS LESS 0)
  message(FATAL_ERROR "generate.c final profile insertion point changed")
endif()

set(KVL_PROFILE_SOURCE "${CMAKE_BINARY_DIR}/generate_profile.c")
file(WRITE "${KVL_PROFILE_SOURCE}" "${KVL_PROFILE_GENERATE}")
set_source_files_properties("${KVL_PROFILE_SOURCE}" PROPERTIES
  GENERATED TRUE
  COMPILE_DEFINITIONS "kvl_matvec_bf16=kvl_matvec_bf16_dump;kvl_moe_token_bf16=kvl_moe_token_auto"
)
add_executable(kvl_generate_profile
  "${KVL_PROFILE_SOURCE}"
  "${KVL_ROOT}/src/logits_dump.c"
)
target_link_libraries(kvl_generate_profile PRIVATE kvl_storage)
