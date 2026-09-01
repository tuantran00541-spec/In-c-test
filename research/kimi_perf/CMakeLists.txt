cmake_minimum_required(VERSION 3.16)
project(kimi_perf_research C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

get_filename_component(KVL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# This lab intentionally compares two schedulers with all other packed-Q8
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
