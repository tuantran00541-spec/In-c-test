cmake_minimum_required(VERSION 3.16)
project(kimi_vl_lowram C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
option(KVL_USE_AVX2 "Use AVX2 BF16/Q8 GEMV backends" OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT MSVC)
  find_package(OpenMP)
endif()

# Keep the old fully-materialized MLA prefill available as a lab oracle while
# routing production callers on this branch to the exact low-memory implementation.
set_source_files_properties(src/ops.c PROPERTIES
  COMPILE_DEFINITIONS "kvl_mla_prefill_bf16=kvl_mla_prefill_materialized_bf16")

add_library(kvl_storage
  src/expert_store.c
  src/expert_cache.c
  src/trunk_store.c
  src/ops.c
  src/mla_prefill_streaming.c
  src/q8_ops.c
  src/mla_state.c
  src/mla_compressed_state.c
  src/vision.c
)
target_include_directories(kvl_storage PUBLIC include)
if (OpenMP_C_FOUND)
  target_link_libraries(kvl_storage PUBLIC OpenMP::OpenMP_C)
endif()
if (KVL_USE_AVX2)
  target_compile_definitions(kvl_storage PRIVATE KVL_USE_AVX2=1)
  if(MSVC)
    target_compile_options(kvl_storage PRIVATE /arch:AVX2)
  else()
    target_compile_options(kvl_storage PRIVATE -mavx2)
  endif()
endif()
if (WIN32)
  target_compile_definitions(kvl_storage PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_link_libraries(kvl_storage PUBLIC m)
endif()

add_executable(kvl_probe src/main.c)
target_link_libraries(kvl_probe PRIVATE kvl_storage)
add_executable(kvl_cache_probe src/cache_probe.c)
target_link_libraries(kvl_cache_probe PRIVATE kvl_storage)
add_executable(kvl_moe_probe src/moe_probe.c)
target_link_libraries(kvl_moe_probe PRIVATE kvl_storage)
target_compile_definitions(kvl_moe_probe PRIVATE kvl_moe_token_bf16=kvl_moe_token_auto)
add_executable(kvl_layer_probe src/layer_probe.c)
target_link_libraries(kvl_layer_probe PRIVATE kvl_storage)
add_executable(kvl_stack_probe src/stack_probe.c)
target_link_libraries(kvl_stack_probe PRIVATE kvl_storage)
add_executable(kvl_logits_probe src/logits_probe.c)
target_link_libraries(kvl_logits_probe PRIVATE kvl_storage)
add_executable(kvl_mla_incremental_probe src/mla_incremental_probe.c)
target_link_libraries(kvl_mla_incremental_probe PRIVATE kvl_storage)
add_executable(kvl_mla_real_probe src/mla_real_probe.c)
target_link_libraries(kvl_mla_real_probe PRIVATE kvl_storage)
add_executable(kvl_mla_streaming_probe src/mla_streaming_probe.c)
target_link_libraries(kvl_mla_streaming_probe PRIVATE kvl_storage)
add_executable(kvl_mla_prefill_bench src/mla_prefill_bench.c)
target_link_libraries(kvl_mla_prefill_bench PRIVATE kvl_storage)
add_executable(kvl_decode_probe src/decode_probe.c)
target_link_libraries(kvl_decode_probe PRIVATE kvl_storage)
add_executable(kvl_generate src/generate.c)
target_link_libraries(kvl_generate PRIVATE kvl_storage)
target_compile_definitions(kvl_generate PRIVATE kvl_moe_token_bf16=kvl_moe_token_auto)
add_executable(kvl_generate_vl src/generate_vl.c)
target_link_libraries(kvl_generate_vl PRIVATE kvl_storage)
target_compile_definitions(kvl_generate_vl PRIVATE kvl_moe_token_bf16=kvl_moe_token_auto)
add_executable(kvl_vision src/vision_probe.c)
target_link_libraries(kvl_vision PRIVATE kvl_storage)
add_executable(kvl_q8_probe src/q8_probe.c)
target_link_libraries(kvl_q8_probe PRIVATE kvl_storage)
