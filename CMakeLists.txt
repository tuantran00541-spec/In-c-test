cmake_minimum_required(VERSION 3.16)
project(kimi_vl_lowram C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(OpenMP)

add_library(kvl_storage
  src/expert_store.c
  src/expert_cache.c
  src/trunk_store.c
  src/ops.c
  src/mla_state.c
  src/mla_compressed_state.c
)
target_include_directories(kvl_storage PUBLIC include)
if (OpenMP_C_FOUND)
  target_link_libraries(kvl_storage PUBLIC OpenMP::OpenMP_C)
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
