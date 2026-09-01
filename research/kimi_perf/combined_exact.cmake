# Combined candidate containing only transformations that have independent
# numerical/exactness gates. Shared-Q8 is intentionally excluded because it is
# lossy and must remain a separate quality arm.

add_library(kvl_q8_expert_parallel_stack STATIC
  "${CMAKE_CURRENT_LIST_DIR}/q8_expert_parallel_stack.c"
)
target_include_directories(kvl_q8_expert_parallel_stack PRIVATE
  "${KVL_ROOT}/include" "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_q8_expert_parallel_stack PUBLIC
  kvl_storage kvl_router_stack
)
target_compile_definitions(kvl_q8_expert_parallel_stack PRIVATE KVL_USE_AVX2=1)
if(MSVC)
  target_compile_options(kvl_q8_expert_parallel_stack PRIVATE /arch:AVX2 /openmp)
else()
  target_compile_options(kvl_q8_expert_parallel_stack PRIVATE -mavx2)
  if(OpenMP_C_FOUND)
    target_link_libraries(kvl_q8_expert_parallel_stack PUBLIC OpenMP::OpenMP_C)
  endif()
endif()

file(READ "${KVL_ROOT}/src/generate.c" KVL_COMBINED_EXACT_GENERATE)
string(REPLACE
  "#include \"kvl/trunk_store.h\"\n"
  "#include \"kvl/trunk_store.h\"\n#include \"mla_prefill_token_parallel.h\"\n"
  KVL_COMBINED_EXACT_GENERATE "${KVL_COMBINED_EXACT_GENERATE}")

set(KVL_COMBINED_PREFILL_NEEDLE
"        if (kvl_mla_prefill_bf16(work_b, work_a, seq_len, &cfg, &aw) != 0 ||\n            kvl_mla_compressed_state_prefill_bf16(work_a, seq_len, &cfg, &aw,\n                                                   &states[layer]) != 0) {")
set(KVL_COMBINED_PREFILL_REPLACEMENT
"        if (kvl_mla_prefill_compressed_token_parallel_bf16(\n                work_b, work_a, seq_len, &cfg, &aw, &states[layer]) != 0) {")
string(REPLACE "${KVL_COMBINED_PREFILL_NEEDLE}" "${KVL_COMBINED_PREFILL_REPLACEMENT}"
       KVL_COMBINED_EXACT_GENERATE "${KVL_COMBINED_EXACT_GENERATE}")
string(FIND "${KVL_COMBINED_EXACT_GENERATE}"
       "kvl_mla_prefill_compressed_token_parallel_bf16" KVL_COMBINED_PREFILL_POS)
if(KVL_COMBINED_PREFILL_POS LESS 0)
  message(FATAL_ERROR "generate.c combined token-parallel insertion point changed")
endif()

set(KVL_COMBINED_EXACT_SOURCE "${CMAKE_BINARY_DIR}/generate_combined_exact.c")
file(WRITE "${KVL_COMBINED_EXACT_SOURCE}" "${KVL_COMBINED_EXACT_GENERATE}")
set_source_files_properties("${KVL_COMBINED_EXACT_SOURCE}" PROPERTIES
  GENERATED TRUE
  COMPILE_DEFINITIONS
    "kvl_matvec_bf16=kvl_matvec_bf16_dump;kvl_moe_token_bf16=kvl_moe_token_q8_expert_parallel_auto;kvl_mla_decode_compressed_bf16=kvl_mla_decode_compressed_reuse_bf16"
)

add_executable(kvl_generate_combined_exact
  "${KVL_COMBINED_EXACT_SOURCE}"
  "${KVL_ROOT}/src/logits_dump.c"
)
target_include_directories(kvl_generate_combined_exact PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}"
)
target_link_libraries(kvl_generate_combined_exact PRIVATE
  kvl_storage
  kvl_mla_prefill_token_parallel
  kvl_mla_decode_reuse
  kvl_q8_expert_parallel_stack
  kvl_router_stack
)
