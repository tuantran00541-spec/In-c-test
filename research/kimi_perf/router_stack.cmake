add_library(kvl_router_stack STATIC
  "${CMAKE_CURRENT_LIST_DIR}/router_stack.c"
)
target_include_directories(kvl_router_stack PRIVATE "${KVL_ROOT}/include")
target_link_libraries(kvl_router_stack PUBLIC kvl_storage)

add_executable(kvl_router_stack_probe
  "${CMAKE_CURRENT_LIST_DIR}/router_stack_probe.c"
)
target_include_directories(kvl_router_stack_probe PRIVATE "${KVL_ROOT}/include")
target_link_libraries(kvl_router_stack_probe PRIVATE kvl_storage kvl_router_stack)
