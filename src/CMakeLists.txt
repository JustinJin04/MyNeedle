set(SOURCES)

list(APPEND SOURCES NDArray.cc kernel_cpu.cc Tensor.cc)

add_library(MyNeedle SHARED ${SOURCES})

target_compile_features(MyNeedle PUBLIC cxx_std_17)