cmake_minimum_required(VERSION 3.10)
project(BookCharacterAI)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(Engine/Tensor)
add_subdirectory(Engine/Attention)
add_subdirectory(Engine/Autograd)
add_subdirectory(Engine/Layers)
add_subdirectory(Engine/Transformer)
add_subdirectory(Engine/Normalization)
add_subdirectory(Engine/Tokenizer)

add_executable(book_ai
    Models/MargaritaModel/main.cpp
    App/book_loader.cpp
    App/trainer.cpp
    App/chat_interface.cpp
)

target_link_libraries(book_ai
    tensor_lib
    autograd_lib
    attention_lib
    layers_lib
    transformer_lib
    normalization_lib
    matmul_lib
    tokenizer_lib
    "-framework Accelerate"
)