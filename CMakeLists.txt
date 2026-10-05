cmake_minimum_required(VERSION 3.14)
project(DocumentSimilarity LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(include)

set(CORE_SOURCES
    src/document.cpp
    src/preprocessor.cpp
    src/tfidf.cpp
    src/similarity_engine.cpp
)

add_executable(doc_similarity src/main.cpp ${CORE_SOURCES})
target_link_libraries(doc_similarity PRIVATE OpenMP::OpenMP_CXX)

add_executable(run_tests tests/test_correctness.cpp ${CORE_SOURCES})
target_link_libraries(run_tests PRIVATE OpenMP::OpenMP_CXX)
