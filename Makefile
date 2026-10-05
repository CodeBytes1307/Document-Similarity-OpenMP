CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra

# OpenMP flag detection for macOS Apple Clang vs GCC
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    BREW_OMP_DIR := $(shell find /opt/homebrew/Cellar/libomp -maxdepth 1 -type d 2>/dev/null | tail -n 1)
    ifneq ($(BREW_OMP_DIR),)
        OMP_FLAGS := -Xpreprocessor -fopenmp -I$(BREW_OMP_DIR)/include -L$(BREW_OMP_DIR)/lib -lomp
    else
        OMP_FLAGS := -fopenmp
    endif
else
    OMP_FLAGS := -fopenmp
endif

INC := -Iinclude
SRC := src/document.cpp src/preprocessor.cpp src/tfidf.cpp src/similarity_engine.cpp
MAIN_SRC := src/main.cpp
TEST_SRC := tests/test_correctness.cpp

TARGET := doc_similarity
TEST_TARGET := run_tests

.PHONY: all clean test

all: $(TARGET) $(TEST_TARGET)

$(TARGET): $(SRC) $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INC) $^ -o $@

$(TEST_TARGET): $(SRC) $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INC) $^ -o $@

test: all
	./$(TEST_TARGET)

clean:
	rm -f $(TARGET) $(TEST_TARGET) *.o
