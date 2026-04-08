# ─── Sudoku Solver – COSC 4397 Final Project ────────────────────────────────
NVCC       := nvcc
CXX_STD    := --std=c++17
ARCH       := -arch=sm_86
OPT        := -O3 --use_fast_math
STACKSIZE  := -Xptxas -v
NVCCFLAGS  := $(CXX_STD) $(ARCH) $(OPT) $(STACKSIZE)
TARGET     := sudoku_bench
SRCS       := sudoku_bench.cu
INCLUDES   := -I.

.PHONY: all run clean info

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^
	@echo ""
	@echo "Build successful → ./$(TARGET)"

run: all
	@echo ""
	./$(TARGET)

info:
	@nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
	            --format=csv,noheader

clean:
	rm -f $(TARGET) *.o *.ptx
