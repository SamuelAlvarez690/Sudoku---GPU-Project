#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util.h"

#define BOARD_DIM 9
#define GROUP_DIM 3
#define BATCH_SIZE 25000

typedef struct board {
  uint16_t cells[BOARD_DIM * BOARD_DIM];
} board_t;

// Host version
uint16_t digit_to_cell(int digit) {
  if (digit == 0) return 0x3FE;
  return 1 << digit;
}

// Device version
__device__ uint16_t d_digit_to_cell(int digit) {
  if (digit == 0) return 0x3FE;
  return 1 << digit;
}

// Host version
int cell_to_digit(uint16_t cell) {
  int lsb = __builtin_ctz(cell);
  if (cell == (uint16_t)(1 << lsb)) return lsb;
  return 0;
}

// Device version
__device__ int d_cell_to_digit(uint16_t cell) {
  int msb = __clz(cell);
  int lsb = sizeof(unsigned int) * 8 - msb - 1;
  if (cell == (uint16_t)(1 << lsb)) return lsb;
  return 0;
}

__device__ bool is_valid(board_t* board, int pos, int digit) {
  int row = pos / 9;
  int col = pos % 9;
  int box_row = (row / 3) * 3;
  int box_col = (col / 3) * 3;

  for (int i = 0; i < 9; i++) {
    if (d_cell_to_digit(board->cells[row * 9 + i]) == digit) return false;
    if (d_cell_to_digit(board->cells[i * 9 + col]) == digit) return false;
    int br = box_row + i / 3;
    int bc = box_col + i % 3;
    if (d_cell_to_digit(board->cells[br * 9 + bc]) == digit) return false;
  }

  return true;
}

__global__ void naive_solver(board_t* boards) {
  board_t* board = &boards[blockIdx.x];

  bool given[81];
  for (int i = 0; i < 81; i++) {
    given[i] = (d_cell_to_digit(board->cells[i]) != 0);
  }

  int guess[81];
  for (int i = 0; i < 81; i++) guess[i] = 0;

  int pos = 0;
  while (pos >= 0 && pos < 81) {
    if (given[pos]) {
      pos++;
      continue;
    }

    int start = guess[pos] + 1;
    bool placed = false;

    for (int digit = start; digit <= 9; digit++) {
      if (is_valid(board, pos, digit)) {
        guess[pos] = digit;
        board->cells[pos] = d_digit_to_cell(digit);
        placed = true;
        break;
      }
    }

    if (placed) {
      pos++;
    } else {
      guess[pos] = 0;
      board->cells[pos] = d_digit_to_cell(0);
      pos--;

      while (pos >= 0 && given[pos]) {
        pos--;
      }
    }
  }
}

void solve_boards(board_t* cpu_boards, size_t num_boards) {
  board_t* gpu_boards;
  if (cudaMalloc(&gpu_boards, sizeof(board_t) * num_boards) != cudaSuccess) {
    perror("cuda malloc failed.");
    exit(2);
  }
  if (cudaMemcpy(gpu_boards, cpu_boards, sizeof(board_t) * num_boards,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    perror("cuda memcpy failed.");
    exit(2);
  }

  naive_solver<<<num_boards, 1>>>(gpu_boards);

  if (cudaDeviceSynchronize() != cudaSuccess) {
    perror("cudaDeviceSynchronize failed.");
    exit(2);
  }
  if (cudaMemcpy(cpu_boards, gpu_boards, sizeof(board_t) * num_boards,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    perror("cuda memcpy failed.");
    exit(2);
  }
  cudaFree(gpu_boards);
}

bool read_board(board_t* output, const char* str) {
  for (int index = 0; index < BOARD_DIM * BOARD_DIM; index++) {
    if (str[index] < '0' || str[index] > '9') return false;
    int value = str[index] - '0';
    output->cells[index] = digit_to_cell(value);
  }
  return true;
}

void check_solutions(board_t* boards, board_t* solutions, size_t num_boards,
                     size_t* solved_count, size_t* error_count) {
  for (int i = 0; i < num_boards; i++) {
    if (memcmp(&boards[i], &solutions[i], sizeof(board_t)) == 0) {
      (*solved_count)++;
    } else {
      bool valid = true;
      for (int j = 0; j < BOARD_DIM * BOARD_DIM; j++) {
        if ((boards[i].cells[j] & solutions[i].cells[j]) == 0) {
          valid = false;
        }
      }
      if (!valid) (*error_count)++;
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <input file name>\n", argv[0]);
    exit(1);
  }

  FILE* input = fopen(argv[1], "r");
  if (input == NULL) {
    fprintf(stderr, "Failed to open input file %s.\n", argv[1]);
    perror(NULL);
    exit(2);
  }

  size_t board_count = 0;
  size_t solved_count = 0;
  size_t error_count = 0;
  size_t solving_time = 0;

  board_t boards[BATCH_SIZE];
  board_t solutions[BATCH_SIZE];
  size_t batch_count = 0;

  char* line = NULL;
  size_t line_capacity = 0;
  while (getline(&line, &line_capacity, input) > 0) {
    if (!read_board(&boards[batch_count], line)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }
    if (!read_board(&solutions[batch_count], line + BOARD_DIM * BOARD_DIM + 1)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }

    batch_count++;
    board_count++;

    if (batch_count == BATCH_SIZE) {
      size_t start_time = time_ms();
      solve_boards(boards, batch_count);
      solving_time += time_ms() - start_time;
      check_solutions(boards, solutions, batch_count, &solved_count, &error_count);
      batch_count = 0;
    }
  }

  if (batch_count > 0) {
    size_t start_time = time_ms();
    solve_boards(boards, batch_count);
    solving_time += time_ms() - start_time;
    check_solutions(boards, solutions, batch_count, &solved_count, &error_count);
  }

  double seconds = (double)solving_time / 1000;
  double solving_rate = (double)solved_count / seconds;
  if (seconds < 0.01) solving_rate = 0;

  printf("Boards: %lu\n", board_count);
  printf("Boards Solved: %lu\n", solved_count);
  printf("Errors: %lu\n", error_count);
  printf("Total Solving Time: %lums\n", solving_time);
  printf("Solving Rate: %.2f sudoku/second\n", solving_rate);

  return 0;
}