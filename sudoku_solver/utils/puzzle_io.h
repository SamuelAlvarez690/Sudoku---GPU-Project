#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// A Sudoku board is stored as a flat array of 81 ints.
// 0 = empty cell, 1-9 = given clue.
using Board = int[81];

// Pretty-print a board to stdout
inline void print_board(const int* b) {
    for (int r = 0; r < 9; r++) {
        if (r % 3 == 0 && r != 0) printf("+-------+-------+-------+\n");
        for (int c = 0; c < 9; c++) {
            if (c % 3 == 0) printf("| ");
            int v = b[r * 9 + c];
            if (v == 0) printf(". ");
            else        printf("%d ", v);
        }
        printf("|\n");
    }
}

// Validate a completed board (all rows/cols/boxes use 1-9 exactly once)
inline bool validate_board(const int* b) {
    for (int r = 0; r < 9; r++) {
        int seen = 0;
        for (int c = 0; c < 9; c++) {
            int v = b[r*9+c];
            if (v < 1 || v > 9) return false;
            if (seen & (1 << v)) return false;
            seen |= (1 << v);
        }
    }
    for (int c = 0; c < 9; c++) {
        int seen = 0;
        for (int r = 0; r < 9; r++) {
            int v = b[r*9+c];
            if (seen & (1 << v)) return false;
            seen |= (1 << v);
        }
    }
    for (int br = 0; br < 3; br++) {
        for (int bc = 0; bc < 3; bc++) {
            int seen = 0;
            for (int dr = 0; dr < 3; dr++)
                for (int dc = 0; dc < 3; dc++) {
                    int v = b[(br*3+dr)*9 + (bc*3+dc)];
                    if (seen & (1 << v)) return false;
                    seen |= (1 << v);
                }
        }
    }
    return true;
}

// Built-in puzzle bank – Easy → Near-hardest
static const char* PUZZLE_BANK[] = {
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "000000907000420180000705026100904000050000040000507009920108000034059000507000000",
    "800000000003600000070090200060005030000403001008070600002006004500008000010000006",
    "000000012000035000000600070700000300000400800100000000000120000080000040050000600",
    "000000000000003085001020000000507000004000100090000000500000073002010000000040009",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000001",
};
static const char* PUZZLE_NAMES[] = { "Easy", "Medium", "Hard", "Expert", "Extreme", "Near-hardest" };
static const int NUM_PUZZLES = 1;

inline void load_puzzle(const char* str, int* board) {
    for (int i = 0; i < 81; i++) {
        char ch = str[i];
        board[i] = (ch >= '1' && ch <= '9') ? (ch - '0') : 0;
    }
}

inline std::vector<std::vector<int>> generate_puzzle_batch(int N) {
    std::vector<std::vector<int>> batch(N, std::vector<int>(81));
    for (int i = 0; i < N; i++)
        load_puzzle(PUZZLE_BANK[i % NUM_PUZZLES], batch[i].data());
    return batch;
}
