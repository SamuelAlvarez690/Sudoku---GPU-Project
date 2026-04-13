#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cassert>
 
using Board = int[81];
 
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
 
// ─── Puzzle bank ─────────────────────────────────────────────────────────────
// All 8 puzzles CPU-verified to solve within 50,000 naive backtracks.
// Iteration counts measured by Python reference solver (same algorithm as
// naive.cuh) — safe for the naive GPU kernel with NAIVE_MAX_ITER=2,000,000.
 
static const char* PUZZLE_BANK[] = {
    // orig-A  ~4,209 iters
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    // orig-B  ~201 iters
    "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
    // orig-C  ~295 iters
    "200080300060070084030500209000105408000000000402706000301007040720040060004010003",
    // orig-D  ~19,023 iters
    "000000907000420180000705026100904000050000040000507009920108000034059000507000000",
    // new-A   ~9,699 iters
    "800000000003600000070090200060005030000403001008070600002006004500008000010000006",
    // new-B   ~5,010 iters
    "290300000000080070000070090000530000670000043000046000010090000060050000000002084",
    // new-C   ~7,141 iters
    "000200090080030000000080300000900600004050200002008000001040000000060050060005000",
    // new-D   ~3,558 iters
    "043080250600000000000001094900004070000608000010200003820500000000000005034090710",
};
static const int NUM_PUZZLES = 8;
 
static const char* PUZZLE_NAMES[] = {
    "Puzzle-0", "Puzzle-1", "Puzzle-2", "Puzzle-3",
    "Puzzle-4", "Puzzle-5", "Puzzle-6", "Puzzle-7",
};
 
// ── Hard bank — bitboard/constraint_prop only, DO NOT use with naive ─────────
static const char* HARD_BANK[] = {
    "000000000010400700000020084900060200000903000007010009680040000002007010000000000",
    "000000000420005080000061007700400050008000600060003004500720000030100096000000000",
};
static const int NUM_HARD = 2;
 
inline void load_puzzle(const char* str, int* board) {
    for (int i = 0; i < 81; i++) {
        char ch = str[i];
        board[i] = (ch >= '1' && ch <= '9') ? (ch - '0') : 0;
    }
}
 
// Default — all verified safe for naive
inline std::vector<std::vector<int>> generate_puzzle_batch(int N) {
    std::vector<std::vector<int>> batch(N, std::vector<int>(81));
    for (int i = 0; i < N; i++)
        load_puzzle(PUZZLE_BANK[i % NUM_PUZZLES], batch[i].data());
    return batch;
}
 
// Hard batch — bitboard/constraint_prop only
inline std::vector<std::vector<int>> generate_hard_batch(int N) {
    std::vector<std::vector<int>> batch(N, std::vector<int>(81));
    for (int i = 0; i < N; i++)
        load_puzzle(HARD_BANK[i % NUM_HARD], batch[i].data());
    return batch;
}
 
inline std::vector<std::vector<int>> generate_batch_at(int N, int start, int count) {
    std::vector<std::vector<int>> batch(N, std::vector<int>(81));
    for (int i = 0; i < N; i++)
        load_puzzle(PUZZLE_BANK[start + (i % count)], batch[i].data());
    return batch;
}