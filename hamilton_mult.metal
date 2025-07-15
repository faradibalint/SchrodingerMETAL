//
//  hamilton_mult.metal
//  Schrodinger
//
//  Created by Farádi Bálint on 2025. 06. 27..
//
#include <metal_stdlib>
using namespace metal;

kernel void hamiltonian_mult(
    device const float* diag        [[buffer(0)]],
    device const float* offDiag     [[buffer(1)]],
    device const float* inputVec    [[buffer(2)]],
    device float* outputVec         [[buffer(3)]],
    constant int& N                 [[buffer(4)]],
    uint2 tid                       [[thread_position_in_grid]]
) {
    int i = tid.y;
    int j = tid.x;
    if (i >= N || j >= N) return;

    int idx = i * N + j;
    int base = idx * 5;

    float center = inputVec[idx];
    float left   = (j > 0)     ? inputVec[idx - 1]   : 0.0f;
    float right  = (j < N - 1) ? inputVec[idx + 1]   : 0.0f;
    float top    = (i > 0)     ? inputVec[idx - N]   : 0.0f;
    float bottom = (i < N - 1) ? inputVec[idx + N]   : 0.0f;

    float result = diag[idx] * center;
    result += offDiag[base + 0] * left;
    result += offDiag[base + 1] * right;
    result += offDiag[base + 2] * top;
    result += offDiag[base + 3] * bottom;

    outputVec[idx] = result;
}

