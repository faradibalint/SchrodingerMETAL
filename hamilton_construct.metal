//
//  hamilton_construct.metal
//  Schrodinger
//
//  Created by Farádi Bálint on 2025. 05. 27..
//

#include <metal_stdlib>
using namespace metal;

kernel void calculate_hamiltonian(
    device float* diag [[buffer(0)]],
    device float* off_diag [[buffer(1)]],
    constant float* V [[buffer(2)]],
    constant float& h [[buffer(3)]],
    constant int& N [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Peremellenőrzés
    if (gid.x >= N || gid.y >= N) return;
    
    int i = gid.x;
    int j = gid.y;
    int idx = i * N + j;
    int base_idx = idx * 5;
    
    // Kinetikus energia együttható
    float kinetic_coeff = -1.0f / (2.0f * h * h);
    
    // 1. Főátló: 4 * (-kinetic_coeff) + V(i,j)
    diag[idx] = 4.0f * (-kinetic_coeff) + V[idx];
    // 2. Szomszédok nullázása
    for (int k = 0; k < 5; ++k) {
        off_diag[base_idx + k] = 0.0f;
    }
    
    // 3. Szomszédok beállítása (bal, jobb, felső, alsó)
    if (j > 0) { // Bal
        off_diag[base_idx + 0] = kinetic_coeff;
    }
    if (j < N-1) { // Jobb
        off_diag[base_idx + 1] = kinetic_coeff;
    }
    if (i > 0) { // Felső
        off_diag[base_idx + 2] = kinetic_coeff;
    }
    if (i < N-1) { // Alsó
        off_diag[base_idx + 3] = kinetic_coeff;
    }
    // base_idx + 4: helykitöltő (0 marad)
}

