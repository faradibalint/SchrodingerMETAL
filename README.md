# SchrödingerMETAL
This project presents a METAL framework accelerated solver for Schrödinger's 2D Time-independent quantum equation.
The equation is solved using the method of finite differences and Lanczos algorithm for the Hamiltonian. The tridiagonal matrix produced by Lanczos iterations is solved with Armadillo to get the eigenvalues and eigenvectors.
The project uses metal-cpp to implement GPU acceleration to multiple parts of the solver. The Hamiltonian construction and the Matrix-Vector multiplication is GPU accelerated.
Currently a simply 2D Harmonic Oscillator is presented as potential.

The project contains a comaparion between the GPU accelerated version and a plain CPU based Lanczos.

This project was created as an assignment for GPU Programming class at Eötvös Loránd University.
