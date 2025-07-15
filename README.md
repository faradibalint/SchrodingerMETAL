# SchrödingerMETAL
This project presents a METAL framework accelerated solver for Schrödinger's 2D Time-independent quantum equation.
The equation is solved using the method of finite differences and Lanczos algorithm for the Hamiltonian.
The project uses metal-cpp to implement GPU acceleration to multiple parts of the solver. The Hamiltonian construction and the Matrix-Vector multiplication is GPU accelerated.
Currently a simply 2D Harmonic Oscillator is presented as potential.

This project was created as an assignment for GPU Programming class at Eötvös Loránd University.
