# SchrödingerMETAL
This project presents a METAL framework accelerated 2D Time-independent Schrödinger equation solver.
The equation is solved using the method of finite differences and Lanczos algiorithm for the Hamiltonian.
The project uses metal-cpp to implement GPU acceleration to multiple parts of the solver. The Hamiltonian construction and the Matrix-Vector multiplication is GPU accelerated.
Currently a simply 2D Harmonic Oscillator is presented as potential.

This project was created as an assignment for GPU Programming class at Eötvös Loránd University.
