//
//  main.cpp
//  Schrodinger
//
//  Created by Farádi Bálint on 2025. 05. 26..
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>
#include <armadillo>
#include <cmath>
#include <algorithm>
#include <lambda_lanczos/lambda_lanczos.hpp>
#include <random>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <fstream>

const int N = 200;
const double L = 5.0;            // Tartomány [-L, L] x [-L, L]
const double h = 2 * L / (N - 1); // Rácslépésköz
const double omega = 1.0;
const int matrixSize = N * N;

enum class LanczosError
{
    MetalNotSupported,
    CommandQueueCreationFailed,
    LibraryCreationFailed,
    FunctionNotFound,
    InvalidDimensions,
    BufferCreationFailed,
    CommandBufferCreationFailed,
    PipelineStateCreationFailed
};

struct LanczosResult
{
    std::vector<std::vector<float>> tridiagonalMatrix;
    std::vector<std::vector<float>> lanczosVectors;
    std::vector<float> alphas;
    std::vector<float> betas;
    int iterations;
    double executionTime;
};

class LanczosMetalSolver
{
private:
    MTL::Device *device;
    MTL::CommandQueue *commandQueue;
    MTL::Library *library;
    MTL::ComputePipelineState *matVecPipeline;
    MTL::ComputePipelineState *vectorOpsPipeline;
    MTL::ComputePipelineState *reductionPipeline;

    NS::AutoreleasePool *autoreleasePool;

    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;

public:
    LanczosMetalSolver()
    {
        autoreleasePool = NS::AutoreleasePool::alloc()->init();

        // Initialize Metal device
        device = MTL::CreateSystemDefaultDevice();
        if (!device)
        {
            throw std::runtime_error("Metal not supported on this device");
        }

        // Create command queue
        commandQueue = device->newCommandQueue();
        if (!commandQueue)
        {
            throw std::runtime_error("Failed to create command queue");
        }

        // Create Metal library from source
        library = device->newDefaultLibrary();


        // Create compute pipeline states
        createPipelineStates();

        std::cout << "Metal device initialized: " << device->name()->utf8String() << std::endl;
    }

    ~LanczosMetalSolver()
    {
        // cleanup();
    }

    void cleanup()
    {
        if (matVecPipeline)
        {
            matVecPipeline->release();
            matVecPipeline = nullptr;
        }
        if (vectorOpsPipeline)
        {
            vectorOpsPipeline->release();
            vectorOpsPipeline = nullptr;
        }
        if (reductionPipeline)
        {
            reductionPipeline->release();
            reductionPipeline = nullptr;
        }
        if (library)
        {
            library->release();
            library = nullptr;
        }
        if (commandQueue)
        {
            commandQueue->release();
            commandQueue = nullptr;
        }
        if (device)
        {
            device->release();
            device = nullptr;
        }
        if (autoreleasePool)
        {
            autoreleasePool->release();
            autoreleasePool = nullptr;
        }
    }

    LanczosResult solveLanczos(MTL::Buffer *diagonal,
                               MTL::Buffer *offDiagonal,
                               int maxIterations,
                               float tolerance)
    {

        startTime = std::chrono::high_resolution_clock::now();
        const int n = (int)diagonal->length() / sizeof(float);

        // buffers for matrix data
        MTL::Buffer *diagonalBuffer = diagonal;
        MTL::Buffer *offDiagonalBuffer = offDiagonal;

        // random starting vector
        std::vector<float> startVector = generateRandomVector(n);
        normalizeVector(startVector);

        // Working buffers
        MTL::Buffer *currentVectorBuffer = createBuffer(startVector.data(), n * sizeof(float));
        MTL::Buffer *previousVectorBuffer = createBuffer(n * sizeof(float));
        MTL::Buffer *resultBuffer = createBuffer(n * sizeof(float));
        MTL::Buffer *tempBuffer = createBuffer(n * sizeof(float));

        // Algorithm state
        std::vector<std::vector<float>> lanczosVectors;
        std::vector<float> alphas;
        std::vector<float> betas = {0.0f}; // β₁ = 0

        lanczosVectors.push_back(startVector);

        std::vector<float> currentVector = startVector;
        std::vector<float> previousVector(n, 0.0f);

        int iteration = 0;
        for (iteration = 0; iteration < maxIterations; ++iteration)
        {
            // Step 1: w' = H * v_j (Matrix-vector multiplication)
            std::vector<float> wPrime = performMatrixVectorMultiply(diagonalBuffer, offDiagonalBuffer, currentVector, n);

            // Step 2: α_j = w'ᵀ * v_j
            float alpha = computeDotProduct(wPrime, currentVector);
            alphas.push_back(alpha);

            // Step 3: w = w' - α_j * v_j - β_j * v_{j-1}
            for (int i = 0; i < n; ++i)
            {
                wPrime[i] = wPrime[i] - alpha * currentVector[i] - betas[iteration] * previousVector[i];
            }

            // Step 4: β_{j+1} = ||w||
            float beta = computeVectorNorm(wPrime);

            betas.push_back(beta);

            // Step 5: v_{j+1} = w / β_{j+1}
            std::vector<float> nextVector(n);
            for (int i = 0; i < n; ++i)
            {
                nextVector[i] = wPrime[i] / beta;
            }

            performReorthogonalization(nextVector, lanczosVectors);

            // Update vectors for next iteration
            previousVector = currentVector;
            currentVector = nextVector;
            lanczosVectors.push_back(nextVector);

            // Log
            if ((iteration + 1) % 10 == 0)
            {
                std::cout << "Completed " << (iteration + 1) << " iterations, current β = " << beta << std::endl;
            }
        }

        // Clean up Metal buffers
        diagonalBuffer->release();
        offDiagonalBuffer->release();
        currentVectorBuffer->release();
        previousVectorBuffer->release();
        resultBuffer->release();
        tempBuffer->release();

        endTime = std::chrono::high_resolution_clock::now();
        double executionTime = std::chrono::duration<double>(endTime - startTime).count();

        // Construct tridiagonal matrix
        auto tridiagonalMatrix = constructTridiagonalMatrix(alphas, std::vector<float>(betas.begin() + 1, betas.end()));

        std::cout << "Lanczos algorithm completed:" << std::endl;
        std::cout << "  Iterations: " << (iteration + 1) << std::endl;
        std::cout << "  Execution time: " << executionTime << " seconds" << std::endl;
        std::cout << "  Final tridiagonal matrix size: " << alphas.size() << "x" << alphas.size() << std::endl;

        return LanczosResult{tridiagonalMatrix, lanczosVectors, alphas, std::vector<float>(betas.begin() + 1, betas.end()), iteration + 1, executionTime};
    }

private:
    void createPipelineStates()
    {
        MTL::Function *matVecFunction = library->newFunction(NS::String::string("hamiltonian2D_matvec", NS::UTF8StringEncoding));

        if (!matVecFunction)
        {
            throw std::runtime_error("Failed to find hamiltonian_matvec function");
        }

        NS::Error *error = nullptr;
        matVecPipeline = device->newComputePipelineState(matVecFunction, &error);
        matVecFunction->release();

        if (!matVecPipeline)
        {
            if (error)
            {
                std::cerr << "Failed to create matrix-vector pipeline: " << error->localizedDescription()->utf8String() << std::endl;
                error->release();
            }
            throw std::runtime_error("Failed to create matrix-vector pipeline");
        }

        // Create vector operations pipeline
        MTL::Function *vectorOpsFunction = library->newFunction(NS::String::string("vector_operations", NS::UTF8StringEncoding));

        if (!vectorOpsFunction)
        {
            throw std::runtime_error("Failed to find vector_operations function");
        }

        vectorOpsPipeline = device->newComputePipelineState(vectorOpsFunction, &error);
        vectorOpsFunction->release();

        if (!vectorOpsPipeline)
        {
            if (error)
            {
                std::cerr << "Failed to create vector operations pipeline: " << error->localizedDescription()->utf8String() << std::endl;
                error->release();
            }
            throw std::runtime_error("Failed to create vector operations pipeline");
        }

        // Create reduction pipeline
        MTL::Function *reductionFunction = library->newFunction(NS::String::string("reduction_sum", NS::UTF8StringEncoding));

        if (!reductionFunction)
        {
            throw std::runtime_error("Failed to find reduction_sum function");
        }

        reductionPipeline = device->newComputePipelineState(reductionFunction, &error);
        reductionFunction->release();

        if (!reductionPipeline)
        {
            if (error)
            {
                std::cerr << "Failed to create reduction pipeline: " << error->localizedDescription()->utf8String() << std::endl;
                error->release();
            }
            throw std::runtime_error("Failed to create reduction pipeline");
        }
    }

    MTL::Buffer *createBuffer(const void *data, size_t size)
    {
        MTL::Buffer *buffer = device->newBuffer(size, MTL::ResourceStorageModeShared);
        if (!buffer)
        {
            throw std::runtime_error("Failed to create Metal buffer");
        }

        if (data)
        {
            memcpy(buffer->contents(), data, size);
        }
        return buffer;
    }

    MTL::Buffer *createBuffer(size_t size)
    {
        return createBuffer(nullptr, size);
    }

    std::vector<float> generateRandomVector(int size)
    {
        std::vector<float> vector(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < size; ++i)
        {
            vector[i] = dis(gen);
        }

        return vector;
    }

    void normalizeVector(std::vector<float> &vector)
    {
        float norm = 0.0f;
        for (float val : vector)
        {
            norm += val * val;
        }
        norm = std::sqrt(norm);

        if (norm > 0.0f)
        {
            for (float &val : vector)
            {
                val /= norm;
            }
        }
    }

    std::vector<float> performMatrixVectorMultiply(MTL::Buffer *diagonalBuffer,
                                                   MTL::Buffer *offDiagonalBuffer,
                                                   const std::vector<float> &vector,
                                                   int size)
    {

        MTL::Buffer *vectorBuffer = createBuffer(vector.data(), size * sizeof(float));
        MTL::Buffer *resultBuffer = createBuffer(size * sizeof(float));

        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(matVecPipeline);
        encoder->setBuffer(diagonalBuffer, 0, 0);
        encoder->setBuffer(offDiagonalBuffer, 0, 1);
        encoder->setBuffer(vectorBuffer, 0, 2);
        encoder->setBuffer(resultBuffer, 0, 3);
        encoder->setBytes(&N, sizeof(int), 4);

        MTL::Size threadsPerGroup = MTL::Size::Make(std::min(static_cast<NS::UInteger>(256), matVecPipeline->maxTotalThreadsPerThreadgroup()), 1, 1);
        MTL::Size gridSize = MTL::Size(N, N, 1);

        encoder->dispatchThreadgroups(gridSize, threadsPerGroup);
        encoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        std::vector<float> result(size);
        memcpy(result.data(), resultBuffer->contents(), size * sizeof(float));

        vectorBuffer->release();
        resultBuffer->release();
        return result;
    }

    float computeDotProduct(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
        {
            throw std::runtime_error("Vector sizes must match for dot product");
        }

        float result = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            result += a[i] * b[i];
        }

        return result;
    }

    float computeVectorNorm(const std::vector<float> &vector)
    {
        float norm = 0.0f;
        for (float val : vector)
        {
            norm += val * val;
        }
        return std::sqrt(norm);
    }

    void performReorthogonalization(std::vector<float> &vector,
                                    const std::vector<std::vector<float>> &previousVectors)
    {
        // Modified Gram-Schmidt reorthogonalization
        for (const auto &prevVector : previousVectors)
        {
            if (prevVector.size() != vector.size())
                continue;

            float dotProduct = computeDotProduct(vector, prevVector);
            for (size_t i = 0; i < vector.size(); ++i)
            {
                vector[i] -= dotProduct * prevVector[i];
            }
        }

        // Renormalize
        float norm = computeVectorNorm(vector);
        if (norm > 0.0f)
        {
            for (float &val : vector)
            {
                val /= norm;
            }
        }
    }

    std::vector<std::vector<float>> constructTridiagonalMatrix(const std::vector<float> &alphas,
                                                               const std::vector<float> &betas)
    {
        int n = static_cast<int>(alphas.size());
        std::vector<std::vector<float>> matrix(n, std::vector<float>(n, 0.0f));

        // Fill diagonal
        for (int i = 0; i < n; ++i)
        {
            matrix[i][i] = alphas[i];
        }

        // Fill off-diagonals
        for (int i = 0; i < std::min(n - 1, static_cast<int>(betas.size())); ++i)
        {
            matrix[i][i + 1] = betas[i];
            matrix[i + 1][i] = betas[i];
        }

        return matrix;
    }
};

class TridiagonalEigenSolver
{
private:
    arma::vec alphas;
    arma::vec betas;
    arma::mat tridiagonal_matrix;

public:
    TridiagonalEigenSolver(const std::vector<double> &alpha_coeffs, const std::vector<double> &beta_coeffs)
    {
        alphas = arma::conv_to<arma::vec>::from(alpha_coeffs);
        betas = arma::conv_to<arma::vec>::from(beta_coeffs);
    }

    // Construct tridiagonal matrix from Lanczos coefficients
    void constructTridiagonalMatrix()
    {
        int n = alphas.n_elem;
        tridiagonal_matrix = arma::mat(n, n, arma::fill::zeros);
        tridiagonal_matrix.diag() = alphas;
        if (betas.n_elem > 0)
        {
            tridiagonal_matrix.diag(1) = betas;
            tridiagonal_matrix.diag(-1) = betas;
        }
    }

    std::pair<arma::vec, arma::mat> solveEigenvalues()
    {
        if (tridiagonal_matrix.is_empty())
        {
            throw std::runtime_error("Tridiagonal matrix not constructed. Call constructTridiagonalMatrix() first.");
        }

        arma::vec eigenvalues;
        arma::mat eigenvectors;

        bool success = eig_sym(eigenvalues, eigenvectors, tridiagonal_matrix);

        if (!success)
        {
            throw std::runtime_error("Eigenvalue decomposition failed");
        }

        std::cout << "Successfully computed eigenvalues using Armadillo's eig_sym" << std::endl;
        //eigenvalues = sort(eigenvalues);

        return {eigenvalues, eigenvectors};
    }
};

float *constructHarmonicPotentialGPU(int N, double omega, double L)
{
    float *V = new float[N * N];

    for (int i = 0; i < N; i++)
    {
        double x = -L + i * h; // x from -L to L
        for (int j = 0; j < N; j++)
        {
            double y = -L + j * h; // y from -L to L
            double rSq = x * x + y * y;
            V[i * N + j] = static_cast<float>(0.5 * omega * omega * rSq);
        }
    }
    return V;
}
// CPU
void constructHamiltonian(
    int N, double omega, double L,
    std::vector<double> &diag, std::vector<double> &offDiag)
{
    
    const double kinetic_coeff = -1.0 / (2.0 * h * h);

    diag.resize(matrixSize);
    offDiag.resize(5 * matrixSize, 0.0);

    std::vector<double> x(N), y(N);
    for (int i = 0; i < N; ++i)
    {
        x[i] = -L + i * h;
        y[i] = -L + i * h;
    }

    // Build Hamiltonian
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            const int idx = i * N + j;
            const int base = 5 * idx;

            const double V = 0.5 * omega * omega * (x[i] * x[i] + y[j] * y[j]);

            diag[idx] = 4.0 * (-kinetic_coeff) + V;

            if (j > 0)
                offDiag[base + 0] = kinetic_coeff;
            if (j < N - 1)
                offDiag[base + 1] = kinetic_coeff;
            if (i > 0)
                offDiag[base + 2] = kinetic_coeff;
            if (i < N - 1)
                offDiag[base + 3] = kinetic_coeff;
        }
    }
}

int main()
{
    try
    {
        // 1. Metal inicializálása
        MTL::Device *device = MTL::CreateSystemDefaultDevice();
        MTL::CommandQueue *commandQueue = device->newCommandQueue();

        // 2. Rácsparaméterek
        const float h = static_cast<float>(2 * L / (N - 1));

        // 3. Pufferek létrehozása
        MTL::Buffer *diagBuffer = device->newBuffer(matrixSize * sizeof(float), MTL::ResourceStorageModeShared);
        MTL::Buffer *offDiagBuffer = device->newBuffer(5 * matrixSize * sizeof(float), MTL::ResourceStorageModeShared);

        // 4. Potenciál inicializálása
        float *V_gpu = constructHarmonicPotentialGPU(N, omega, L);
        MTL::Buffer *VBuffer = device->newBuffer(V_gpu, matrixSize * sizeof(float), MTL::ResourceStorageModeManaged);

        // 5. Kernel betöltése
        NS::Error *error = nullptr;
        MTL::Library *library = device->newDefaultLibrary();
        MTL::Function *kernelFunction = library->newFunction(NS::String::string("calculate_hamiltonian", NS::UTF8StringEncoding));
        MTL::ComputePipelineState *pipeline = device->newComputePipelineState(kernelFunction, &error);

        // 6. Kernel futtatása (2D grid!)

        auto t_ham_start = std::chrono::high_resolution_clock::now();
        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(diagBuffer, 0, 0);
        encoder->setBuffer(offDiagBuffer, 0, 1);
        encoder->setBuffer(VBuffer, 0, 2);
        encoder->setBytes(&h, sizeof(float), 3);
        encoder->setBytes(&N, sizeof(int), 4);

        // 2D grid: N x N pont
        MTL::Size gridSize = MTL::Size(N, N, 1);
        MTL::Size threadgroupSize = MTL::Size(32, 8, 1);

        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        auto t_ham_end = std::chrono::high_resolution_clock::now();
        std::cout << "[TIMER] Hamiltonian construction finished in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_ham_end - t_ham_start).count()
                  << " ms" << std::endl;

        // Creating Lanczos solver
        LanczosMetalSolver solver;

        auto result1 = solver.solveLanczos(diagBuffer, offDiagBuffer, 1200, 1e-8f);

        std::cout << "=== Armadillo-Based Tridiagonal Eigenvalue Solver ===" << std::endl;
        std::vector<double> alphas(result1.alphas.begin(), result1.alphas.end());
        std::vector<double> betas(result1.betas.begin(), result1.betas.end());
        betas.resize(alphas.size() - 1);

        TridiagonalEigenSolver solverr(alphas, betas);
        solverr.constructTridiagonalMatrix();
        // Solve for eigenvalues
        std::cout << "\nSolving eigenvalue problem..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        auto  [eigenvalues, eigenvectors] = solverr.solveEigenvalues();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::cout << "Eigenvalue computation completed in " << duration.count() << " microseconds" << std::endl;
        
        // Print eigenvalues

        std::cout << std::fixed << std::setprecision(8);
        for (arma::uword i = 0; i < 5; ++i)
        {
            std::cout << "λ[" << std::setw(2) << i << "] = " << std::setw(12) << eigenvalues(i) << std::endl;
        }
        std::vector<float> psi0(N*N, 0.0f);
        for (size_t j = 0; j < result1.lanczosVectors.size()-1; ++j) {
            float coeff = eigenvectors(j, 2);
            const std::vector<float>& vj = result1.lanczosVectors[j];
            for (int i = 0; i < N*N; ++i) {
                psi0[i] += coeff * vj[i];
            }
        }
        
        std::ofstream out("/tmp/psi2.csv");
        if (!out) {
            std::cerr << "Failed to open psi0.csv for writing!" << std::endl;
        }
        for (size_t i = 0; i < psi0.size(); ++i) {
            out << psi0[i] << "\n";
        }
        out.close();
    }
    catch (const std::exception &e)
    {
        arma::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    

    std::cout << "CPU" << std::endl;
    {
        
        std::vector<double> diag;
        std::vector<double> offDiag;
        auto t_ham_start = std::chrono::high_resolution_clock::now();

        constructHamiltonian(N, omega, L, diag, offDiag);
        auto t_ham_end = std::chrono::high_resolution_clock::now();
        std::cout << "[TIMER] Hamiltonian construction finished in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_ham_end - t_ham_start).count()
                  << " ms" << std::endl;
        auto SMV_implementation = [&](const std::vector<float> &in, std::vector<float> &out)
        {
            out.resize(N * N);
            std::fill(out.begin(), out.end(), 0.0);

            for (int idx = 0; idx < N * N; idx++)
            {
                // Get 2D grid coordinates
                const int i = idx / N;
                const int j = idx % N;
                const int base = 5 * idx;
                //diagonal
                out[idx] = diag[idx] * in[idx];

                if (j > 0)
                    out[idx] += offDiag[base + 0] * in[idx - 1]; // Left
                if (j < N - 1)
                    out[idx] += offDiag[base + 1] * in[idx + 1]; // Right
                if (i > 0)
                    out[idx] += offDiag[base + 2] * in[idx - N]; // Top
                if (i < N - 1)
                    out[idx] += offDiag[base + 3] * in[idx + N]; // Bottom
            }
        };

        lambda_lanczos::LambdaLanczos<float> engine(SMV_implementation, N * N, false, 5);

        auto t_eig_start = std::chrono::high_resolution_clock::now();
        auto [eigenvalues, eigenvectors] = engine.run();
        auto t_eig_end = std::chrono::high_resolution_clock::now();

        std::cout << "[TIMER] Eigenvalue computation finished in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_eig_end - t_eig_start).count()
                  << " ms" << std::endl;

        // Print result
        for (int i = 0; i < eigenvalues.size(); ++i)
        {
            std::cout << "Eigenvalue: " << std::setprecision(16) << eigenvalues[i] << std::endl;
        }
    }

    return 0;
}
