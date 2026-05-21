#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>

using namespace std;

__global__ void multiplyKernel(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int total = 0;
        for (int k = 0; k < N; ++k) {
            total += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = total;
    }
}

class Matrix {
private:
    vector<int> data;
    int size;

public:
    Matrix(int n = 0) : size(n) {
        data.resize(n * n, 0);
    }

    vector<int>& getData() { return data; }
    const vector<int>& getData() const { return data; }

    int get(int i, int j) const {
        return data[i * size + j];
    }

    void set(int i, int j, int val) {
        data[i * size + j] = val;
    }

    int getSize() const {
        return size;
    }

    void randomFill() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dist(0, 100);
        
        for (int i = 0; i < size * size; i++) {
            data[i] = dist(gen);
        }
    }

    void saveToFile(const string& filename) const {
        ofstream file(filename);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                file << get(i, j);
                if (j < size - 1) file << " ";
            }
            if (i < size - 1) file << "\n";
        }
        file.close();
    }

    bool loadFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int val;
                file >> val;
                set(i, j, val);
            }
        }
        return true;
    }

    Matrix gpuMultiply(const Matrix& other, int bx, int by, float& gpuTime) const {
        Matrix result(size);
        
        int bytesCount = size * size * sizeof(int);
        
        int* devA = nullptr;
        int* devB = nullptr;
        int* devC = nullptr;
        
        cudaMalloc(&devA, bytesCount);
        cudaMalloc(&devB, bytesCount);
        cudaMalloc(&devC, bytesCount);
        
        cudaMemcpy(devA, data.data(), bytesCount, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, other.data.data(), bytesCount, cudaMemcpyHostToDevice);
        
        dim3 blockSize(bx, by);
        dim3 gridSize((size + bx - 1) / bx, (size + by - 1) / by);
        
        cudaEvent_t timeStart, timeStop;
        cudaEventCreate(&timeStart);
        cudaEventCreate(&timeStop);
        
        cudaEventRecord(timeStart);
        multiplyKernel<<<gridSize, blockSize>>>(devA, devB, devC, size);
        cudaEventRecord(timeStop);
        cudaEventSynchronize(timeStop);
        
        cudaEventElapsedTime(&gpuTime, timeStart, timeStop);
        
        cudaMemcpy(result.data.data(), devC, bytesCount, cudaMemcpyDeviceToHost);
        
        cudaEventDestroy(timeStart);
        cudaEventDestroy(timeStop);
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
        
        return result;
    }
};

int main() {
    vector<int> dimensions = {200, 400, 800, 1200, 1600, 2000};
    
    vector<pair<int, int>> blockVariants = {
        {8, 8}, {16, 8}, {16, 16}, {32, 8}, {32, 16}, {32, 32}
    };
    
    int gpuCount;
    cudaGetDeviceCount(&gpuCount);
    
    cout << "\n============================================================" << endl;
    cout << "GPU MATRIX MULTIPLICATION" << endl;
    cout << "============================================================" << endl;
    
    if (gpuCount > 0) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cout << "GPU: " << deviceProp.name << endl;
        cout << "Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "============================================================\n" << endl;
    }
    
    ofstream outputLog("cuda_results.txt");
    outputLog << "CUDA MATRIX MULTIPLICATION RESULTS\n";
    outputLog << "==================================================\n\n";
    
    ofstream timingData("timing_data.txt");
    timingData << "Size\tBlockX\tBlockY\tTime_ms\n";
    
    for (int n : dimensions) {
        cout << "Processing " << n << "x" << n << "... " << flush;
        
        Matrix matA(n);
        Matrix matB(n);
        
        string fileA = "A_" + to_string(n) + ".txt";
        string fileB = "B_" + to_string(n) + ".txt";
        
        ifstream checkA(fileA);
        if (!checkA.is_open()) {
            matA.randomFill();
            matB.randomFill();
            matA.saveToFile(fileA);
            matB.saveToFile(fileB);
        } else {
            checkA.close();
            matA.loadFromFile(fileA);
            matB.loadFromFile(fileB);
        }
        
        outputLog << "Size: " << n << "x" << n << "\n";
        outputLog << "----------------------------------------\n";
        
        float bestDuration = 9999999.0f;
        int bestBlockX = 0, bestBlockY = 0;
        
        for (auto& block : blockVariants) {
            float kernelTime = 0.0f;
            
            cudaEvent_t totalStart, totalEnd;
            cudaEventCreate(&totalStart);
            cudaEventCreate(&totalEnd);
            
            cudaEventRecord(totalStart);
            Matrix matC = matA.gpuMultiply(matB, block.first, block.second, kernelTime);
            cudaEventRecord(totalEnd);
            cudaEventSynchronize(totalEnd);
            
            float totalTime = 0.0f;
            cudaEventElapsedTime(&totalTime, totalStart, totalEnd);
            
            cudaEventDestroy(totalStart);
            cudaEventDestroy(totalEnd);
            
            if (block == blockVariants[0]) {
                string fileC = "C_" + to_string(n) + ".txt";
                matC.saveToFile(fileC);
            }
            
            cout << "\n  " << block.first << "x" << block.second 
                 << ": " << fixed << setprecision(3) << totalTime << " ms";
            
            outputLog << "  Block " << block.first << "x" << block.second 
                      << ": " << fixed << setprecision(3) << totalTime << " ms\n";
            
            timingData << n << "\t" << block.first << "\t" 
                       << block.second << "\t" << totalTime << "\n";
            
            if (totalTime < bestDuration) {
                bestDuration = totalTime;
                bestBlockX = block.first;
                bestBlockY = block.second;
            }
        }
        
        outputLog << "  BEST: " << bestBlockX << "x" << bestBlockY 
                  << " -> " << fixed << setprecision(3) << bestDuration << " ms\n\n";
        
        cout << "\n  BEST: " << bestBlockX << "x" << bestBlockY 
             << " (" << bestDuration << " ms)" << endl;
    }
    
    outputLog << "==================================================\n";
    outputLog.close();
    timingData.close();
    
    cout << "\n============================================================" << endl;
    cout << "RESULTS SAVED TO:" << endl;
    cout << "  - cuda_results.txt" << endl;
    cout << "  - timing_data.txt" << endl;
    cout << "============================================================\n" << endl;
    
    return 0;
}
