#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <mpi.h>


using namespace std;

class Matrix {
private:
    vector<int> data;
    int n;

public:
    Matrix(int size = 0) : n(size) {
        data.resize(size * size, 0);
    }

    vector<int>& getData() { return data; }
    const vector<int>& getData() const { return data; }

    int get(int i, int j) const {
        return data[i * n + j];
    }

    void set(int i, int j, int val) {
        data[i * n + j] = val;
    }

    int size() const {
        return n;
    }

    void randomFill() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dist(0, 100);

        for (int i = 0; i < n * n; i++) {
            data[i] = dist(gen);
        }
    }

    void save(const string& filename) const {
        ofstream file(filename);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                file << get(i, j);
                if (j < n - 1) file << " ";
            }
            if (i < n - 1) file << "\n";
        }
        file.close();
    }

    bool load(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int val;
                file >> val;
                set(i, j, val);
            }
        }
        return true;
    }

    Matrix parallelMultiply(const Matrix& B) const {
        int procRank, procCount;
        MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
        MPI_Comm_size(MPI_COMM_WORLD, &procCount);

        int matrixSize = n;
        Matrix result(matrixSize);

        vector<int> flatB(matrixSize * matrixSize);
        if (procRank == 0) {
            flatB = B.getData();
        }
        MPI_Bcast(flatB.data(), matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

        int rows_per_proc = matrixSize / procCount;
        int remainder = matrixSize % procCount;

        vector<int> sendcounts(procCount), displs(procCount);
        int offset = 0;
        for (int i = 0; i < procCount; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * matrixSize;
            displs[i] = offset;
            offset += sendcounts[i];
        }

        int local_rows = sendcounts[procRank] / matrixSize;
        vector<int> localA(local_rows * matrixSize);

        MPI_Scatterv(getData().data(), sendcounts.data(), displs.data(), MPI_INT,
            localA.data(), local_rows * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> localC(local_rows * matrixSize, 0);
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < matrixSize; j++) {
                long long sum = 0;
                for (int k = 0; k < matrixSize; k++) {
                    sum += (long long)localA[i * matrixSize + k] * (long long)flatB[k * matrixSize + j];
                }
                localC[i * matrixSize + j] = (int)sum;
            }
        }

        vector<int> recvcounts(procCount), recvdispls(procCount);
        offset = 0;
        for (int i = 0; i < procCount; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = rows * matrixSize;
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }

        if (procRank == 0) {
            result.getData().resize(matrixSize * matrixSize);
        }

        MPI_Gatherv(localC.data(), local_rows * matrixSize, MPI_INT,
            result.getData().data(), recvcounts.data(), recvdispls.data(), MPI_INT,
            0, MPI_COMM_WORLD);

        return result;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int procRank, procCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };

    if (procRank == 0) {
        cout << "\n============================================================" << endl;
        cout << "MPI MATRIX MULTIPLICATION WITH " << procCount << " PROCESSES" << endl;
        cout << "============================================================\n" << endl;
    }

    for (int currentSize : sizes) {
        Matrix A(currentSize);
        Matrix B(currentSize);

        string aFilename = "A_" + to_string(currentSize) + ".txt";
        string bFilename = "B_" + to_string(currentSize) + ".txt";

        if (procRank == 0) {
            ifstream checkA(aFilename);
            if (!checkA.is_open()) {
                cout << "Generating A_" << currentSize << " and B_" << currentSize << "... " << flush;
                A.randomFill();
                B.randomFill();
                A.save(aFilename);
                B.save(bFilename);
                cout << "done" << endl;
            }
            else {
                checkA.close();
                cout << "Loading A_" << currentSize << " and B_" << currentSize << "... " << flush;
                ifstream fileA(aFilename);
                ifstream fileB(bFilename);
                for (int i = 0; i < currentSize; i++) {
                    for (int j = 0; j < currentSize; j++) {
                        int val;
                        fileA >> val;
                        A.set(i, j, val);
                        fileB >> val;
                        B.set(i, j, val);
                    }
                }
                cout << "done" << endl;
            }
        }

        vector<int> flatA(currentSize * currentSize);
        vector<int> flatB(currentSize * currentSize);

        if (procRank == 0) {
            flatA = A.getData();
            flatB = B.getData();
        }

        MPI_Bcast(flatA.data(), currentSize * currentSize, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(flatB.data(), currentSize * currentSize, MPI_INT, 0, MPI_COMM_WORLD);

        Matrix A_all(currentSize);
        Matrix B_all(currentSize);
        A_all.getData() = flatA;
        B_all.getData() = flatB;

        MPI_Barrier(MPI_COMM_WORLD);
        auto timeStart = chrono::high_resolution_clock::now();

        Matrix C = A_all.parallelMultiply(B_all);

        auto timeEnd = chrono::high_resolution_clock::now();
        double elapsedSeconds = chrono::duration<double>(timeEnd - timeStart).count();

        if (procRank == 0) {
            string outputFile = "C_" + to_string(currentSize) + "_" + to_string(procCount) + "procs.txt";
            C.save(outputFile);

            cout << "C_" << currentSize << "_" << procCount << "procs.txt - "
                << fixed << setprecision(5) << elapsedSeconds << " sec" << endl;

            ofstream resultsFile("all_results.txt", ios::app);
            resultsFile << currentSize << "x" << currentSize << "\t"
                << procCount << "\t"
                << fixed << setprecision(5) << elapsedSeconds << "\n";
            resultsFile.close();
        }
    }

    if (procRank == 0) {
        cout << "\n============================================================" << endl;
        cout << "FINISHED WITH " << procCount << " PROCESSES" << endl;
        cout << "============================================================\n" << endl;
    }

    MPI_Finalize();
    return 0;
}