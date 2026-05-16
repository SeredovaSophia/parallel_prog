#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <omp.h>

using namespace std;

class Matrix {
private:
    vector<int> data;
    int n;

public:
    Matrix(int size = 0) : n(size) {
        data.resize(size * size, 0);
    }

    int get(int i, int j) const {
        return data[i * n + j];
    }

    void set(int i, int j, int val) {
        data[i * n + j] = val;
    }

    int size() const { return n; }

    void randomFill() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dist(0, 100);
        for (int i = 0; i < n * n; i++) {
            data[i] = dist(gen);
        }
    }

    Matrix operator*(const Matrix& other) const {
        Matrix result(n);

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += data[i * n + k] * other.data[k * n + j];
                }
                result.data[i * n + j] = sum;
            }
        }
        return result;
    }

    void save(const string& filename) const {
        ofstream file(filename);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                file << data[i * n + j];
                if (j < n - 1) file << " ";
            }
            if (i < n - 1) file << "\n";
        }
        file.close();
    }
};

int main() {
    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
    vector<int> threads_list = { 1, 2, 4, 8 };

    ofstream results_files[4];
    string thread_files[4] = {
        "results_1_thread.txt",
        "results_2_thread.txt",
        "results_4_thread.txt",
        "results_8_thread.txt"
    };

    for (int i = 0; i < 4; i++) {
        results_files[i].open(thread_files[i]);
        results_files[i] << "Matrix size\tTime (seconds)\n";
    }

    cout << "\n" << string(60, '=') << "\n";
    cout << "PARALLEL MATRIX MULTIPLICATION (OpenMP)\n";
    cout << string(60, '=') << "\n\n";

    for (int size : sizes) {
        cout << "Generating matrices " << size << "x" << size << "...\n";

        Matrix A(size), B(size);
        A.randomFill();
        B.randomFill();

        A.save("A_" + to_string(size) + ".txt");
        B.save("B_" + to_string(size) + ".txt");

        cout << "Size: " << size << "x" << size << "\n";
        cout << string(40, '-') << "\n";

        for (int idx = 0; idx < threads_list.size(); idx++) {
            int threads = threads_list[idx];
            omp_set_num_threads(threads);

            cout << "  Threads: " << setw(2) << threads << " ... ";
            cout.flush();

            auto start = chrono::high_resolution_clock::now();
            Matrix C = A * B;
            auto end = chrono::high_resolution_clock::now();

            double seconds = chrono::duration<double>(end - start).count();

            if (threads == 1) {
                C.save("C_" + to_string(size) + ".txt");
            }

            results_files[idx] << size << "x" << size << "\t"
                << fixed << setprecision(5) << seconds << "\n";

            cout << seconds << " sec\n";
        }
        cout << string(40, '-') << "\n\n";
    }

    for (int i = 0; i < 4; i++) {
        results_files[i].close();
    }

    cout << string(60, '=') << "\n";
    cout << "RESULTS SAVED IN FILES:\n";
    for (int i = 0; i < 4; i++) {
        cout << "  - " << thread_files[i] << "\n";
    }
    cout << string(60, '=') << "\n\n";

    return 0;
}