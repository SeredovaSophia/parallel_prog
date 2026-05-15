#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>

using namespace std;

class Matrix {
private:
    vector<vector<int>> data;
    int n;

public:
    Matrix(int size = 0) : n(size) {
        data.resize(size, vector<int>(size, 0));
    }

    int get(int i, int j) const {
        return data[i][j];
    }

    void set(int i, int j, int val) {
        data[i][j] = val;
    }

    int size() const {
        return n;
    }

    void randomFill() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dist(0, 100);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = dist(gen);
            }
        }
    }

   Matrix operator*(const Matrix& other) const {
        Matrix result(n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

     void save(const string& filename) const {
        ofstream file(filename);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                file << data[i][j];
                if (j < n - 1) file << " ";
            }
            if (i < n - 1) file << "\n";
        }
        file.close();
    }
};

int main() {
    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };

    ofstream results("results.txt");
    results << "Matrix size\tTime (seconds)\n";
   
    for (int size : sizes) {
        cout << "Working with matrices " << size << "x" << size << "... ";

        Matrix A(size);
        Matrix B(size);

        A.randomFill();
        B.randomFill();

        string fileA = "A_" + to_string(size) + ".txt";
        string fileB = "B_" + to_string(size) + ".txt";
        A.save(fileA);
        B.save(fileB);

        auto start = chrono::high_resolution_clock::now();
        Matrix C = A * B;
        auto end = chrono::high_resolution_clock::now();

        double seconds = chrono::duration<double>(end - start).count();

        string fileC = "C_" + to_string(size) + ".txt";
        C.save(fileC);

        results << size << "x" << size << "\t" << fixed << setprecision(5) << seconds << "\n";

        cout << "done (" << seconds << " sec)\n";
    }

    results.close();

    cout << "DONE!\n";
    cout << "Files created: A_200.txt, B_200.txt, C_200.txt, etc.\n";
    cout << "Measurement results in file results.txt\n";
   
    return 0;
}