import numpy as np
import os


def read_matrix(filename):
    """Reads matrix from text file"""
    if not os.path.exists(filename):
        return None
    return np.loadtxt(filename, dtype=int)


def check_multiplication(size):
    """Checks multiplication for given size"""

    A = read_matrix(f"A_{size}.txt")
    B = read_matrix(f"B_{size}.txt")
    C_cuda = read_matrix(f"C_{size}.txt")

    if A is None or B is None or C_cuda is None:
        return False, "FILE NOT FOUND", 0

    C_numpy = np.dot(A, B)

    if np.array_equal(C_cuda, C_numpy):
        return True, "CORRECT", 0
    else:
        max_diff = np.max(np.abs(C_cuda - C_numpy))
        return False, "WRONG", max_diff


def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]

    print("\n" + "=" * 60)
    print("VERIFICATION OF CUDA MATRIX MULTIPLICATION RESULTS")
    print("Comparison: CUDA vs Python (NumPy)")
    print("=" * 60 + "\n")

    results = []
    all_ok = True

    for size in sizes:
        ok, status, diff = check_multiplication(size)
        if ok:
            print(f"{size}x{size}: {status}")
        else:
            if status == "FILE NOT FOUND":
                print(f"{size}x{size}: {status}")
            else:
                print(f"{size}x{size}: {status} (diff: {diff:.2e})")
            all_ok = False
        results.append((size, status, diff))

    # Save to file
    with open("verify_results_cuda.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("VERIFICATION RESULTS\n")
        f.write("Comparison: CUDA vs Python (NumPy)\n")
        f.write("=" * 60 + "\n\n")

        for size, status, diff in results:
            if status == "WRONG":
                f.write(f"{size}x{size}: {status} (diff: {diff:.2e})\n")
            else:
                f.write(f"{size}x{size}: {status}\n")

        f.write("\n" + "=" * 60 + "\n")
        if all_ok:
            f.write("ALL CHECKS PASSED SUCCESSFULLY!\n")
        else:
            f.write("ERRORS FOUND! Check results.\n")
        f.write("=" * 60 + "\n")

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL VERIFICATIONS PASSED")
    else:
        print("SOME VERIFICATIONS FAILED")
    print("=" * 60)
    print("\nResults saved to verify_results_cuda.txt")


if __name__ == "__main__":
    main()