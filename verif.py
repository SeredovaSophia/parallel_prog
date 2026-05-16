import numpy as np

def read_matrix(filename):
    return np.loadtxt(filename, dtype=int)


def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]
    processes = [1, 2, 4, 8]

    print("\n" + "=" * 60)
    print("ПРОВЕРКА РЕЗУЛЬТАТОВ УМНОЖЕНИЯ МАТРИЦ")
    print("Сравнение C++ (MPI) и Python (NumPy)")
    print("=" * 60 + "\n")

    all_ok = True
    results = []

    for size in sizes:
        A = read_matrix(f"A_{size}.txt")
        B = read_matrix(f"B_{size}.txt")
        C_numpy = np.dot(A, B)

        for proc in processes:
            filename = f"C_{size}_{proc}procs.txt"
            try:
                C_cpp = read_matrix(filename)
                if np.array_equal(C_cpp, C_numpy):
                    status = "CORRECT"
                    print(f"{size}x{size} | {proc} processes: CORRECT")
                else:
                    max_diff = np.max(np.abs(C_cpp - C_numpy))
                    status = f"WRONG (diff: {max_diff:.2e})"
                    print(f"{size}x{size} | {proc} processes: WRONG (diff: {max_diff:.2e})")
                    all_ok = False
            except FileNotFoundError:
                status = "FILE NOT FOUND"
                print(f"{size}x{size} | {proc} processes: FILE NOT FOUND")
                all_ok = False
            results.append((size, proc, status))

    with open("verify_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("РЕЗУЛЬТАТЫ ВЕРИФИКАЦИИ\n")
        f.write("Сравнение C++ (MPI) и Python (NumPy)\n")
        f.write("=" * 60 + "\n\n")

        for size, proc, status in results:
            f.write(f"{size}x{size} | {proc} processes: {status}\n")

        f.write("\n" + "=" * 60 + "\n")
        if all_ok:
            f.write("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО\n")
        else:
            f.write("ЕСТЬ ОШИБКИ! ПРОВЕРЬТЕ РЕЗУЛЬТАТЫ\n")
        f.write("=" * 60 + "\n")

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL VERIFICATIONS PASSED")
    else:
        print("SOME VERIFICATIONS FAILED")
    print("=" * 60)
    print("\nРезультаты сохранены в файл verify_results.txt")


if __name__ == "__main__":
    main()