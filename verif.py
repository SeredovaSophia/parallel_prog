import numpy as np

def read_matrix(filename):
    """Читает матрицу из текстового файла"""
    return np.loadtxt(filename, dtype=int)


def check_multiplication(size):
    """Проверяет умножение для заданного размера"""

    A = read_matrix(f"A_{size}.txt")
    B = read_matrix(f"B_{size}.txt")
    C_cpp = read_matrix(f"C_{size}.txt")

    C_numpy = np.dot(A, B)

    if np.array_equal(C_cpp, C_numpy):
        return True, 0
    else:
        max_diff = np.max(np.abs(C_cpp - C_numpy))
        return False, max_diff


def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]

    print("\n" + "=" * 60)
    print("ПРОВЕРКА РЕЗУЛЬТАТОВ УМНОЖЕНИЯ МАТРИЦ")
    print("Сравнение C++ и Python (NumPy)")
    print("=" * 60 + "\n")

    results = []

    for size in sizes:
        ok, diff = check_multiplication(size)
        if ok:
            status = "СОВПАДАЕТ"
            print(f"{size}x{size}: {status}")
        else:
            status = f"НЕ СОВПАДАЕТ (разница: {diff:.2e})"
            print(f"{size}x{size}: {status}")
        results.append((size, status))

    # Сохраняем в файл
    with open("verify_results.txt", "w", encoding="utf-8") as f:
        f.write("РЕЗУЛЬТАТЫ ВЕРИФИКАЦИИ\n")
        f.write("=" * 50 + "\n")
        f.write("Сравнение C++ и Python (NumPy)\n\n")

        for size, status in results:
            f.write(f"{size}x{size}: {status}\n")

        f.write("\n" + "=" * 50 + "\n")
        all_ok = all("СОВПАДАЕТ" in s for _, s in results)
        if all_ok:
            f.write("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!\n")
        else:
            f.write("ЕСТЬ ОШИБКИ! Проверьте результаты.\n")

    print("\n" + "=" * 60)
    print("Результаты сохранены в verify_results.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()