import matplotlib.pyplot as plt

# Данные из results.txt (нужно ввести вручную или прочитать из файла)
# Но проще ввести вручную после запуска main.cpp
sizes = [200, 400, 800, 1200, 1600, 2000]

# !!! ЗДЕСЬ НУЖНО ВСТАВИТЬ СВОИ ЗНАЧЕНИЯ ИЗ results.txt !!!
# Например:
times = [0.15067, 0.93180, 8.50731, 30.64400, 78.31968, 167.00958]

# Теоретическая кривая O(n³) - нормируем по первому значению
theoretical = [times[0] * (sizes[i] / sizes[0])**3 for i in range(len(sizes))]

# Строим график
plt.figure(figsize=(12, 7))

# Реальные замеры
plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8,
         label='Реальные замеры (C++)', color='blue')

# Теоретическая кривая
plt.plot(sizes, theoretical, 'r--', linewidth=2,
         label='Теоретическая сложность O(n³)', color='red')

# Оформление
plt.xlabel('Размер матрицы (n × n)', fontsize=12)
plt.ylabel('Время выполнения (секунды)', fontsize=12)
plt.title('Зависимость времени умножения матриц от размера', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Подписываем точки
for x, y in zip(sizes, times):
    plt.annotate(f'{y:.2f} с', (x, y), xytext=(5, 5),
                 textcoords='offset points', fontsize=9)

# Сохраняем график
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("График сохранён как plot.png")