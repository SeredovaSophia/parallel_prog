import matplotlib.pyplot as plt

sizes = [200, 400, 800, 1200, 1600, 2000]
times = [0.15067, 0.93180, 8.50731, 30.64400, 78.31968, 167.00958]

theoretical = [times[0] * (sizes[i] / sizes[0])**3 for i in range(len(sizes))]

plt.figure(figsize=(12, 7))

plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8,
         label='Реальные замеры (C++)', color='blue')

plt.plot(sizes, theoretical, 'r--', linewidth=2,
         label='Теоретическая сложность O(n³)', color='red')

plt.xlabel('Размер матрицы (n × n)', fontsize=12)
plt.ylabel('Время выполнения (секунды)', fontsize=12)
plt.title('Зависимость времени умножения матриц от размера', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

for x, y in zip(sizes, times):
    plt.annotate(f'{y:.2f} с', (x, y), xytext=(5, 5),
                 textcoords='offset points', fontsize=9)

plt.savefig('plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("График сохранён как plot.png")
