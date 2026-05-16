import matplotlib.pyplot as plt

threads = [1, 2, 4, 8]
sizes = [200, 400, 800, 1200, 1600, 2000]

time_data = {t: [] for t in threads}

for t in threads:
    filename = f"results_{t}_thread.txt"
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                if line.strip():
                    size_str, time_str = line.strip().split('\t')
                    size = int(size_str.split('x')[0])
                    time_data[t].append(float(time_str))
    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
        exit(1)

plt.figure(figsize=(12, 8))

colors = ['blue', 'green', 'orange', 'red']
labels = ['1 поток', '2 потока', '4 потока', '8 потоков']

for i, t in enumerate(threads):
    plt.plot(sizes, time_data[t], color=colors[i], linewidth=2, label=labels[i])

plt.xlabel('Размер матрицы (n × n)', fontsize=12)
plt.ylabel('Время выполнения (секунды)', fontsize=12)
plt.title('Время умножения матриц (OpenMP)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

for i, t in enumerate(threads):
    for x, y in zip(sizes, time_data[t]):
        if y > 0.1:
            plt.annotate(f'{y:.1f}', (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('plot_time.png', dpi=150, bbox_inches='tight')
plt.show()

print("График сохранён как plot_time.png")