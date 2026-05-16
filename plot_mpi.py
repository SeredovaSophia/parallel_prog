import matplotlib.pyplot as plt

def read_results(filename):
    times = {1: [], 2: [], 4: [], 8: []}
    sizes_set = set()

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                size_str = parts[0]
                size = int(size_str.split('x')[0])
                proc = int(parts[1])
                time_val = float(parts[2])
                sizes_set.add(size)
                times[proc].append((size, time_val))

    sizes = sorted(sizes_set)
    times_sorted = {}
    for proc in times:
        time_dict = {size: time_val for size, time_val in times[proc]}
        times_sorted[proc] = [time_dict.get(size, None) for size in sizes]

    return sizes, times_sorted

sizes, times = read_results("all_results.txt")

processes = [1, 2, 4, 8]
colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(12, 8))

for i, proc in enumerate(processes):
    x = [s for s, t in zip(sizes, times[proc]) if t is not None]
    y = [t for t in times[proc] if t is not None]
    if x and y:
        plt.plot(x, y, color=colors[i], linewidth=2, label=f'{proc} processes')

plt.xlabel('Matrix size (n × n)', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Parallel Matrix Multiplication Performance (MPI)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.savefig('mpi_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("Graph saved as mpi_plot.png")