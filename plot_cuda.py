import matplotlib.pyplot as plt
import numpy as np


def read_all_times(filename):
    sizes = []
    block_configs = []
    times = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines) > 0 and 'Size\tBlockX\tBlockY\tTime_ms' in lines[0]:
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    size = int(parts[0])
                    block_x = int(parts[1])
                    block_y = int(parts[2])
                    time_ms = float(parts[3])

                    if block_x == 8 and block_y == 8 and size == 200:
                        continue

                    sizes.append(size)
                    block_configs.append(f"{block_x}x{block_y}")
                    times.append(time_ms)

    return sizes, block_configs, times


def group_by_block(sizes, block_configs, times):
    block_data = {}
    for size, block, time in zip(sizes, block_configs, times):
        if block not in block_data:
            block_data[block] = {'sizes': [], 'times': []}
        block_data[block]['sizes'].append(size)
        block_data[block]['times'].append(time)
    return block_data


sizes, block_configs, times = read_all_times("timing_data.txt")
block_data = group_by_block(sizes, block_configs, times)

colors = {
    '8x8': 'blue',
    '16x8': 'green',
    '16x16': 'orange',
    '32x8': 'red',
    '32x16': 'purple',
    '32x32': 'brown'
}

plt.figure(figsize=(14, 10))

for block in ['8x8', '16x8', '16x16', '32x8', '32x16', '32x32']:
    if block in block_data:
        x = block_data[block]['sizes']
        y = block_data[block]['times']

        if len(x) >= 4:
            coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(coeffs)
            x_smooth = np.linspace(min(x), max(x), 100)
            y_smooth = poly(x_smooth)
            plt.plot(x_smooth, y_smooth, color=colors[block], linewidth=2, label=f'Block {block}')
        else:
            plt.plot(x, y, color=colors[block], linewidth=2, label=f'Block {block}')

plt.xlabel('Matrix size (n × n)', fontsize=12)
plt.ylabel('Time (milliseconds)', fontsize=12)
plt.title('CUDA Matrix Multiplication: Time vs Matrix Size\n(All Block Configurations)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.savefig('cuda_all_blocks_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("Graph saved as cuda_all_blocks_plot.png")

print("\n" + "=" * 60)
print("SUMMARY BY BLOCK CONFIGURATION")
print("=" * 60)
for block in ['8x8', '16x8', '16x16', '32x8', '32x16', '32x32']:
    if block in block_data:
        times_list = block_data[block]['times']
        sizes_list = block_data[block]['sizes']
        print(f"\nBlock {block}:")
        for size, t in zip(sizes_list, times_list):
            print(f"  {size}x{size}: {t:.3f} ms")