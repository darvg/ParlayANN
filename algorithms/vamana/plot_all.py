import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Patterns to extract data
patterns = {
    'detected_points': re.compile(r'Detected (\d+) points'),
    'graph_stats': re.compile(r'Graph has average degree ([\d\.]+) and maximum degree (\d+)'),
    'recall_data': re.compile(
        r'For (\d+)@(\d+) recall = ([\d\.]+), recall 1@(\d+) = ([\d\.]+), QPS = ([\d\.]+), Latency = ([\d\.e-]+), ndcg = ([\d\.]+), max_max_approximation = ([\d\.]+), max_mean_approximation = ([\d\.]+), max_avg_approximation = ([\d\.]+), mean_max_approximation = ([\d\.]+), mean_mean_approximation = ([\d\.]+), mean_avg_approximation = (-?\d+\.\d+|-nan)'
    ),
    'parlay_time': re.compile(r'Parlay time: ([\d\.]+)')
}

# Data storage
parsed_data = defaultdict(lambda: {
    'detected_points': [],
    'graph_stats': [],
    'recall_data': [],
    'parlay_time': None
})
current_block = None

# Function to parse data from a single line
def parse_line(line):
    global current_block
    for key, pattern in patterns.items():
        match = pattern.search(line)
        if match:
            if key == 'detected_points':
                parsed_data[current_block]['detected_points'].append(int(match.group(1)))
            elif key == 'graph_stats':
                parsed_data[current_block]['graph_stats'].append({
                    'average_degree': float(match.group(1)),
                    'maximum_degree': int(match.group(2))
                })
            elif key == 'recall_data':
                parsed_data[current_block]['recall_data'].append({
                    'Q': int(match.group(1)),
                    'recall': float(match.group(3)),
                    'recall_1': float(match.group(4)),
                    'QPS': float(match.group(5)),
                    'Latency': float(match.group(6)),
                    'ndcg': float(match.group(7)),
                    'max_max_approximation': float(match.group(8)),
                    'max_mean_approximation': float(match.group(9)),
                    'max_avg_approximation': float(match.group(10)),
                    'mean_max_approximation': float(match.group(11)),
                    'mean_mean_approximation': float(match.group(12)),
                    'mean_avg_approximation': float(match.group(13))
                })
            elif key == 'parlay_time':
                parsed_data[current_block]['parlay_time'] = float(match.group(1))

# Read and parse data from the file
file_path = 'data.txt'  # Replace with your actual file path
with open(file_path, 'r') as file:
    for line in file:
        if 'GREP_ME' in line:
            current_block = line.strip()  # Use the identifier as the block key
        parse_line(line)

# Plotting function
def plot_comparison(data1, data2, label1, label2):
    metrics = [
        'max_max_approximation', 'max_mean_approximation', 'max_avg_approximation',
        'mean_max_approximation', 'mean_mean_approximation', 'mean_avg_approximation',
        'recall'
    ]
    
    plt.figure(figsize=(14, 8))
    
    for metric in metrics:
        qps1 = [d['QPS'] for d in data1['recall_data'] if metric in d]
        values1 = [d[metric] for d in data1['recall_data'] if metric in d]
        
        qps2 = [d['QPS'] for d in data2['recall_data'] if metric in d]
        values2 = [d[metric] for d in data2['recall_data'] if metric in d]
        
        plt.plot(qps1, values1, marker='o', label=f'{label1} - {metric}')
        plt.plot(qps2, values2, marker='s', label=f'{label2} - {metric}')

    plt.xlabel('QPS')
    plt.ylabel('Metrics')
    plt.title(f'Comparison of Metrics for {label1} vs {label2}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Extracting data for sorted and unsorted
sorted_data = parsed_data.get('GREP_ME spacev1_sorted') or parsed_data.get('GREP_ME bigann_sorted')
unsorted_data = parsed_data.get('GREP_ME spacev1_unsorted') or parsed_data.get('GREP_ME bigann_unsorted')

# Plotting the comparison
plot_comparison(unsorted_data, sorted_data, 'Unsorted', 'Sorted')
