# Install OpenCV if not installed

import cv2
import os
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, LpBinary, LpStatus, value,lpSum
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def extract_cell_colors(img_rgb, rows, cols):
    height, width, _ = img_rgb.shape
    cell_height, cell_width = height // rows, width // cols
    colors = []
    positions = []

    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * cell_height, (row + 1) * cell_height
            x1, x2 = col * cell_width, (col + 1) * cell_width
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            if cy < height and cx < width:
                colors.append(img_rgb[cy, cx])
                positions.append((row, col))
    return np.array(colors), positions

def find_best_grid(img_rgb, fixed_clusters=10, grid_range=(2, 12)):
    best_score = float('inf')
    best_combo = None
    best_colors = None
    best_positions = None

    for size in range(grid_range[0], grid_range[1] + 1):
        colors, positions = extract_cell_colors(img_rgb, size, size)
        if fixed_clusters >= len(colors):
            continue
        kmeans = KMeans(n_clusters=fixed_clusters, random_state=0, n_init=5).fit(colors)
        if kmeans.inertia_ < best_score:
            best_score = kmeans.inertia_
            best_combo = (size, size)
            best_colors = colors
            best_positions = positions

    return best_colors, best_positions, best_combo

def find_best_clusters(colors, positions, cluster_range=(9, 14)):
    best_score = -1
    best_clusters = None
    best_df = None

    for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
        if n_clusters >= len(colors):
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5).fit(colors)
        labels = kmeans.labels_
        score = silhouette_score(colors, labels)
        if score > best_score:
            best_score = score
            centroids = kmeans.cluster_centers_.astype(int)
            data = [{
                "Row": row,
                "Column": col,
                "Cluster": label,
                "RGB Color": tuple(centroids[label])
            } for (row, col), label in zip(positions, labels)]
            best_df = pd.DataFrame(data)
            best_clusters = n_clusters

    return best_df, best_clusters, best_score

def plot_grid(df, grid_size, n_clusters):
    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(6, 6))
    for _, row in df.iterrows():
        r, c = int(row['Row']), int(row['Column'])
        color = np.array(row['RGB Color']) / 255.0
        rect = plt.Rectangle((c, rows - 1 - r), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols + 1))
    ax.set_yticks(np.arange(0, rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.grid(True, which='both', color='black', linewidth=0.5)
    plt.title(f"Best Grid: {rows}x{cols}, Clusters: {n_clusters}")
    plt.tight_layout()
    
    return fig  # ✅ Return the figure instead of plt.show()


#######


from pulp import LpStatus

from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, LpStatus
import numpy as np

from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, LpStatus
import numpy as np

def solver(n, df):
    # Create the ILP model
    model = LpProblem("Reduced_Queens_With_Clusters", LpMaximize)

    # Decision variables
    x = {(i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i in range(n) for j in range(n)}

    # 1. One queen per row
    for i in range(n):
        model += lpSum(x[i, j] for j in range(n)) == 1

    # 2. One queen per column
    for j in range(n):
        model += lpSum(x[i, j] for i in range(n)) == 1

    # 3. No queens on same diagonals
    for i in range(n - 1):
        for j in range(n - 1):
            model += x[i, j] + x[i + 1, j + 1] <= 1
    for i in range(n - 1, 0, -1):
        for j in range(n - 1, 0, -1):
            model += x[i, j] + x[i - 1, j - 1] <= 1


    # Top-right to bottom-left diagonals (new)
    for i in range(n - 1):
        for j in range(1, n):
            model += x[i, j] + x[i + 1, j - 1] <= 1

    # Bottom-left to top-right diagonals (new)
    for i in range(1, n):
        for j in range(n - 1):
            model += x[i, j] + x[i - 1, j + 1] <= 1


    # 4. One queen per cluster
    for cluster_id in df['Cluster'].unique():
        cluster_cells = df[df['Cluster'] == cluster_id][['Row', 'Column']].values
        model += lpSum(x[i, j] for i, j in cluster_cells) == 1

    # Objective (optional)
    model += lpSum(x[i, j] for i in range(n) for j in range(n))

    # 🔍 SOLVE AND CHECK STATUS
    status = model.solve()

    if LpStatus[status] != "Optimal":
        print("Solver status:", LpStatus[status])
        return None  # ❌ Return None if unsolvable

    # ✅ Extract solution
    solution = {
        (i, j): int(x[i, j].varValue) if x[i, j].varValue is not None else 0
        for i in range(n) for j in range(n)
    }

    solution_grid = np.array([[solution[i, j] for j in range(n)] for i in range(n)])
    return solution_grid



def plot_grid_sol(df, grid_size, n_clusters, solution_grid):
    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(6, 6))
    for _, row in df.iterrows():
        r, c = int(row['Row']), int(row['Column'])
        color = np.array(row['RGB Color']) / 255.0
        rect = plt.Rectangle((c, rows - 1 - r), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols + 1))
    ax.set_yticks(np.arange(0, rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.grid(True, which='both', color='black', linewidth=0.5)
    plt.title(f"Solved Grid: {rows}x{cols}, Clusters: {n_clusters}")
    plt.tight_layout()

    for i in range(len(solution_grid)):
        for j in range(len(solution_grid)):
            if solution_grid[i][j] == 1:
                ax.plot(j + 0.5, rows - 1 - i + 0.5, 'x', color='black', markersize=12, markeredgewidth=2)

    return fig  # ✅ Return the figure
