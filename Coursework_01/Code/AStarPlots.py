import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

bins = list(range(1, 27))

bfs_cells = [1720, 231, 485, 650, 658, 384, 1734, 216, 450, 244, 506, 255, 563, 254, 626, 240, 1077, 354, 458, 315, 364, 271, 297, 233, 268, 197]
dfs_cells = [282, 1868, 1868, 1865, 1744, 442, 184, 1868, 1007, 61, 1003, 75, 1799, 91, 1834, 105, 1775, 1565, 1768, 1565, 1651, 1565, 1622, 1564, 1558, 1567]
astar_cells = [1537, 93, 170, 142, 138, 115, 1461, 66, 51, 66, 51, 66, 51, 66, 51, 66, 253, 62, 47, 62, 47, 62, 47, 62, 47, 62]

cumulative_bfs = np.cumsum(bfs_cells)
cumulative_dfs = np.cumsum(dfs_cells)
cumulative_astar = np.cumsum(astar_cells)

sns.set_theme(style="darkgrid", palette="muted")

plt.figure(figsize=(12, 7))

sns.lineplot(x=bins, y=cumulative_astar, label='A* Cumulative Cells', marker='o', linewidth=2)
sns.lineplot(x=bins, y=cumulative_bfs, label='BFS Cumulative Cells', marker='s', linewidth=2)
sns.lineplot(x=bins, y=cumulative_dfs, label='DFS Cumulative Cells', marker='^', linewidth=2)

plt.xlabel('Bin Number', fontsize=12)
plt.ylabel('Cumulative Cells Visited', fontsize=12)
plt.xticks(bins)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()