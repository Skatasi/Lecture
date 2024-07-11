import numpy as np
from matplotlib import pyplot as plt

data = np.array([[0,0],[1,0],[2,1],[3,0],[4,2],[5,0],[6,0],[8,0]])
eps = 1
min_pts = 2

# Initialize flags for each point; 0 means unvisited
flags = np.zeros(len(data), dtype=int)

def get_neighbors(point_index):
    neighbors = []
    for i in range(len(data)):
        if np.linalg.norm(data[point_index] - data[i]) < eps:
            neighbors.append(i)
    return neighbors

clusters = []
noise = []
outliers = []

for i in range(len(data)):
    if flags[i] == 0:  # Point has not been visited
        flags[i] = 1  # Mark as visited
        neighbors = get_neighbors(i)
        if len(neighbors) < min_pts:
            noise.append(data[i])
        else:
            cluster = []
            for n in neighbors:
                if flags[n] == 0:
                    flags[n] = 1  # Mark as visited
                    further_neighbors = get_neighbors(n)
                    if len(further_neighbors) >= min_pts:
                        neighbors.extend(further_neighbors)
                if n not in cluster:
                    cluster.append(data[n])
            clusters.append(cluster)

# Calculate outliers based on the score
for cluster in clusters:
    if len(cluster) > 0:
        centroid = np.mean(cluster, axis=0)
        for point in cluster:
            distance = np.linalg.norm(point - centroid)
            score = distance / len(cluster)
            if score > 1.5:
                outliers.append(point)

print("Clusters:", clusters)
print("Noise:", noise)
print("Outliers:", outliers)