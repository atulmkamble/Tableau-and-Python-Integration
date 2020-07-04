# Sample Data
x = [6.35, 6.40, 6.65, 8.60, 8.90, 9.00, 9.10]
y = [1.95, 1.95, 2.05, 3.05, 3.05, 3.10, 3.15]

# Test the clustering model on sample data
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
X = np.column_stack([x, y])
X = StandardScaler().fit_transform(X)
db = DBSCAN(eps=0.3, min_samples=3).fit(X)
print(f'Cluster Labels: {db.labels_.tolist()}')