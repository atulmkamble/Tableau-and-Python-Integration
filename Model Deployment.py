# Import tabpy client for deployment
from tabpy.tabpy_tools.client import Client

# Server URL (This would be the host and port on which you are running the TabPy server)
client = Client('http://localhost:9004/')

# Define the function tested above
def clustering(x, y):
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    X = np.column_stack([x, y])
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=3).fit(X)
    return db.labels_.tolist()

# Deploy the model to TabPy server
# Add Override = True if you are deploying the model again
client.deploy('clustering', clustering,
              'Returns cluster Ids for each data point specified by the pairs in x and y')

"""
Check if the model is model is deployed on the TabPy server at the URL below:
Server URL (This would be the host and port on which you are running the TabPy server):
http://localhost:9004/endpoints
"""
# Sample Data
x = [6.35, 6.40, 6.65, 8.60, 8.90, 9.00, 9.10]
y = [1.95, 1.95, 2.05, 3.05, 3.05, 3.10, 3.15]

# Test the deployed model
print(client.query('clustering', x, y))

# To delete the deployed model from TabPy server
# client.remove('clustering')