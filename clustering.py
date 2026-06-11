from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, k=3):
        self.k = k

    def create_clusters(self, X):
        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        return clusters, kmeans