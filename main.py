from data_loader import DataLoader
from preprocessing import Preprocessing
from DataTransformation import DataTransformation
from clustering import Clustering
from tuner import ModelTuner
from prediction_Validation_Insertion import ModelValidator

from sklearn.model_selection import train_test_split

loader = DataLoader("wafer_data.csv")
df = loader.load_data()

pre = Preprocessing(df)
df = pre.handle_missing()
X, y = pre.separate_features()

transformer = DataTransformation()
X_scaled = transformer.fit_transform(X)

cluster = Clustering(k=3)
clusters, _ = cluster.create_clusters(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

tuner = ModelTuner()
model = tuner.tune(X_train, y_train)

validator = ModelValidator()
validator.evaluate(model, X_test, y_test)