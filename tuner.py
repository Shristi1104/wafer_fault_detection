from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class ModelTuner:
    def tune(self, X, y):
        models = {
            "RandomForest": (
                RandomForestClassifier(),
                {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
            ),
            "LogisticRegression": (
                LogisticRegression(max_iter=1000),
                {"C": [0.1, 1, 10]}
            )
        }

        best_model = None
        best_score = 0

        for name, (model, params) in models.items():
            grid = GridSearchCV(model, params, cv=3, scoring="accuracy")
            grid.fit(X, y)

            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_

        return best_model