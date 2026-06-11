from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelValidator:
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(acc)
        print(prec)
        print(rec)
        print(f1)