class Preprocessing:
    def __init__(self, df):
        self.df = df

    def handle_missing(self):
        return self.df.fillna(self.df.mean(numeric_only=True))

    def separate_features(self):
        X = self.df.drop("Pass/Fail", axis=1)
        y = self.df["Pass/Fail"]
        y = y.replace(-1, 0)
        return X, y