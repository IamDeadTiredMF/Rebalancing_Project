import pandas as pd
import joblib
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def get_data():
    from src.features import load_features, get_feature_columns
    from src.labels import load_labels
    
    X_raw, y_raw = load_features().sort_index(), load_labels().sort_index()
    f_cols = [c for c in get_feature_columns() if c in X_raw.columns]
    
    idx = X_raw.index.intersection(y_raw.index)
    return X_raw.loc[idx, f_cols].dropna(), y_raw.loc[idx], f_cols

def train():
    X, y, f_cols = get_data()
    print(f"Data Loaded: {len(X)} rows. Positives: {y.sum()}")

    cutoff = pd.to_datetime(config.train_date)
    xt, yt = X[X.index <= cutoff], y[y.index <= cutoff]
    xv, yv = X[X.index > cutoff], y[y.index > cutoff]

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, class_weight='balanced', random_state=42)
    clf.fit(xt, yt)
    
    score = precision_score(yv, clf.predict(xv), zero_division=0)
    print(f"Precision: {score:.4f}")
    out = {"model": clf, "cols": f_cols}
    joblib.dump(out, config.get_model_file_path())

    imp = pd.Series(clf.feature_importances_, index=f_cols).sort_values(ascending=False)
    print("\nTop Features:\n", imp.head(8))

if __name__ == "__main__":
    train()