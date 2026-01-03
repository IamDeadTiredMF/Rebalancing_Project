import pandas as pd
import joblib
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def get_data():
    from src.features import load_features, get_feature_columns
    from src.labels import load_labels

    X_raw, y_raw = load_features().sort_index(), load_labels().sort_index()
    f_cols = [c for c in get_feature_columns() if c in X_raw.columns]
    idx = X_raw.index.intersection(y_raw.index)

    X = X_raw.loc[idx, f_cols].dropna()
    y = y_raw.loc[X.index].astype(int)

    return X, y, f_cols


def train():
    X, y, f_cols = get_data()
    print(f"Data Loaded: {len(X)} rows. Positives: {int(y.sum())}")

    cutoff = pd.to_datetime(config.train_date)
    xt, yt = X[X.index <= cutoff], y[y.index <= cutoff]
    xv, yv = X[X.index > cutoff], y[y.index > cutoff]

    clf = RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_sample_split,
        min_samples_leaf=config.rf_samples_leaf,
        class_weight="balanced",
        random_state=config.random_seed,
    )
    clf.fit(xt, yt)

    probs = clf.predict_proba(xv)[:, 1]
    s = pd.Series(probs)

    roc = roc_auc_score(yv, probs) if len(pd.unique(yv)) > 1 else 0.0
    ap = average_precision_score(yv, probs) if int(yv.sum()) > 0 else 0.0

    print("test prob stats:", float(s.min()), float(s.quantile(0.5)), float(s.quantile(0.9)), float(s.quantile(0.95)), float(s.max()))
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {ap:.4f}")

    out = {"model": clf, "cols": f_cols}
    joblib.dump(out, config.get_model_file_path())

    imp = pd.Series(clf.feature_importances_, index=f_cols).sort_values(ascending=False)
    print("\nTop Features:\n", imp.head(8))


if __name__ == "__main__":
    train()







