import numpy as np
import pandas as pd
import joblib
import config
from pandas.tseries.offsets import BDay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit


def get_data():
    from src.features import load_features, get_feature_columns
    from src.labels import load_labels

    X_raw = load_features().sort_index()
    y_raw = load_labels().sort_index()

    f_cols = [c for c in get_feature_columns() if c in X_raw.columns]
    idx = X_raw.index.intersection(y_raw.index)

    X = X_raw.loc[idx, f_cols].dropna()
    y = y_raw.loc[X.index].astype(int)

    return X, y, f_cols


def _clf():
    return RandomForestClassifier(n_estimators=config.rf_n_estimators, max_depth=config.rf_max_depth, min_samples_split=config.rf_sample_split, min_samples_leaf=config.rf_samples_leaf, class_weight=getattr(config, "rf_class_weight", "balanced"), random_state=config.random_seed, n_jobs=-1)


def _pos_proba(clf, X_):
    classes = list(getattr(clf, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else 1
    return clf.predict_proba(X_)[:, pos_idx], classes


def train(show_cv=True, show_cv_folds=False):
    X, y, f_cols = get_data()
    print(f"Data Loaded: {len(X)} rows. Positives: {int(y.sum())}")

    cutoff = pd.to_datetime(config.train_date)

    gap_days = int(getattr(config, "horizon", 0)) + int(getattr(config, "trade_lag_days", 0))
    safe_cutoff = cutoff - BDay(gap_days)

    xt, yt = X[X.index <= safe_cutoff], y[y.index <= safe_cutoff]
    xv, yv = X[X.index > cutoff], y[y.index > cutoff]

    if show_cv and int(getattr(config, "cv_folds", 0)) >= 2:
        gap = int(getattr(config, "horizon", 0)) + int(getattr(config, "trade_lag_days", 0))
        tscv = TimeSeriesSplit(n_splits=int(config.cv_folds), gap=gap, test_size=252)

        rocs, prs = [], []
        for k, (tr_idx, va_idx) in enumerate(tscv.split(xt), start=1):
            clf = _clf()
            clf.fit(xt.iloc[tr_idx], yt.iloc[tr_idx])

            p_va, _ = _pos_proba(clf, xt.iloc[va_idx])
            y_va = yt.iloc[va_idx]

            roc = roc_auc_score(y_va, p_va) if len(pd.unique(y_va)) > 1 else np.nan
            pr = average_precision_score(y_va, p_va) if int(y_va.sum()) > 0 else np.nan

            rocs.append(roc)
            prs.append(pr)

            if show_cv_folds:
                rtxt = f"{roc:.4f}" if np.isfinite(roc) else "nan"
                ptxt = f"{pr:.4f}" if np.isfinite(pr) else "nan"
                print(f"cv fold {k}: roc={rtxt} pr={ptxt} pos_rate={float(y_va.mean()):.3f}")

        rocs = np.array([v for v in rocs if np.isfinite(v)])
        prs = np.array([v for v in prs if np.isfinite(v)])

        if len(rocs):
            print(f"cv roc mean={rocs.mean():.4f} std={rocs.std(ddof=0):.4f}")
        if len(prs):
            print(f"cv pr mean={prs.mean():.4f} std={prs.std(ddof=0):.4f}")

    clf = _clf()
    clf.fit(xt, yt)

    probs, classes = _pos_proba(clf, xv)

    s = pd.Series(probs)
    roc = roc_auc_score(yv, probs) if len(pd.unique(yv)) > 1 else 0.0
    ap = average_precision_score(yv, probs) if int(yv.sum()) > 0 else 0.0

    print("classes_:", classes)
    print("test prob stats:", float(s.min()), float(s.quantile(0.5)), float(s.quantile(0.9)), float(s.quantile(0.95)), float(s.max()))
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {ap:.4f}")

    joblib.dump({"model": clf, "cols": f_cols}, config.get_model_file_path())

    imp = pd.Series(clf.feature_importances_, index=f_cols).sort_values(ascending=False)
    print("\nTop Features:\n", imp.head(8))


if __name__ == "__main__":
    train()














