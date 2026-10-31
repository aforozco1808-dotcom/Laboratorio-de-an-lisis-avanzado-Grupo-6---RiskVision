import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from proyecto.features.build import PACKAGE_E
from proyecto.models.train import build_pipeline

PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def _clients(n: int = 300, seed: int = 0):
    """Clientes sintéticos con las columnas del dataset; el atraso reciente aumenta la probabilidad de incumplir."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"LIMIT_BAL": rng.integers(10_000, 500_000, n).astype(float),
                      "SEX": rng.integers(1, 3, n), "EDUCATION": rng.integers(1, 5, n), "MARRIAGE": rng.integers(1, 4, n),
                      "AGE": rng.integers(21, 70, n).astype(float)})
    for col in PAY:
        X[col] = rng.integers(-2, 4, n)
    for i in range(1, 7):
        X[f"BILL_AMT{i}"] = rng.integers(-1000, 200_000, n).astype(float)
        X[f"PAY_AMT{i}"] = rng.integers(0, 20_000, n).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X["PAY_0"] - 1)))).astype(int)
    return X, y


def test_pipeline_predicts_probabilities_from_raw_rows():
    X, y = _clients()
    pipe = build_pipeline(LogisticRegression(max_iter=500), features=PACKAGE_E, linear=True, pay_as_category=True).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_learns_medians_only_from_the_rows_it_is_fitted_on():
    X, y = _clients()
    X.loc[:49, "AGE"] = np.nan                              # vacíos en las primeras 50 filas
    train = X.iloc[:150]
    pipe = build_pipeline(LogisticRegression(max_iter=500), linear=True).fit(train, y[:150])
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    columns = pipe.named_steps["prep"].transformers_[1][2]
    learned = imputer.statistics_[list(columns).index("AGE")]
    assert learned == train["AGE"].median()                 # no la mediana de las 300 filas
    assert learned != X["AGE"].median()


def test_pipeline_handles_missing_values_and_unseen_categories_at_prediction_time():
    X, y = _clients()
    pipe = build_pipeline(LogisticRegression(max_iter=500), features=PACKAGE_E, linear=True).fit(X, y)
    new = X.iloc[:5].copy()
    new.loc[new.index[0], "LIMIT_BAL"] = np.nan
    new.loc[new.index[1], "EDUCATION"] = 9                  # categoría que no vio al entrenar
    assert not np.isnan(pipe.predict_proba(new)).any()


def test_drop_removes_the_column_from_the_model_input():
    X, y = _clients()
    pipe = build_pipeline(LogisticRegression(max_iter=500), drop=("SEX", "PAY_0"), linear=True).fit(X, y)
    seen = pipe.named_steps["features"].transform(X).columns
    assert "SEX" not in seen and "PAY_0" not in seen


def test_features_requested_by_the_pipeline_are_computed_inside_it():
    X, y = _clients()
    pipe = build_pipeline(LogisticRegression(max_iter=500), features=["n_delayed_months"], linear=True).fit(X, y)
    assert "n_delayed_months" in pipe.named_steps["features"].transform(X).columns
    with pytest.raises(ValueError):
        build_pipeline(LogisticRegression(), features=["no_existe"]).fit(X, y)


def test_scaler_and_categories_are_learned_only_from_the_training_rows():
    """Escala y categorías también se ajustan solo con entrenamiento, no solo la mediana."""
    X, y = _clients()
    X["EDUCATION"] = 1
    X.loc[X.index[200:], "EDUCATION"] = 4                   # la categoría 4 solo existe en las filas de validación
    train = X.iloc[:200]
    pipe = build_pipeline(LogisticRegression(max_iter=500), linear=True).fit(train, y[:200])
    prep = pipe.named_steps["prep"]
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    numeric = list(prep.transformers_[1][2])
    assert scaler.mean_[numeric.index("AGE")] == pytest.approx(train["AGE"].mean())
    assert scaler.mean_[numeric.index("AGE")] != pytest.approx(X["AGE"].mean())
    onehot = prep.named_transformers_["cat"].named_steps["onehot"]
    education_categories = list(onehot.categories_[list(prep.transformers_[0][2]).index("EDUCATION")])
    assert 4 not in education_categories
