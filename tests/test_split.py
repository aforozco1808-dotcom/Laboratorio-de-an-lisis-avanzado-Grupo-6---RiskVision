import numpy as np
import pandas as pd
import pytest

from proyecto.data.split import assert_disjoint_ids, assert_unique_ids, make_xy, split_stratified, stratified_cv

TARGET = "default payment next month"


def _clients(n: int = 1000, positives: int = 220) -> pd.DataFrame:
    """n clientes distintos; `positives` incumplen (22 % como en los datos reales)."""
    y = np.zeros(n, dtype=int)
    y[:positives] = 1
    return pd.DataFrame({"ID": range(1, n + 1), "x": np.arange(n), TARGET: y})


def test_no_client_in_two_partitions_and_no_row_is_lost():
    df = _clients()
    train, test = split_stratified(df, test_size=0.2, seed=42)
    assert set(train["ID"]).isdisjoint(set(test["ID"]))      # ningún cliente en los dos lados
    assert len(train) + len(test) == len(df)                  # no se pierde ninguna fila
    assert len(test) == 200


def test_split_keeps_the_default_rate_in_both_partitions():
    """Falla si alguien parte sin estratificar: con 22 % de positivos, test debe tener exactamente 44 de 200."""
    train, test = split_stratified(_clients(), test_size=0.2, seed=42)
    assert test[TARGET].sum() == 44
    assert train[TARGET].sum() == 176


def test_split_is_reproducible_with_the_same_seed():
    a, _ = split_stratified(_clients(), seed=42)
    b, _ = split_stratified(_clients(), seed=42)
    assert a["ID"].tolist() == b["ID"].tolist()


def test_repeated_client_is_rejected_because_it_would_need_a_group_split():
    df = pd.concat([_clients(10, 3), _clients(10, 3).iloc[:2]])
    with pytest.raises(AssertionError):
        assert_unique_ids(df)
    with pytest.raises(AssertionError):
        split_stratified(df)


def test_assert_disjoint_raises_on_overlap():
    df = _clients(10, 3)
    with pytest.raises(AssertionError):
        assert_disjoint_ids(df.iloc[:6], df.iloc[4:])


def test_make_xy_drops_the_id_and_the_target_from_the_predictors():
    X, y = make_xy(_clients(10, 3))
    assert list(X.columns) == ["x"]
    assert y.name == TARGET


def test_cv_folds_are_stratified_and_do_not_share_rows():
    df = _clients()
    for train_idx, valid_idx in stratified_cv(5, seed=42).split(df, df[TARGET]):
        assert set(train_idx).isdisjoint(valid_idx)
        assert df.iloc[valid_idx][TARGET].sum() == 44
