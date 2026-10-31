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


@pytest.mark.parametrize("value", [None, np.nan, "", "  ", np.inf, -np.inf, 1.5, True])
def test_missing_or_invalid_id_is_rejected(value):
    df = _clients(10, 3)
    df["ID"] = df["ID"].astype(object)
    df.loc[0, "ID"] = value
    for function in (assert_unique_ids, split_stratified, make_xy):
        with pytest.raises(AssertionError, match="ID"):
            function(df)


def test_disjoint_check_also_rejects_invalid_ids_within_a_partition():
    with pytest.raises(AssertionError, match="ID"):
        assert_disjoint_ids(pd.DataFrame({"ID": [None]}))


def test_nonempty_text_ids_are_preserved():
    df = _clients()
    df["ID"] = df["ID"].map(lambda value: f"client-{value}")
    train, test = split_stratified(df)
    assert set(train["ID"]) | set(test["ID"]) == set(df["ID"])
    assert train["ID"].dtype == df["ID"].dtype


@pytest.mark.parametrize("value", [2, -1, None, np.nan, np.inf, "1", True])
def test_split_and_make_xy_reject_invalid_target(value):
    df = _clients()
    df[TARGET] = df[TARGET].astype(object)
    df.loc[0, TARGET] = value
    for function in (split_stratified, make_xy):
        with pytest.raises(ValueError, match="objetivo"):
            function(df)


@pytest.mark.parametrize("function", [assert_unique_ids, split_stratified, make_xy])
def test_duplicate_columns_are_rejected(function):
    df = _clients()
    df = pd.concat([df, df[["ID"]]], axis=1)
    with pytest.raises(ValueError, match="duplicados"):
        function(df)


def test_split_preserves_all_values_and_both_partitions_are_reproducible():
    df = _clients()
    original = df.copy(deep=True)
    train, test = split_stratified(df)
    other_train, other_test = split_stratified(df)
    pd.testing.assert_frame_equal(train, other_train)
    pd.testing.assert_frame_equal(test, other_test)
    restored = pd.concat([train, test]).sort_values("ID").reset_index(drop=True)
    pd.testing.assert_frame_equal(restored, original)
    pd.testing.assert_frame_equal(df, original)


def test_categorical_ids_and_target_with_unused_categories_are_accepted():
    df = _clients()
    df["ID"] = df["ID"].astype("category")
    df[TARGET] = pd.Categorical(df[TARGET], categories=[0, 1])
    train, test = split_stratified(df)
    assert len(train) + len(test) == len(df)
    make_xy(train)


@pytest.mark.parametrize("column", ["ID", TARGET])
def test_split_and_make_xy_reject_missing_identity_or_target_columns(column):
    for function in (split_stratified, make_xy):
        with pytest.raises(ValueError, match="Faltan columnas"):
            function(_clients().drop(columns=column))
