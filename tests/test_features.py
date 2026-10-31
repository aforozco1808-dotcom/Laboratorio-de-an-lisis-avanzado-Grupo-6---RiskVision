import numpy as np
import pandas as pd
import pytest

from proyecto.features.build import FEATURES, PACKAGE_D, PACKAGE_E, build_features

PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
AMT = [f"PAY_AMT{i}" for i in range(1, 7)]


def _frame(**overrides) -> pd.DataFrame:
    """Dos clientes: el 0 sin atrasos y con la tarjeta tranquila; el 1 con atraso reciente y la tarjeta casi llena."""
    base = {"LIMIT_BAL": [100000, 10000],
            "default payment next month": [0, 1]}
    for col, values in zip(PAY, [[-1, 2], [-1, 2], [0, 0], [0, 1], [-2, 0], [0, 0]]):
        base[col] = values
    for col, values in zip(BILL, [[1000, 9000], [2000, 9500], [0, 8000], [0, 8000], [0, 7000], [0, 7000]]):
        base[col] = values
    for col, values in zip(AMT, [[500, 0], [0, 100], [0, 0], [0, 0], [0, 0], [0, 0]]):
        base[col] = values
    base.update(overrides)
    return pd.DataFrame(base)


def test_utilization_features_divide_by_the_limit():
    df = _frame()
    assert FEATURES["utilization_last"](df).tolist() == [0.01, 0.9]
    assert FEATURES["utilization_mean"](df).iloc[1] == pytest.approx(np.mean([9000, 9500, 8000, 8000, 7000, 7000]) / 10000)
    assert FEATURES["utilization_max"](df).tolist() == [0.02, 0.95]


def test_ratios_are_missing_not_infinite_when_the_limit_is_not_positive():
    df = _frame(LIMIT_BAL=[0, np.nan])
    for name in ["utilization_last", "utilization_mean", "utilization_max", "pending_balance_last", "log_limit"]:
        assert FEATURES[name](df).isna().all(), name


def test_pending_balance_last_is_negative_when_the_payment_exceeds_the_bill():
    df = _frame(BILL_AMT2=[2000, 9500], PAY_AMT1=[500, 12000])
    assert FEATURES["pending_balance_last"](df).tolist() == [(2000 - 500) / 100000, (9500 - 12000) / 10000]


def test_delay_streak_counts_consecutive_late_months_from_the_most_recent():
    df = _frame(PAY_0=[0, 2], PAY_2=[3, 1], PAY_3=[3, 0], PAY_4=[0, 4], PAY_5=[0, 0], PAY_6=[0, 0])
    # cliente 0: atrasos viejos pero el mes más reciente está al día -> 0; cliente 1: dos meses seguidos y se corta
    assert FEATURES["delay_streak"](df).tolist() == [0, 2]


def test_delay_features_ignore_the_non_delay_states():
    df = _frame(PAY_0=[-2, 3], PAY_2=[-1, 2], PAY_3=[0, 1], PAY_4=[0, 0], PAY_5=[0, 0], PAY_6=[0, -1])
    assert FEATURES["n_delayed_months"](df).tolist() == [0, 3]
    assert FEATURES["delay_severity"](df).tolist() == [0, 6]    # 3 + 2 + 1; el -1 no resta
    assert FEATURES["severe_delay_months"](df).tolist() == [0, 2]       # solo los de 2 o más meses


def test_n_zero_payments_counts_only_exact_zero_payments():
    df = _frame(PAY_AMT1=[0, 5], PAY_AMT2=[0, 0], PAY_AMT3=[10, 0], PAY_AMT4=[0, 0], PAY_AMT5=[1, 1], PAY_AMT6=[2, 0])
    assert FEATURES["n_zero_payments"](df).tolist() == [3, 4]   # cliente 1: meses 2, 3, 4 y 6


def test_n_minimum_payments_needs_a_positive_bill_and_a_payment_of_at_most_five_percent():
    df = _frame(BILL_AMT2=[1000, 1000], PAY_AMT1=[50, 51],       # 5 % exacto cuenta; 5,1 % no
                BILL_AMT3=[0, -500], PAY_AMT2=[0, 0],            # factura cero o negativa no cuenta
                BILL_AMT4=[0, 0], PAY_AMT3=[0, 0], BILL_AMT5=[0, 0], PAY_AMT4=[0, 0], BILL_AMT6=[0, 0], PAY_AMT5=[0, 0])
    assert FEATURES["n_minimum_payments"](df).tolist() == [1, 0]


def test_log_limit_of_one_is_zero():
    assert FEATURES["log_limit"](_frame(LIMIT_BAL=[1, np.e])).tolist() == pytest.approx([0.0, 1.0])


def test_build_features_keeps_rows_and_the_original_and_adds_only_the_requested_columns():
    df = _frame()
    copy = df.copy()
    out = build_features(df, PACKAGE_E)
    pd.testing.assert_frame_equal(df, copy)
    assert len(out) == len(df)
    assert list(out.columns) == list(df.columns) + list(PACKAGE_E)


def test_build_features_rejects_unknown_repeated_and_existing_names():
    df = _frame()
    with pytest.raises(ValueError):
        build_features(df, ["no_existe"])
    with pytest.raises(ValueError):
        build_features(df, ["n_delayed_months", "n_delayed_months"])
    with pytest.raises(ValueError):
        build_features(build_features(df, ["n_delayed_months"]), ["n_delayed_months"])


def test_packages_only_use_registered_features():
    assert set(PACKAGE_E) <= set(PACKAGE_D) <= set(FEATURES)


def test_features_do_not_use_the_target():
    """Si una variable mirara el objetivo, cambiarlo alteraría el resultado."""
    df = _frame()
    altered = df.assign(**{"default payment next month": [1, 0]})
    a = build_features(df, list(FEATURES)).drop(columns=["default payment next month"])
    b = build_features(altered, list(FEATURES)).drop(columns=["default payment next month"])
    pd.testing.assert_frame_equal(a, b)


def _random_clients(n: int = 60, seed: int = 3) -> pd.DataFrame:
    """Clientes sintéticos con todas las columnas que usan las variables."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"LIMIT_BAL": rng.integers(10_000, 500_000, n).astype(float)})
    for col in PAY:
        df[col] = rng.integers(-2, 5, n)
    for col in BILL:
        df[col] = rng.integers(-2_000, 300_000, n).astype(float)
    for col in AMT:
        df[col] = rng.choice([0, 0, 500, 5_000, 40_000], n).astype(float)
    return df


@pytest.mark.parametrize("name", sorted(FEATURES))
def test_every_feature_depends_only_on_its_own_row(name):
    """Calculada con toda la tabla o con solo unas filas, la variable da lo mismo (no usa medias ni datos de otros clientes)."""
    df = _random_clients()
    subset = df.iloc[10:25]
    full = FEATURES[name](df).iloc[10:25]
    alone = FEATURES[name](subset)
    pd.testing.assert_series_equal(full, alone, check_names=False)
