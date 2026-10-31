import numpy as np
import pandas as pd
import pytest

from proyecto.data.cleaning import (
    clean,
    group_rare_education,
    group_rare_marriage,
    load_raw,
    mark_impossible_as_missing,
    to_category,
)


def _frame(**overrides) -> pd.DataFrame:
    """Tres clientes con valores normales; `overrides` reemplaza columnas completas."""
    base = {
        "ID": [1, 2, 3],
        "LIMIT_BAL": [20000, 120000, 90000],
        "SEX": [2, 1, 2],
        "EDUCATION": [1, 2, 3],
        "MARRIAGE": [1, 2, 2],
        "AGE": [24, 35, 60],
        "BILL_AMT1": [3913, -200, 0],
        "default payment next month": [1, 0, 0],
    }
    for i in range(1, 7):
        base[f"PAY_AMT{i}"] = [0, 689, 1500]
        base.setdefault(f"BILL_AMT{i}", [1000, 0, -200])
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        base[col] = [-2, -1, 0]
    base.update(overrides)
    return pd.DataFrame(base)


# Decisión #3: códigos de EDUCATION sin documentar -> "otros"
def test_education_rare_codes_become_others_and_normal_stay():
    result = group_rare_education(pd.Series([0, 1, 2, 3, 4, 5, 6]))
    assert result.tolist() == [4, 1, 2, 3, 4, 4, 4]


# Decisión #3: código 0 de MARRIAGE -> "otros"
def test_marriage_zero_becomes_others_and_normal_stay():
    result = group_rare_marriage(pd.Series([0, 1, 2, 3]))
    assert result.tolist() == [3, 1, 2, 3]


# Decisión #3: SEX, EDUCATION y MARRIAGE son etiquetas, no cantidades
def test_to_category_marks_label_columns_only():
    result = to_category(_frame())
    assert all(isinstance(result[c].dtype, pd.CategoricalDtype) for c in ["SEX", "EDUCATION", "MARRIAGE"])
    assert not isinstance(result["AGE"].dtype, pd.CategoricalDtype)


# Lo imposible se marca como vacío (caso normal: no cambia nada; caso sucio: se vacía)
def test_impossible_values_become_missing_and_normal_ones_stay():
    normal = mark_impossible_as_missing(_frame())
    assert normal.isna().sum().sum() == 0

    sucio = mark_impossible_as_missing(_frame(AGE=[24, 250, 60], SEX=[2, 7, 2], LIMIT_BAL=[20000, -5, 90000], PAY_AMT1=[0, -10, 1500]))
    assert pd.isna(sucio.loc[1, "AGE"]) and pd.isna(sucio.loc[1, "SEX"])
    assert pd.isna(sucio.loc[1, "LIMIT_BAL"]) and pd.isna(sucio.loc[1, "PAY_AMT1"])
    assert sucio.loc[0, "AGE"] == 24 and sucio.loc[2, "PAY_AMT1"] == 1500


# Decisión #5: un BILL_AMT negativo es un saldo a favor, no un error: se conserva
def test_negative_bill_is_kept():
    assert mark_impossible_as_missing(_frame()).loc[1, "BILL_AMT1"] == -200


# Decisión #8: la limpieza no borra filas y no modifica el original
def test_clean_keeps_every_row_and_does_not_touch_the_original():
    raw = _frame(EDUCATION=[0, 5, 6], MARRIAGE=[0, 1, 2])
    copy = raw.copy()
    result = clean(raw)
    assert len(result) == len(raw)
    pd.testing.assert_frame_equal(raw, copy)
    assert result["EDUCATION"].astype(int).tolist() == [4, 4, 4]
    assert result["MARRIAGE"].astype(int).tolist() == [3, 1, 2]


def test_clean_leaves_the_id_and_the_target_untouched():
    raw = _frame()
    result = clean(raw)
    assert result["ID"].tolist() == raw["ID"].tolist()
    assert result["default payment next month"].tolist() == raw["default payment next month"].tolist()
    assert not np.isnan(result["LIMIT_BAL"]).any()


@pytest.mark.parametrize("column", ["ID", "default payment next month", "BILL_AMT6", "PAY_0", "AGE"])
def test_clean_rejects_missing_required_columns(column):
    with pytest.raises(ValueError, match="Faltan columnas"):
        clean(_frame().drop(columns=column))


def test_clean_rejects_duplicate_column_names():
    raw = pd.concat([_frame(), _frame()[["AGE"]]], axis=1)
    with pytest.raises(ValueError, match="duplicados"):
        clean(raw)


@pytest.mark.parametrize("value", [2, -1, np.nan, np.inf, "1", True])
def test_clean_rejects_invalid_target(value):
    raw = _frame(**{"default payment next month": pd.Series([value, 0, 1], dtype=object)})
    with pytest.raises(ValueError, match="objetivo"):
        clean(raw)


@pytest.mark.parametrize("values", [[1, 1, 3], [1, None, 3], ["a", " ", "c"], [1, np.inf, 3]])
def test_clean_rejects_invalid_ids(values):
    with pytest.raises(AssertionError, match="ID"):
        clean(_frame(ID=values))


@pytest.mark.parametrize("column,value", [("EDUCATION", 99), ("MARRIAGE", 4), ("PAY_0", -3), ("PAY_6", 10), ("EDUCATION", 1.5)])
def test_clean_rejects_codes_outside_fixed_domains(column, value):
    with pytest.raises(ValueError, match=column):
        clean(_frame(**{column: [value, 1, 2]}))


@pytest.mark.parametrize("function,values", [(group_rare_education, [1, 99]), (group_rare_marriage, [1, 99])])
def test_grouping_functions_reject_unknown_codes(function, values):
    with pytest.raises(ValueError, match="dominio"):
        function(pd.Series(values))


def test_valid_categories_may_be_absent_and_payment_code_nine_is_valid():
    result = clean(_frame(EDUCATION=[1, 1, 1], MARRIAGE=[2, 2, 2], PAY_0=[-2, 0, 9]))
    assert result["EDUCATION"].tolist() == [1, 1, 1]
    assert result["MARRIAGE"].tolist() == [2, 2, 2]
    assert result["PAY_0"].tolist() == [-2, 0, 9]


@pytest.mark.parametrize("column", ["AGE", "LIMIT_BAL", "BILL_AMT1", "PAY_AMT6", "SEX", "PAY_0"])
@pytest.mark.parametrize("value", [np.inf, -np.inf, "12", True])
def test_clean_rejects_nonfinite_or_nonnumeric_predictors(column, value):
    with pytest.raises(ValueError, match=column):
        clean(_frame(**{column: pd.Series([value, 1, 2], dtype=object)}))


def test_fractional_age_is_rejected_instead_of_rounded():
    with pytest.raises(ValueError, match="AGE.*entera"):
        mark_impossible_as_missing(_frame(AGE=[24.5, 35, 60]))


def test_clean_converts_only_fixed_impossible_values_and_keeps_missing_predictors():
    raw = _frame(AGE=[17, 101, 24], SEX=[7, 1, 2], LIMIT_BAL=[0, -1, 90000],
                 PAY_AMT1=[-1, 0, np.nan], EDUCATION=[np.nan, 1, 2])
    original = raw.copy(deep=True)
    result = clean(raw)
    assert result["AGE"].isna().tolist() == [True, True, False]
    assert result["SEX"].isna().tolist() == [True, False, False]
    assert result["LIMIT_BAL"].isna().tolist() == [True, True, False]
    assert result["PAY_AMT1"].isna().tolist() == [True, False, True]
    assert pd.isna(result.loc[0, "EDUCATION"])
    assert result.loc[1, "BILL_AMT1"] == -200
    assert len(result) == len(raw)
    pd.testing.assert_frame_equal(raw, original)


def test_clean_is_idempotent_and_accepts_numeric_categorical_columns():
    once = clean(_frame(EDUCATION=[0, 5, 6], MARRIAGE=[0, 1, 2]))
    pd.testing.assert_frame_equal(clean(once), once)


def test_nullable_predictors_and_absent_categories_are_accepted():
    raw = _frame()
    raw["AGE"] = pd.Series([24, pd.NA, 60], dtype="Int64")
    raw["EDUCATION"] = pd.Series([1, pd.NA, 1], dtype="Int64")
    raw["MARRIAGE"] = pd.Categorical([2, 2, 2], categories=[0, 1, 2, 3])
    raw["BILL_AMT6"] = np.nan
    result = clean(raw)
    assert pd.isna(result.loc[1, "AGE"])
    assert pd.isna(result.loc[1, "EDUCATION"])
    assert result["MARRIAGE"].tolist() == [2, 2, 2]
    assert result["BILL_AMT6"].isna().all()


def test_to_category_rejects_invalid_label_codes_without_converting_them():
    with pytest.raises(ValueError, match="EDUCATION"):
        to_category(_frame(EDUCATION=[1, 99, 2]))


def test_load_raw_rejects_missing_header_columns(monkeypatch):
    raw = _frame().drop(columns="PAY_AMT6")
    monkeypatch.setattr(pd, "read_excel", lambda path, **kwargs: pd.DataFrame([raw.columns.tolist()]))
    with pytest.raises(ValueError, match="PAY_AMT6"):
        load_raw("example.xls")


def test_load_raw_validates_header_and_preserves_valid_data(monkeypatch):
    raw = _frame()

    def read_excel(path, **kwargs):
        if kwargs.get("header") is None:
            return pd.DataFrame([raw.columns.tolist()])
        return raw.copy()

    monkeypatch.setattr(pd, "read_excel", read_excel)
    pd.testing.assert_frame_equal(load_raw("example.xls"), raw)


def test_load_raw_rejects_duplicate_headers_before_pandas_renames_them(monkeypatch):
    calls = []

    def read_excel(path, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame([[*_frame().columns, "AGE"]])

    monkeypatch.setattr(pd, "read_excel", read_excel)
    with pytest.raises(ValueError, match="duplicados"):
        load_raw("example.xls")
    assert len(calls) == 1


def test_load_raw_rejects_invalid_target_after_reading(monkeypatch):
    raw = _frame(**{"default payment next month": [0, 2, 1]})
    monkeypatch.setattr(pd, "read_excel", lambda path, **kwargs:
                        pd.DataFrame([raw.columns.tolist()]) if kwargs.get("header") is None else raw)
    with pytest.raises(ValueError, match="objetivo"):
        load_raw("example.xls")
