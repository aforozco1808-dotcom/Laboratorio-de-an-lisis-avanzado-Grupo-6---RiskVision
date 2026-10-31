import numpy as np
import pandas as pd

from proyecto.data.cleaning import (
    clean,
    group_rare_education,
    group_rare_marriage,
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
