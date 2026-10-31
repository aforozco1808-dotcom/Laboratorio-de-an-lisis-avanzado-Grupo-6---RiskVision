"""Limpieza del dataset crudo Default of Credit Card Clients.

Cada regla sale de una decisión del EDA (notebook 02, reports/01_eda.md) y es una función pequeña
con su prueba en tests/test_cleaning.py:

  #3  Los códigos de EDUCATION sin documentar (0, 5, 6) pasan a "otros" (4); el código 0 de
      MARRIAGE pasa a "otros" (3).
  #3  SEX, EDUCATION y MARRIAGE son etiquetas, no cantidades: se guardan como categorías.
  --  Lo imposible se marca como vacío (no se borran filas). En estos datos no hay casos, pero la
      regla protege cuando lleguen datos nuevos.
  #8  No se borra ninguna fila: la limpieza devuelve exactamente las filas que recibe.

Lo que NO se hace aquí (depende de los datos de entrenamiento y va dentro del Pipeline, etapa 4):
imputar con la mediana, estandarizar, codificar one-hot y transformar los montos (#4, #5).
"""
from numbers import Real

import numpy as np
import pandas as pd

from proyecto.config import CATEGORICAL, ID_COL, RAW_FILE, TARGET
from proyecto.data.split import _validate_columns, _validate_target, assert_unique_ids

# Códigos que la documentación de UCI no define, y a qué categoría documentada se agrupan
EDUCATION_RARE = {0: 4, 5: 4, 6: 4}   # 4 = "otros"
MARRIAGE_RARE = {0: 3}                # 3 = "otros"

# Lo que es posible en el mundo real (no es el rango observado)
VALID_VALUES = {"SEX": {1, 2}}
VALID_RANGE = {"AGE": (18, 100), "LIMIT_BAL": (1, np.inf)}
NON_NEGATIVE = [f"PAY_AMT{i}" for i in range(1, 7)]   # un pago no puede ser negativo (BILL_AMT sí: saldo a favor)
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PREDICTORS = ["LIMIT_BAL", *CATEGORICAL, "AGE", *PAY_COLS, *BILL_COLS, *NON_NEGATIVE]
REQUIRED_COLUMNS = [ID_COL, *PREDICTORS, TARGET]

# Los códigos adicionales observados se conservan o agrupan, no se interpretan como cantidades.
# UCI también documenta PAY=9 (nueve meses o más), aunque esté ausente en este archivo.
# Fuente: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
RAW_DOMAINS = {"EDUCATION": set(range(7)), "MARRIAGE": {0, 1, 2, 3}}
PAY_DOMAIN = {-2, -1, 0, *range(1, 10)}


def _validate_numeric(series: pd.Series) -> None:
    """Admite números reales y faltantes, pero no texto, booleanos ni infinitos."""
    values = series.dropna()
    numeric = values.map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_)))
    if not numeric.to_numpy(dtype=bool).all():
        raise ValueError(f"{series.name}: se requieren números reales o valores faltantes")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{series.name}: no se admiten valores infinitos")


def _validate_codes(series: pd.Series, allowed) -> None:
    """Un dominio no exige que todas sus categorías aparezcan en la muestra."""
    _validate_numeric(series)
    if not series.dropna().isin(allowed).all():
        raise ValueError(f"{series.name}: hay códigos fuera del dominio permitido {sorted(allowed)}")


def _validate_predictors(df: pd.DataFrame) -> None:
    """Valida los predictores presentes, sin aprender rangos de la muestra."""
    _validate_columns(df, [])
    for col in PREDICTORS:
        if col in df:
            _validate_numeric(df[col])
    if "AGE" in df and (df["AGE"].dropna() % 1 != 0).any():
        raise ValueError("AGE: la edad debe ser entera")
    for col, allowed in RAW_DOMAINS.items():
        if col in df:
            _validate_codes(df[col], allowed)
    for col in PAY_COLS:
        if col in df:
            _validate_codes(df[col], PAY_DOMAIN)


def _validate_training_frame(df: pd.DataFrame) -> None:
    """Contrato completo del dataset; los faltantes en predictores se conservan."""
    _validate_columns(df, REQUIRED_COLUMNS)
    assert_unique_ids(df)
    _validate_target(df)
    _validate_predictors(df)


def load_raw(path=RAW_FILE) -> pd.DataFrame:
    """Lee el archivo crudo. La primera fila del .xls es auxiliar (X1...X23/Y); los nombres reales están en la segunda.

    No hay datos personales: solo un ID anónimo, que se conserva para verificar la partición.
    """
    # Revisar la cabecera original evita que pandas oculte duplicados al renombrarlos.
    header = pd.read_excel(path, header=None, skiprows=1, nrows=1)
    if header.empty:
        raise ValueError("El archivo no contiene la cabecera esperada")
    _validate_columns(pd.DataFrame(columns=header.iloc[0].tolist()), REQUIRED_COLUMNS)
    df = pd.read_excel(path, header=1)
    _validate_training_frame(df)
    return df


def group_rare_education(series: pd.Series) -> pd.Series:
    """EDUCATION 0, 5 y 6 (sin documentar) pasan a 4 ('otros')."""
    _validate_codes(series, RAW_DOMAINS["EDUCATION"])
    return series.replace(EDUCATION_RARE)


def group_rare_marriage(series: pd.Series) -> pd.Series:
    """MARRIAGE 0 (sin documentar) pasa a 3 ('otros')."""
    _validate_codes(series, RAW_DOMAINS["MARRIAGE"])
    return series.replace(MARRIAGE_RARE)


def to_category(df: pd.DataFrame, columns=CATEGORICAL) -> pd.DataFrame:
    """Marca las columnas como categorías (sus números son etiquetas, no cantidades)."""
    _validate_columns(df, columns)
    for col in columns:
        if col in RAW_DOMAINS:
            _validate_codes(df[col], RAW_DOMAINS[col])
        elif col in VALID_VALUES:
            _validate_codes(df[col], VALID_VALUES[col])
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype("category")
    return df


def mark_impossible_as_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte en vacío lo que no puede ocurrir (edad fuera de 18-100, SEX distinto de 1 o 2, pagos negativos...).

    No borra filas: un vacío se imputa después, dentro del Pipeline.
    """
    _validate_columns(df, [*VALID_VALUES, *VALID_RANGE, *NON_NEGATIVE])
    _validate_predictors(df)
    df = df.copy()
    for col, allowed in VALID_VALUES.items():
        df[col] = df[col].where(df[col].isin(allowed))
    for col, (low, high) in VALID_RANGE.items():
        df[col] = df[col].where(df[col].between(low, high))
    for col in NON_NEGATIVE:
        df[col] = df[col].where(df[col] >= 0)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el dataset completo y devuelve una copia limpia sin eliminar filas.

    Rechaza errores de esquema, ID, objetivo, tipos, infinitos y códigos desconocidos.
    Conserva faltantes y facturas negativas. Solo las reglas fijas de valores imposibles
    convierten a faltante (sexo inválido, edad entera fuera de rango, límite o pagos inválidos).
    No imputa, codifica ni ajusta escalas o umbrales de la muestra.
    """
    _validate_training_frame(df)
    out = df.copy()
    out["EDUCATION"] = group_rare_education(out["EDUCATION"])
    out["MARRIAGE"] = group_rare_marriage(out["MARRIAGE"])
    out = mark_impossible_as_missing(out)
    out = to_category(out)
    return out
