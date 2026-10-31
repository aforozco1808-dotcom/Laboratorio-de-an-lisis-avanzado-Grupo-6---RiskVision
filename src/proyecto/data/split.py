"""Partición de los datos.

Unidad de análisis: una fila es un cliente distinto (los 30.000 ID son únicos, ver reports/01_eda.md),
así que partir por fila es válido: ningún cliente puede quedar a los dos lados. Se estratifica por la
variable objetivo porque solo el 22,12 % incumple y la clase rara debe tener la misma proporción en
entrenamiento, prueba y cada pliegue de la validación cruzada.

La prueba queda cerrada: no se mira hasta la validación final (etapa 7).
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from proyecto.config import ID_COL, RANDOM_STATE, TARGET, TEST_SIZE


def assert_unique_ids(df: pd.DataFrame, id_col: str = ID_COL) -> None:
    """Falla si algún cliente aparece en más de una fila (en ese caso habría que partir por cliente)."""
    repeated = int(df[id_col].duplicated().sum())
    if repeated:
        raise AssertionError(f"{repeated} filas repiten un {id_col}: la partición debe hacerse por {id_col}, no por fila")


def assert_disjoint_ids(*frames: pd.DataFrame, id_col: str = ID_COL) -> None:
    """Falla si algún cliente aparece en más de una partición."""
    seen: set = set()
    for i, frame in enumerate(frames):
        ids = set(frame[id_col])
        overlap = seen & ids
        if overlap:
            raise AssertionError(f"{len(overlap)} clientes aparecen en más de una partición (partición {i})")
        seen |= ids


def split_stratified(df: pd.DataFrame, target: str = TARGET, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE):
    """Separa entrenamiento y prueba por fila, estratificado por el objetivo. Verifica que no se repita ningún cliente."""
    assert_unique_ids(df)
    train, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    assert_disjoint_ids(train, test)
    return train, test


def make_xy(df: pd.DataFrame, target: str = TARGET, id_col: str = ID_COL):
    """Separa los predictores (sin ID ni objetivo) de la variable objetivo."""
    return df.drop(columns=[id_col, target]), df[target]


def stratified_cv(n_splits: int = 5, seed: int = RANDOM_STATE) -> StratifiedKFold:
    """Validación cruzada estratificada con la misma semilla en todo el proyecto."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
