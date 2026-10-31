# Tareas · 04 Variables (features)

**Entrada:** `especificaciones/04-features.md`, `specs/04-features/plan.md`
**Regla:** una tarea se marca hecha solo cuando su prueba pasa. No se escribe código hasta que el grupo apruebe este archivo y el plan.

## Fase 0 · Aprobación

- [x] T000 Integrar a `main` la rama `revision-h1-riskvision` (commit `557cd11`, Juan David): validaciones nuevas en limpieza y partición, más 14+ pruebas. Verificado en copia temporal: 112 pruebas pasan y la partición sale idéntica.
- [x] T001 El grupo lee y aprueba `especificaciones/04-features.md` (en especial la regla de la §3 y el supuesto del 5 % de `n_minimum_payments`).
- [x] T002 El grupo aprueba `plan.md` y este `tasks.md`.

## Fase 1 · Funciones de variables (`src/proyecto/features/build.py`, `tests/test_features.py`)

Cada tarea incluye su función y su prueba (un caso normal y uno borde).

- [x] T101 `add_credit_utilization` → `utilization_last`, `utilization_mean`, `utilization_max`. Prueba: límite de 0 o vacío da vacío, no infinito.
- [x] T102 `add_pending_balance` → `pending_balance_last`. Prueba: pago mayor que la factura da saldo negativo y no falla.
- [x] T103 `add_delay_run` → `delay_streak`. Prueba: `PAY_0 = 0` con atrasos antiguos da 0; atraso solo en `PAY_0` y `PAY_2` da 2.
- [x] T104 `add_delay_severity` → `delay_severity`, `severe_delay_months`, `n_delayed_months`. Prueba: los estados -2 y -1 no suman.
- [x] T105 `add_zero_payments` → `n_zero_payments`. Prueba: contar solo `PAY_AMT*` iguales a 0.
- [x] T106 `add_minimum_payments` → `n_minimum_payments`. Prueba: factura negativa o cero con pago cero cuenta como definido en la especificación.
- [x] T107 `add_log_limit` → `log_limit`. Prueba: límite 1 da 0.
- [x] T108 `build_features(df, names)` agrupa las funciones. Prueba: no cambia el número de filas ni modifica el original, y no repite columnas.
- [x] T109 `test_features_do_not_use_the_target`: las variables salen idénticas si se altera el objetivo.

## Fase 2 · Pipeline (`src/proyecto/models/train.py`, `tests/test_pipeline.py`)

- [x] T201 `build_pipeline(model, features)` con variables → imputación → escala o codificación según el modelo → modelo.
- [x] T202 Prueba: el `Pipeline` entrenado con el entrenamiento predice sobre filas crudas y devuelve probabilidades entre 0 y 1.
- [x] T203 Prueba: no se ajusta nada con las filas de validación (las medianas y escalas salen solo de las filas de entrenamiento de cada pliegue).

## Fase 3 · Medición (`notebooks/04_features.ipynb`)

- [x] T301 Cargar entrenamiento y crear los pliegues 5 × 3 una vez, con comprobación de que ninguna fila queda a los dos lados.
- [x] T302 Medir la referencia y cada idea en LightGBM y en la logística (AUC, AUC-PR, costo).
- [x] T303 Tabla de mejora / ruido y % de pliegues, con la regla de la §3 aplicada.
- [x] T304 Medir los paquetes (E, D y el que resulte de las ideas conservadas) y las variantes sin `SEX` y sin `PAY_0`.
- [x] T305 Elegir el paquete final y guardar la lista de variables en `src/proyecto/features/build.py` (constante `FINAL_FEATURES`).

## Fase 4 · Informe y cierre

- [x] T401 `reports/04_features.md`: una fila por idea con su cifra y su decisión, incluidas las descartadas.
- [x] T402 Actualizar `reports/01_eda.md` / ficha si algún supuesto cambia (5:1, 5 %).
- [x] T403 `pytest` en verde desde la raíz; `git status` sin cambios en `data/raw/` ni en `test.joblib`.
- [ ] T404 Auditoría de fuga (`auditor-de-fuga`) y revisión del código (`revisor-de-codigo-ia`); cada integrante puede explicar una función.
- [ ] T405 Commit y push.

## Dependencias

T001 → T002 → Fase 1 → Fase 2 → Fase 3 → Fase 4. Dentro de la Fase 1, T101-T107 son independientes entre sí pero comparten módulo y archivo de pruebas: se hacen una a una.
T304 incluye repetir la medición del paquete elegido con otras semillas de pliegues, o con validación anidada si el grupo la aprueba (sesgo de selección, §3 de la especificación y revisión H1).
Antes de la Fase 1: traer a `main` la rama `revision-h1-riskvision` (T000), porque cambia `cleaning.py` y `split.py`, de los que dependen las variables.

## Notas de cierre (2026-10-31)
- T001/T002: se tomaron como aprobadas cuando el grupo pidió completar la etapa y eligió la regla pareada (opción B). Conviene que las confirmen por escrito.
- T304: se hizo con otras semillas (7 y 99). La validación anidada **no** se hizo.
- T402 (actualizar `reports/01_eda.md`): se agregó la referencia a la etapa 4. T404 (auditoría de fuga y revisión de código) y T405 (commit y push) siguen pendientes.
