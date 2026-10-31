# Plan · 04 Variables (features)

**Entrada:** `especificaciones/04-features.md` (reglas, ideas y criterio de decisión)   **Estado:** ☒ ejecutado el 2026-10-31 (regla de decisión actualizada a mejora pareada; ver `reports/04_features.md`)

## Resumen

Crear las variables candidatas como funciones puras, medirlas con los mismos pliegues en LightGBM y en regresión logística, y conservar solo las que cumplen la regla de decisión. Es una etapa de **medición**: el resultado esperado es un paquete corto de variables y una tabla con la cifra de cada idea, incluidas las descartadas.

## Contexto técnico

- Python 3.12, pandas, scikit-learn, LightGBM, pytest (ya en `requirements.txt`; no se agrega nada).
- Datos: `data/processed/train.joblib`, 24.000 filas. La prueba no se abre.
- Verificación de la constitución y de `CLAUDE.md`: partición ya hecha y verificada; sin datos personales; preprocesamiento dentro de `Pipeline`; prueba por función; decisiones escritas en `reports/`.

## Diseño

| Decisión | Motivo |
|---|---|
| Funciones puras fila a fila en `src/proyecto/features/build.py`, una por idea | No aprenden de los datos, así que no hay fuga y se prueban con tres filas |
| `build_features(df)` junto a funciones sueltas | El notebook y el `Pipeline` llaman a la misma función |
| Se aplican con `FunctionTransformer` **dentro** del `Pipeline` | La app y la inferencia reciben los datos crudos del cliente y recalculan las variables |
| `build_pipeline(model, features)` en `src/proyecto/models/train.py` | Lo exige el notebook 04 y se reutiliza en 05, 06 y 07 |
| Dos modelos de referencia (LightGBM y logística) | Una variable puede ayudar a uno y no al otro; el paquete final debe servir al modelo que se elija en el torneo |
| Medición por paquete, además de por idea | Las ideas aisladas casi no mueven LightGBM; juntas sí cumplen la regla en la logística |
| Pliegues repetidos (5 × 3) | Con 5 pliegues el ruido (±0,007) tapa mejoras de 0,002 |

## Estructura que se agrega

```text
src/proyecto/features/build.py     # funciones de variables
src/proyecto/models/train.py       # build_pipeline
src/proyecto/models/evaluate.py    # pliegues repetidos, métricas, mejora pareada y regla de decisión
tests/test_features.py             # una prueba por función + prueba contra uso del objetivo
tests/test_pipeline.py             # el Pipeline recibe datos crudos y devuelve probabilidades
tests/test_evaluate.py             # penalización, pliegues y regla de decisión
notebooks/04_features.ipynb        # orquesta la medición
reports/04_features.md             # tabla de decisiones con cifras
```

## Riesgos

| Riesgo | Mitigación |
|---|---|
| Fuga al calcular razones con datos de todo el conjunto | Las funciones solo usan la fila; prueba con el objetivo alterado |
| Elegir variables por ruido (el AUC cambia ±0,007 entre pliegues) | Mejora pareada (cada pliegue contra sí mismo), 15 valores y % de pliegues |
| La regla de decisión se fijó después de una medición preliminar | Está dicho en la §3 de la especificación; el grupo la aprueba o la cambia antes de ejecutar |
| `n_minimum_payments` depende de un 5 % supuesto | Se marca como supuesto y se mide con 5 % solamente; se descarta si no aporta |
| Dependencia de `PAY_0` | Se mide el modelo sin ella y se documenta en el informe |
| Dividir entre un límite vacío o cero | Devuelve vacío; prueba específica |
