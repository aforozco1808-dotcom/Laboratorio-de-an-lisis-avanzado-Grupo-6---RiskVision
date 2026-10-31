# Tareas · 05 Torneo de modelos

**Regla:** una tarea se marca hecha solo cuando su prueba pasa o cuando se ve funcionando.

## Fase 0 · Aprobación
- [ ] T001 El grupo revisa `especificaciones/05-torneo.md` (candidatos, métricas, regla de finalistas).
- [ ] T002 El grupo aprueba `plan.md` y este archivo.

## Fase 1 · Código (`src/proyecto/models/evaluate.py`, `tests/test_evaluate.py`)
- [x] T101 `technical_metrics` con prueba de valores calculados a mano (AUC 15/16, AUC-PR, penalización, recall por clase).
- [x] T102 Prueba de que `recall_top` y `lift_top` no dependen del umbral.
- [x] T103 Prueba de que funciona con una línea base constante.
- [x] T104 `score_folds_detailed` con prueba (una fila por pliegue, todas las métricas y el tiempo).

## Fase 2 · Notebook (`notebooks/05_torneo.ipynb`)
- [x] T201 Cargar entrenamiento y crear los mismos 15 pliegues de la etapa 4.
- [x] T202 Definir 5 modelos y 2 líneas base con parámetros fijos.
- [x] T203 Correr el torneo y guardar una fila por pliegue en `reports/05_torneo_por_pliegue.csv`.
- [x] T204 Tabla resumen, gráfico, métricas técnicas y tabla de rankings.
- [x] T205 Comparación pareada de los tres mejores.
- [x] T206 Traducir la diferencia a volumen por cada 1.000 clientes.

## Fase 2b · Ampliación (opciones 1, 3 y 2)
- [x] T211 Penalización con 2:1 y 10:1 y captura al revisar 5 % y 10 %, con prueba de valores calculados a mano.
- [x] T212 `PayDelayRule` (regla `PAY_0 ≥ 2`) como tercera línea base, con prueba.
- [x] T213 Calibración por tramos (`calibration_table`, `expected_calibration_error`, `observed_rate_between`) con pruebas.
- [x] T214 Variante LightGBM calibrado y secciones 5.3 y 5.4 del notebook.

- [x] T215 Comparación con el notebook de selección de modelo y afinación exploratoria (informe, sección «Comparación con el notebook de selección de modelo»).

## Fase 3 · Informe y cierre
- [x] T301 `reports/05_torneo.md` con las tablas, los finalistas y lo descartado.
- [x] T302 `pytest` en verde y `git status` sin cambios en `data/raw/` ni en `test.joblib`.
- [x] T303a Auditoría de fuga de la primera versión (sin fuga; correcciones aplicadas).
- [ ] T303b Nueva pasada de `auditor-de-fuga` y `revisor-de-codigo-ia` sobre la ampliación.
- [ ] T304 Commit y push.

## Dependencias
T001 → T002 → Fase 1 → Fase 2 → Fase 3.
Esta versión se escribió y ejecutó antes de la aprobación de T001/T002. Además de lo previsto, se agregó la comparación de LightGBM con paquete D frente a E en pliegues nuevos y la del boosting frente a la logística (sección 6.1 del notebook); si el grupo cambia los candidatos o la regla, se repite la Fase 2.
