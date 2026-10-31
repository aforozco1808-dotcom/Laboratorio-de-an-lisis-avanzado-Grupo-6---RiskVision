# Especificación del proyecto

> Plantilla. Complétenla en el grupo y manténganla actualizada. Claude la lee antes de generar
> código: lo que escriban aquí manda sobre cualquier otra suposición.

## 1. Grupo

| Integrante | Rol (datos / modelado / app / negocio) |
|---|---|
| Juan David Ardila Arboleda | Analista de Datos |
| Lina Isabel Vasquez Villegas | Analista de Datos |
| Andres Felipe Orozco Roa | Analista de Datos |
| Cristina Garces | Analista de Datos |
| Estefania Cervera | Analista de Datos |

Nombre de la "startup": RiskVision   Repositorio (público): https://github.com/aforozco1808-dotcom/Laboratorio-de-an-lisis-avanzado-Grupo-6---RiskVision

## 2. Caso de negocio

- **Pregunta de negocio** (una frase, que alguien sin formación técnica entienda): ¿Qué clientes de tarjeta de crédito tienen alta probabilidad de no pagar el próximo mes?
- **¿Quién usará el resultado y qué decisión tomará con él?** El área de riesgo de crédito del banco. Con la probabilidad de impago de cada cliente decide si ajusta el límite de crédito, hace seguimiento preventivo o ofrece reestructurar la deuda.
- **¿Qué cuesta equivocarse?** (cuál error es peor y por qué): Es peor no detectar a un cliente que sí va a incumplir (falso negativo), porque el banco pierde el dinero prestado. Equivocarse en el otro sentido (falso positivo) solo genera un seguimiento innecesario o un límite más bajo para un buen cliente.

## 3. Datos

- **Fuente** (enlace de Kaggle, empresa, etc.): UCI Machine Learning Repository, "Default of Credit Card Clients" (clientes de un banco de Taiwán, abril a septiembre de 2005). Archivo: `data/raw/default of credit card clients.xls`.
- **Filas / columnas:** 30.000 filas y 25 columnas (24 variables más `ID`). Sin valores nulos.
- **Variable objetivo** (columna que queremos predecir o describir): `default payment next month` (1 = incumple, 0 = paga). Clase desbalanceada: 22,1 % incumple (6.636 de 30.000).
- **Tipo de problema:** ☒ clasificación  ☐ regresión  ☐ clusterización
- **¿Una misma entidad (cliente, paciente, empresa) aparece en varias filas?** ☒ no  ☐ sí → la partición
  será por: fila (cada fila es un cliente distinto; `ID` no se repite), estratificada por la variable objetivo: 80 % entrenamiento (24.000 filas) y 20 % prueba (6.000 filas), con semilla 42 y validación cruzada de 5 folds estratificados.
- **Datos personales o confidenciales presentes:** ☐ no  ☒ sí → se eliminarán/anonimizarán así: no hay nombres ni documentos, solo un `ID` anónimo. `SEX`, `AGE` y `MARRIAGE` son atributos sensibles: se revisará su efecto y no se usarán para decidir a quién se le niega crédito. `ID` se excluye del modelo.
- **Licencia o permiso de uso de los datos:** Datos públicos del repositorio UCI. Verificar la licencia indicada en la página del dataset antes de la entrega final.

## 4. Métrica

- **Métrica principal:** ROC AUC (definida en `model_selection_classification_REVISION_FINAL.ipynb`)   **Por qué esa y no otra:** las clases están desbalanceadas, así que la exactitud engaña: un modelo que dice "todos pagan" acierta 77,9 % y no sirve. El ROC AUC mide qué tan bien el modelo ordena a los clientes de menor a mayor riesgo, sin fijar aún un umbral, porque el dataset no trae los costos de negocio. 0,5 equivale al azar y 1,0 a una separación perfecta.
- **Métricas secundarias:** recall de la clase "incumple" (por el costo del falso negativo), precision, F1 y AUC-PR.
- **Línea base** (qué resultado obtiene una regla simple o un modelo trivial): modelo que siempre predice la clase más frecuente ("paga"): ROC AUC 0,5000, exactitud 77,9 % y recall 0 %. Resultados del notebook de selección de modelo: Gradient Boosting inicial 0,7815 ± 0,0058 en validación cruzada; optimizado 0,7850; **prueba final 0,7784**. Todos los modelos probados superan la línea base (KNN 0,7589, árbol de decisión 0,7589 y regresión logística 0,7272).

## 5. Riesgos

- Fuga de información posible: escalar o imputar antes de partir los datos, y usar la prueba para elegir el umbral o el modelo. Ajustar todo solo con entrenamiento.
- Sesgos o equidad: `SEX`, `AGE` y `MARRIAGE` pueden producir decisiones discriminatorias; comparar errores por grupo en la etapa de validación.
- Datos faltantes o de mala calidad: no hay nulos, pero `EDUCATION` tiene códigos 0, 5 y 6 y `MARRIAGE` el código 0, que no están en la documentación; `PAY_*` mezcla estados de pago con códigos como -2 y 0. Documentar cómo se tratan.
- Límites de lo que el modelo puede decir: los datos son de un solo banco de Taiwán en 2005; predice probabilidad de impago, no causas, y puede no servir en otro país o época.

## 6. Entregables y fechas

| Hito | Contenido | Fecha |
|---|---|---|
| H1 | Ficha aprobada, EDA y preprocesamiento reproducible | Cierre de la clase 2 (vie 30 oct) |
| H2 | Torneo, optimización con tracking, modelo registrado, repo modular, app local | Cierre de la clase 4 (jue 5 nov) |
| H3 | App desplegada y presentación (≤ 5 diapositivas) | Clase 5 (vie 6 nov) |
| H4 | Informe técnico final (hasta 8 días tras la última clase) | Domingo 15 nov |

## 7. Decisiones tomadas (agréguenlas a medida que avancen)

| Fecha | Decisión | Motivo |
|---|---|---|
| 2026-10-07 | Métrica ROC AUC para torneo, optimización y prueba final | Clases desbalanceadas y sin costos de negocio para fijar un umbral |
| 2026-10-14 | Gradient Boosting pasa a optimización (30 trials con Optuna) | Mayor ROC AUC promedio en validación cruzada; `max_depth=5` quedó en el límite del rango |
| 2026-10-14 | Montos en escala original, sin `log`; imputación por mediana y estandarización dentro del pipeline | Hay ceros y negativos en `BILL_AMT`, así que `log(x)` no sirve; evita fuga de información |
| 2026-10-30 | Copiar el dataset a `data/raw/` y dejar el original en la raíz | Los compañeros lo reciben por GitHub, porque `data/raw/` no se sube |
