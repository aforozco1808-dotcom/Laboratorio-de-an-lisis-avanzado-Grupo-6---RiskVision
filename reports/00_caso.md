# Ficha de caso

> Estados: `borrador` → (`devuelta` → `borrador`)* → `aprobada`. Solo con la ficha **aprobada** se avanza al EDA.

- **case_id:** riskvision
- **business_question:** El área de riesgo de crédito de un banco decide qué clientes de tarjeta de crédito reciben seguimiento preventivo, un ajuste de límite o una oferta de reestructuración. Con el historial de pagos de los últimos 6 meses, los montos facturados y pagados, el límite y datos demográficos, el modelo ordena a los clientes según su probabilidad de no pagar el próximo mes. El modelo prioriza; la decisión final la toma una persona.
- **dataset_source:** Default of Credit Card Clients, UCI Machine Learning Repository (clientes de un banco de Taiwán, abril a septiembre de 2005). Licencia: CC BY 4.0 (verificada en UCI el 2026-10-30). Cita: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- **dataset_rows:** 30.000 (volumen aceptado para el proyecto)
- **dataset_columns:** 25 (24 variables más `ID`)
- **target:** `default payment next month` (1 = incumple, 0 = paga; 22,12 % incumple)
- **problem_type:** clasificación (binaria, desbalanceada)
- **primary_metric:** ROC AUC, porque ordena a los clientes por riesgo sin fijar un umbral y es adecuada con clases desbalanceadas. La línea base (clase más frecuente) tiene 0,5000. Una referencia previa del grupo, con Gradient Boosting, dio 0,7784 en prueba.
- **business_cost_note:** El error más caro es no detectar a un cliente que sí incumple (falso negativo): el banco pierde lo prestado. El error barato es una falsa alarma: un seguimiento innecesario. Se trabaja con una proporción supuesta de 5 a 1 (supuesto del grupo, no dato del dataset), que debe validarse. Costo esperado por cliente con esa matriz: siempre «paga» 1,11; azar 1,03; marcar a todos 0,78.
- **status:** borrador
- **reviewer_notes:** <!-- las llena el instructor -->
