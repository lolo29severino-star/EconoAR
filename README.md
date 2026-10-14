# econoAR

Tablero macroeconómico federal de la Argentina: país, regiones, las 24 provincias y los 31 aglomerados urbanos de la EPH, con pronóstico de inflación, modelos econométricos, agrupamiento de aglomerados y un asistente con IA.

Incluye empleo registrado por modalidad, masa salarial, ingresos de los hogares (microdatos EPH), crédito y mora (nacional y por provincia), Producto Bruto Geográfico y la situación fiscal de cada provincia.

Segundo parcial de **Programación para Economistas** (UCA).

## Qué hay en la carpeta

| Archivo | Para qué sirve |
|---|---|
| `app.py` | La página en Python (Streamlit): resumen, precios, actividad y salarios, trabajo, empleo e ingresos, crédito y mora, pobreza, fiscal y sector externo, provincias (mapa con 44 indicadores), PBG y finanzas provinciales, modelos y asistente IA. |
| `notebooks/econoAR_analisis.ipynb` | El notebook que pide la consigna: importación, procesamiento, modelos y gráficos, con los apartados 1 a 7 del entregable al principio. Ya está ejecutado. |
| `econoar/series.py` | Catálogo de identificadores de la API de Series de Tiempo y el cliente que las descarga. |
| `econoar/datos.py` | Carga del snapshot, actualización en vivo y cálculo de indicadores (inflación, salario real, fiscal en % del PIB, empleo registrado, masa salarial, crédito real y mora, PBG, ratios fiscales provinciales, tabla provincial). |
| `econoar/eph.py` | Descarga de los microdatos de la EPH y cálculo de ingresos, informalidad y desocupación por aglomerado con los ponderadores del INDEC. |
| `econoar/modelos.py` | Pronóstico de inflación (ingenuo, media móvil, ARIMA y Ridge con validación rolling origin), Ley de Okun, regresiones de pobreza y de mora, convergencia regional del PBG y K-means de aglomerados. |
| `econoar/ia.py` | Asistente y generador de informes: usa Claude si hay clave de Anthropic y un motor local si no. |
| `econoar/graficos.py` | Mapa coroplético de provincias que funciona sin internet. |
| `data/` | Snapshot de datos al 14-10-2026 (CSV y GeoJSON). |

## Cómo correrlo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en el navegador (http://localhost:8501). En la barra lateral se elige:

* **Snapshot (sin internet)**: usa los CSV de `data/`. Arranca al instante.
* **API en vivo (datos.gob.ar)**: vuelve a bajar todas las series (tarda uno o dos minutos) y combina lo nuevo con el snapshot.

Para que el asistente y el informe los redacte Claude, cargá la clave en la barra lateral o antes de arrancar:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

Sin clave todo funciona igual, con un motor de respuestas local.

### En Google Colab

Comprimí la carpeta como `econoAR.zip`, subila a la sesión de Colab, abrí `notebooks/econoAR_analisis.ipynb` y ejecutá todo. La primera celda descomprime el proyecto e instala lo que falte.

## Fuentes

* API de Series de Tiempo de datos.gob.ar: IPC (INDEC), EMAE y PIB (INDEC), EPH (INDEC), índice de salarios (INDEC), IMIG (Ministerio de Economía), recaudación (ARCA), reservas, tipo de cambio y BADLAR (BCRA), ICA (INDEC), SIPA por provincia y por modalidad, remuneraciones SIPA (Secretaría de Trabajo), préstamos, tasas e irregularidad de cartera (BCRA), IPI, ISAC, supermercados y Gini (INDEC).
* Microdatos de la EPH 1T-2026 (INDEC): ingresos, informalidad y desocupación por aglomerado (cálculo propio, coincide con lo publicado).
* CEPAL y Ministerio de Economía, «Desagregación provincial del valor agregado bruto de la Argentina, base 2004», 2004-2023 (descargado de Argendata, Fundar).
* Dirección Nacional de Asuntos Provinciales (Ministerio de Economía): ejecución presupuestaria 2025 de las provincias (APNF, datos provisorios; La Pampa sin dato).
* Centro de Economía y Comunicación (Mapa de la Deuda, con la Central de Deudores del BCRA) y CEPA: mora de personas por provincia, por edad y por tipo de crédito.
* Secretaría de Trabajo (OEDE): salario promedio de los asalariados privados registrados por provincia, marzo 2026.
* Informes técnicos del INDEC: mercado de trabajo EPH 2T-2026, pobreza e indigencia 1S-2026, IPC septiembre 2026, origen provincial de las exportaciones 1S-2026.
* Censo 2022, resultados definitivos (INDEC).
* Coeficientes de la Ley 23.548 y variación real de las transferencias automáticas ene-ago 2026 (Fundación Encuentro con datos del Ministerio de Economía).
* Contornos de provincias: Natural Earth (dominio público), con Malvinas dentro de Tierra del Fuego, AIAS.

## Detalles que conviene saber

* La API de Series de Tiempo devuelve huecos al azar cuando se piden varias series juntas. El cliente pide de a una serie y une varios pedidos.
* Las series trimestrales de PIB de la API están en niveles anualizados: el PIB del año es el promedio de los cuatro trimestres, no la suma.
* La EPH mide aglomerados urbanos. Para comparar provincias se usa el aglomerado principal de cada una (tabla `AGLOMERADO_PRINCIPAL` en `econoar/datos.py`).
* El índice de salarios no registrados se publica con cinco meses de rezago; por eso no se compara con el IPC del mismo mes.
* No hay un PBG oficial homogéneo posterior a 2023. El PBG 2025 en pesos o dólares es una estimación: participación de la CEPAL en 2023 × PIB nominal 2025.
* La remuneración SIPA de junio y diciembre incluye el medio aguinaldo: la masa salarial se compara contra el mismo mes del año anterior o en sumas de 12 meses.
* El monotributo social cae de 653 mil a 259 mil inscriptos en diciembre de 2024 (quiebre administrativo de la serie).
