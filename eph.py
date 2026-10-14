"""
Microdatos de la Encuesta Permanente de Hogares (EPH, INDEC).

  * descargar_eph(): baja la base usuaria trimestral (zip con usu_individual y
    usu_hogar) desde la web del INDEC.
  * indicadores_ingreso(): calcula, por aglomerado, región y total, el ingreso
    per cápita familiar (media y mediana), el ingreso de la ocupación principal,
    la informalidad de los asalariados y la desocupación, con los ponderadores
    que indica el INDEC.

Con la base del 1T-2026 el total de los 31 aglomerados da: ingreso per cápita
familiar medio $728.008 y mediano $500.000, ingreso medio de la ocupación
principal $1.104.227, 37,9% de asalariados sin descuento jubilatorio y 7,8% de
desocupación, los mismos valores que publica el INDEC. Esos resultados están
guardados en data/eph_ingresos_1T2026.csv para usar el tablero sin internet.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

URL_EPH = "https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/EPH_usu_{t}_Trim_{anio}_txt.zip"

REGIONES_EPH = {1: "Gran Buenos Aires", 40: "Noroeste", 41: "Noreste", 42: "Cuyo", 43: "Pampeana", 44: "Patagonia"}

AGLOMERADOS_EPH = {
    2: "Gran La Plata", 3: "Bahía Blanca-Cerri", 4: "Gran Rosario", 5: "Gran Santa Fe", 6: "Gran Paraná",
    7: "Posadas", 8: "Gran Resistencia", 9: "Comodoro Rivadavia-Rada Tilly", 10: "Gran Mendoza",
    12: "Corrientes", 13: "Gran Córdoba", 14: "Concordia", 15: "Formosa", 17: "Neuquén-Plottier",
    18: "Santiago del Estero-La Banda", 19: "Jujuy-Palpalá", 20: "Río Gallegos", 22: "Gran Catamarca",
    23: "Salta", 25: "La Rioja", 26: "San Luis-El Chorrillo", 27: "Gran San Juan",
    29: "Gran Tucumán-Tafí Viejo", 30: "Santa Rosa-Toay", 31: "Ushuaia-Río Grande",
    32: "Ciudad Autónoma de Buenos Aires", 33: "Partidos del Gran Buenos Aires", 34: "Mar del Plata",
    36: "Río Cuarto", 38: "San Nicolás-Villa Constitución", 91: "Rawson-Trelew",
    93: "Viedma-Carmen de Patagones",
}

COLUMNAS = ["AGLOMERADO", "REGION", "PONDERA", "ESTADO", "CAT_OCUP", "PP07H", "P21", "PONDIIO", "IPCF", "PONDIH"]


def descargar_eph(trimestre: int, anio: int, timeout: float = 60) -> pd.DataFrame:
    """Descarga la base individual de la EPH (requiere internet)."""
    import requests

    r = requests.get(URL_EPH.format(t=trimestre, anio=anio), timeout=timeout)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        nombre = next(n for n in z.namelist() if "individual" in n.lower() and n.lower().endswith(".txt"))
        with z.open(nombre) as f:
            return leer_individual(f)


def leer_individual(origen) -> pd.DataFrame:
    """Lee el txt usu_individual (separador ';', decimales con coma)."""
    return pd.read_csv(origen, sep=";", decimal=",", usecols=COLUMNAS, low_memory=False)


def _media_ponderada(x: pd.Series, w: pd.Series) -> float:
    m = (w > 0) & x.notna()
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def _mediana_ponderada(x: pd.Series, w: pd.Series) -> float:
    m = (w > 0) & x.notna()
    if not m.any():
        return np.nan
    orden = np.argsort(x[m].values, kind="stable")
    v, p = x[m].values[orden], w[m].values[orden]
    acum = np.cumsum(p)
    return float(v[np.searchsorted(acum, acum[-1] / 2)])


def _indicadores(d: pd.DataFrame) -> dict:
    ocup = d[(d["ESTADO"] == 1) & (d["P21"] > 0)]
    asal = d[(d["ESTADO"] == 1) & (d["CAT_OCUP"] == 3)]
    pea = d[d["ESTADO"].isin([1, 2])]
    return {
        "casos_muestra": len(d),
        # ingreso per cápita familiar: todas las personas, ponderador de ingresos del hogar (PONDIH)
        "ipcf_medio": _media_ponderada(d["IPCF"], d["PONDIH"]),
        "ipcf_mediana": _mediana_ponderada(d["IPCF"], d["PONDIH"]),
        # ocupación principal: ocupados con ingreso positivo, ponderador PONDIIO
        "p21_medio": _media_ponderada(ocup["P21"], ocup["PONDIIO"]),
        # informalidad: asalariados sin descuento jubilatorio (PP07H = 2)
        "informalidad_asal": asal.loc[asal["PP07H"] == 2, "PONDERA"].sum() / asal["PONDERA"].sum() * 100,
        "desocupacion": pea.loc[pea["ESTADO"] == 2, "PONDERA"].sum() / pea["PONDERA"].sum() * 100,
    }


def indicadores_ingreso(ind: pd.DataFrame) -> pd.DataFrame:
    """Indicadores de ingreso y empleo por aglomerado, región y total de 31 aglomerados."""
    filas = []
    for cod, g in ind.groupby("AGLOMERADO"):
        filas.append({"nombre": AGLOMERADOS_EPH.get(int(cod), str(cod)), "tipo": "aglomerado", "codigo": cod, **_indicadores(g)})
    for cod, g in ind.groupby("REGION"):
        filas.append({"nombre": REGIONES_EPH.get(int(cod), str(cod)), "tipo": "region", "codigo": cod, **_indicadores(g)})
    filas.append({"nombre": "Total 31 aglomerados", "tipo": "total", "codigo": np.nan, **_indicadores(ind)})
    out = pd.DataFrame(filas)
    for c in ["ipcf_medio", "ipcf_mediana", "p21_medio"]:
        out[c] = out[c].round(0)
    for c in ["informalidad_asal", "desocupacion"]:
        out[c] = out[c].round(1)
    return out
