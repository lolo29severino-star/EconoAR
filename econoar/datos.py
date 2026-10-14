"""
Carga y procesamiento de datos de econoAR.

Dos fuentes posibles:
  1. Snapshot local (carpeta data/): CSV armados el 14-10-2026 a partir de la
     API de Series de Tiempo, informes técnicos del INDEC (EPH, pobreza, IPC,
     OPEX), el Censo 2022, la desagregación provincial del VAB de la CEPAL,
     las ejecuciones presupuestarias provinciales (Min. Economía), el Mapa de
     la Deuda (CEC) y los salarios por provincia del OEDE. Siempre disponible.
  2. API en vivo: `actualizar_desde_api()` vuelve a bajar las series
     nacionales y las provinciales del SIPA y pisa el snapshot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import series as S

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REGIONES_IPC = {
    "ipc_gba": "GBA", "ipc_pampeana": "Pampeana", "ipc_noreste": "Noreste",
    "ipc_noroeste": "Noroeste", "ipc_cuyo": "Cuyo", "ipc_patagonia": "Patagonia",
}

# Aglomerado EPH de referencia para cada provincia (el más poblado).
# La EPH es urbana: el dato del aglomerado se usa como aproximación provincial.
AGLOMERADO_PRINCIPAL = {
    "Buenos Aires": "Partidos del Gran Buenos Aires", "CABA": "Ciudad Autónoma de Buenos Aires",
    "Catamarca": "Gran Catamarca", "Chaco": "Gran Resistencia", "Chubut": "Comodoro Rivadavia-Rada Tilly",
    "Córdoba": "Gran Córdoba", "Corrientes": "Corrientes", "Entre Ríos": "Gran Paraná",
    "Formosa": "Formosa", "Jujuy": "Jujuy-Palpalá", "La Pampa": "Santa Rosa-Toay",
    "La Rioja": "La Rioja", "Mendoza": "Gran Mendoza", "Misiones": "Posadas",
    "Neuquén": "Neuquén-Plottier", "Río Negro": "Viedma-Carmen de Patagones", "Salta": "Salta",
    "San Juan": "Gran San Juan", "San Luis": "San Luis-El Chorrillo", "Santa Cruz": "Río Gallegos",
    "Santa Fe": "Gran Rosario", "Santiago del Estero": "Santiago del Estero-La Banda",
    "Tierra del Fuego": "Ushuaia-Río Grande", "Tucumán": "Gran Tucumán-Tafí Viejo",
}


@dataclass
class Datos:
    mensual: pd.DataFrame
    trimestral: pd.DataFrame
    aglomerados: pd.DataFrame
    pobreza_hist: pd.DataFrame
    ipc_aperturas: pd.DataFrame
    canastas: pd.DataFrame
    sipa_mensual: pd.DataFrame
    sipa_anual: pd.DataFrame
    provincias: pd.DataFrame
    geojson: dict
    # --- ampliación: empleo, masa salarial, crédito, mora, PBG y finanzas provinciales ---
    ampliado: pd.DataFrame = field(default_factory=pd.DataFrame)       # mensual (SIPA, BCRA, INDEC)
    pbg: pd.DataFrame = field(default_factory=pd.DataFrame)            # VAB provincial 2004-2023 (CEPAL)
    fiscal_prov: pd.DataFrame = field(default_factory=pd.DataFrame)    # ejecución 2025 por provincia
    mora_prov: pd.DataFrame = field(default_factory=pd.DataFrame)      # mora de personas por provincia
    mora_edades: pd.DataFrame = field(default_factory=pd.DataFrame)
    mora_tipos: pd.DataFrame = field(default_factory=pd.DataFrame)
    salarios_prov: pd.DataFrame = field(default_factory=pd.DataFrame)  # salario privado por provincia
    ingresos_aglo: pd.DataFrame = field(default_factory=pd.DataFrame)  # ingresos EPH por aglomerado
    origen: str = "snapshot 14-10-2026"
    avisos: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def _leer_mensual(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["fecha"] + "-01")
    return df.set_index("fecha").sort_index()


def cargar_snapshot(data_dir: Path | str = DATA_DIR) -> Datos:
    d = Path(data_dir)
    mensual = _leer_mensual(d / "mensual.csv")
    trim = pd.read_csv(d / "trimestral.csv")
    trim["fecha"] = pd.to_datetime(trim["fecha"] + "-01")
    trim = trim.set_index("fecha").sort_index()

    eph = pd.read_csv(d / "eph_aglomerados_2T2026.csv")
    pob = pd.read_csv(d / "pobreza_1S2026.csv")
    aglo = eph.merge(
        pob[["nombre", "hogares_pobreza", "personas_pobreza", "hogares_indigencia", "personas_indigencia"]],
        on="nombre", how="left",
    )
    pob_hist = pd.read_csv(d / "pobreza_historica.csv", parse_dates=["fecha"])
    ipc_ap = pd.read_csv(d / "ipc_sep2026_aperturas.csv")
    canastas = pd.read_csv(d / "canastas_jun2026.csv")
    sipa_m = _leer_mensual(d / "sipa_provincias_mensual.csv")
    sipa_a = pd.read_csv(d / "sipa_provincias_anual.csv").set_index("provincia")
    base = pd.read_csv(d / "provincias_base.csv")
    expo = pd.read_csv(d / "exportaciones_provincia_1S2026.csv")
    geo = json.loads((d / "provincias.geojson").read_text(encoding="utf-8"))

    datos = Datos(mensual, trim, aglo, pob_hist, ipc_ap, canastas, sipa_m, sipa_a,
                  provincias=pd.DataFrame(), geojson=geo)
    datos.ampliado = _leer_mensual(d / "mensual_ampliado.csv")
    datos.pbg = pd.read_csv(d / "pbg_provincias_cepal.csv")
    datos.fiscal_prov = pd.read_csv(d / "fiscal_provincias_2025.csv")
    datos.mora_prov = pd.read_csv(d / "mora_provincias_jun2026.csv")
    datos.mora_edades = pd.read_csv(d / "mora_edades_ago2026.csv")
    datos.mora_tipos = pd.read_csv(d / "mora_tipos_2026.csv")
    datos.salarios_prov = pd.read_csv(d / "salarios_provincias_mar2026.csv")
    datos.ingresos_aglo = pd.read_csv(d / "eph_ingresos_1T2026.csv")
    datos.provincias = ampliar_provincias(construir_provincias(base, expo, sipa_m, aglo), datos)
    return datos


# ---------------------------------------------------------------------------
# Actualización en vivo
# ---------------------------------------------------------------------------
def actualizar_desde_api(datos: Datos, progreso=None) -> Datos:
    """Baja las series nacionales y del SIPA y las combina con el snapshot.

    Las celdas que la API trae se usan; las que no trae (huecos intermitentes)
    se completan con el snapshot. Si no hay internet, devuelve el snapshot.
    """
    avisos = []
    try:
        m = S.descargar_varias(S.MENSUALES, start_date="2017-01-01", progreso=progreso)
        if m.notna().sum().sum() == 0:
            raise ConnectionError("La API no devolvió datos")
        m.index = m.index.to_period("M").to_timestamp()
        datos.mensual = m.combine_first(datos.mensual)
        t = S.descargar_varias(S.TRIMESTRALES, start_date="2016-01-01")
        for c in ("desocupacion", "actividad", "empleo"):
            t[c] = t[c] * 100
        datos.trimestral = t.combine_first(datos.trimestral)
        sipa = S.descargar_varias(S.SIPA_PROVINCIAS, start_date="2023-01-01")
        sipa.index = sipa.index.to_period("M").to_timestamp()
        datos.sipa_mensual = sipa.combine_first(datos.sipa_mensual)
        amp = S.descargar_varias(S.AMPLIADAS, start_date="2017-01-01")
        amp.index = amp.index.to_period("M").to_timestamp()
        datos.ampliado = amp.combine_first(datos.ampliado)
        base = pd.read_csv(DATA_DIR / "provincias_base.csv")
        expo = pd.read_csv(DATA_DIR / "exportaciones_provincia_1S2026.csv")
        datos.provincias = ampliar_provincias(
            construir_provincias(base, expo, datos.sipa_mensual, datos.aglomerados), datos)
        datos.origen = "API datos.gob.ar (en vivo) + snapshot"
    except Exception as e:  # sin conexión o API caída
        avisos.append(f"No se pudo actualizar desde la API ({e}). Se usa el snapshot local.")
    datos.avisos = avisos
    return datos


# ---------------------------------------------------------------------------
# Indicadores derivados
# ---------------------------------------------------------------------------
def var_mensual(s: pd.Series) -> pd.Series:
    return (s / s.shift(1) - 1) * 100


def var_interanual(s: pd.Series, periodos: int = 12) -> pd.Series:
    return (s / s.shift(periodos) - 1) * 100


def completar_indice(s: pd.Series) -> pd.Series:
    """Interpola en logaritmos los huecos de un índice de precios (solo internos)."""
    return np.exp(np.log(s).interpolate(limit_area="inside"))


def inflacion(datos: Datos) -> pd.DataFrame:
    m = datos.mensual
    out = pd.DataFrame(index=m.index)
    for col, nombre in [("ipc_nivel_general", "general"), ("ipc_nucleo", "nucleo"),
                        ("ipc_regulados", "regulados"), ("ipc_estacionales", "estacionales")]:
        s = completar_indice(m[col])
        out[f"{nombre}_mm"] = var_mensual(s)
        out[f"{nombre}_ia"] = var_interanual(s)
    for col, reg in REGIONES_IPC.items():
        s = completar_indice(m[col])
        out[f"{reg}_ia"] = var_interanual(s)
        out[f"{reg}_mm"] = var_mensual(s)
    return out


def salario_real(datos: Datos, base: str = "2023-11-01") -> pd.DataFrame:
    """Índices de salario real (deflactados por IPC), base = 100 en `base`."""
    m = datos.mensual
    ipc = completar_indice(m["ipc_nivel_general"])
    out = pd.DataFrame(index=m.index)
    for c in ["salarios_total", "salarios_reg_privado", "salarios_publico", "salarios_no_reg"]:
        real = completar_indice(m[c]) / ipc
        if pd.Timestamp(base) in real.index and pd.notna(real.loc[base]):
            real = real / real.loc[base] * 100
        out[c.replace("salarios_", "real_")] = real
    return out


def fiscal_acumulado(datos: Datos, anio: int) -> dict:
    m = datos.mensual
    sub = m[m.index.year == anio]
    meses = sub["resultado_primario"].dropna()
    return {
        "anio": anio,
        "meses": len(meses),
        "primario": sub["resultado_primario"].sum(min_count=1),
        "financiero": sub["resultado_financiero"].sum(min_count=1),
        "intereses": sub["intereses"].sum(min_count=1),
        "ultimo_mes": meses.index.max(),
    }


def fiscal_pct_pib(datos: Datos) -> pd.DataFrame:
    """Resultado primario y financiero anual como % del PIB nominal."""
    m = datos.mensual
    t = datos.trimestral
    anual_f = m[["resultado_primario", "resultado_financiero", "intereses"]].groupby(m.index.year).sum(min_count=12)
    # Las series trimestrales de PIB de la API están expresadas en niveles anualizados:
    # el PIB del año es el PROMEDIO de los cuatro trimestres (no la suma).
    g = t["pib_nominal"].groupby(t.index.year)
    pib = g.mean().where(g.count() == 4)
    out = anual_f.div(pib, axis=0) * 100
    return out.dropna(how="all")


def pib_variaciones(datos: Datos) -> pd.DataFrame:
    t = datos.trimestral.copy()
    t["pib_ia"] = var_interanual(t["pib_const"], 4)
    g = t["pib_const"].groupby(t.index.year)
    anual = g.mean().where(g.count() == 4)  # niveles anualizados -> promedio
    return t, (anual.pct_change() * 100).dropna()


def pib_nominal_anual(datos: Datos) -> pd.Series:
    """PIB a precios corrientes por año (millones de $), solo años completos."""
    t = datos.trimestral
    g = t["pib_nominal"].groupby(t.index.year)
    return g.mean().where(g.count() == 4).dropna()


def _ipc_completo(datos: Datos, indice: pd.DatetimeIndex) -> pd.Series:
    return completar_indice(datos.mensual["ipc_nivel_general"]).reindex(indice)


def _deflactar(serie: pd.Series, ipc: pd.Series) -> pd.Series:
    """Lleva una serie nominal a pesos del último mes con IPC disponible."""
    base = ipc.dropna().iloc[-1]
    return serie * base / ipc


# ---------------------------------------------------------------------------
# Empleo registrado, masa salarial, crédito y mora (nacional)
# ---------------------------------------------------------------------------
MODALIDADES_SIPA = {
    "sipa_asal_priv": "Asalariados privados", "sipa_asal_pub": "Asalariados públicos",
    "sipa_casas": "Casas particulares", "sipa_autonomos": "Autónomos",
    "sipa_monotributo": "Monotributistas", "sipa_monot_social": "Monotributo social",
}


def empleo_registrado(datos: Datos) -> pd.DataFrame:
    """Trabajadores registrados por modalidad (SIPA, miles) y variaciones interanuales."""
    a = datos.ampliado
    cols = ["sipa_total"] + list(MODALIDADES_SIPA)
    out = a[cols].copy()
    # la serie de asalariados públicos tiene huecos sueltos en la API: se interpolan los internos
    out = out.interpolate(limit_area="inside", limit=2)
    for c in cols:
        out[f"{c}_ia"] = var_interanual(out[c])
    return out


def masa_salarial(datos: Datos) -> pd.DataFrame:
    """Masa salarial del empleo asalariado privado registrado.

    masa = asalariados privados (SIPA) x remuneración bruta promedio (SIPA).
    La remuneración de junio y diciembre incluye el medio aguinaldo, por eso
    se comparan meses iguales (variación interanual) o sumas de 12 meses.
    Los valores reales están en pesos del último mes con IPC.
    """
    a = datos.ampliado
    ipc = _ipc_completo(datos, a.index)
    masa = a["sipa_asal_priv"] * 1000 * a["rem_promedio"]
    real = _deflactar(masa, ipc)
    out = pd.DataFrame(index=a.index)
    out["masa_nominal_mm"] = masa / 1e6
    out["masa_real_mm"] = real / 1e6
    out["masa_real_ia"] = var_interanual(real)
    out["masa_real_12m_bill"] = real.rolling(12, min_periods=12).sum() / 1e12
    out["masa_real_12m_ia"] = var_interanual(out["masa_real_12m_bill"])
    out["rem_real"] = _deflactar(a["rem_promedio"], ipc)
    out["rem_real_ia"] = var_interanual(out["rem_real"])
    out["rem_mediana_real"] = _deflactar(a["rem_mediana"], ipc)
    out["empleo_ia"] = var_interanual(a["sipa_asal_priv"])
    return out


def credito_real(datos: Datos) -> pd.DataFrame:
    """Préstamos en pesos al sector privado: saldos reales, variaciones y % del PIB."""
    a = datos.ampliado
    ipc = _ipc_completo(datos, a.index)
    out = pd.DataFrame(index=a.index)
    for c in ["prest_pesos", "prest_consumo", "prest_comerciales", "prest_personales"]:
        real = _deflactar(a[c], ipc)
        out[f"{c}_real_bill"] = real / 1e6          # billones de $ del último mes
        out[f"{c}_real_ia"] = var_interanual(real)
        out[f"{c}_real_mm"] = var_mensual(real)
    # Crédito / PIB: saldo sobre el PIB nominal de los últimos 4 trimestres. El PIB de 4 trimestres
    # se fecha en el último mes del trimestre y se interpola en logaritmos para tener un dato mensual.
    t = datos.trimestral["pib_nominal"].dropna()
    pib_4t = t.rolling(4).mean().dropna()  # niveles anualizados -> promedio de 4 trimestres = PIB del último año
    pib_4t.index = pib_4t.index + pd.DateOffset(months=2)
    idx = pib_4t.index.union(a.index)
    pib_m = np.exp(np.log(pib_4t.reindex(idx)).interpolate(limit_area="inside")).ffill().reindex(a.index)
    out["credito_pct_pib"] = a["prest_pesos"] / pib_m * 100
    out["tasa_tarjetas"] = a["tasa_tarjetas"]
    return out


def mora_nacional(datos: Datos) -> pd.DataFrame:
    """Irregularidad de la cartera de préstamos al sector privado (% , BCRA)."""
    return datos.ampliado[["irreg_total", "irreg_publicos", "irreg_privados"]].dropna(how="all")


def actividad_sectorial(datos: Datos) -> pd.DataFrame:
    a = datos.ampliado
    out = pd.DataFrame(index=a.index)
    out["ipi_ia"] = var_interanual(a["ipi"])
    out["isac_ia"] = var_interanual(a["isac"])
    out["super_ia"] = var_interanual(a["super_desest"])
    out["emae_ia"] = var_interanual(datos.mensual["emae"]).reindex(a.index)
    return out


def ultimo_valor(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan, None
    return s.iloc[-1], s.index[-1]


# ---------------------------------------------------------------------------
# Provincias
# ---------------------------------------------------------------------------
def _valor_cercano(s: pd.Series, fecha: str, ventana: int = 2):
    """Valor en `fecha` o, si falta, el más cercano dentro de +/- `ventana` meses."""
    f = pd.Timestamp(fecha)
    s = s.dropna()
    if f in s.index:
        return s.loc[f]
    cerca = s[(s.index >= f - pd.DateOffset(months=ventana)) & (s.index <= f + pd.DateOffset(months=ventana))]
    if cerca.empty:
        return np.nan
    idx = np.argmin(np.abs((cerca.index - f).days))
    return cerca.iloc[idx]


def construir_provincias(base: pd.DataFrame, expo: pd.DataFrame, sipa_m: pd.DataFrame,
                         aglo: pd.DataFrame) -> pd.DataFrame:
    p = base.copy()
    # Empleo privado registrado (SIPA): último dato disponible y variaciones
    filas = []
    for prov in p["provincia"]:
        s = sipa_m[prov].dropna() if prov in sipa_m else pd.Series(dtype=float)
        if s.empty:
            filas.append((prov, np.nan, None, np.nan, np.nan))
            continue
        ult_fecha = s.index[-1]
        ult = s.iloc[-1]
        hace_un_anio = _valor_cercano(s, str((ult_fecha - pd.DateOffset(years=1)).date()), 1)
        nov23 = _valor_cercano(s, "2023-11-01", 2)
        filas.append((prov, ult, ult_fecha, (ult / hace_un_anio - 1) * 100, (ult / nov23 - 1) * 100))
    sipa = pd.DataFrame(filas, columns=["provincia", "empleo_privado_miles", "empleo_privado_fecha",
                                        "empleo_privado_var_ia", "empleo_privado_var_nov23"])
    p = p.merge(sipa, on="provincia", how="left")
    p["empleo_privado_cada_100_hab"] = p["empleo_privado_miles"] * 1000 / p["poblacion_2022"] * 100
    # Exportaciones (OPEX 1S2026)
    p = p.merge(expo[["provincia", "expo_1S2026_musd", "var_pct"]].rename(columns={"var_pct": "expo_var_ia"}),
                on="provincia", how="left")
    p["expo_usd_por_hab"] = p["expo_1S2026_musd"] * 1e6 / p["poblacion_2022"]
    # EPH y pobreza del aglomerado principal
    p["aglomerado_ref"] = p["provincia"].map(AGLOMERADO_PRINCIPAL)
    cols = ["nombre", "tasa_actividad", "tasa_empleo", "tasa_desocupacion", "tasa_subocupacion",
            "personas_pobreza", "personas_indigencia"]
    p = p.merge(aglo[cols].rename(columns={"nombre": "aglomerado_ref"}), on="aglomerado_ref", how="left")
    p["crec_poblacion_2010_2022"] = (p["poblacion_2022"] / p["poblacion_2010"] - 1) * 100
    return p


# ---------------------------------------------------------------------------
# Producto bruto geográfico (CEPAL) y finanzas públicas provinciales
# ---------------------------------------------------------------------------
def resumen_pbg(datos: Datos) -> pd.DataFrame:
    """Indicadores del Producto Bruto Geográfico por provincia.

    Fuente: CEPAL, «Desagregación provincial del valor agregado bruto de la
    Argentina, base 2004» (VAB a precios básicos, millones de $ de 2004).
    El PBG en pesos corrientes del último año con PIB completo es una
    ESTIMACIÓN: participación provincial del último año CEPAL x PIB nominal.
    """
    pbg = datos.pbg
    w = pbg.pivot(index="provincia", columns="anio", values="vab_const_2004")
    sh = pbg.pivot(index="provincia", columns="anio", values="participacion")
    ult = int(w.columns.max())
    pop = pd.read_csv(DATA_DIR / "provincias_base.csv").set_index("provincia")
    out = pd.DataFrame(index=w.index)
    out["pbg_participacion"] = sh[ult]
    out["pbg_var_ultimo"] = (w[ult] / w[ult - 1] - 1) * 100
    out["pbg_crec_anual_2004"] = ((w[ult] / w[2004]) ** (1 / (ult - 2004)) - 1) * 100
    out["pbg_var_vs_2019"] = (w[ult] / w[2019] - 1) * 100
    # VAB per cápita relativo: censos 2010 y 2022 (índice país = 100)
    p22 = pop["poblacion_2022"].reindex(w.index)
    out["pbg_pc_indice"] = (sh[ult] / 100) / (p22 / p22.sum()) * 100
    p10 = pop["poblacion_2010"].reindex(w.index)
    pc10, pc22 = w[2010] / p10, w[2022] / p22
    out["pbg_pc_crec_2010_2022"] = ((pc22 / pc10) ** (1 / 12) - 1) * 100
    pib = pib_nominal_anual(datos)
    anio_pib = int(pib.index.max())
    out["pbg_estimado_mm"] = sh[ult] / 100 * pib.loc[anio_pib]
    out["pbg_pc_estimado"] = out["pbg_estimado_mm"] * 1e6 / p22
    tc = datos.mensual["tipo_cambio"]
    out["pbg_pc_usd"] = out["pbg_pc_estimado"] / tc[tc.index.year == anio_pib].mean()
    out.attrs.update(anio_pbg=ult, anio_pib=anio_pib)
    return out.reset_index()


def pbg_largo(datos: Datos) -> pd.DataFrame:
    """VAB provincial en formato ancho (años x provincias), millones de $ de 2004."""
    return datos.pbg.pivot(index="anio", columns="provincia", values="vab_const_2004")


def fiscal_provincial(datos: Datos, pbg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ratios de solvencia y autonomía de las provincias (ejecución 2025, APNF).

    Fuente: Dirección Nacional de Asuntos Provinciales (Min. Economía), datos
    provisorios. La Pampa no informó la ejecución 2025 al cierre de la base.
    """
    f = datos.fiscal_prov[datos.fiscal_prov["provincia"] != "Total provincias"].copy()
    it, gt = f["ingresos_totales"], f["gastos_totales"]
    f["resultado_primario_pct_ing"] = f["resultado_primario"] / it * 100
    f["resultado_financiero_pct_ing"] = f["resultado_financiero"] / it * 100
    f["autonomia_fiscal"] = f["trib_propios"] / it * 100
    f["dependencia_nacional"] = (f["trib_nacionales"] + f["transf_corrientes_recibidas"]) / it * 100
    f["regalias_pct_ing"] = f["regalias"] / it * 100
    f["intereses_pct_ing"] = f["intereses"] / it * 100
    f["personal_pct_gasto"] = f["gasto_personal"] / gt * 100
    f["inversion_pct_gasto"] = f["inversion_real"] / gt * 100
    pop = pd.read_csv(DATA_DIR / "provincias_base.csv")[["provincia", "poblacion_2022"]]
    f = f.merge(pop, on="provincia", how="left")
    f["gasto_por_habitante"] = gt * 1e6 / f["poblacion_2022"]
    f["recursos_propios_por_hab"] = f["trib_propios"] * 1e6 / f["poblacion_2022"]
    if pbg is None and not datos.pbg.empty:
        pbg = resumen_pbg(datos)
    if pbg is not None and pbg.attrs.get("anio_pib") == 2025:
        f = f.merge(pbg[["provincia", "pbg_estimado_mm"]], on="provincia", how="left")
        f["gasto_pct_pbg"] = gt / f["pbg_estimado_mm"] * 100
        f["resultado_financiero_pct_pbg"] = f["resultado_financiero"] / f["pbg_estimado_mm"] * 100
        f = f.drop(columns="pbg_estimado_mm")
    return f.drop(columns="poblacion_2022")


def ampliar_provincias(p: pd.DataFrame, datos: Datos) -> pd.DataFrame:
    """Agrega a la tabla provincial: PBG, finanzas públicas, mora, salarios, masa salarial e ingresos EPH."""
    p = p.copy()
    if not datos.pbg.empty:
        pb = resumen_pbg(datos)
        p = p.merge(pb, on="provincia", how="left")
    else:
        pb = None
    if not datos.fiscal_prov.empty:
        fp = fiscal_provincial(datos, pb)
        cols = ["provincia", "resultado_primario_pct_ing", "resultado_financiero_pct_ing", "autonomia_fiscal",
                "dependencia_nacional", "regalias_pct_ing", "intereses_pct_ing", "personal_pct_gasto",
                "inversion_pct_gasto", "gasto_por_habitante", "recursos_propios_por_hab"]
        cols += [c for c in ("gasto_pct_pbg", "resultado_financiero_pct_pbg") if c in fp]
        p = p.merge(fp[cols], on="provincia", how="left")
    if not datos.mora_prov.empty:
        p = p.merge(datos.mora_prov, on="provincia", how="left")
    if not datos.salarios_prov.empty:
        sal = datos.salarios_prov
        nac = sal.loc[sal["provincia"] == "Argentina", "salario_privado_mar2026"]
        p = p.merge(sal, on="provincia", how="left")
        p["salario_privado_rel"] = p["salario_privado_mar2026"] / (nac.iloc[0] if len(nac) else np.nan) * 100
        asal_mar = [
            _valor_cercano(datos.sipa_mensual[prov], "2026-03-01", 1) if prov in datos.sipa_mensual else np.nan
            for prov in p["provincia"]
        ]
        p["masa_salarial_priv_mm"] = np.array(asal_mar) * 1000 * p["salario_privado_mar2026"] / 1e6
        p["masa_salarial_priv_por_hab"] = p["masa_salarial_priv_mm"] * 1e6 / p["poblacion_2022"]
    if not datos.ingresos_aglo.empty:
        ing = datos.ingresos_aglo.rename(columns={"nombre": "aglomerado_ref"})
        p = p.merge(ing[["aglomerado_ref", "ipcf_medio", "ipcf_mediana", "p21_medio", "informalidad_asal"]],
                    on="aglomerado_ref", how="left")
    return p


INDICADORES_PROVINCIALES = {
    "tasa_desocupacion": ("Desocupación (%) · EPH 2T-2026, aglomerado principal", "%"),
    "tasa_actividad": ("Tasa de actividad (%) · oferta laboral, EPH 2T-2026", "%"),
    "tasa_empleo": ("Tasa de empleo (%) · EPH 2T-2026", "%"),
    "tasa_subocupacion": ("Subocupación (%) · EPH 2T-2026", "%"),
    "personas_pobreza": ("Pobreza en personas (%) · EPH 1S-2026", "%"),
    "personas_indigencia": ("Indigencia en personas (%) · EPH 1S-2026", "%"),
    "empleo_privado_miles": ("Asalariados privados registrados (miles) · SIPA", "miles"),
    "empleo_privado_var_ia": ("Empleo privado registrado, var. interanual (%) · SIPA", "%"),
    "empleo_privado_var_nov23": ("Empleo privado registrado, var. desde nov-2023 (%) · SIPA", "%"),
    "empleo_privado_cada_100_hab": ("Asalariados privados registrados cada 100 habitantes", "c/100 hab"),
    "expo_1S2026_musd": ("Exportaciones 1S-2026 (millones de USD) · OPEX", "M USD"),
    "expo_var_ia": ("Exportaciones, var. 1S-2026 vs 1S-2025 (%) · OPEX", "%"),
    "expo_usd_por_hab": ("Exportaciones por habitante 1S-2026 (USD)", "USD"),
    "coef_copa_ley23548": ("Coeficiente de coparticipación (Ley 23.548, %)", "%"),
    "copa_var_real_ene_ago_2026": ("Transferencias automáticas, var. real ene-ago 2026 (%)", "%"),
    "poblacion_2022": ("Población (Censo 2022)", "personas"),
    "crec_poblacion_2010_2022": ("Crecimiento poblacional 2010-2022 (%)", "%"),
    # Producción (CEPAL)
    "pbg_participacion": ("PBG: participación en el VAB nacional 2023 (%) · CEPAL", "%"),
    "pbg_pc_indice": ("PBG por habitante 2023 (país = 100) · CEPAL y Censo 2022", "índice"),
    "pbg_pc_usd": ("PBG por habitante estimado 2025 (USD oficiales)", "USD"),
    "pbg_var_ultimo": ("PBG: variación real 2023 vs 2022 (%) · CEPAL", "%"),
    "pbg_crec_anual_2004": ("PBG: crecimiento real promedio anual 2004-2023 (%)", "%"),
    "pbg_var_vs_2019": ("PBG: variación real 2023 vs 2019 (%)", "%"),
    "pbg_pc_crec_2010_2022": ("PBG por habitante: crecimiento anual 2010-2022 (%)", "%"),
    # Finanzas públicas provinciales (ejecución 2025)
    "resultado_primario_pct_ing": ("Resultado primario 2025 (% de los ingresos) · DNAP", "%"),
    "resultado_financiero_pct_ing": ("Resultado financiero 2025 (% de los ingresos) · DNAP", "%"),
    "resultado_financiero_pct_pbg": ("Resultado financiero 2025 (% del PBG estimado)", "% PBG"),
    "gasto_pct_pbg": ("Gasto provincial 2025 (% del PBG estimado)", "% PBG"),
    "autonomia_fiscal": ("Autonomía fiscal: recursos tributarios propios / ingresos 2025 (%)", "%"),
    "dependencia_nacional": ("Dependencia de Nación: coparticipación y transferencias / ingresos 2025 (%)", "%"),
    "regalias_pct_ing": ("Regalías / ingresos totales 2025 (%)", "%"),
    "intereses_pct_ing": ("Intereses de la deuda / ingresos 2025 (%)", "%"),
    "personal_pct_gasto": ("Gasto en personal / gasto total 2025 (%)", "%"),
    "inversion_pct_gasto": ("Inversión real directa / gasto total 2025 (%)", "%"),
    "gasto_por_habitante": ("Gasto provincial por habitante 2025 ($)", "$"),
    # Crédito y mora
    "mora_personas_jun2026": ("Mora: personas con deudas de más de 90 días (%) · CEC jun-2026", "%"),
    # Salarios, masa salarial e ingresos
    "salario_privado_mar2026": ("Salario promedio privado registrado mar-2026 ($) · OEDE", "$"),
    "salario_privado_rel": ("Salario privado relativo al promedio nacional (país = 100)", "índice"),
    "masa_salarial_priv_mm": ("Masa salarial privada registrada mar-2026 (millones de $)", "M $"),
    "masa_salarial_priv_por_hab": ("Masa salarial privada registrada por habitante mar-2026 ($)", "$"),
    "ipcf_medio": ("Ingreso per cápita familiar medio 1T-2026 ($) · EPH, aglomerado principal", "$"),
    "ipcf_mediana": ("Ingreso per cápita familiar mediano 1T-2026 ($) · EPH", "$"),
    "p21_medio": ("Ingreso medio de la ocupación principal 1T-2026 ($) · EPH", "$"),
    "informalidad_asal": ("Asalariados sin descuento jubilatorio 1T-2026 (%) · EPH", "%"),
}

# Agrupación para los selectores de la app y del tablero web
GRUPOS_PROVINCIALES = {
    "Trabajo": ["tasa_desocupacion", "tasa_actividad", "tasa_empleo", "tasa_subocupacion", "informalidad_asal",
                "empleo_privado_miles", "empleo_privado_var_ia", "empleo_privado_var_nov23", "empleo_privado_cada_100_hab"],
    "Ingresos, salarios y pobreza": ["ipcf_medio", "ipcf_mediana", "p21_medio", "salario_privado_mar2026",
                                     "salario_privado_rel", "masa_salarial_priv_mm", "masa_salarial_priv_por_hab",
                                     "personas_pobreza", "personas_indigencia"],
    "Producción (PBG)": ["pbg_participacion", "pbg_pc_indice", "pbg_pc_usd", "pbg_var_ultimo", "pbg_crec_anual_2004",
                         "pbg_var_vs_2019", "pbg_pc_crec_2010_2022", "expo_1S2026_musd", "expo_var_ia", "expo_usd_por_hab"],
    "Finanzas públicas provinciales": ["resultado_primario_pct_ing", "resultado_financiero_pct_ing",
                                       "resultado_financiero_pct_pbg", "gasto_pct_pbg", "autonomia_fiscal",
                                       "dependencia_nacional", "regalias_pct_ing", "intereses_pct_ing",
                                       "personal_pct_gasto", "inversion_pct_gasto", "gasto_por_habitante",
                                       "coef_copa_ley23548", "copa_var_real_ene_ago_2026"],
    "Crédito y mora": ["mora_personas_jun2026"],
    "Población": ["poblacion_2022", "crec_poblacion_2010_2022"],
}

# Indicadores donde un valor alto es "malo" (escala roja en los mapas)
MALO_SI_ALTO = {"tasa_desocupacion", "tasa_subocupacion", "personas_pobreza", "personas_indigencia",
                "mora_personas_jun2026", "informalidad_asal", "dependencia_nacional", "intereses_pct_ing"}
# Indicadores con escala divergente centrada en 0
DIVERGENTES = {"empleo_privado_var_ia", "empleo_privado_var_nov23", "expo_var_ia", "copa_var_real_ene_ago_2026",
               "pbg_var_ultimo", "pbg_var_vs_2019", "pbg_crec_anual_2004", "pbg_pc_crec_2010_2022",
               "resultado_primario_pct_ing", "resultado_financiero_pct_ing", "resultado_financiero_pct_pbg"}
