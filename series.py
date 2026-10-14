"""
Catálogo de series y cliente de la API de Series de Tiempo de Argentina
(https://apis.datos.gob.ar/series/api/), mantenida por la Secretaría de
Innovación / datos.gob.ar con datos de INDEC, BCRA, Ministerio de Economía
y Secretaría de Trabajo.

Nota práctica sobre la API (verificado en octubre 2026):
  * Si se piden varias series en una misma llamada (ids=a,b,c) la API a veces
    devuelve celdas vacías de forma intermitente. Por eso este cliente pide
    las series DE A UNA y, además, repite cada pedido un par de veces y une
    los resultados (los valores, cuando aparecen, son siempre los mismos).
  * La API acepta CORS (Access-Control-Allow-Origin: *), así que también se
    puede consultar desde el navegador.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

API_URL = "https://apis.datos.gob.ar/series/api/series/"


@dataclass(frozen=True)
class Serie:
    id: str
    nombre: str
    unidad: str
    fuente: str
    frecuencia: str  # "M" mensual, "T" trimestral


# ---------------------------------------------------------------------------
# Catálogo nacional (mensual)
# ---------------------------------------------------------------------------
MENSUALES: dict[str, Serie] = {
    # Precios (INDEC, IPC base dic-2016 = 100)
    "ipc_nivel_general": Serie("148.3_INIVELNAL_DICI_M_26", "IPC Nivel general - Nacional", "índice", "INDEC", "M"),
    "ipc_nucleo": Serie("148.3_INUCLEONAL_DICI_M_19", "IPC Núcleo - Nacional", "índice", "INDEC", "M"),
    "ipc_regulados": Serie("148.3_IREGULANAL_DICI_M_22", "IPC Regulados - Nacional", "índice", "INDEC", "M"),
    "ipc_estacionales": Serie("148.3_IESTACINAL_DICI_M_25", "IPC Estacionales - Nacional", "índice", "INDEC", "M"),
    "ipc_gba": Serie("148.3_INIVELGBA_DICI_M_21", "IPC Nivel general - GBA", "índice", "INDEC", "M"),
    "ipc_pampeana": Serie("148.3_INIVELANA_DICI_M_26", "IPC Nivel general - Pampeana", "índice", "INDEC", "M"),
    "ipc_noreste": Serie("148.3_INIVELNEA_DICI_M_21", "IPC Nivel general - Noreste", "índice", "INDEC", "M"),
    "ipc_noroeste": Serie("148.3_INIVELNOA_DICI_M_21", "IPC Nivel general - Noroeste", "índice", "INDEC", "M"),
    "ipc_cuyo": Serie("148.3_INIVELUYO_DICI_M_22", "IPC Nivel general - Cuyo", "índice", "INDEC", "M"),
    "ipc_patagonia": Serie("148.3_INIVELNIA_DICI_M_27", "IPC Nivel general - Patagonia", "índice", "INDEC", "M"),
    # Actividad
    "emae": Serie("143.3_NO_PR_2004_A_21", "EMAE (original)", "índice 2004=100", "INDEC", "M"),
    "emae_desest": Serie("143.3_NO_PR_2004_A_31", "EMAE desestacionalizado", "índice 2004=100", "INDEC", "M"),
    # Salarios (índice oct-2016 = 100)
    "salarios_total": Serie("149.1_TL_INDIIOS_OCTU_0_21", "Índice de salarios - Total", "índice", "INDEC", "M"),
    "salarios_reg_privado": Serie("149.1_SOR_PRIADO_OCTU_0_25", "Índice de salarios - Registrado privado", "índice", "INDEC", "M"),
    "salarios_publico": Serie("149.1_SOR_PUBICO_OCTU_0_14", "Índice de salarios - Público", "índice", "INDEC", "M"),
    "salarios_no_reg": Serie("149.1_SOR_PRIADO_OCTU_0_28", "Índice de salarios - No registrado", "índice", "INDEC", "M"),
    # Sector externo y monetario (BCRA)
    "tipo_cambio": Serie("92.1_TCV_0_0_21", "Tipo de cambio de valuación (promedio mensual)", "$ por USD", "BCRA", "M"),
    "reservas_musd": Serie("92.1_RID_0_0_32", "Reservas internacionales del BCRA", "millones de USD", "BCRA", "M"),
    "badlar": Serie("89.1_TIB_0_0_20", "Tasa BADLAR bancos privados", "% TNA", "BCRA", "M"),
    "expo_musd": Serie("74.3_IET_0_M_16", "Exportaciones (ICA)", "millones de USD", "INDEC", "M"),
    "impo_musd": Serie("74.3_IIT_0_M_25", "Importaciones (ICA)", "millones de USD", "INDEC", "M"),
    # Fiscal (Sector Público Nacional no financiero, base caja - IMIG)
    "resultado_primario": Serie("452.3_RESULTADO_RIO_0_M_18_54", "Resultado primario SPN", "millones de $", "Min. Economía", "M"),
    "resultado_financiero": Serie("452.3_RESULTADO_ERO_0_M_20_25", "Resultado financiero SPN", "millones de $", "Min. Economía", "M"),
    "intereses": Serie("452.3_INTERESES_TOS_0_M_15_62", "Intereses netos de la deuda", "millones de $", "Min. Economía", "M"),
    "recaudacion": Serie("172.3_TL_RECAION_M_0_0_17", "Recaudación tributaria total", "millones de $", "ARCA", "M"),
}

# ---------------------------------------------------------------------------
# Catálogo nacional (trimestral)
# ---------------------------------------------------------------------------
TRIMESTRALES: dict[str, Serie] = {
    "pib_const": Serie("4.2_OGP_2004_T_17", "PIB a precios constantes de 2004", "millones de $ de 2004", "INDEC", "T"),
    "pib_nominal": Serie("4.4_OGP_2004_T_17", "PIB a precios corrientes", "millones de $", "INDEC", "T"),
    "desocupacion": Serie("45.2_ECTDT_0_T_33", "Tasa de desocupación (31 aglomerados)", "proporción", "INDEC-EPH", "T"),
    "actividad": Serie("43.2_ECTAT_0_T_33", "Tasa de actividad (31 aglomerados)", "proporción", "INDEC-EPH", "T"),
    "empleo": Serie("44.2_ECTET_0_T_30", "Tasa de empleo (31 aglomerados)", "proporción", "INDEC-EPH", "T"),
}

# Desocupación por aglomerado (EPH). Código -> (id de la API, nombre)
DESOCUPACION_AGLOMERADOS: dict[str, tuple[str, str]] = {
    "gran_la_plata": ("45.2_ECTDTGLP_0_T_47", "Gran La Plata"),
    "bahia_blanca": ("45.2_ECTDTBB_0_T_46", "Bahía Blanca-Cerri"),
    "gran_rosario": ("45.2_ECTDTGR_0_T_46", "Gran Rosario"),
    "gran_santa_fe": ("45.2_ECTDTGSF_0_T_47", "Gran Santa Fe"),
    "gran_parana": ("45.2_ECTDTGP_0_T_45", "Gran Paraná"),
    "posadas": ("45.2_ECTDTP_0_T_41", "Posadas"),
    "gran_resistencia": ("45.2_ECTDTGR_0_T_50", "Gran Resistencia"),
    "comodoro": ("45.2_ECTDTCR_0_T_52", "Comodoro Rivadavia-Rada Tilly"),
    "gran_mendoza": ("45.2_ECTDTGM_0_T_46", "Gran Mendoza"),
    "corrientes": ("45.2_ECTDTC_0_T_44", "Corrientes"),
    "gran_cordoba": ("45.2_ECTDTGC_0_T_46", "Gran Córdoba"),
    "concordia": ("45.2_ECTDTC_0_T_43", "Concordia"),
    "formosa": ("45.2_ECTDTF_0_T_41", "Formosa"),
    "neuquen": ("45.2_ECTDTN_0_T_41", "Neuquén-Plottier"),
    "santiago": ("45.2_ECTDTSDE_0_T_53", "Santiago del Estero-La Banda"),
    "jujuy": ("45.2_ECTDTJ_0_T_39", "Jujuy-Palpalá"),
    "rio_gallegos": ("45.2_ECTDTRG_0_T_46", "Río Gallegos"),
    "gran_catamarca": ("45.2_ECTDTGC_0_T_48", "Gran Catamarca"),
    "salta": ("45.2_ECTDTS_0_T_39", "Salta"),
    "la_rioja": ("45.2_ECTDTLR_0_T_42", "La Rioja"),
    "san_luis": ("45.2_ECTDTSL_0_T_42", "San Luis-El Chorrillo"),
    "gran_san_juan": ("45.2_ECTDTGSJ_0_T_47", "Gran San Juan"),
    "gran_tucuman": ("45.2_ECTDTGT_0_T_46", "Gran Tucumán-Tafí Viejo"),
    "santa_rosa": ("45.2_ECTDTSR_0_T_44", "Santa Rosa-Toay"),
    "ushuaia": ("45.2_ECTDTURG_0_T_52", "Ushuaia-Río Grande"),
    "caba": ("45.2_ECTDTC_0_T_38", "Ciudad Autónoma de Buenos Aires"),
    "partidos_gba": ("45.2_ECTDTPG_0_T_46", "Partidos del Gran Buenos Aires"),
    "mar_del_plata": ("45.2_ECTDTMDP_0_T_47", "Mar del Plata"),
    "rio_cuarto": ("45.2_ECTDTRC_0_T_44", "Río Cuarto"),
    "san_nicolas": ("45.2_ECTDTSN_0_T_45", "San Nicolás-Villa Constitución"),
    "rawson_trelew": ("45.2_ECTDTRT_0_T_47", "Rawson-Trelew"),
    "viedma": ("45.2_ECTDTVCP_0_T_57", "Viedma-Carmen de Patagones"),
}

# ---------------------------------------------------------------------------
# Catálogo ampliado (mensual): empleo, masa salarial, crédito, mora y actividad
# ---------------------------------------------------------------------------
AMPLIADAS: dict[str, Serie] = {
    # Trabajadores registrados por modalidad (SIPA, miles, sin estacionalidad)
    "sipa_total": Serie("151.1_TL_SIN_TAC_2012_M_15", "Trabajadores registrados - Total", "miles", "Sec. de Trabajo (SIPA)", "M"),
    "sipa_asal_priv": Serie("151.1_AARIADOTAC_2012_M_26", "Asalariados del sector privado", "miles", "Sec. de Trabajo (SIPA)", "M"),
    "sipa_asal_pub": Serie("151.1_AARIADOTAC_2012_M_25", "Asalariados del sector público", "miles", "Sec. de Trabajo (SIPA)", "M"),
    "sipa_casas": Serie("151.1_AARIADOTAC_2012_M_40", "Asalariados de casas particulares", "miles", "Sec. de Trabajo (SIPA)", "M"),
    "sipa_autonomos": Serie("151.1_IPENDIETAC_2012_M_34", "Independientes autónomos", "miles", "Sec. de Trabajo (SIPA)", "M"),
    "sipa_monotributo": Serie("151.1_IPENDIETAC_2012_M_36", "Independientes monotributo", "miles", "Sec. de Trabajo (SIPA)", "M"),
    "sipa_monot_social": Serie("151.1_IPENDIETAC_2012_M_43", "Independientes monotributo social", "miles", "Sec. de Trabajo (SIPA)", "M"),
    # Remuneración de los asalariados registrados del sector privado (SIPA, $ corrientes;
    # junio y diciembre incluyen el medio aguinaldo)
    "rem_promedio": Serie("153.1_RNERACIDIO_2009_M_21", "Remuneración bruta promedio, asalariados privados (SIPA)", "$", "Sec. de Trabajo (SIPA)", "M"),
    "rem_mediana": Serie("153.1_RNERACIANA_2009_M_20", "Remuneración bruta mediana, asalariados privados (SIPA)", "$", "Sec. de Trabajo (SIPA)", "M"),
    # Mora: irregularidad de las financiaciones al sector privado (BCRA, % de la cartera)
    "irreg_total": Serie("332.2_SISTEMA_FIADA__53", "Irregularidad de la cartera privada - sistema financiero", "%", "BCRA", "M"),
    "irreg_publicos": Serie("332.2_BANCOS_PUBADA__55", "Irregularidad de la cartera privada - bancos públicos", "%", "BCRA", "M"),
    "irreg_privados": Serie("332.2_BANCOS_PRIADA__51", "Irregularidad de la cartera privada - bancos privados", "%", "BCRA", "M"),
    # Crédito en pesos al sector privado (BCRA, saldos en millones de $)
    "prest_pesos": Serie("91.1_DETALLE_PRPOT_0_0_25", "Préstamos en pesos al sector privado", "millones de $", "BCRA", "M"),
    "prest_consumo": Serie("91.1_DETALLE_PREND_0_0_33", "Préstamos al consumo (personales y tarjetas)", "millones de $", "BCRA", "M"),
    "prest_comerciales": Serie("91.1_PRESTAMOS_LES_0_0_37", "Préstamos comerciales (adelantos + documentos)", "millones de $", "BCRA", "M"),
    "prest_personales": Serie("91.1_DETALLE_PRLES_0_0_52", "Préstamos personales", "millones de $", "BCRA", "M"),
    "tasa_tarjetas": Serie("89.1_TS_INTETAS_0_M_30", "Tasa de interés de tarjetas de crédito", "% TNA", "BCRA", "M"),
    # Actividad sectorial (INDEC)
    "ipi": Serie("453.1_SERIE_ORIGNAL_0_0_14_46", "Índice de producción industrial manufacturero (IPI)", "índice 2016=100", "INDEC", "M"),
    "isac": Serie("33.2_ISAC_NIVELRAL_0_M_18_63", "Indicador sintético de la actividad de la construcción (ISAC)", "índice 2004=100", "INDEC", "M"),
    "super_desest": Serie("455.1_VENTAS_PREADA_0_M_44_44", "Ventas en supermercados a precios constantes (desest.)", "índice 2017=100", "INDEC", "M"),
    # Distribución del ingreso (EPH, trimestral)
    "gini": Serie("65.1_CGI_0_0_21", "Coeficiente de Gini del ingreso per cápita familiar", "coeficiente", "INDEC-EPH", "T"),
}

# Asalariados registrados del sector privado por provincia (SIPA, miles)
SIPA_PROVINCIAS: dict[str, str] = {
    "Buenos Aires": "154.1_BOS_AIRRES_C_0_0_12", "CABA": "154.1_CABA_C_0_0_4",
    "Catamarca": "154.1_CMARCARCA_C_0_0_9", "Chaco": "154.1_COACO_C_0_0_5",
    "Chubut": "154.1_CUTBUT_C_0_0_6", "Córdoba": "154.1_COBAOBA_C_0_0_7",
    "Corrientes": "154.1_CIENTESTES_C_0_0_10", "Entre Ríos": "154.1_EE_RIOSIOS_C_0_0_10",
    "Formosa": "154.1_FOSAOSA_C_0_0_7", "Jujuy": "154.1_JYJUY_C_0_0_5",
    "La Pampa": "154.1_LAMPAMPA_C_0_0_8", "La Rioja": "154.1_LIOJAOJA_C_0_0_8",
    "Mendoza": "154.1_MOZAOZA_C_0_0_7", "Misiones": "154.1_MONESNES_C_0_0_8",
    "Neuquén": "154.1_NUENUEN_C_0_0_7", "Río Negro": "154.1_RNEGROGRO_C_0_0_9",
    "Salta": "154.1_SALTA_C_0_0_5", "San Juan": "154.1_SJUANUAN_C_0_0_8",
    "San Luis": "154.1_SLUISUIS_C_0_0_8", "Santa Cruz": "154.1_SA_CRUZRUZ_C_0_0_10",
    "Santa Fe": "154.1_SA_FE_FE_C_0_0_8", "Santiago del Estero": "154.1_SIAGO_EERO_C_0_0_15",
    "Tierra del Fuego": "154.1_TRA_FUEEGO_C_0_0_12", "Tucumán": "154.1_TMANMAN_C_0_0_7",
}


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------
def _pedir(id_serie: str, start_date: str | None, timeout: float) -> dict:
    import requests

    params = {"ids": id_serie, "format": "json", "limit": 1000}
    if start_date:
        params["start_date"] = start_date
    r = requests.get(API_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def descargar_serie(id_serie: str, start_date: str | None = None, intentos: int = 3,
                    timeout: float = 25.0, pausa: float = 0.3) -> pd.Series:
    """Descarga UNA serie y devuelve un pd.Series indexado por fecha (inicio de período).

    Repite el pedido `intentos` veces y une los valores no nulos, porque la API
    a veces omite observaciones de manera intermitente.
    """
    valores: dict[pd.Timestamp, float] = {}
    ultimo_error: Exception | None = None
    for _ in range(max(1, intentos)):
        try:
            data = _pedir(id_serie, start_date, timeout).get("data", [])
            for fecha, valor in data:
                if valor is not None:
                    valores[pd.Timestamp(fecha)] = float(valor)
        except Exception as e:  # red caída, timeout, 5xx...
            ultimo_error = e
        time.sleep(pausa)
    if not valores and ultimo_error is not None:
        raise ConnectionError(f"No se pudo descargar {id_serie}: {ultimo_error}")
    return pd.Series(valores, name=id_serie).sort_index()


def descargar_varias(catalogo: dict[str, Serie] | dict[str, str], start_date: str | None = None,
                     intentos: int = 2, claves: Iterable[str] | None = None,
                     progreso=None) -> pd.DataFrame:
    """Descarga varias series (de a una) y las junta en un DataFrame ancho."""
    claves = list(claves or catalogo.keys())
    columnas = {}
    for i, k in enumerate(claves):
        v = catalogo[k]
        id_serie = v.id if isinstance(v, Serie) else (v[0] if isinstance(v, tuple) else v)
        try:
            columnas[k] = descargar_serie(id_serie, start_date=start_date, intentos=intentos)
        except ConnectionError:
            columnas[k] = pd.Series(dtype=float)
        if progreso:
            progreso((i + 1) / len(claves), k)
    df = pd.DataFrame(columnas)
    df.index.name = "fecha"
    return df
