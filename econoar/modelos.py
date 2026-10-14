"""
Modelos estadísticos y de machine learning de econoAR.

  * pronosticar_inflacion(): compara modelos (ingenuo, media móvil, ARIMA y
    Ridge con rezagos) con una validación "rolling origin" y pronostica la
    inflación mensual con el mejor (ARIMA da intervalos de confianza).
  * ley_de_okun(): MCO de la variación interanual de la desocupación contra el
    crecimiento interanual del PIB (errores HAC).
  * regresion_pobreza(): corte transversal de los 31 aglomerados de la EPH.
  * clusters_aglomerados(): K-means sobre indicadores laborales y de pobreza,
    con PCA para graficar.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Pronóstico de inflación
# ---------------------------------------------------------------------------
@dataclass
class ResultadoPronostico:
    historia: pd.Series            # inflación mensual observada (%)
    pronostico: pd.DataFrame       # columnas: media, inf_80, sup_80, inf_95, sup_95
    ranking: pd.DataFrame          # MAE / RMSE por modelo en la validación
    modelo_elegido: str
    orden_arima: tuple
    desde: str


def _features_ml(df: pd.DataFrame, rezagos: int = 3) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for k in range(1, rezagos + 1):
        X[f"infl_l{k}"] = df["infl"].shift(k)
    X["nucleo_l1"] = df["nucleo"].shift(1)
    X["regulados_l1"] = df["regulados"].shift(1)
    X["tc_l1"] = df["tc"].shift(1)
    X["mes"] = df.index.month
    X = pd.get_dummies(X, columns=["mes"], drop_first=True, dtype=float)
    return X


def _ajustar_arima(y: pd.Series, orden: tuple):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(y.values, order=orden, trend="c").fit()


def elegir_orden_arima(y: pd.Series, max_p: int = 2, max_q: int = 2) -> tuple:
    mejor, mejor_aic = (1, 0, 0), np.inf
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            aic = _ajustar_arima(y, (p, 0, q)).aic
        except Exception:
            continue
        if aic < mejor_aic:
            mejor, mejor_aic = (p, 0, q), aic
    return mejor


def pronosticar_inflacion(mensual: pd.DataFrame, horizonte: int = 6, desde: str = "2024-06-01",
                          meses_validacion: int = 12) -> ResultadoPronostico:
    """Pronóstico de la inflación mensual (var. % m/m del IPC nivel general).

    Se estima sobre el régimen posterior a la desinflación (desde jun-2024 por
    defecto): incluir 2023-24 haría que los modelos "recuerden" el salto de
    dic-2023 y sobreestimen la inflación de los próximos meses.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def mm(col):
        s = np.exp(np.log(mensual[col]).interpolate(limit_area="inside"))
        return (s / s.shift(1) - 1) * 100

    df = pd.DataFrame({
        "infl": mm("ipc_nivel_general"), "nucleo": mm("ipc_nucleo"),
        "regulados": mm("ipc_regulados"),
        "tc": (mensual["tipo_cambio"] / mensual["tipo_cambio"].shift(1) - 1) * 100,
    }).dropna(subset=["infl"])
    df["tc"] = df["tc"].ffill()
    y_full = df["infl"]
    y = y_full[y_full.index >= desde]

    # --- Validación rolling-origin (pronóstico a 1 paso) ---
    errores = {"Ingenuo (último dato)": [], "Media móvil 3m": [], "ARIMA": [], "Ridge con rezagos (ML)": []}
    X_all = _features_ml(df).dropna()
    fechas_test = y.index[-meses_validacion:]
    orden = elegir_orden_arima(y.iloc[:-meses_validacion])
    for f in fechas_test:
        hist = y[y.index < f]
        real = y.loc[f]
        errores["Ingenuo (último dato)"].append(real - hist.iloc[-1])
        errores["Media móvil 3m"].append(real - hist.iloc[-3:].mean())
        try:
            fc = _ajustar_arima(hist, orden).forecast(1)[0]
        except Exception:
            fc = hist.mean()
        errores["ARIMA"].append(real - fc)
        # Ridge: entrena con toda la historia disponible (desde 2017) hasta f
        Xtr = X_all[X_all.index < f]
        ytr = y_full.loc[Xtr.index]
        if f in X_all.index:
            modelo = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(Xtr, ytr)
            errores["Ridge con rezagos (ML)"].append(real - modelo.predict(X_all.loc[[f]])[0])
        else:
            errores["Ridge con rezagos (ML)"].append(np.nan)

    ranking = pd.DataFrame({
        k: {"MAE (p.p.)": np.nanmean(np.abs(v)), "RMSE (p.p.)": np.sqrt(np.nanmean(np.square(v)))}
        for k, v in errores.items()
    }).T.sort_values("MAE (p.p.)")
    elegido = ranking.index[0]

    # --- Pronóstico final con ARIMA (para tener intervalos) ---
    orden = elegir_orden_arima(y)
    res = _ajustar_arima(y, orden)
    pred = res.get_forecast(horizonte)
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizonte, freq="MS")
    ci80 = pred.conf_int(alpha=0.20)
    ci95 = pred.conf_int(alpha=0.05)
    fc = pd.DataFrame({
        "media": pred.predicted_mean, "inf_80": ci80[:, 0], "sup_80": ci80[:, 1],
        "inf_95": ci95[:, 0], "sup_95": ci95[:, 1],
    }, index=idx)
    # Si otro modelo ganó la validación, su trayectoria pasa a ser la "media" y las
    # bandas salen de sus errores fuera de muestra (RMSE de la validación x raíz de h).
    if elegido != "ARIMA":
        if elegido == "Media móvil 3m":
            media = np.repeat(y.iloc[-3:].mean(), horizonte)
        elif elegido == "Ingenuo (último dato)":
            media = np.repeat(y.iloc[-1], horizonte)
        else:  # Ridge: pronóstico recursivo no implementado -> se usa ARIMA como trayectoria
            media = fc["media"].values
        rmse = ranking.loc[elegido, "RMSE (p.p.)"]
        h = np.sqrt(np.arange(1, horizonte + 1))
        fc = pd.DataFrame({
            "media": media,
            "inf_80": media - 1.2816 * rmse * h, "sup_80": media + 1.2816 * rmse * h,
            "inf_95": media - 1.96 * rmse * h, "sup_95": media + 1.96 * rmse * h,
        }, index=idx)
        fc["media_arima"] = pred.predicted_mean
    else:
        fc["media_arima"] = fc["media"]
    return ResultadoPronostico(y_full, fc, ranking, elegido, orden, desde)


def inflacion_acumulada(serie_mm: pd.Series) -> float:
    return (np.prod(1 + serie_mm / 100) - 1) * 100


# ---------------------------------------------------------------------------
# Ley de Okun
# ---------------------------------------------------------------------------
def ley_de_okun(trimestral: pd.DataFrame, excluir_2020: bool = True):
    import statsmodels.api as sm

    t = trimestral.copy()
    t["crec_pib"] = (t["pib_const"] / t["pib_const"].shift(4) - 1) * 100
    t["delta_u"] = t["desocupacion"] - t["desocupacion"].shift(4)
    d = t[["crec_pib", "delta_u"]].dropna()
    if excluir_2020:
        d = d[(d.index.year != 2020) & ~((d.index.year == 2021) & (d.index.quarter <= 2))]
    X = sm.add_constant(d["crec_pib"])
    modelo = sm.OLS(d["delta_u"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return modelo, d


# ---------------------------------------------------------------------------
# Corte transversal de aglomerados
# ---------------------------------------------------------------------------
def regresion_pobreza(aglomerados: pd.DataFrame):
    import statsmodels.formula.api as smf

    d = aglomerados[aglomerados["tipo"] == "aglomerado"].dropna(subset=["personas_pobreza"])
    modelo = smf.ols("personas_pobreza ~ tasa_desocupacion + tasa_empleo + tasa_subocupacion",
                     data=d).fit(cov_type="HC1")
    return modelo, d


VARIABLES_CLUSTER = ["tasa_actividad", "tasa_empleo", "tasa_desocupacion", "tasa_subocupacion",
                     "personas_pobreza", "personas_indigencia"]


def clusters_aglomerados(aglomerados: pd.DataFrame, k: int = 4, semilla: int = 7):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    d = aglomerados[aglomerados["tipo"] == "aglomerado"].dropna(subset=VARIABLES_CLUSTER).copy()
    Z = StandardScaler().fit_transform(d[VARIABLES_CLUSTER])
    siluetas = {}
    for kk in range(2, 7):
        etiquetas = KMeans(n_clusters=kk, n_init=20, random_state=semilla).fit_predict(Z)
        siluetas[kk] = silhouette_score(Z, etiquetas)
    km = KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(Z)
    d["cluster"] = km.labels_
    pca = PCA(n_components=2, random_state=semilla).fit(Z)
    comp = pca.transform(Z)
    d["pc1"], d["pc2"] = comp[:, 0], comp[:, 1]
    perfiles = d.groupby("cluster")[VARIABLES_CLUSTER].mean().round(1)
    nombres = _nombrar_clusters(perfiles)
    d["perfil"] = d["cluster"].map(nombres)
    perfiles.index = perfiles.index.map(nombres)
    return d, perfiles, pd.Series(siluetas, name="silueta"), pca.explained_variance_ratio_


def _nombrar_clusters(perfiles: pd.DataFrame) -> dict:
    """Pone nombres legibles a los clusters según su centroide (relativo a la media)."""
    z = (perfiles - perfiles.mean()) / perfiles.std(ddof=0).replace(0, 1)
    nombres = {}
    usados = set()
    for c, fila in z.iterrows():
        if fila["personas_pobreza"] > 0.8:
            n = "Alta pobreza"
        elif fila["tasa_desocupacion"] > 0.8:
            n = "Alto desempleo y subocupación" if fila["tasa_subocupacion"] > 0.5 else "Alto desempleo"
        elif fila["tasa_empleo"] > 0.8 and fila["personas_pobreza"] < 0:
            n = "Mercado laboral dinámico"
        elif fila["tasa_actividad"] < -0.8:
            n = "Baja participación laboral"
        elif fila["tasa_subocupacion"] > 0.8:
            n = "Alta subocupación"
        else:
            n = "Situación intermedia"
        base, i = n, 2
        while n in usados:
            n = f"{base} {i}"
            i += 1
        usados.add(n)
        nombres[c] = n
    return nombres


# ---------------------------------------------------------------------------
# Mora provincial
# ---------------------------------------------------------------------------
def regresion_mora(provincias: pd.DataFrame):
    """¿Por qué la mora de las familias difiere entre provincias?

    Corte transversal de 23 provincias (Río Negro sin dato de mora):
      (1) mora ~ log(ingreso per cápita familiar medio, EPH 1T-2026)
      (2) se agregan desocupación e informalidad como controles.
    Errores robustos (HC1). Con n = 23 los resultados son descriptivos.
    """
    import statsmodels.formula.api as smf

    d = provincias.dropna(subset=["mora_personas_jun2026", "ipcf_medio"]).copy()
    d["log_ipcf"] = np.log(d["ipcf_medio"])
    simple = smf.ols("mora_personas_jun2026 ~ log_ipcf", data=d).fit(cov_type="HC1")
    controles = smf.ols("mora_personas_jun2026 ~ log_ipcf + tasa_desocupacion + informalidad_asal",
                        data=d).fit(cov_type="HC1")
    return simple, controles, d


# ---------------------------------------------------------------------------
# Convergencia regional (PBG per cápita)
# ---------------------------------------------------------------------------
@dataclass
class ResultadoConvergencia:
    modelo: object
    datos: pd.DataFrame
    beta: float
    p_valor: float
    velocidad: float        # λ: velocidad de convergencia anual implícita
    vida_media: float       # años para cerrar la mitad de la brecha
    sigma_inicial: float    # desvío del log del PBG per cápita en el año inicial
    sigma_final: float


def convergencia_pbg(vab: pd.DataFrame, poblacion: pd.DataFrame, anio_ini: int = 2010,
                     anio_fin: int = 2022) -> ResultadoConvergencia:
    """Convergencia β y σ del PBG per cápita entre provincias (años censales 2010 y 2022).

    vab: años x provincias (millones de $ de 2004, CEPAL); poblacion: provincia,
    poblacion_2010, poblacion_2022. Se estima
        g_i = a + b * ln(y_i,2010) + e_i,  g = crecimiento % anual del PBG per cápita
    b < 0 indica que las provincias más pobres crecieron más rápido.
    """
    import statsmodels.formula.api as smf

    pop = poblacion.set_index("provincia")
    y0 = vab.loc[anio_ini] * 1e6 / pop[f"poblacion_{anio_ini}"]
    y1 = vab.loc[anio_fin] * 1e6 / pop[f"poblacion_{anio_fin}"]
    T = anio_fin - anio_ini
    d = pd.DataFrame({"pbg_pc_ini": y0, "pbg_pc_fin": y1})
    d["crec_anual"] = ((y1 / y0) ** (1 / T) - 1) * 100
    d["log_pbg_pc_ini"] = np.log(y0)
    d = d.dropna().rename_axis("provincia").reset_index()
    m = smf.ols("crec_anual ~ log_pbg_pc_ini", data=d).fit(cov_type="HC1")
    b = m.params["log_pbg_pc_ini"]
    arg = 1 + b * T / 100
    lam = -np.log(arg) / T if arg > 0 else np.nan
    vida = np.log(2) / lam if lam and lam > 0 else np.inf
    return ResultadoConvergencia(m, d, b, m.pvalues["log_pbg_pc_ini"], lam, vida,
                                 float(np.log(y0).std(ddof=0)), float(np.log(y1).std(ddof=0)))
