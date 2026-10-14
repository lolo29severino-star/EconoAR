"""
econoAR · Tablero macroeconómico de la Argentina (nación, regiones, provincias y aglomerados)

Ejecutar:   streamlit run app.py
Opcional:   export ANTHROPIC_API_KEY=...   (activa el asistente con Claude; sin clave usa el motor local)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from econoar import datos as D
from econoar import ia
from econoar import modelos as M
from econoar import graficos as G

st.set_page_config(page_title="econoAR", page_icon="📈", layout="wide")

AZUL, CELESTE, NARANJA, VERDE, ROJO, GRIS = "#2563eb", "#0ea5e9", "#ea580c", "#16a34a", "#dc2626", "#64748b"
COLORES_REG = {"GBA": "#2563eb", "Pampeana": "#16a34a", "Noreste": "#ea580c", "Noroeste": "#a855f7",
               "Cuyo": "#e11d48", "Patagonia": "#0891b2"}
PLANTILLA = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                 legend=dict(orientation="h", y=-0.15), font=dict(size=13))


def fmt(x, dec=1, suf=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "s/d"
    s = f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s + suf


def md(texto: str) -> str:
    """Escapa '$' para que Streamlit no lo interprete como LaTeX."""
    return texto.replace("$", "\\$")


def mes(f):
    return ia._mes(f) if f is not None else "s/d"


# ---------------------------------------------------------------------------
# Datos (cacheados)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar(en_vivo: bool):
    d = D.cargar_snapshot()
    if en_vivo:
        d = D.actualizar_desde_api(d)
    return d


@st.cache_data(show_spinner=False)
def correr_modelos(_d: D.Datos, clave: str):
    pron = M.pronosticar_inflacion(_d.mensual)
    okun, okun_d = M.ley_de_okun(_d.trimestral)
    rp, rp_d = M.regresion_pobreza(_d.aglomerados)
    cl, perf, sil, ev = M.clusters_aglomerados(_d.aglomerados, k=4)
    mora = M.regresion_mora(_d.provincias)
    conv = M.convergencia_pbg(D.pbg_largo(_d), pd.read_csv(D.DATA_DIR / "provincias_base.csv"))
    return pron, (okun, okun_d), (rp, rp_d), (cl, perf, sil, ev), mora, conv


def selector_indicador(clave: str, grupo_inicial: str = "Trabajo", indicador_inicial: str | None = None):
    """Dos selectores encadenados: grupo temático e indicador provincial."""
    grupos = list(D.GRUPOS_PROVINCIALES)
    c1, c2 = st.columns([1, 2])
    g = c1.selectbox("Tema", grupos, index=grupos.index(grupo_inicial), key=f"{clave}_g")
    opciones = D.GRUPOS_PROVINCIALES[g]
    idx = opciones.index(indicador_inicial) if indicador_inicial in opciones else 0
    return c2.selectbox("Indicador", opciones, index=idx, key=f"{clave}_i",
                        format_func=lambda k: D.INDICADORES_PROVINCIALES[k][0])


def mapa_y_ranking(p: pd.DataFrame, ind: str, altura: int = 720):
    nombre, unidad = D.INDICADORES_PROVINCIALES[ind]
    divergente = ind in D.DIVERGENTES
    escala = "Reds" if ind in D.MALO_SI_ALTO else ("RdBu" if divergente else "Blues")
    grandes = p[ind].abs().max() >= 1000
    col1, col2 = st.columns([3, 2])
    fig = G.mapa_provincias(p, datos.geojson, ind, escala=escala, unidad=unidad,
                            medio=0 if divergente else None, decimales=0 if grandes else 1, altura=altura)
    col1.plotly_chart(fig, width="stretch")
    r = p[["provincia", ind]].dropna().sort_values(ind, ascending=False)
    rango = None
    if divergente:
        amp = float(np.percentile(r[ind].abs(), 90)) or 1.0
        rango = (-amp, amp)
    fig = px.bar(r, x=ind, y="provincia", orientation="h", color=ind, color_continuous_scale=escala,
                 labels={ind: unidad, "provincia": ""}, range_color=rango)
    fig.update_layout(**{**PLANTILLA, "hovermode": "closest"}, height=altura, coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed"))
    col2.plotly_chart(fig, width="stretch")
    sin_dato = p.loc[p[ind].isna(), "provincia"].tolist()
    if sin_dato:
        st.caption(f"Sin dato: {', '.join(sin_dato)}.")


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 econoAR")
    st.caption("Macroeconomía argentina: país, regiones, provincias y aglomerados.")
    fuente = st.radio("Fuente de datos", ["Snapshot (sin internet)", "API en vivo (datos.gob.ar)"], index=0,
                      help="La API en vivo baja las series una por una; tarda ~1–2 minutos.")
    seccion = st.radio("Sección", ["Resumen", "Precios", "Actividad y salarios", "Trabajo", "Empleo e ingresos",
                                   "Crédito y mora", "Pobreza", "Fiscal y sector externo", "Provincias",
                                   "PBG y finanzas provinciales", "Modelos e IA predictiva", "Asistente IA"])
    st.divider()
    api_key = st.text_input("Clave de Anthropic (opcional)", type="password",
                            value=os.environ.get("ANTHROPIC_API_KEY", ""),
                            help="Activa respuestas e informes redactados por Claude. Sin clave se usa un motor local.")

with st.spinner("Cargando datos…"):
    datos = cargar(fuente.startswith("API"))
for aviso in datos.avisos:
    st.warning(aviso)

(pron, (okun, okun_d), (rp, rp_d), (clusters, perfiles, siluetas, var_pca),
 (mora_s, mora_c, mora_d), conv) = correr_modelos(datos, datos.origen)
hechos = ia.hechos_clave(datos, pron)
infl = D.inflacion(datos)
m, t = datos.mensual, datos.trimestral

st.sidebar.caption(f"Origen: {datos.origen}")

# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------
if seccion == "Resumen":
    st.title("Panorama macroeconómico")
    v, f = D.ultimo_valor(infl["general_mm"])
    ia_, _ = D.ultimo_valor(infl["general_ia"])
    u, fu = D.ultimo_valor(t["desocupacion"])
    u_prev = t["desocupacion"].dropna().iloc[-5]
    tp, pib_anual = D.pib_variaciones(datos)
    g, fg = D.ultimo_valor(tp["pib_ia"])
    pob = datos.pobreza_hist.iloc[-1]
    pob_prev = datos.pobreza_hist.iloc[-3]
    fa = D.fiscal_acumulado(datos, int(m["resultado_primario"].dropna().index.max().year))
    res, fres = D.ultimo_valor(m["reservas_musd"])
    tc, ftc = D.ultimo_valor(m["tipo_cambio"])

    c = st.columns(4)
    c[0].metric(f"Inflación {mes(f)}", fmt(v, 1, "%"), f"{fmt(ia_, 1)}% interanual", delta_color="off", delta_arrow="off",
                border=True, chart_data=infl["general_mm"].dropna().iloc[-18:].round(2).tolist(), chart_type="bar")
    c[1].metric(f"Desocupación {ia._trim(fu)}", fmt(u, 1, "%"), f"{fmt(u - u_prev, 1)} p.p. vs. año anterior",
                delta_color="inverse", border=True, chart_data=t["desocupacion"].dropna().iloc[-12:].tolist())
    c[2].metric("Pobreza (personas) 1S-2026", fmt(pob["personas_pobreza"], 1, "%"),
                f"{fmt(pob['personas_pobreza'] - pob_prev['personas_pobreza'], 1)} p.p. vs. 1S-2025",
                delta_color="inverse", border=True)
    c[3].metric(f"PIB {ia._trim(fg)}", fmt(g, 1, "%"), "interanual, precios constantes", delta_color="off", delta_arrow="off", border=True)
    c = st.columns(4)
    c[0].metric(f"Resultado primario {fa['anio']}", f"$ {fmt(fa['primario'] / 1e6, 2)} bill.",
                f"financiero $ {fmt(fa['financiero'] / 1e6, 2)} bill. (a {mes(fa['ultimo_mes'])})", delta_color="off", delta_arrow="off", border=True)
    c[1].metric(f"Reservas BCRA ({mes(fres)})", f"USD {fmt(res / 1000, 1)} mil M", "promedio mensual", delta_color="off", delta_arrow="off", border=True)
    c[2].metric(f"Dólar oficial ({mes(ftc)})", f"$ {fmt(tc, 0)}", "tipo de cambio de valuación", delta_color="off", delta_arrow="off", border=True)
    p0 = pron.pronostico.iloc[0]
    c[3].metric(f"Pronóstico inflación {mes(pron.pronostico.index[0])}", fmt(p0['media'], 1, "%"),
                f"IC 80%: {fmt(p0['inf_80'], 1)}–{fmt(p0['sup_80'], 1)}%", delta_color="off", delta_arrow="off", border=True)
    er = D.empleo_registrado(datos)
    ms = D.masa_salarial(datos)
    mo = D.mora_nacional(datos)
    cr = D.credito_real(datos)
    tot, ftot = D.ultimo_valor(er["sipa_total"])
    m12, fm12 = D.ultimo_valor(ms["masa_real_12m_ia"])
    irr, firr = D.ultimo_valor(mo["irreg_total"])
    cia, fcia = D.ultimo_valor(cr["prest_pesos_real_ia"])
    c = st.columns(4)
    c[0].metric(f"Trabajadores registrados ({mes(ftot)})", f"{fmt(tot / 1000, 2)} M",
                f"{fmt(er['sipa_total_ia'].loc[ftot])}% interanual", border=True,
                chart_data=er["sipa_total"].dropna().iloc[-24:].round(1).tolist())
    c[1].metric(f"Masa salarial real privada (12 m a {mes(fm12)})", fmt(m12, 1, "%"), "interanual · SIPA",
                delta_color="off", delta_arrow="off", border=True,
                chart_data=ms["masa_real_12m_ia"].dropna().iloc[-24:].round(2).tolist())
    c[2].metric(f"Mora bancaria ({mes(firr)})", fmt(irr, 1, "%"),
                f"{fmt(irr - mo['irreg_total'].shift(12).loc[firr], 1)} p.p. en un año", delta_color="inverse", border=True,
                chart_data=mo["irreg_total"].iloc[-24:].round(2).tolist())
    c[3].metric(f"Crédito en pesos ({mes(fcia)})", fmt(cia, 1, "%"), "variación real interanual",
                delta_color="off", delta_arrow="off", border=True,
                chart_data=cr["prest_pesos_real_bill"].dropna().iloc[-24:].round(1).tolist())

    st.subheader("Informe automático")
    with st.spinner("Redactando…"):
        texto, motor = ia.generar_informe(hechos, api_key=api_key or None)
    st.markdown(md(texto))
    st.caption("Redactado por Claude a partir de los datos del tablero." if motor == "claude"
               else "Resumen generado con plantilla (cargá una clave de Anthropic para que lo redacte Claude).")

    st.subheader("Desocupación y pobreza por aglomerado")
    a = datos.aglomerados[datos.aglomerados["tipo"] == "aglomerado"]
    fig = px.scatter(a, x="tasa_desocupacion", y="personas_pobreza", size="tasa_actividad", color="tasa_empleo",
                     hover_name="nombre", color_continuous_scale="Blues",
                     labels={"tasa_desocupacion": "Desocupación 2T-2026 (%)", "personas_pobreza": "Pobreza 1S-2026 (% personas)",
                             "tasa_empleo": "Tasa de empleo"})
    fig.update_layout(**PLANTILLA, height=460)
    st.plotly_chart(fig, width="stretch")

elif seccion == "Precios":
    st.title("Precios")
    desde = st.slider("Desde", 2018, 2026, 2022)
    i = infl[infl.index.year >= desde]
    col1, col2 = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=i.index, y=i["general_mm"], name="Nivel general", marker_color=AZUL)
    fig.add_scatter(x=i.index, y=i["nucleo_mm"], name="Núcleo", line=dict(color=NARANJA, width=2))
    fig.update_layout(title="Inflación mensual (%)", **PLANTILLA, height=380)
    col1.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    for c_, n_, col_ in [("general_ia", "Nivel general", AZUL), ("nucleo_ia", "Núcleo", NARANJA),
                         ("regulados_ia", "Regulados", VERDE), ("estacionales_ia", "Estacionales", GRIS)]:
        fig.add_scatter(x=i.index, y=i[c_], name=n_, line=dict(color=col_, width=2))
    fig.update_layout(title="Inflación interanual por categoría (%)", **PLANTILLA, height=380)
    col2.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    fig = go.Figure()
    for reg, color in COLORES_REG.items():
        fig.add_scatter(x=i.index, y=i[f"{reg}_ia"], name=reg, line=dict(color=color, width=2), connectgaps=True)
    fig.update_layout(title="Inflación interanual por región (%)", **PLANTILLA, height=380)
    col1.plotly_chart(fig, width="stretch")
    ap = datos.ipc_aperturas[datos.ipc_aperturas["seccion"] == "division"].sort_values("var_mensual")
    fig = px.bar(ap, x="var_mensual", y="item", orientation="h", color="var_interanual", color_continuous_scale="Oranges",
                 labels={"var_mensual": "Var. mensual agosto 2026 (%)", "item": "", "var_interanual": "Interanual (%)"})
    fig.update_layout(title="Divisiones del IPC – agosto 2026", **{**PLANTILLA, "hovermode": "closest"}, height=380)
    col2.plotly_chart(fig, width="stretch")

    st.subheader("Canasta básica por región (junio 2026, por adulto equivalente)")
    cb = datos.canastas.melt(id_vars="region", var_name="canasta", value_name="pesos")
    cb["canasta"] = cb["canasta"].map({"cba_adulto_equivalente": "Alimentaria (CBA)", "cbt_adulto_equivalente": "Total (CBT)"})
    fig = px.bar(cb, x="region", y="pesos", color="canasta", barmode="group", color_discrete_sequence=[NARANJA, AZUL])
    fig.update_layout(**{**PLANTILLA, "hovermode": "closest"}, height=360)
    st.plotly_chart(fig, width="stretch")

elif seccion == "Actividad y salarios":
    st.title("Actividad y salarios")
    tp, pib_anual = D.pib_variaciones(datos)
    col1, col2 = st.columns(2)
    x = tp.dropna(subset=["pib_ia"])
    fig = go.Figure(go.Bar(x=x.index, y=x["pib_ia"], marker_color=np.where(x["pib_ia"] >= 0, VERDE, ROJO)))
    fig.update_layout(title="PIB – variación interanual por trimestre (%)", **PLANTILLA, height=380)
    col1.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=m.index, y=m["emae"], name="Original", line=dict(color="#cbd5e1"))
    fig.add_scatter(x=m.index, y=m["emae_desest"], name="Desestacionalizado", line=dict(color=AZUL, width=3))
    fig.update_layout(title="EMAE (2004 = 100)", **PLANTILLA, height=380)
    col2.plotly_chart(fig, width="stretch")
    st.subheader("Salario real (nov-2023 = 100)")
    sr = D.salario_real(datos)
    sr = sr[sr.index >= "2022-01-01"]
    fig = go.Figure()
    for c_, n_, col_ in [("real_reg_privado", "Registrado privado", AZUL), ("real_publico", "Público", NARANJA),
                         ("real_total", "Total", GRIS)]:
        fig.add_scatter(x=sr.index, y=sr[c_], name=n_, line=dict(color=col_, width=2))
    fig.add_hline(y=100, line_dash="dot", line_color=GRIS)
    fig.update_layout(**PLANTILLA, height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption("Índice de salarios INDEC deflactado por IPC nacional. El índice de salarios no registrados se publica "
               "con 5 meses de rezago, por eso no se muestra en esta comparación.")
    st.subheader("Actividad por sector · variación interanual (%)")
    ac = D.actividad_sectorial(datos)
    ac = ac[ac.index >= "2023-01-01"]
    fig = go.Figure()
    for c_, n_, col_ in [("emae_ia", "EMAE (economía total)", "#111827"), ("ipi_ia", "Industria (IPI)", AZUL),
                         ("isac_ia", "Construcción (ISAC)", NARANJA), ("super_ia", "Supermercados (ventas reales)", VERDE)]:
        fig.add_scatter(x=ac.index, y=ac[c_], name=n_, line=dict(color=col_, width=2.5 if c_ == "emae_ia" else 1.8))
    fig.add_hline(y=0, line_color=GRIS, line_width=1)
    fig.update_layout(**PLANTILLA, height=400)
    st.plotly_chart(fig, width="stretch")
    st.caption("INDEC: IPI manufacturero, ISAC y encuesta de supermercados (serie desestacionalizada a precios constantes).")

elif seccion == "Trabajo":
    st.title("Mercado de trabajo")
    tt = t.dropna(subset=["desocupacion"])
    fig = go.Figure()
    fig.add_scatter(x=tt.index, y=tt["actividad"], name="Actividad (oferta laboral)", line=dict(color=AZUL, width=2))
    fig.add_scatter(x=tt.index, y=tt["empleo"], name="Empleo", line=dict(color=VERDE, width=2))
    fig.add_bar(x=tt.index, y=tt["desocupacion"], name="Desocupación", marker_color=NARANJA, yaxis="y2", opacity=0.6)
    fig.update_layout(title="Tasas básicas EPH – 31 aglomerados urbanos (%)", **PLANTILLA, height=420,
                      yaxis=dict(title="Actividad / empleo"), yaxis2=dict(title="Desocupación", overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Por aglomerado y región · 2T-2026")
    nivel = st.radio("Nivel", ["Aglomerados", "Regiones"], horizontal=True)
    a = datos.aglomerados[datos.aglomerados["tipo"] == ("aglomerado" if nivel == "Aglomerados" else "region")]
    ind = st.selectbox("Indicador", ["tasa_desocupacion", "tasa_actividad", "tasa_empleo", "tasa_subocupacion"],
                       format_func=lambda k: {"tasa_desocupacion": "Desocupación", "tasa_actividad": "Actividad (oferta laboral)",
                                              "tasa_empleo": "Empleo", "tasa_subocupacion": "Subocupación"}[k])
    a = a.sort_values(ind)
    total = datos.aglomerados.loc[datos.aglomerados["tipo"] == "total", ind].iloc[0]
    fig = px.bar(a, x=ind, y="nombre", orientation="h", color=ind, color_continuous_scale="Blues",
                 labels={ind: "%", "nombre": ""})
    fig.add_vline(x=total, line_dash="dot", annotation_text=f"Total {fmt(total)}%")
    fig.update_layout(**{**PLANTILLA, "hovermode": "closest"}, height=max(380, 22 * len(a)), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Fuente: INDEC, EPH 2T-2026. Estimaciones de Santiago del Estero-La Banda y Viedma-Carmen de Patagones "
               "con coeficiente de variación > 25% (poco confiables).")

    st.subheader("Empleo privado registrado por provincia (SIPA)")
    prov = st.multiselect("Provincias", list(datos.sipa_mensual.columns), default=["Neuquén", "Córdoba", "Santa Fe", "Tierra del Fuego"])
    if prov:
        s = datos.sipa_mensual[prov].interpolate(limit_area="inside")
        base = s.apply(lambda c: D._valor_cercano(c, "2023-11-01", 2))
        fig = px.line((s / base * 100), labels={"value": "nov-2023 = 100", "fecha": "", "variable": ""})
        fig.update_layout(**PLANTILLA, height=380)
        st.plotly_chart(fig, width="stretch")

elif seccion == "Empleo e ingresos":
    st.title("Empleo registrado, masa salarial e ingresos")
    er = D.empleo_registrado(datos)
    ms = D.masa_salarial(datos)
    a_ = datos.ampliado
    tot, ft = D.ultimo_valor(er["sipa_total"])
    c = st.columns(4)
    c[0].metric(f"Trabajadores registrados ({mes(ft)})", f"{fmt(tot, 0)} mil", f"{fmt(er['sipa_total_ia'].loc[ft])}% i.a.", border=True)
    c[1].metric("Asalariados privados", f"{fmt(er['sipa_asal_priv'].loc[ft], 0)} mil", f"{fmt(er['sipa_asal_priv_ia'].loc[ft])}% i.a.", border=True)
    m12, fm = D.ultimo_valor(ms["masa_real_12m_ia"])
    c[2].metric(f"Masa salarial real (12 m a {mes(fm)})", f"$ {fmt(ms['masa_real_12m_bill'].loc[fm], 1)} bill.",
                f"{fmt(m12)}% i.a.", border=True)
    g_, fg_ = D.ultimo_valor(a_["gini"])
    c[3].metric(f"Coeficiente de Gini ({ia._trim(fg_)})", fmt(g_, 3),
                f"{fmt(g_ - a_['gini'].dropna().iloc[-5], 3)} vs. un año antes", delta_color="inverse", border=True)

    st.subheader("Trabajadores registrados por modalidad (nov-2023 = 100)")
    ee = er[[k for k in D.MODALIDADES_SIPA if k != "sipa_monot_social"]]
    ee = ee[ee.index >= "2022-01-01"]
    base_ = ee.apply(lambda col: D._valor_cercano(col, "2023-11-01", 1))
    fig = px.line(ee / base_ * 100, labels={"value": "nov-2023 = 100", "fecha": "", "variable": ""})
    fig.for_each_trace(lambda tr: tr.update(name=D.MODALIDADES_SIPA[tr.name]))
    fig.add_hline(y=100, line_dash="dot", line_color=GRIS)
    fig.update_layout(**PLANTILLA, height=400)
    st.plotly_chart(fig, width="stretch")
    st.caption("SIPA (Secretaría de Trabajo), series sin estacionalidad, en miles de trabajadores. Una persona con más de un "
               "empleo cuenta una vez por modalidad. El monotributo social no se grafica: cayó de 653 mil a 259 mil inscriptos "
               "entre noviembre y diciembre de 2024, un quiebre administrativo de la serie que también baja el total.")
    tabla_mod = pd.DataFrame({
        "Miles (último dato)": [er[k].dropna().iloc[-1] for k in D.MODALIDADES_SIPA],
        "Var. % interanual": [er[f"{k}_ia"].dropna().iloc[-1] for k in D.MODALIDADES_SIPA],
        "Período": [mes(er[k].dropna().index[-1]) for k in D.MODALIDADES_SIPA],
    }, index=list(D.MODALIDADES_SIPA.values())).round(1)
    st.dataframe(tabla_mod, width="stretch")

    col1, col2 = st.columns(2)
    mm_ = ms[ms.index >= "2019-01-01"]
    fig = go.Figure()
    fig.add_scatter(x=mm_.index, y=mm_["masa_real_12m_bill"], name="Masa real, suma 12 meses (bill. $)", line=dict(color=AZUL, width=3))
    fig.add_bar(x=mm_.index, y=mm_["masa_real_ia"], name="Var. % i.a. del mes", marker_color=np.where(mm_["masa_real_ia"] >= 0, VERDE, ROJO),
                yaxis="y2", opacity=0.45)
    fig.update_layout(title="Masa salarial real del empleo privado registrado", **PLANTILLA, height=400,
                      yaxis=dict(title="billones de $ del último mes"),
                      yaxis2=dict(title="% i.a.", overlaying="y", side="right", showgrid=False, tickformat=".0f", zeroline=True))
    col1.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=mm_.index, y=mm_["rem_real"].rolling(12).mean(), name="Promedio (media móvil 12 m)", line=dict(color=AZUL, width=2.5))
    fig.add_scatter(x=mm_.index, y=mm_["rem_mediana_real"].rolling(12).mean(), name="Mediana (media móvil 12 m)", line=dict(color=NARANJA, width=2.5))
    fig.update_layout(title="Remuneración bruta real ($ del último mes)", **PLANTILLA, height=400)
    col2.plotly_chart(fig, width="stretch")
    st.caption("Masa salarial = asalariados privados registrados × remuneración bruta promedio (SIPA), deflactada por IPC. Junio y "
               "diciembre incluyen el medio aguinaldo; por eso se usan sumas o promedios de 12 meses y comparaciones interanuales.")

    st.subheader("Ingresos de los hogares por aglomerado · EPH 1T-2026")
    ing = datos.ingresos_aglo
    ia_ = ing[ing["tipo"] == "aglomerado"].sort_values("ipcf_medio")
    tot_ = ing[ing["tipo"] == "total"].iloc[0]
    var_ing = st.radio("Medida", ["Ingreso per cápita familiar", "Ingreso de la ocupación principal"], horizontal=True)
    col1, col2 = st.columns([3, 2])
    if var_ing.startswith("Ingreso per"):
        fig = go.Figure()
        fig.add_bar(x=ia_["ipcf_medio"], y=ia_["nombre"], orientation="h", name="Media", marker_color=AZUL)
        fig.add_scatter(x=ia_["ipcf_mediana"], y=ia_["nombre"], mode="markers", name="Mediana", marker=dict(color=NARANJA, size=9, symbol="diamond"))
        fig.add_vline(x=tot_["ipcf_medio"], line_dash="dot", annotation_text=f"Media 31 aglom. $ {fmt(tot_['ipcf_medio'], 0)}")
    else:
        ia_ = ia_.sort_values("p21_medio")
        fig = go.Figure(go.Bar(x=ia_["p21_medio"], y=ia_["nombre"], orientation="h", marker_color=VERDE))
        fig.add_vline(x=tot_["p21_medio"], line_dash="dot", annotation_text=f"Media $ {fmt(tot_['p21_medio'], 0)}")
    fig.update_layout(**{**PLANTILLA, "hovermode": "closest"}, height=760, xaxis_title="$ por mes")
    col1.plotly_chart(fig, width="stretch")
    fig = px.scatter(ia_, x="ipcf_medio", y="informalidad_asal", text="nombre", size="casos_muestra",
                     labels={"ipcf_medio": "Ingreso per cápita familiar medio ($)", "informalidad_asal": "Asalariados sin aportes (%)"})
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(**{**PLANTILLA, "hovermode": "closest"}, height=760, title="Ingreso e informalidad")
    col2.plotly_chart(fig, width="stretch")
    st.caption("Cálculo propio con los microdatos de la EPH 1T-2026 (INDEC), ponderados por PONDIH (ingreso familiar) y PONDIIO "
               "(ocupación principal). Informalidad: asalariados sin descuento jubilatorio (PP07H). El total de los 31 aglomerados "
               "coincide con lo publicado por INDEC.")

    st.subheader("Desigualdad: coeficiente de Gini del ingreso per cápita familiar")
    gg = a_["gini"].dropna()
    fig = go.Figure(go.Scatter(x=gg.index, y=gg, mode="lines+markers", line=dict(color=AZUL, width=2)))
    fig.update_layout(**PLANTILLA, height=320, yaxis_title="Gini (0 = igualdad)")
    st.plotly_chart(fig, width="stretch")

elif seccion == "Crédito y mora":
    st.title("Crédito y mora")
    mo = D.mora_nacional(datos)
    cr = D.credito_real(datos)
    irr, fi = D.ultimo_valor(mo["irreg_total"])
    tipos = datos.mora_tipos.set_index("indicador")
    c = st.columns(4)
    c[0].metric(f"Mora del sistema financiero ({mes(fi)})", fmt(irr, 2, "%"),
                f"{fmt(irr - mo['irreg_total'].shift(12).loc[fi], 1)} p.p. en un año", delta_color="inverse", border=True)
    c[1].metric("Mora de las familias (jun-2026)", fmt(tipos.loc["Familias (préstamos en situación irregular)", "valor"], 1, "%"),
                f"empresas {fmt(tipos.loc['Empresas', 'valor'], 1)}%", delta_color="off", delta_arrow="off", border=True)
    c[2].metric("Personas con deudas > 90 días", fmt(tipos.loc["Personas humanas con deuda en mora (más de 90 días)", "valor"], 1, "%"),
                "jun-2026 · Mapa de la Deuda", delta_color="off", delta_arrow="off", border=True)
    cia, fc = D.ultimo_valor(cr["prest_pesos_real_ia"])
    c[3].metric(f"Crédito en pesos / PIB ({mes(fc)})", fmt(cr['credito_pct_pib'].loc[fc], 1, "%"), f"{fmt(cia)}% real i.a.", border=True)

    col1, col2 = st.columns(2)
    fig = go.Figure()
    for c_, n_, col_ in [("irreg_total", "Sistema financiero", "#111827"), ("irreg_publicos", "Bancos públicos", AZUL),
                         ("irreg_privados", "Bancos privados", NARANJA)]:
        fig.add_scatter(x=mo.index, y=mo[c_], name=n_, line=dict(color=col_, width=2.5 if c_ == "irreg_total" else 1.6))
    fig.update_layout(title="Irregularidad de la cartera de préstamos al sector privado (%)", **PLANTILLA, height=400)
    col1.plotly_chart(fig, width="stretch")
    crr = cr[cr.index >= "2019-01-01"]
    fig = go.Figure()
    for c_, n_, col_ in [("prest_consumo_real_bill", "Consumo (personales + tarjetas)", NARANJA),
                         ("prest_comerciales_real_bill", "Comerciales", AZUL)]:
        fig.add_scatter(x=crr.index, y=crr[c_], name=n_, stackgroup="uno", line=dict(color=col_))
    fig.add_scatter(x=crr.index, y=crr["credito_pct_pib"], name="Crédito total (% del PIB)", yaxis="y2",
                    line=dict(color="#111827", width=2, dash="dot"))
    fig.update_layout(title="Préstamos en pesos al sector privado (billones de $ del último mes)", **PLANTILLA, height=400,
                      yaxis2=dict(title="% del PIB", overlaying="y", side="right"))
    col2.plotly_chart(fig, width="stretch")
    st.caption("BCRA: irregularidad = financiaciones en situación 3 a 6 sobre el total (Informe sobre Entidades Financieras); "
               "saldos promedio mensuales de préstamos deflactados por IPC. Crédito/PIB: saldo sobre el PIB nominal de los "
               "últimos cuatro trimestres.")

    st.subheader("Mora de las personas por provincia · junio 2026")
    p = datos.provincias
    mapa_y_ranking(p, "mora_personas_jun2026", altura=640)
    st.caption("Centro de Economía y Comunicación (Mapa de la Deuda) con datos de la Central de Deudores del BCRA: % de personas "
               "humanas con deudas con más de 90 días de atraso. Río Negro no figura en el informe.")
    col1, col2 = st.columns(2)
    me = datos.mora_edades[datos.mora_edades["grupo_edad"] != "Total"]
    fig = px.bar(me, x="grupo_edad", y="mora_pct_monto", color="mora_pct_monto", color_continuous_scale="Reds",
                 labels={"grupo_edad": "Edad", "mora_pct_monto": "% del monto adeudado en mora"})
    fig.update_layout(title="Mora por edad (ago-2026, % del monto)", **{**PLANTILLA, "hovermode": "closest"}, height=380,
                      coloraxis_showscale=False)
    col1.plotly_chart(fig, width="stretch")
    mt = datos.mora_tipos.sort_values("valor")
    fig = px.bar(mt, x="valor", y="indicador", orientation="h", hover_data=["periodo", "fuente"],
                 labels={"valor": "% en mora", "indicador": ""}, color_discrete_sequence=[ROJO])
    fig.update_layout(title="Distintas mediciones de mora (%)", **{**PLANTILLA, "hovermode": "closest"}, height=380)
    col2.plotly_chart(fig, width="stretch")

    st.subheader("¿Dónde hay más mora? Ingreso de los hogares y morosidad")
    fig = px.scatter(mora_d, x="ipcf_medio", y="mora_personas_jun2026", text="provincia", trendline="ols",
                     trendline_options=dict(log_x=True), log_x=True,
                     labels={"ipcf_medio": "Ingreso per cápita familiar medio 1T-2026 ($, escala log)",
                             "mora_personas_jun2026": "Personas en mora (%)"})
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(**{**PLANTILLA, "hovermode": "closest"}, height=460)
    st.plotly_chart(fig, width="stretch")
    b_ = mora_s.params["log_ipcf"]
    st.markdown(f"Un ingreso per cápita familiar **10% más alto** se asocia con **{fmt(-b_ * np.log(1.1), 2)} p.p. menos** de personas "
                f"en mora (MCO, {int(mora_s.nobs)} provincias, p = {mora_s.pvalues['log_ipcf']:.3f}, R² = {fmt(mora_s.rsquared, 2)}). "
                "El ingreso es el del aglomerado principal de cada provincia (EPH).")

elif seccion == "Pobreza":
    st.title("Pobreza e indigencia")
    col1, col2 = st.columns([1, 2])
    ph = datos.pobreza_hist
    fig = go.Figure()
    fig.add_bar(x=ph["semestre"], y=ph["personas_pobreza"], name="Pobreza", marker_color=NARANJA)
    fig.add_bar(x=ph["semestre"], y=ph["personas_indigencia"], name="Indigencia", marker_color=ROJO)
    fig.update_layout(title="% de personas – 31 aglomerados", barmode="overlay", **PLANTILLA, height=380)
    col1.plotly_chart(fig, width="stretch")
    a = datos.aglomerados[datos.aglomerados["tipo"] == "aglomerado"].sort_values("personas_pobreza")
    fig = px.bar(a, x="personas_pobreza", y="nombre", orientation="h", color="personas_indigencia",
                 color_continuous_scale="Reds", labels={"personas_pobreza": "Pobreza (% personas)", "nombre": "",
                                                        "personas_indigencia": "Indigencia (%)"})
    fig.update_layout(title="Pobreza por aglomerado – 1S-2026", **{**PLANTILLA, "hovermode": "closest"}, height=760)
    col2.plotly_chart(fig, width="stretch")

elif seccion == "Fiscal y sector externo":
    st.title("Fiscal y sector externo")
    mm = m[m.index >= "2023-01-01"]
    fig = go.Figure()
    fig.add_bar(x=mm.index, y=mm["resultado_primario"] / 1e6, name="Primario", marker_color=AZUL)
    fig.add_bar(x=mm.index, y=mm["resultado_financiero"] / 1e6, name="Financiero", marker_color=NARANJA)
    fig.update_layout(title="Resultado del Sector Público Nacional (billones de $, mensual)", barmode="group", **PLANTILLA, height=380)
    st.plotly_chart(fig, width="stretch")
    col1, col2 = st.columns(2)
    fp = D.fiscal_pct_pib(datos)
    fig = go.Figure()
    fig.add_bar(x=fp.index, y=fp["resultado_primario"], name="Primario", marker_color=AZUL)
    fig.add_bar(x=fp.index, y=fp["resultado_financiero"], name="Financiero", marker_color=NARANJA)
    fig.update_layout(title="Resultado anual (% del PIB)", barmode="group", **PLANTILLA, height=360)
    col1.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=m.index, y=m["reservas_musd"] / 1000, name="Reservas (miles de M USD)", line=dict(color=VERDE, width=2))
    fig.add_scatter(x=m.index, y=m["tipo_cambio"], name="Tipo de cambio ($/USD)", line=dict(color=GRIS), yaxis="y2")
    fig.update_layout(title="Reservas del BCRA y tipo de cambio", **PLANTILLA, height=360,
                      yaxis2=dict(overlaying="y", side="right"))
    col2.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    fig.add_bar(x=mm.index, y=mm["expo_musd"], name="Exportaciones", marker_color=VERDE)
    fig.add_bar(x=mm.index, y=-mm["impo_musd"], name="Importaciones", marker_color=ROJO)
    fig.add_scatter(x=mm.index, y=mm["expo_musd"] - mm["impo_musd"], name="Saldo", line=dict(color="#111827", width=2))
    fig.update_layout(title="Comercio exterior (millones de USD)", barmode="relative", **PLANTILLA, height=380)
    st.plotly_chart(fig, width="stretch")

elif seccion == "Provincias":
    st.title("Las 24 jurisdicciones")
    p = datos.provincias
    ind = selector_indicador("prov")
    mapa_y_ranking(p, ind)
    st.caption("La EPH releva aglomerados urbanos: para desocupación, pobreza, ingresos e informalidad se muestra el aglomerado "
               "principal de cada provincia. Malvinas, Georgias y Sandwich del Sur integran Tierra del Fuego, AIAS.")
    st.subheader("Ficha provincial")
    elegida = st.selectbox("Provincia", p["provincia"].tolist(), index=p["provincia"].tolist().index("Córdoba"))
    fila = p[p["provincia"] == elegida].iloc[0]
    c = st.columns(4)
    c[0].metric("Población (Censo 2022)", fmt(fila["poblacion_2022"], 0), f"{fmt(fila['crec_poblacion_2010_2022'])}% vs 2010", delta_color="off", delta_arrow="off", border=True)
    c[1].metric(f"Desocupación ({fila['aglomerado_ref']})", fmt(fila["tasa_desocupacion"], 1, "%"),
                f"actividad {fmt(fila['tasa_actividad'])}%", delta_color="off", delta_arrow="off", border=True)
    c[2].metric("Empleo privado registrado", f"{fmt(fila['empleo_privado_miles'])} mil",
                f"{fmt(fila['empleo_privado_var_ia'])}% interanual", border=True)
    c[3].metric("Exportaciones 1S-2026", f"USD {fmt(fila['expo_1S2026_musd'], 0)} M", f"{fmt(fila['expo_var_ia'])}%", border=True)
    c = st.columns(4)
    c[0].metric("PBG 2023 (CEPAL)", f"{fmt(fila['pbg_participacion'], 1)}% del país",
                f"por habitante {fmt(fila['pbg_pc_indice'], 0)} (país = 100)", delta_color="off", delta_arrow="off", border=True)
    c[1].metric("Salario privado registrado (mar-2026)", f"$ {fmt(fila['salario_privado_mar2026'], 0)}",
                f"{fmt(fila['salario_privado_rel'] - 100, 1)}% vs. promedio país", border=True)
    c[2].metric("Resultado financiero 2025", f"{fmt(fila['resultado_financiero_pct_ing'], 1)}% de los ingresos",
                f"autonomía fiscal {fmt(fila['autonomia_fiscal'], 1)}%", delta_color="off", delta_arrow="off", border=True)
    c[3].metric("Personas en mora (jun-2026)", fmt(fila["mora_personas_jun2026"], 1, "%"),
                f"{fmt(fila['mora_personas_jun2026'] - 17.4, 1)} p.p. vs. país", delta_color="inverse", border=True)
    st.subheader("Tabla completa")
    g_tab = st.selectbox("Tema de la tabla", list(D.GRUPOS_PROVINCIALES), key="tabla_g")
    cols = D.GRUPOS_PROVINCIALES[g_tab]
    st.dataframe(p.set_index("provincia")[cols].rename(columns={k: D.INDICADORES_PROVINCIALES[k][0] for k in cols}).round(1),
                 width="stretch")

elif seccion == "PBG y finanzas provinciales":
    st.title("Producto Bruto Geográfico y finanzas de las provincias")
    tab_pbg, tab_fis = st.tabs(["Producto Bruto Geográfico", "Situación fiscal provincial 2025"])
    with tab_pbg:
        pb = D.resumen_pbg(datos)
        w = D.pbg_largo(datos)
        c = st.columns(3)
        top3 = pb.sort_values("pbg_participacion", ascending=False).head(3)
        c[0].metric("Concentración: 3 mayores (2023)", fmt(top3["pbg_participacion"].sum(), 1, "%"),
                    ", ".join(top3["provincia"]), delta_color="off", delta_arrow="off", border=True)
        mx, mn = pb.loc[pb["pbg_pc_indice"].idxmax()], pb.loc[pb["pbg_pc_indice"].idxmin()]
        c[1].metric("Brecha de PBG por habitante", f"{fmt(mx['pbg_pc_indice'] / mn['pbg_pc_indice'], 1)} veces",
                    f"{mx['provincia']} vs. {mn['provincia']}", delta_color="off", delta_arrow="off", border=True)
        c[2].metric("Convergencia β 2010-2022", fmt(conv.beta, 2), f"p = {conv.p_valor:.3f} · vida media {fmt(conv.vida_media, 0)} años",
                    delta_color="off", delta_arrow="off", border=True)
        col1, col2 = st.columns([2, 3])
        r = pb.sort_values("pbg_participacion")
        fig = px.bar(r, x="pbg_participacion", y="provincia", orientation="h", color="pbg_pc_indice",
                     color_continuous_scale=[[0, "#93c5fd"], [1, "#1e3a8a"]],
                     labels={"pbg_participacion": "% del VAB nacional 2023", "provincia": "",
                             "pbg_pc_indice": "Per cápita (país = 100)"})
        fig.update_layout(title="Participación en el PBG (color: PBG por habitante)", **{**PLANTILLA, "hovermode": "closest"},
                          height=720, coloraxis_showscale=False)
        col1.plotly_chart(fig, width="stretch")
        elegidas = col2.multiselect("Provincias", list(w.columns), default=["Neuquén", "CABA", "Buenos Aires", "Córdoba", "Santa Fe", "Catamarca"])
        if elegidas:
            idx = w[elegidas] / w[elegidas].loc[2004] * 100
            fig = px.line(idx, labels={"value": "2004 = 100", "anio": "", "provincia": ""}, markers=True)
            nac = w.sum(axis=1) / w.sum(axis=1).loc[2004] * 100
            fig.add_scatter(x=nac.index, y=nac, name="Total país", line=dict(color="#111827", width=3, dash="dot"))
            fig.update_layout(title="PBG a precios constantes (2004 = 100)", **PLANTILLA, height=420)
            col2.plotly_chart(fig, width="stretch")
        fig = px.scatter(conv.datos, x="log_pbg_pc_ini", y="crec_anual", text="provincia", trendline="ols",
                         labels={"log_pbg_pc_ini": "log PBG por habitante 2010 ($ de 2004)", "crec_anual": "% anual 2010-22"})
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(title="Convergencia β: crecimiento del PBG per cápita vs. nivel inicial", **{**PLANTILLA, "hovermode": "closest"}, height=380)
        col2.plotly_chart(fig, width="stretch")
        st.dataframe(pb.set_index("provincia")[["pbg_participacion", "pbg_pc_indice", "pbg_pc_usd", "pbg_var_ultimo",
                                                 "pbg_var_vs_2019", "pbg_crec_anual_2004", "pbg_pc_crec_2010_2022"]]
                     .rename(columns={k: D.INDICADORES_PROVINCIALES[k][0] for k in D.INDICADORES_PROVINCIALES}).round(1),
                     width="stretch")
        st.caption(f"CEPAL y Ministerio de Economía, «Desagregación provincial del valor agregado bruto de la Argentina, base 2004» "
                   f"(serie 2004-{pb.attrs['anio_pbg']}, VAB a precios básicos de 2004, vía Argendata-Fundar). No existe un PBG oficial "
                   f"homogéneo más reciente: el PBG por habitante en USD {pb.attrs['anio_pib']} es una estimación (participación 2023 × "
                   f"PIB nominal {pb.attrs['anio_pib']} ÷ población 2022 ÷ dólar oficial promedio). 2023 incluye el efecto de la sequía.")
    with tab_fis:
        fp = D.fiscal_provincial(datos)
        tot = datos.fiscal_prov[datos.fiscal_prov["provincia"] == "Total provincias"].iloc[0]
        c = st.columns(4)
        c[0].metric("Resultado primario consolidado 2025", f"$ {fmt(tot['resultado_primario'] / 1e6, 2)} bill.",
                    f"{fmt(tot['resultado_primario'] / tot['ingresos_totales'] * 100, 1)}% de los ingresos", delta_color="off", delta_arrow="off", border=True)
        c[1].metric("Resultado financiero consolidado 2025", f"$ {fmt(tot['resultado_financiero'] / 1e6, 2)} bill.",
                    f"{fmt(tot['resultado_financiero'] / tot['ingresos_totales'] * 100, 1)}% de los ingresos", delta_color="off", delta_arrow="off", border=True)
        n_sup = int((fp["resultado_financiero"] > 0).sum())
        c[2].metric("Provincias con superávit financiero", f"{n_sup} de {fp['resultado_financiero'].notna().sum()}",
                    "sin La Pampa (no informó)", delta_color="off", delta_arrow="off", border=True)
        c[3].metric("Recursos de origen nacional", f"{fmt((tot['trib_nacionales'] + tot['transf_corrientes_recibidas']) / tot['ingresos_totales'] * 100, 1)}%",
                    "de los ingresos totales", delta_color="off", delta_arrow="off", border=True)
        ind = st.selectbox("Indicador fiscal", D.GRUPOS_PROVINCIALES["Finanzas públicas provinciales"],
                           format_func=lambda k: D.INDICADORES_PROVINCIALES[k][0], key="fis_i")
        mapa_y_ranking(datos.provincias, ind, altura=640)
        col1, col2 = st.columns(2)
        comp = fp.set_index("provincia")[["trib_propios", "trib_nacionales", "transf_corrientes_recibidas", "regalias", "contrib_seg_social"]]
        otros = fp.set_index("provincia")["ingresos_totales"] - comp.sum(axis=1)
        comp = comp.assign(otros=otros).div(fp.set_index("provincia")["ingresos_totales"], axis=0).mul(100).dropna()
        comp = comp.sort_values("trib_propios")
        nombres = {"trib_propios": "Tributos propios", "trib_nacionales": "Coparticipación y otros tributos nacionales",
                   "transf_corrientes_recibidas": "Transferencias corrientes", "regalias": "Regalías",
                   "contrib_seg_social": "Contribuciones a la seguridad social", "otros": "Otros"}
        fig = go.Figure()
        for c_, col_ in zip(nombres, [AZUL, NARANJA, "#f59e0b", VERDE, "#a855f7", "#cbd5e1"]):
            fig.add_bar(y=comp.index, x=comp[c_], orientation="h", name=nombres[c_], marker_color=col_)
        fig.update_layout(title="¿De dónde salen los ingresos? (% de los ingresos totales 2025)", barmode="stack",
                          **{**PLANTILLA, "hovermode": "closest"}, height=700)
        col1.plotly_chart(fig, width="stretch")
        fig = px.scatter(fp.dropna(subset=["autonomia_fiscal"]), x="autonomia_fiscal", y="resultado_financiero_pct_ing",
                         size="gastos_totales", text="provincia", color="personal_pct_gasto", color_continuous_scale="Oranges",
                         labels={"autonomia_fiscal": "Autonomía fiscal: tributos propios / ingresos (%)",
                                 "resultado_financiero_pct_ing": "Resultado financiero (% ingresos)",
                                 "personal_pct_gasto": "Personal / gasto (%)"})
        fig.add_hline(y=0, line_color=GRIS)
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(title="Autonomía, resultado y peso del gasto en personal", **{**PLANTILLA, "hovermode": "closest"}, height=700)
        col2.plotly_chart(fig, width="stretch")
        st.caption("Dirección Nacional de Asuntos Provinciales (Ministerio de Economía), ejecución presupuestaria 2025 de la "
                   "Administración Pública No Financiera, datos provisorios a junio de 2026, en millones de pesos. La Pampa no "
                   "informó; Santiago del Estero informa el gasto en la etapa de compromiso. El PBG 2025 usado como denominador es "
                   "una estimación (ver pestaña PBG).")

elif seccion == "Modelos e IA predictiva":
    st.title("Modelos")
    st.subheader("1 · Pronóstico de inflación mensual")
    h = pron.historia[pron.historia.index >= "2024-06-01"]
    fc = pron.pronostico
    fig = go.Figure()
    fig.add_scatter(x=h.index, y=h, name="Observada", line=dict(color=AZUL, width=2))
    fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc["sup_95"]) + list(fc["inf_95"][::-1]),
                    fill="toself", fillcolor="rgba(234,88,12,0.12)", line=dict(width=0), mode="lines", name="IC 95%")
    fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc["sup_80"]) + list(fc["inf_80"][::-1]),
                    fill="toself", fillcolor="rgba(234,88,12,0.25)", line=dict(width=0), mode="lines", name="IC 80%")
    fig.add_scatter(x=fc.index, y=fc["media"], name=f"Pronóstico ({pron.modelo_elegido})", line=dict(color=NARANJA, width=3, dash="dot"))
    fig.add_scatter(x=fc.index, y=fc["media_arima"], name=f"ARIMA{pron.orden_arima}", line=dict(color=GRIS, width=1.5, dash="dash"))
    fig.update_layout(**PLANTILLA, height=420, yaxis_title="% mensual")
    st.plotly_chart(fig, width="stretch")
    col1, col2 = st.columns(2)
    col1.markdown("**Validación fuera de muestra** (últimos 12 meses, pronóstico a un paso):")
    col1.dataframe(pron.ranking.round(3), width="stretch")
    col2.markdown(f"Modelo elegido por menor MAE: **{pron.modelo_elegido}**. Inflación acumulada proyectada para los próximos "
                  f"6 meses: **{fmt(M.inflacion_acumulada(fc['media']))}%**. Estimación sobre el régimen posterior a la "
                  f"desinflación (desde {pron.desde[:7]}).")

    st.subheader("2 · Ley de Okun")
    b = okun.params
    fig = px.scatter(okun_d, x="crec_pib", y="delta_u", trendline="ols",
                     labels={"crec_pib": "Crecimiento del PIB, % interanual", "delta_u": "Variación de la desocupación, p.p. interanual"})
    fig.update_layout(**{**PLANTILLA, "hovermode": "closest"}, height=380)
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"Coeficiente de Okun: **{fmt(b['crec_pib'], 3)}** (p = {okun.pvalues['crec_pib']:.4f}, R² = {fmt(okun.rsquared, 2)}). "
                f"Cada punto extra de crecimiento se asocia con una baja de {fmt(-b['crec_pib'], 2)} p.p. en la desocupación. "
                f"Crecimiento que mantiene estable el desempleo: {fmt(-b['const'] / b['crec_pib'], 1)}%.")

    st.subheader("3 · ¿Qué explica la pobreza entre aglomerados?")
    st.dataframe(pd.DataFrame({"coef.": rp.params, "error est. (HC1)": rp.bse, "p-valor": rp.pvalues}).round(3),
                 width="stretch")
    st.caption(f"MCO con {int(rp.nobs)} aglomerados, R² = {fmt(rp.rsquared, 2)}. Variable dependiente: % de personas pobres (1S-2026).")

    st.subheader("4 · Tipología de aglomerados (K-means, aprendizaje no supervisado)")
    fig = px.scatter(clusters, x="pc1", y="pc2", color="perfil", text="nombre",
                     labels={"pc1": f"Componente 1 ({var_pca[0] * 100:.0f}% var.)", "pc2": f"Componente 2 ({var_pca[1] * 100:.0f}% var.)"})
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(**{**PLANTILLA, "hovermode": "closest"}, height=560)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(perfiles, width="stretch")

    st.subheader("5 · Mora e ingresos de los hogares (corte transversal de provincias)")
    tabla = pd.DataFrame({
        "(1) coef.": mora_s.params, "(1) p-valor": mora_s.pvalues,
        "(2) coef.": mora_c.params, "(2) p-valor": mora_c.pvalues,
    }).round(3)
    st.dataframe(tabla, width="stretch")
    st.caption(f"Variable dependiente: % de personas con deudas de más de 90 días (jun-2026). MCO con errores HC1, "
               f"{int(mora_s.nobs)} provincias. R²: (1) {fmt(mora_s.rsquared, 2)}, (2) {fmt(mora_c.rsquared, 2)}. "
               "log_ipcf = logaritmo del ingreso per cápita familiar medio (EPH 1T-2026).")

    st.subheader("6 · Convergencia regional del PBG per cápita (2010-2022)")
    fig = px.scatter(conv.datos, x="log_pbg_pc_ini", y="crec_anual", text="provincia", trendline="ols",
                     labels={"log_pbg_pc_ini": "log PBG por habitante 2010", "crec_anual": "Crecimiento anual 2010-2022 (%)"})
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(**{**PLANTILLA, "hovermode": "closest"}, height=420)
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"β = **{fmt(conv.beta, 3)}** (p = {conv.p_valor:.3f}, R² = {fmt(conv.modelo.rsquared, 2)}): las provincias más pobres "
                f"en 2010 crecieron algo más rápido. La velocidad implícita es de {fmt(conv.velocidad * 100, 2)}% anual, así que cerrar "
                f"la mitad de la brecha llevaría unos **{fmt(conv.vida_media, 0)} años**. La dispersión (σ del log del PBG per cápita) "
                f"pasó de {fmt(conv.sigma_inicial, 3)} a {fmt(conv.sigma_final, 3)}. Años censales para usar población observada.")

elif seccion == "Asistente IA":
    st.title("Asistente econoAR")
    motor_txt = "Claude (Anthropic)" if (api_key or os.environ.get("ANTHROPIC_API_KEY")) else "motor local de reglas"
    st.caption(f"Responde solo con los datos del tablero, citando período y fuente. Motor: {motor_txt}.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(md(msg["content"]))
    sugerencias = ["¿Cuánto fue la inflación del último mes?", "¿Cómo está el empleo en Neuquén?",
                   "¿Cómo viene la mora?", "¿Hay superávit fiscal?"]
    cols = st.columns(len(sugerencias))
    elegida = None
    for c_, s_ in zip(cols, sugerencias):
        if c_.button(s_):
            elegida = s_
    pregunta = st.chat_input("Preguntá sobre inflación, empleo, pobreza, provincias…") or elegida
    if pregunta:
        st.session_state.chat.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)
        with st.chat_message("assistant"):
            with st.spinner("Pensando…"):
                resp, motor = ia.responder(pregunta, hechos, historial=st.session_state.chat[:-1], api_key=api_key or None)
            st.markdown(md(resp))
        st.session_state.chat.append({"role": "assistant", "content": resp})

st.divider()
st.caption("Fuentes: INDEC (IPC, EPH, pobreza, EMAE, PIB, IPI, ISAC, supermercados, OPEX, Censo 2022), BCRA (préstamos, "
           "irregularidad), Ministerio de Economía (IMIG y ejecuciones provinciales), ARCA, Secretaría de Trabajo (SIPA, OEDE), "
           "CEPAL (PBG), Centro de Economía y Comunicación y CEPA (mora), vía API de Series de Tiempo de datos.gob.ar y "
           "publicaciones oficiales. Datos al 14-10-2026.")
