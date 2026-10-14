"""
Componente de IA de econoAR.

  * hechos_clave(): resume los últimos datos en una lista de "hechos" con
    fecha y fuente. Es el contexto que recibe el modelo de lenguaje.
  * responder(): asistente conversacional. Si hay ANTHROPIC_API_KEY usa
    Claude (API de Anthropic) con los hechos como contexto ("RAG" liviano:
    el modelo solo puede responder con esos datos). Sin clave, usa un motor
    local de reglas que busca el indicador o la provincia en la pregunta.
  * generar_informe(): resumen ejecutivo automático (LLM o plantilla).
"""
from __future__ import annotations

import os
import re
import unicodedata

import numpy as np
import pandas as pd

from . import datos as D

MODELO_POR_DEFECTO = os.environ.get("ECONOAR_MODELO", "claude-sonnet-5-5")

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


def _mes(f) -> str:
    f = pd.Timestamp(f)
    return f"{MESES[f.month - 1]} {f.year}"


def _trim(f) -> str:
    f = pd.Timestamp(f)
    return f"{f.quarter}T-{f.year}"


def _n(x, dec=1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "s/d"
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Hechos
# ---------------------------------------------------------------------------
def hechos_clave(datos: D.Datos, pronostico=None) -> dict[str, list[str]]:
    infl = D.inflacion(datos)
    m = datos.mensual
    h: dict[str, list[str]] = {}

    v, f = D.ultimo_valor(infl["general_mm"])
    ia, _ = D.ultimo_valor(infl["general_ia"])
    nuc, _ = D.ultimo_valor(infl["nucleo_mm"])
    acum = (m["ipc_nivel_general"].dropna().iloc[-1] /
            m.loc[f"{f.year - 1}-12-01", "ipc_nivel_general"] - 1) * 100
    h["inflacion"] = [
        f"Inflación (IPC INDEC) de {_mes(f)}: {_n(v)}% mensual, {_n(ia)}% interanual, "
        f"{_n(acum)}% acumulada en el año. Núcleo: {_n(nuc)}% mensual.",
    ]
    reg = []
    for col, nombre in D.REGIONES_IPC.items():
        r, fr = D.ultimo_valor(infl[f"{nombre}_ia"])
        if fr is not None:
            reg.append(f"{nombre} {_n(r)}%")
    h["inflacion"].append("Inflación interanual por región: " + ", ".join(reg) + ".")
    if pronostico is not None:
        p = pronostico.pronostico
        h["inflacion"].append(
            f"Pronóstico econoAR ({pronostico.modelo_elegido}): {_n(p['media'].iloc[0])}% para "
            f"{_mes(p.index[0])} (IC 80%: {_n(p['inf_80'].iloc[0])}–{_n(p['sup_80'].iloc[0])}%)."
        )

    t = datos.trimestral
    u, fu = D.ultimo_valor(t["desocupacion"])
    a, _ = D.ultimo_valor(t["actividad"])
    e, _ = D.ultimo_valor(t["empleo"])
    h["empleo"] = [
        f"Desocupación (EPH, 31 aglomerados) {_trim(fu)}: {_n(u)}%. Tasa de actividad (oferta laboral): "
        f"{_n(a)}%. Tasa de empleo: {_n(e)}%. Subocupación 2T-2026: 11,5%. Informalidad laboral: 45,0%."
    ]
    tp, pib_anual = D.pib_variaciones(datos)
    g, fg = D.ultimo_valor(tp["pib_ia"])
    em, fe = D.ultimo_valor(m["emae_desest"])
    em_ia = (m["emae"].dropna().iloc[-1] / m["emae"].dropna().shift(12).iloc[-1] - 1) * 100
    h["actividad"] = [
        f"PIB {_trim(fg)}: {_n(g)}% interanual (precios constantes). PIB 2025: {_n(pib_anual.get(2025, np.nan))}%.",
        f"EMAE {_mes(fe)}: {_n(em_ia)}% interanual (serie original).",
    ]
    ph = datos.pobreza_hist.iloc[-1]
    h["pobreza"] = [
        f"Pobreza 1S-2026: {_n(ph['personas_pobreza'])}% de las personas y {_n(ph['hogares_pobreza'])}% de los hogares; "
        f"indigencia {_n(ph['personas_indigencia'])}% de las personas (31 aglomerados urbanos)."
    ]
    f_anio = D.fiscal_acumulado(datos, int(m["resultado_primario"].dropna().index.max().year))
    h["fiscal"] = [
        f"Sector Público Nacional, acumulado {f_anio['anio']} hasta {_mes(f_anio['ultimo_mes'])}: resultado primario "
        f"${_n(f_anio['primario'] / 1e6, 2)} billones; intereses ${_n(f_anio['intereses'] / 1e6, 2)} billones; "
        f"resultado financiero ${_n(f_anio['financiero'] / 1e6, 2)} billones.",
    ]
    rec, fr = D.ultimo_valor(m["recaudacion"])
    h["fiscal"].append(f"Recaudación tributaria de {_mes(fr)}: ${_n(rec / 1e6, 2)} billones.")
    tc, ftc = D.ultimo_valor(m["tipo_cambio"])
    res, fres = D.ultimo_valor(m["reservas_musd"])
    ex, fex = D.ultimo_valor(m["expo_musd"])
    im, _ = D.ultimo_valor(m["impo_musd"])
    h["externo"] = [
        f"Tipo de cambio oficial (valuación BCRA, promedio de {_mes(ftc)}): ${_n(tc, 2)} por dólar.",
        f"Reservas internacionales BCRA (promedio de {_mes(fres)}): USD {_n(res, 0)} millones.",
        f"Comercio exterior {_mes(fex)}: exportaciones USD {_n(ex, 0)} M, importaciones USD {_n(im, 0)} M, "
        f"saldo USD {_n(ex - im, 0)} M.",
    ]
    sr = D.salario_real(datos)
    s_real, fs = D.ultimo_valor(sr["real_reg_privado"])
    h["salarios"] = [f"Salario real registrado privado ({_mes(fs)}): índice {_n(s_real)} (nov-2023 = 100)."]

    if not datos.ampliado.empty:
        a = datos.ampliado
        er = D.empleo_registrado(datos)
        tot, ft = D.ultimo_valor(er["sipa_total"])
        h["empleo"].append(
            f"Trabajadores registrados (SIPA, {_mes(ft)}): {_n(tot / 1000, 2)} millones ({_n(er['sipa_total_ia'].loc[ft])}% i.a.); "
            f"asalariados privados {_n(er['sipa_asal_priv'].loc[ft], 0)} mil ({_n(er['sipa_asal_priv_ia'].loc[ft])}% i.a.); "
            f"monotributistas {_n(er['sipa_monotributo'].loc[ft], 0)} mil ({_n(er['sipa_monotributo_ia'].loc[ft])}% i.a.)."
        )
        ms = D.masa_salarial(datos)
        m12, fm = D.ultimo_valor(ms["masa_real_12m_ia"])
        h["salarios"].append(
            f"Masa salarial real del empleo privado registrado (SIPA, 12 meses a {_mes(fm)}): {_n(m12)}% interanual; "
            f"remuneración bruta promedio real {_n(ms['rem_real_ia'].loc[fm])}% i.a. y empleo asalariado privado "
            f"{_n(ms['empleo_ia'].loc[fm])}% i.a."
        )
        g, fg = D.ultimo_valor(a["gini"])
        h["pobreza"].append(f"Coeficiente de Gini del ingreso per cápita familiar {_trim(fg)}: {_n(g, 3)} (EPH, INDEC).")
        cr = D.credito_real(datos)
        c_ia, fc = D.ultimo_valor(cr["prest_pesos_real_ia"])
        mo = D.mora_nacional(datos)
        irr, fi = D.ultimo_valor(mo["irreg_total"])
        h["credito"] = [
            f"Préstamos en pesos al sector privado ({_mes(fc)}): {_n(c_ia)}% real interanual; consumo "
            f"{_n(cr['prest_consumo_real_ia'].loc[fc])}% real i.a.; crédito total {_n(cr['credito_pct_pib'].loc[fc])}% del PIB.",
            f"Mora (irregularidad de la cartera privada del sistema financiero, BCRA, {_mes(fi)}): {_n(irr, 2)}% "
            f"(un año antes {_n(mo['irreg_total'].shift(12).loc[fi], 2)}%).",
        ]
        for _, r in datos.mora_tipos.iterrows():
            h["credito"].append(f"{r['indicador']}: {_n(r['valor'])}% ({r['periodo']}, {r['fuente']}).")
        mp = datos.mora_prov.dropna().sort_values("mora_personas_jun2026")
        mp = mp[mp["provincia"] != "Argentina"]
        h["credito"].append(
            f"Mora de personas por provincia (deudas de más de 90 días, jun-2026, CEC): más alta en "
            f"{mp.iloc[-1]['provincia']} ({_n(mp.iloc[-1]['mora_personas_jun2026'])}%), más baja en "
            f"{mp.iloc[0]['provincia']} ({_n(mp.iloc[0]['mora_personas_jun2026'])}%); promedio país 17,4%."
        )
        ac = D.actividad_sectorial(datos)
        ipi, fipi = D.ultimo_valor(ac["ipi_ia"])
        isac, fis = D.ultimo_valor(ac["isac_ia"])
        h["actividad"].append(f"Industria (IPI, {_mes(fipi)}): {_n(ipi)}% i.a.; construcción (ISAC, {_mes(fis)}): {_n(isac)}% i.a.")

    if not datos.pbg.empty:
        pb = D.resumen_pbg(datos).sort_values("pbg_participacion", ascending=False)
        top = ", ".join(f"{r.provincia} {_n(r.pbg_participacion)}%" for r in pb.head(5).itertuples())
        h["actividad"].append(
            f"Producto Bruto Geográfico (CEPAL, {pb.attrs['anio_pbg']}): las cinco mayores participaciones en el VAB son {top}. "
            f"PBG por habitante más alto: {pb.sort_values('pbg_pc_indice').iloc[-1]['provincia']}; más bajo: "
            f"{pb.sort_values('pbg_pc_indice').iloc[0]['provincia']}."
        )
    if not datos.fiscal_prov.empty:
        tot = datos.fiscal_prov[datos.fiscal_prov["provincia"] == "Total provincias"].iloc[0]
        h["fiscal"].append(
            f"Provincias 2025 (ejecución APNF, 23 jurisdicciones sin La Pampa, Min. Economía): resultado primario "
            f"${_n(tot['resultado_primario'] / 1e6, 2)} billones y financiero ${_n(tot['resultado_financiero'] / 1e6, 2)} "
            f"billones ({_n(tot['resultado_financiero'] / tot['ingresos_totales'] * 100)}% de los ingresos)."
        )

    prov = []
    for _, r in datos.provincias.iterrows():
        linea = (
            f"{r['provincia']}: desocupación {_n(r['tasa_desocupacion'])}% y pobreza {_n(r['personas_pobreza'])}% "
            f"(aglomerado {r['aglomerado_ref']}); empleo privado registrado {_n(r['empleo_privado_miles'])} mil "
            f"({_n(r['empleo_privado_var_ia'])}% i.a.); exportaciones 1S-2026 USD {_n(r['expo_1S2026_musd'], 0)} M "
            f"({_n(r['expo_var_ia'])}%); población {_n(r['poblacion_2022'], 0)}."
        )
        if "pbg_participacion" in r:
            linea += (
                f" PBG 2023 {_n(r['pbg_participacion'])}% del país, por habitante {_n(r['pbg_pc_indice'], 0)} (país = 100)."
                f" Salario privado mar-2026 ${_n(r.get('salario_privado_mar2026'), 0)}; ingreso per cápita familiar medio "
                f"1T-2026 ${_n(r.get('ipcf_medio'), 0)}; informalidad asalariada {_n(r.get('informalidad_asal'))}%."
                f" Mora de personas jun-2026 {_n(r.get('mora_personas_jun2026'))}%."
                f" Finanzas 2025: resultado financiero {_n(r.get('resultado_financiero_pct_ing'))}% de los ingresos, "
                f"autonomía fiscal {_n(r.get('autonomia_fiscal'))}%."
            )
        prov.append(linea)
    h["provincias"] = prov
    aglo = []
    for _, r in datos.aglomerados[datos.aglomerados["tipo"].isin(["aglomerado", "region"])].iterrows():
        aglo.append(
            f"{r['nombre']} ({r['tipo']}, 2T-2026): actividad {_n(r['tasa_actividad'])}%, empleo {_n(r['tasa_empleo'])}%, "
            f"desocupación {_n(r['tasa_desocupacion'])}%, subocupación {_n(r['tasa_subocupacion'])}%; "
            f"pobreza 1S-2026 {_n(r['personas_pobreza'])}%."
        )
    h["aglomerados"] = aglo
    return h


def contexto_texto(hechos: dict[str, list[str]]) -> str:
    partes = []
    for k, v in hechos.items():
        partes.append(f"## {k}\n" + "\n".join(f"- {x}" for x in v))
    return "\n\n".join(partes)


# ---------------------------------------------------------------------------
# Asistente
# ---------------------------------------------------------------------------
SISTEMA = (
    "Sos econoAR, un asistente que explica la macroeconomía argentina a nivel nacional, provincial y por "
    "aglomerado. Respondé en español rioplatense, claro y breve (máximo 6 oraciones salvo que pidan más). "
    "Usá EXCLUSIVAMENTE los datos del contexto; citá siempre el período y la fuente (INDEC, BCRA, Ministerio "
    "de Economía, SIPA, CEPAL, CEC). Si el dato no está en el contexto, decí que no lo tenés en lugar de inventarlo. "
    "Cuando compares provincias aclarate que la EPH mide aglomerados urbanos, no la provincia completa."
)


def _cliente_anthropic(api_key: str | None):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(api_key=key)


def responder(pregunta: str, hechos: dict[str, list[str]], historial: list[dict] | None = None,
              api_key: str | None = None, modelo: str | None = None) -> tuple[str, str]:
    """Devuelve (respuesta, motor) donde motor es 'claude' o 'local'."""
    cliente = _cliente_anthropic(api_key)
    if cliente is not None:
        mensajes = list(historial or [])[-8:]
        mensajes.append({"role": "user", "content": f"CONTEXTO DE DATOS:\n{contexto_texto(hechos)}\n\nPREGUNTA: {pregunta}"})
        try:
            r = cliente.messages.create(model=modelo or MODELO_POR_DEFECTO, max_tokens=700,
                                        system=SISTEMA, messages=mensajes)
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text"), "claude"
        except Exception as e:
            return responder_local(pregunta, hechos) + f"\n\n_(No se pudo usar Claude: {e})_", "local"
    return responder_local(pregunta, hechos), "local"


TEMAS = {
    "inflacion": ["inflacion", "ipc", "precios", "nucleo", "regulados", "pronostico", "proyeccion"],
    "empleo": ["desempleo", "desocupacion", "empleo", "trabajo", "laboral", "actividad laboral", "oferta laboral",
               "informalidad", "subocupacion", "paro"],
    "actividad": ["pbi", "pib", "producto", "emae", "crecimiento", "recesion", "actividad economica"],
    "pobreza": ["pobreza", "pobres", "indigencia", "canasta"],
    "fiscal": ["fiscal", "superavit", "deficit", "gasto", "recaudacion", "impuestos", "intereses", "resultado"],
    "externo": ["dolar", "tipo de cambio", "reservas", "exportaciones", "importaciones", "balanza", "bcra"],
    "salarios": ["salario", "sueldo", "salarios", "poder adquisitivo", "masa salarial", "remuneracion", "ingreso"],
    "credito": ["mora", "morosidad", "deuda", "deudas", "credito", "prestamo", "prestamos", "tarjeta", "irregularidad"],
}


def responder_local(pregunta: str, hechos: dict[str, list[str]]) -> str:
    q = _norm(pregunta)
    # ¿menciona provincias o aglomerados?
    encontrados = []
    for linea in hechos.get("provincias", []) + hechos.get("aglomerados", []):
        nombre = _norm(linea.split(":")[0].split(" (")[0])
        if len(nombre) > 3 and re.search(r"\b" + re.escape(nombre) + r"\b", q):
            encontrados.append(linea)
    temas = [t for t, claves in TEMAS.items() if any(c in q for c in claves)]
    partes = []
    if encontrados:
        partes += encontrados[:4]
    if temas and not encontrados:
        for t in temas:
            partes += hechos.get(t, [])
    if not partes:
        if any(p in q for p in ["provincia", "provincias", "ranking", "mayor", "menor", "peor", "mejor"]):
            return ("Puedo comparar provincias: preguntame por una (por ejemplo «¿cómo está el empleo en Neuquén?») "
                    "o mirá la pestaña Provincias, donde el mapa ordena las 24 jurisdicciones por el indicador que elijas.")
        partes = hechos["inflacion"][:1] + hechos["empleo"] + hechos["actividad"][:1] + hechos["pobreza"]
        partes.insert(0, "No encontré un tema puntual en tu pregunta; este es el panorama general:")
    texto = "\n".join(f"• {p}" if not p.endswith(":") else p for p in partes)
    if encontrados:
        texto += "\n\nNota: la EPH mide aglomerados urbanos; el dato del aglomerado principal se usa como referencia provincial."
    return texto


# ---------------------------------------------------------------------------
# Informe automático
# ---------------------------------------------------------------------------
def generar_informe(hechos: dict[str, list[str]], api_key: str | None = None, modelo: str | None = None) -> tuple[str, str]:
    cliente = _cliente_anthropic(api_key)
    if cliente is not None:
        prompt = ("Con estos datos, escribí un resumen ejecutivo de la coyuntura macroeconómica argentina en 3 párrafos "
                  "(precios y salarios; actividad, empleo y pobreza; frente fiscal, crédito y externo) y cerrá con 3 riesgos a "
                  "monitorear. Citá períodos. No agregues datos que no estén en el contexto.\n\n" + contexto_texto(hechos))
        try:
            r = cliente.messages.create(model=modelo or MODELO_POR_DEFECTO, max_tokens=1200, system=SISTEMA,
                                        messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text"), "claude"
        except Exception:
            pass
    secciones = [
        ("Precios y salarios", hechos["inflacion"] + hechos["salarios"]),
        ("Actividad, empleo y pobreza", hechos["actividad"] + hechos["empleo"] + hechos["pobreza"]),
        ("Frente fiscal y externo", hechos["fiscal"] + hechos["externo"]),
    ]
    if "credito" in hechos:
        secciones.append(("Crédito y mora", hechos["credito"][:3]))
    texto = "\n\n".join(f"**{t}.** " + " ".join(v) for t, v in secciones)
    return texto, "plantilla"
