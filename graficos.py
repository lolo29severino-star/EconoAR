"""
Gráficos reutilizables (app y notebook).

mapa_provincias() dibuja el mapa coroplético con polígonos de Plotly sobre ejes
cartesianos (proyección equirectangular corregida por latitud). A diferencia de
px.choropleth no necesita descargar el mapa base de cdn.plot.ly ni WebGL, así que
funciona sin internet y se exporta bien a imagen.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

COS_LAT = math.cos(math.radians(38))  # corrige el estiramiento horizontal a ~38° S


def _anillos(geom: dict):
    if geom["type"] == "Polygon":
        yield geom["coordinates"][0]
    else:
        for poly in geom["coordinates"]:
            yield poly[0]


def mapa_provincias(provincias: pd.DataFrame, geojson: dict, columna: str, titulo: str = "",
                    escala: str = "Blues", unidad: str = "", medio: float | None = None,
                    altura: int = 720, decimales: int = 1, recortar: bool = True) -> go.Figure:
    """Mapa coroplético de las 24 jurisdicciones.

    Con `medio` la escala es divergente y centrada en ese valor. Con `recortar`,
    la amplitud de la escala divergente se fija en el percentil 90 de los desvíos,
    para que un valor extremo no deje al resto del mapa sin contraste (los valores
    por fuera se pintan con el color del extremo).
    """
    valores = provincias.set_index("provincia")[columna]
    vmin, vmax = np.nanmin(valores.values), np.nanmax(valores.values)
    if medio is not None:  # escala divergente centrada en `medio`
        desvios = np.abs(valores.dropna().values - medio)
        amp = (np.percentile(desvios, 90) if recortar and len(desvios) > 5 else desvios.max()) or 1
        vmin, vmax = medio - amp, medio + amp
    rango = (vmax - vmin) or 1

    def _color(v):
        return sample_colorscale(escala, [min(1.0, max(0.0, (v - vmin) / rango))])[0]
    fig = go.Figure()
    for feat in geojson["features"]:
        nombre = feat["properties"]["provincia"]
        v = valores.get(nombre, np.nan)
        color = "#e5e7eb" if pd.isna(v) else _color(v)
        etiqueta = "s/d" if pd.isna(v) else f"{v:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        xs, ys = [], []
        for anillo in _anillos(feat["geometry"]):
            xs += [p[0] * COS_LAT for p in anillo] + [None]
            ys += [p[1] for p in anillo] + [None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself", fillcolor=color,
            line=dict(color="white", width=0.8), name=nombre, showlegend=False,
            hoveron="fills", hoverinfo="text", text=f"<b>{nombre}</b><br>{etiqueta} {unidad}",
        ))
    # CABA es demasiado chica para verse: se marca con un círculo del mismo color
    if "CABA" in valores.index:
        v = valores["CABA"]
        color = "#e5e7eb" if pd.isna(v) else _color(v)
        etiqueta = "s/d" if pd.isna(v) else f"{v:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        fig.add_trace(go.Scatter(
            x=[-58.44 * COS_LAT], y=[-34.61], mode="markers+text", text=["CABA"], textposition="middle right",
            textfont=dict(size=10, color="#334155"), showlegend=False, hoverinfo="text",
            hovertext=f"<b>CABA</b><br>{etiqueta} {unidad}",
            marker=dict(size=13, color=color, line=dict(color="#334155", width=1)),
        ))
    # barra de colores
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=escala, cmin=vmin, cmax=vmax, color=[vmin],
                    colorbar=dict(title=unidad, thickness=12, len=0.6, x=0.92)),
    ))
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=titulo, height=altura, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=0, r=0, t=40 if titulo else 10, b=0), hovermode="closest")
    return fig
