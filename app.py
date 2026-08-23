"""
Dashboard generalizzata di simulazione scenari — AI Hybrid Observer
Motore Monte Carlo riutilizzabile su più temi editoriali.

Per lanciarla in locale:
    pip install -r requirements.txt
    streamlit run app.py

Per pubblicarla gratis:
    1. Crea un repo GitHub con app.py + requirements.txt
    2. Vai su https://share.streamlit.io , collega il repo
    3. Incorpora l'URL risultante su WordPress con un iframe
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="AI Hybrid Observer — Simulatore di Scenari", layout="wide")

# ---------------------------------------------------------------------------
# 1. CONFIGURAZIONE DEI TEMI EDITORIALI
# Ogni tema è un modulo indipendente: per aggiungerne uno nuovo basta
# aggiungere una voce al dizionario THEMES, senza toccare il motore di calcolo.
# ---------------------------------------------------------------------------

THEMES = {
    "Siccità e agricoltura (Sicilia)": dict(
        driver_label="Indice severità siccità (0-100)",
        impact_label="Perdita di raccolto vs baseline (%)",
        impact_unit="%",
        economic_baseline=4282,  # milioni € valore aggiunto agricolo Sicilia 2024
        economic_label="Perdita economica stimata (milioni €/anno)",
        real_event=dict(year=2024, driver_value=66, impact_value=-10,
                         label="Siccità reale 2024 (66% colture non irrigue in stato severo-estremo)"),
        sensitivity_label="Sensibilità colture (scenario peggiore, 2070)",
        sensitivity_unit="% perdita produttività",
        sensitivity=[("Grano duro", 32), ("Ortaggi", 28), ("Vite", 22), ("Agrumi", 18), ("Olivo", 15)],
        scenarios={
            "Ottimistico — mitigazione forte": dict(color="#2ecc71",
                driver_anchors=([2025, 2040, 2070, 2100], [45, 48, 50, 52]),
                impact_anchors=([2025, 2040, 2070, 2100], [-5, -8, -10, -12])),
            "Intermedio — politiche attuali": dict(color="#f39c12",
                driver_anchors=([2025, 2040, 2070, 2100], [48, 55, 65, 70]),
                impact_anchors=([2025, 2040, 2070, 2100], [-8, -15, -22, -25])),
            "Pessimistico — nessuna mitigazione": dict(color="#e74c3c",
                driver_anchors=([2025, 2040, 2070, 2100], [50, 62, 78, 88]),
                impact_anchors=([2025, 2040, 2070, 2100], [-8, -19, -28, -32])),
        },
        source_note="Fonti: IPCC AR6, CREA, Regione Siciliana, Zampieri et al. 2020",
    ),

    "AI e mercato del lavoro": dict(
        driver_label="Esposizione dei lavori all'automazione AI (%)",
        impact_label="Variazione netta occupazione (%)",
        impact_unit="%",
        economic_baseline=25000,  # milioni € proxy massa salariale settori esposti (illustrativo)
        economic_label="Costo/beneficio netto occupazionale stimato (milioni €/anno)",
        real_event=dict(year=2024, driver_value=40, impact_value=-2,
                         label="Esposizione media globale già nel 2024 (40% dei lavori, 60% economie avanzate)"),
        sensitivity_label="Esposizione per categoria professionale (scenario accelerato, 2035)",
        sensitivity_unit="% mansioni automatizzabili",
        sensitivity=[("Impiegati amministrativi", 55), ("Programmatori junior", 48),
                     ("Servizio clienti", 45), ("Analisti finanziari", 38), ("Professioni manuali", 12)],
        scenarios={
            "Adattamento guidato": dict(color="#2ecc71",
                driver_anchors=([2025, 2030, 2040, 2060], [40, 45, 50, 55]),
                impact_anchors=([2025, 2030, 2040, 2060], [-1, 0, 2, 5])),
            "Transizione turbolenta": dict(color="#f39c12",
                driver_anchors=([2025, 2030, 2040, 2060], [40, 52, 62, 68]),
                impact_anchors=([2025, 2030, 2040, 2060], [-2, -6, -8, -6])),
            "Sostituzione rapida": dict(color="#e74c3c",
                driver_anchors=([2025, 2030, 2040, 2060], [40, 60, 75, 85]),
                impact_anchors=([2025, 2030, 2040, 2060], [-3, -12, -20, -22])),
        },
        source_note="Fonti: rapporti su esposizione lavorativa all'AI (2024-2026), McKinsey Global Institute",
    ),

    "Energia e transizione ecologica": dict(
        driver_label="Quota rinnovabili sui consumi finali (%)",
        impact_label="Riduzione emissioni CO2 vs 1990 (%)",
        impact_unit="%",
        economic_baseline=15000,  # milioni € investimenti annui stimati transizione energetica Italia
        economic_label="Investimenti/costi netti stimati (milioni €/anno)",
        real_event=dict(year=2024, driver_value=29, impact_value=-30,
                         label="Quota rinnovabili installata a fine 2024 (~29% dell'obiettivo 2030 PNIEC)"),
        sensitivity_label="Contributo per fonte al 2030 (scenario target PNIEC)",
        sensitivity_unit="% del mix rinnovabile",
        sensitivity=[("Fotovoltaico", 45), ("Eolico", 25), ("Idroelettrico", 18), ("Biomasse", 8), ("Altro", 4)],
        scenarios={
            "Target PNIEC raggiunto": dict(color="#2ecc71",
                driver_anchors=([2025, 2030, 2040, 2050], [32, 39, 55, 75]),
                impact_anchors=([2025, 2030, 2040, 2050], [-25, -55, -75, -95])),
            "Traiettoria attuale": dict(color="#f39c12",
                driver_anchors=([2025, 2030, 2040, 2050], [30, 34, 42, 55]),
                impact_anchors=([2025, 2030, 2040, 2050], [-22, -40, -55, -70])),
            "Ritardo strutturale": dict(color="#e74c3c",
                driver_anchors=([2025, 2030, 2040, 2050], [29, 31, 35, 42]),
                impact_anchors=([2025, 2030, 2040, 2050], [-20, -28, -35, -45])),
        },
        source_note="Fonti: PNIEC 2024, Legambiente, RePowerEU",
    ),

    "Rischio idrico urbano": dict(
        driver_label="Indice di stress idrico (0-100)",
        impact_label="Danni economici da eventi estremi (miliardi $/anno)",
        impact_unit="mld $",
        economic_baseline=1,  # già in miliardi, non serve riscalare
        economic_label="Danni economici stimati (miliardi $/anno)",
        real_event=dict(year=2024, driver_value=44, impact_value=-1.5,
                         label="Italia già classificata ad alto stress idrico (26% popolazione esposta)"),
        sensitivity_label="Esposizione per tipo di rischio idrico (scenario peggiore, 2050)",
        sensitivity_unit="miliardi $ danni stimati",
        sensitivity=[("Alluvioni fluviali", 2.8), ("Alluvioni costiere", 3.6),
                     ("Siccità urbana", 1.9), ("Reti idriche colabrodo", 1.2), ("Subsidenza", 0.6)],
        scenarios={
            "Gestione resiliente": dict(color="#2ecc71",
                driver_anchors=([2025, 2030, 2040, 2050], [44, 46, 48, 50]),
                impact_anchors=([2025, 2030, 2040, 2050], [-1.5, -1.6, -1.9, -2.1])),
            "Traiettoria attuale": dict(color="#f39c12",
                driver_anchors=([2025, 2030, 2040, 2050], [44, 50, 58, 65]),
                impact_anchors=([2025, 2030, 2040, 2050], [-1.5, -1.8, -2.4, -2.8])),
            "Crisi idrica cronica": dict(color="#e74c3c",
                driver_anchors=([2025, 2030, 2040, 2050], [44, 55, 68, 80]),
                impact_anchors=([2025, 2030, 2040, 2050], [-1.5, -2.0, -3.0, -3.6])),
        },
        source_note="Fonti: WRI Aqueduct, Legambiente, ISPRA",
    ),

    "Rischio geopolitico dell'AI": dict(
        driver_label="Indice di tensione tecnologica USA-Cina (0-100)",
        impact_label="Rischio di interruzione filiera chip avanzati (%)",
        impact_unit="%",
        economic_baseline=2887,  # miliardi $ spesa militare globale 2025 (SIPRI), usato come proxy di scala del rischio
        economic_label="Costo geopolitico stimato (proxy, miliardi $/anno)",
        real_event=dict(year=2025, driver_value=55, impact_value=-15,
                         label="Regime USA a tre livelli sull'export di chip AI (gennaio 2025)"),
        sensitivity_label="Esposizione per attore della filiera (scenario escalation, 2040)",
        sensitivity_unit="% rischio di interruzione produzione",
        sensitivity=[("Taiwan — chip avanzati (TSMC)", 68), ("Cina — autosufficienza forzata", 55),
                     ("Corea del Sud — memory/logic", 42), ("USA — reshoring produzione", 30),
                     ("Europa — equipaggiamenti litografia (ASML)", 25)],
        scenarios={
            "Distensione parziale": dict(color="#2ecc71",
                driver_anchors=([2025, 2030, 2040, 2050], [50, 45, 40, 35]),
                impact_anchors=([2025, 2030, 2040, 2050], [-15, -10, -8, -5])),
            "Competizione strutturata (attuale)": dict(color="#f39c12",
                driver_anchors=([2025, 2030, 2040, 2050], [55, 65, 70, 72]),
                impact_anchors=([2025, 2030, 2040, 2050], [-15, -22, -28, -30])),
            "Escalation / crisi Taiwan": dict(color="#e74c3c",
                driver_anchors=([2025, 2030, 2040, 2050], [60, 80, 90, 95]),
                impact_anchors=([2025, 2030, 2040, 2050], [-15, -35, -55, -70])),
        },
        source_note="Fonti: SIPRI 2026 (spesa militare globale), regolamenti export USA su AI chip 2025-2026, "
                     "Reuters/Bloomberg su controlli semiconduttori Cina-Taiwan",
    ),
}

# ---------------------------------------------------------------------------
# 2. MOTORE MONTE CARLO (riutilizzabile per qualsiasi tema)
# ---------------------------------------------------------------------------

@st.cache_data
def run_simulation(theme_key: str, n_sims: int, seed: int = 7):
    theme = THEMES[theme_key]
    all_years = sorted({y for sc in theme["scenarios"].values() for y in sc["driver_anchors"][0]})
    years = np.arange(min(all_years), max(all_years) + 1)
    rng = np.random.default_rng(seed)

    results = {}
    for name, sc in theme["scenarios"].items():
        d_central = np.interp(years, *sc["driver_anchors"])
        i_central = np.interp(years, *sc["impact_anchors"])
        noise_d = np.linspace(abs(d_central[0]) * 0.03 + 0.5, abs(d_central[-1]) * 0.12 + 1, len(years))
        noise_i = np.linspace(abs(i_central[0]) * 0.03 + 0.3, abs(i_central[-1]) * 0.12 + 0.5, len(years))

        driver = (d_central[None, :]
                  + rng.normal(0, 1, (n_sims, 1)) * noise_d[None, :] * 0.6
                  + rng.normal(0, 1, (n_sims, len(years))) * noise_d[None, :] * 0.4)
        impact = (i_central[None, :]
                  + rng.normal(0, 1, (n_sims, 1)) * noise_i[None, :] * 0.6
                  + rng.normal(0, 1, (n_sims, len(years))) * noise_i[None, :] * 0.4)

        results[name] = dict(driver=driver, impact=impact)
    return years, results


def build_dashboard(theme_key: str, years, results):
    theme = THEMES[theme_key]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(theme["driver_label"], theme["impact_label"],
                         theme["economic_label"], theme["sensitivity_label"]),
    )

    for name, sc in theme["scenarios"].items():
        color = sc["color"]
        d, im = results[name]["driver"], results[name]["impact"]
        d10, d50, d90 = np.percentile(d, 10, axis=0), np.percentile(d, 50, axis=0), np.percentile(d, 90, axis=0)
        i10, i50, i90 = np.percentile(im, 10, axis=0), np.percentile(im, 50, axis=0), np.percentile(im, 90, axis=0)

        fig.add_trace(go.Scatter(x=list(years) + list(years[::-1]), y=list(d90) + list(d10[::-1]),
                                  fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0),
                                  showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=years, y=d50, line=dict(color=color, width=3), name=name), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(years) + list(years[::-1]), y=list(i90) + list(i10[::-1]),
                                  fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0),
                                  showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=years, y=i50, line=dict(color=color, width=3), showlegend=False), row=1, col=2)

    re = theme.get("real_event")
    if re:
        fig.add_annotation(x=re["year"], y=re["driver_value"], text=re["label"], showarrow=True,
                            arrowhead=2, row=1, col=1, font=dict(color="white", size=10))

    bar_labels = [f"{y}" for y in [years[0], years[len(years)//2], years[-1]]]
    for name, sc in theme["scenarios"].items():
        i_mean = np.mean(results[name]["impact"], axis=0)
        idxs = [0, len(years)//2, -1]
        vals = [abs(i_mean[i]) / 100 * theme["economic_baseline"] if theme["impact_unit"] == "%"
                else abs(i_mean[i]) for i in idxs]
        fig.add_trace(go.Bar(x=bar_labels, y=vals, marker_color=sc["color"], showlegend=False), row=2, col=1)

    labels, values = zip(*theme["sensitivity"])
    fig.add_trace(go.Bar(x=list(values), y=list(labels), orientation="h",
                          marker=dict(color=list(values), colorscale="YlOrRd"), showlegend=False), row=2, col=2)

    fig.update_layout(template="plotly_dark", height=800, barmode="group",
                       legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
                       margin=dict(t=110))
    return fig


# ---------------------------------------------------------------------------
# 3. INTERFACCIA STREAMLIT
# ---------------------------------------------------------------------------

st.title("Simulatore di Scenari — AI Hybrid Observer")
st.caption("Motore Monte Carlo generalizzato per esplorare traiettorie future su più temi editoriali.")

col_a, col_b = st.columns([2, 1])
with col_a:
    theme_key = st.selectbox("Scegli il tema editoriale", list(THEMES.keys()))
with col_b:
    n_sims = st.slider("Numero di simulazioni Monte Carlo", 500, 10000, 3000, step=500)

years, results = run_simulation(theme_key, n_sims)
fig = build_dashboard(theme_key, years, results)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"*{THEMES[theme_key]['source_note']}*")

with st.expander("Scarica i dati della simulazione (CSV)"):
    rows = []
    for name, sc in THEMES[theme_key]["scenarios"].items():
        d, im = results[name]["driver"], results[name]["impact"]
        for i, y in enumerate(years):
            rows.append({
                "scenario": name, "anno": y,
                "driver_P10": np.percentile(d[:, i], 10), "driver_P50": np.percentile(d[:, i], 50),
                "driver_P90": np.percentile(d[:, i], 90),
                "impatto_P10": np.percentile(im[:, i], 10), "impatto_P50": np.percentile(im[:, i], 50),
                "impatto_P90": np.percentile(im[:, i], 90),
            })
    df = pd.DataFrame(rows).round(2)
    st.dataframe(df, use_container_width=True)
    st.download_button("Scarica CSV", df.to_csv(index=False).encode("utf-8"),
                        file_name=f"simulazione_{theme_key.split()[0].lower()}.csv")

st.markdown("---")
st.caption("Nota metodologica: i valori centrali sono ancorati a dati reali o a proiezioni pubblicate; "
           "le bande di incertezza (P10-P90) sono generate con simulazione Monte Carlo e vanno intese "
           "come illustrative, non come previsioni di precisione.")
