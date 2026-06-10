'''this is the flow chart of the project'''

import os
from graphviz import Digraph

os.makedirs("diagrams", exist_ok=True)

g = Digraph(
    format="png",
    graph_attr={"rankdir": "TB", "splines": "ortho", "nodesep": "0.6",
                "ranksep": "0.7", "fontname": "Helvetica", "bgcolor": "white", "pad": "0.4"},
    node_attr={"fontname": "Helvetica", "fontsize": "11", "style": "filled",
               "penwidth": "1.2", "margin": "0.18,0.12"},
    edge_attr={"penwidth": "1.0", "color": "#888780", "arrowsize": "0.7"},
)

C = {
    "purple": {"fill": "#EEEDFE", "border": "#534AB7", "font": "#3C3489"},
    "coral":  {"fill": "#FAECE7", "border": "#993C1D", "font": "#712B13"},
    "teal":   {"fill": "#E1F5EE", "border": "#0F6E56", "font": "#085041"},
    "amber":  {"fill": "#FAEEDA", "border": "#854F0B", "font": "#633806"},
    "blue":   {"fill": "#E6F1FB", "border": "#185FA5", "font": "#0C447C"},
    "green":  {"fill": "#EAF3DE", "border": "#3B6D11", "font": "#27500A"},
    "gray":   {"fill": "#F1EFE8", "border": "#5F5E5A", "font": "#444441"},
}

def n(g, i, l, c):
    g.node(i, label=l, fillcolor=C[c]["fill"], color=C[c]["border"], fontcolor=C[c]["font"])

with g.subgraph() as s:
    s.attr(rank="same")
    n(s, "garch",  "GJR-GARCH\nPredicted vol",          "purple")
    n(s, "credit", "Credit\nLevel · shock · accel",      "coral")
    n(s, "liq",    "Liquidity\nLevel · shock · accel",   "coral")
    n(s, "corr",   "Correlation\nPartial · rolling",     "teal")

with g.subgraph() as s:
    s.attr(rank="same")
    n(s, "stress", "Stress index\nCredit + liquidity",   "amber")

with g.subgraph() as s:
    s.attr(rank="same")
    n(s, "ukf",    "UKF Kalman\nTrend · velocity",       "purple")

with g.subgraph() as s:
    s.attr(rank="same")
    g.node("bus",
           label="Signal bus\nKalman · GARCH · credit · liquidity · correlation",
           shape="rectangle", style="filled,dashed",
           fillcolor=C["gray"]["fill"], color=C["gray"]["border"],
           fontcolor=C["gray"]["font"], fontname="Helvetica",
           fontsize="11", margin="0.2,0.15")

with g.subgraph() as s:
    s.attr(rank="same")
    n(s, "bocpd", "BOCPD\nP(changepoint)",       "blue")
    n(s, "cusum", "CUSUM\nThreshold sum",          "blue")
    n(s, "hmm",   "HMM (Gaussian)\nLatent regime", "teal")

with g.subgraph() as s:
    s.attr(rank="same")
    n(s, "out", "Regime classification\nBear · bull · sideways", "green")

solid  = {"color": "#888780", "penwidth": "1.2"}
bypass = {"color": "#B4B2A9", "penwidth": "1.0", "style": "dashed"}

g.edge("credit", "stress", **solid)
g.edge("liq",    "stress", **solid)
g.edge("garch",  "ukf",    **solid)
g.edge("stress", "ukf",    **solid)
g.edge("garch",  "bus",    **solid)
g.edge("credit", "bus",    **solid)
g.edge("liq",    "bus",    **solid)
g.edge("corr",   "bus",    **solid)
g.edge("ukf",    "bus",    **solid)
g.edge("bus",    "bocpd",  **solid)
g.edge("bus",    "cusum",  **solid)
g.edge("bus",    "hmm",    **solid)
g.edge("bocpd",  "out",    **bypass)
g.edge("cusum",  "out",    **bypass)
g.edge("hmm",    "out",    **solid)

g.render("diagrams/regime_pipeline", cleanup=True, view=False)
print("Saved → diagrams/regime_pipeline.png")