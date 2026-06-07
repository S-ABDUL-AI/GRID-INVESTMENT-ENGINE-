"""
Grid Investment Prioritization Engine
Benefit-Cost Optimization for Utility Infrastructure Capital Decisions
Author: Sherriff Abdul-Hamid | poverty360.org

Applies expected-value economics and portfolio optimization to grid
infrastructure investment decisions — bridging development economics
methodology with utility asset management.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import html
from datetime import date
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ── DESIGN TOKENS ─────────────────────────────────────────────────────────────
NAVY      = "#0A1F44"
NAVY_MID  = "#152B5C"
GOLD      = "#C9A84C"
GOLD_LT   = "#E8C97A"
INK       = "#1A1A1A"
BODY      = "#2C3E50"
MUTED     = "#6B7280"
RED       = "#C8382A"
AMBER     = "#B8560A"
GREEN     = "#1A7A2E"
TEAL      = "#0E7490"
RULE      = "#E2E6EC"
OFF_WHITE = "#F8F6F1"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grid Investment Prioritization Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] {{
    font-family: 'Source Sans 3', sans-serif; background: {OFF_WHITE};
  }}
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, {NAVY_MID} 100%);
  }}
  [data-testid="stSidebar"] * {{ color: #D8E2F0 !important; }}
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stCheckbox label,
  [data-testid="stSidebar"] .stRadio label,
  [data-testid="stSidebar"] .stMultiSelect label {{
    color: #E8EEF8 !important; font-weight: 600 !important;
  }}
  .hero-wrap {{
    background: linear-gradient(135deg, {NAVY} 0%, {NAVY_MID} 65%, #1E3A6E 100%);
    border-left: 6px solid {GOLD}; border-radius: 6px;
    padding: 34px 40px 28px; margin-bottom: 22px;
  }}
  .hero-eye   {{ font-size: 11px; font-weight: 700; letter-spacing: 2.5px;
                 color: {GOLD}; text-transform: uppercase; margin-bottom: 10px; }}
  .hero-title {{ font-size: 26px; font-weight: 700; color: #FFFFFF;
                 line-height: 1.3; margin-bottom: 10px; }}
  .hero-sub   {{ font-size: 13.5px; color: #B0BFD8; line-height: 1.65; max-width: 920px; }}
  .sec-lbl {{ font-size: 10px; font-weight: 700; letter-spacing: 2px;
              color: {GOLD}; text-transform: uppercase; margin-bottom: 3px; }}
  .sec-ttl {{ font-size: 19px; font-weight: 700; color: {NAVY}; margin-bottom: 3px; }}
  .sec-sub {{ font-size: 13px; color: {MUTED}; margin-bottom: 14px; }}
  .kpi-card {{
    background: #FFFFFF; border: 1px solid {RULE}; border-top: 3px solid {NAVY};
    border-radius: 4px; padding: 16px 18px; text-align: center;
  }}
  .kpi-value {{ font-size: 1.85rem; font-weight: 700; color: {NAVY}; margin: 0; line-height: 1.1; }}
  .kpi-label {{ font-size: 10px; font-weight: 700; color: {MUTED}; margin: 6px 0 0;
                text-transform: uppercase; letter-spacing: 1px; }}
  .kpi-green {{ border-top-color: {GREEN}; }}
  .kpi-amber {{ border-top-color: {AMBER}; }}
  .kpi-red   {{ border-top-color: {RED}; }}
  .section-header {{
    font-size: 11px; font-weight: 700; color: {NAVY}; letter-spacing: 1.5px;
    text-transform: uppercase; border-bottom: 2px solid {GOLD};
    padding-bottom: 6px; margin-bottom: 14px; margin-top: 18px;
  }}
  .insight-box {{
    background: #F0F4FF; border-left: 4px solid {NAVY}; border-radius: 4px;
    padding: 14px 18px; font-size: 13px; color: {BODY}; line-height: 1.65; margin: 12px 0;
  }}
  .formula-box {{
    background: #FFFFFF; border: 1px solid {RULE}; border-left: 4px solid {GOLD};
    border-radius: 4px; padding: 14px 18px; font-family: 'Courier New', monospace;
    font-size: 12.5px; color: {BODY}; margin: 10px 0; line-height: 1.8;
  }}
  .report-card {{
    background: #FFFFFF; border: 1px solid {RULE}; border-left: 5px solid {GOLD};
    border-radius: 6px; padding: 20px 24px; margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(10, 31, 68, 0.06);
  }}
  .report-title {{ font-size: 16px; font-weight: 700; color: {NAVY}; margin-bottom: 4px; }}
  .report-sub   {{ font-size: 13px; color: {MUTED}; line-height: 1.55; margin-bottom: 12px; }}
  .report-loc   {{ font-size: 11.5px; color: {AMBER}; font-weight: 600;
                   letter-spacing: 0.3px; margin-bottom: 10px; }}
  .byline {{
    background: {NAVY}; border-radius: 4px; padding: 16px 22px;
    font-size: 11.5px; color: #B0BFD8; line-height: 1.8; margin-top: 28px;
  }}
  .byline a {{ color: {GOLD}; text-decoration: none; }}
  button[data-baseweb="tab"] {{
    font-weight: 600 !important; font-size: 13px !important; color: {NAVY} !important;
  }}
  div[data-testid="stDownloadButton"] > button {{
    background: {NAVY}; color: #FFFFFF; border: none; border-radius: 4px;
    font-weight: 600; letter-spacing: 0.3px;
  }}
</style>
""", unsafe_allow_html=True)


# ── DATA GENERATION ───────────────────────────────────────────────────────────
@st.cache_data
def generate_projects(seed=42):
    rng = np.random.default_rng(seed)

    categories = {
        "Wildfire Mitigation":    {"n": 22, "cost_range": (0.8e6,  18e6),  "base_bcr": 3.8},
        "Grid Hardening":         {"n": 18, "cost_range": (1.5e6,  25e6),  "base_bcr": 2.9},
        "Predictive Maintenance": {"n": 16, "cost_range": (0.3e6,  6e6),   "base_bcr": 4.2},
        "Undergrounding":         {"n": 12, "cost_range": (5e6,    60e6),  "base_bcr": 1.8},
        "Substation Upgrades":    {"n": 10, "cost_range": (3e6,    40e6),  "base_bcr": 2.4},
        "Vegetation Management":  {"n": 14, "cost_range": (0.5e6,  8e6),   "base_bcr": 5.1},
        "Climate Resilience":     {"n":  8, "cost_range": (2e6,    20e6),  "base_bcr": 2.2},
    }

    regions = [
        "LA Foothills", "San Bernardino Mtns", "Inland Empire",
        "San Gabriel Valley", "Orange County North",
        "Coachella Valley", "Pomona–Ontario",
    ]

    rows = []
    proj_id = 1
    for cat, cfg in categories.items():
        for _ in range(cfg["n"]):
            cost = rng.uniform(*cfg["cost_range"])
            bcr_noise = rng.normal(0, 0.6)
            bcr = max(0.5, cfg["base_bcr"] + bcr_noise)

            # Risk reduction: total benefit = BCR × cost
            total_benefit = bcr * cost
            # Split benefit: direct risk reduction vs co-benefits
            direct_frac = rng.uniform(0.55, 0.82)
            risk_reduction = total_benefit * direct_frac
            cobenefit      = total_benefit * (1 - direct_frac)

            # Failure probability reduction
            pf_reduction = np.clip(rng.normal(0.18, 0.07), 0.04, 0.40)

            # Timeline
            duration_months = int(rng.choice([6, 12, 18, 24, 36],
                                  p=[0.15, 0.30, 0.25, 0.20, 0.10]))

            # Regulatory requirement
            reg_required = rng.random() < (
                0.7 if cat in ["Wildfire Mitigation", "Vegetation Management"] else 0.25
            )

            # Assets protected
            assets_protected = int(rng.integers(5, 280))

            # Customers protected (weighted by region)
            customers = int(rng.integers(500, 85_000))

            # NPV at 7% discount rate
            annual_benefit = risk_reduction / (duration_months / 12)
            years = duration_months / 12
            npv = sum(annual_benefit / (1.07 ** t) for t in range(1, int(years) + 2)) - cost

            # Emissions reduction (CO2e tonnes)
            co2_reduction = int(rng.integers(10, 2500))

            rows.append({
                "project_id":          f"INV-{proj_id:03d}",
                "project_name":        _project_name(cat, proj_id, rng),
                "category":            cat,
                "region":              rng.choice(regions),
                "cost_usd":            round(cost, 0),
                "risk_reduction_usd":  round(risk_reduction, 0),
                "cobenefit_usd":       round(cobenefit, 0),
                "total_benefit_usd":   round(total_benefit, 0),
                "bcr":                 round(bcr, 2),
                "npv_usd":             round(npv, 0),
                "pf_reduction":        round(pf_reduction, 3),
                "duration_months":     duration_months,
                "assets_protected":    assets_protected,
                "customers_protected": customers,
                "regulatory_required": reg_required,
                "co2_reduction_tonnes":co2_reduction,
                "cost_per_customer":   round(cost / customers, 2),
                "risk_per_dollar":     round(risk_reduction / cost, 3),
            })
            proj_id += 1

    return pd.DataFrame(rows)


def _project_name(cat, pid, rng):
    names = {
        "Wildfire Mitigation":    ["Covered Conductor Replacement", "PSPS Automation Upgrade",
                                   "Fire Hardening Program", "Ignition Prevention Initiative",
                                   "High Fire Risk Zone Retrofit"],
        "Grid Hardening":         ["Transmission Reinforcement", "Distribution Upgrade",
                                   "Storm Hardening Program", "Infrastructure Modernization",
                                   "Grid Resilience Initiative"],
        "Predictive Maintenance": ["Sensor Deployment Program", "AI-Driven Inspection Rollout",
                                   "Condition Monitoring Upgrade", "Smart Asset Diagnostics",
                                   "Remote Monitoring Initiative"],
        "Undergrounding":         ["Underground Conversion", "Overhead-to-Underground Migration",
                                   "Urban Circuit Undergrounding", "Residential Underground Program"],
        "Substation Upgrades":    ["Substation Automation", "Transformer Replacement Program",
                                   "Protection System Upgrade", "Control System Modernization"],
        "Vegetation Management":  ["Enhanced Patrol Program", "Rapid Response Vegetation",
                                   "LiDAR Vegetation Mapping", "Fuel Load Reduction",
                                   "Clearance Enhancement Program"],
        "Climate Resilience":     ["Extreme Heat Adaptation", "Drought Resilience Program",
                                   "Climate-Ready Infrastructure", "Long-Term Resilience Plan"],
    }
    return f"{rng.choice(names[cat])} {pid}"


@st.cache_data
def run_monte_carlo(projects_df, selected_ids, n_sims=2000, seed=99):
    rng = np.random.default_rng(seed)
    sel = projects_df[projects_df["project_id"].isin(selected_ids)]
    if sel.empty:
        return None
    results = []
    for _ in range(n_sims):
        noise      = rng.normal(1.0, 0.15, len(sel))
        sim_bcr    = (sel["bcr"].values * noise).mean()
        noise2     = rng.normal(1.0, 0.18, len(sel))
        sim_risk   = (sel["risk_reduction_usd"].values * noise2).sum()
        results.append({"sim_bcr": sim_bcr, "sim_risk_reduction": sim_risk})
    return pd.DataFrame(results)


def _apply_chart_theme(fig, height=None):
    layout = dict(
        font=dict(family="Source Sans 3, sans-serif", color=BODY, size=12),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFBFC",
        margin=dict(l=12, r=12, t=36, b=40),
    )
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=RULE, zerolinecolor=RULE)
    fig.update_yaxes(gridcolor=RULE, zerolinecolor=RULE)
    return fig


def greedy_optimize(df_cand, budget, obj_col, budget_col="cost_usd"):
    df_sorted = df_cand.sort_values(obj_col, ascending=False).copy()
    selected, spent = [], 0
    for _, row in df_sorted.iterrows():
        if spent + row[budget_col] <= budget:
            selected.append(row["project_id"])
            spent += row[budget_col]
    return selected


OBJ_MAP = {
    "Maximize Risk Reduction":      "risk_per_dollar",
    "Maximize BCR":                 "bcr",
    "Maximize Customers Protected": "customers_protected",
}


def build_scenarios(dff, scenario_budget, prioritize_by, include_reg):
    obj_col = OBJ_MAP[prioritize_by]

    scen_nothing = {
        "name": "Do Nothing", "ids": [], "cost": 0,
        "risk_red": 0, "customers": 0, "bcr": 0,
    }

    if include_reg:
        reg_projects = dff[dff["regulatory_required"]].copy()
        reg_cost = reg_projects["cost_usd"].sum()
        remaining = scenario_budget - reg_cost
        opt_cand = dff[~dff["regulatory_required"]].copy()
        opt_ids = reg_projects["project_id"].tolist()
        if remaining > 0:
            opt_ids += greedy_optimize(opt_cand, remaining, obj_col)
    else:
        opt_ids = greedy_optimize(dff, scenario_budget, obj_col)

    opt_sel = dff[dff["project_id"].isin(opt_ids)]
    scen_opt = {
        "name": "Optimized Portfolio",
        "ids": opt_ids,
        "cost": opt_sel["cost_usd"].sum(),
        "risk_red": opt_sel["risk_reduction_usd"].sum(),
        "customers": opt_sel["customers_protected"].sum(),
        "bcr": (
            opt_sel["total_benefit_usd"].sum() / opt_sel["cost_usd"].sum()
            if opt_sel["cost_usd"].sum() > 0 else 0
        ),
    }

    reg_sel = dff[dff["regulatory_required"]]
    scen_reg = {
        "name": "Regulatory Minimum",
        "ids": reg_sel["project_id"].tolist(),
        "cost": reg_sel["cost_usd"].sum(),
        "risk_red": reg_sel["risk_reduction_usd"].sum(),
        "customers": reg_sel["customers_protected"].sum(),
        "bcr": (
            reg_sel["total_benefit_usd"].sum() / reg_sel["cost_usd"].sum()
            if reg_sel["cost_usd"].sum() > 0 else 0
        ),
    }

    return [scen_nothing, scen_reg, scen_opt], opt_sel, opt_ids, scen_opt, scen_reg


def _scenario_summary_rows(scenarios, dff):
    rows = []
    baseline_risk = dff["risk_reduction_usd"].sum()
    for s in scenarios:
        roi = (s["risk_red"] / s["cost"] * 100) if s["cost"] > 0 else 0
        pct = (s["risk_red"] / baseline_risk * 100) if baseline_risk > 0 else 0
        rows.append({
            "Scenario": s["name"],
            "Projects Funded": len(s["ids"]),
            "Total Investment": f"${s['cost']/1e6:.1f}M",
            "Risk Reduction": f"${s['risk_red']/1e6:.1f}M",
            "% of Max Risk Red.": f"{pct:.1f}%",
            "Portfolio BCR": f"{s['bcr']:.2f}x",
            "ROI (Risk/Cost)": f"{roi:.0f}%",
            "Customers Protected": f"{s['customers']:,}",
        })
    return rows


def _build_report_html(dff, scenarios, scen_opt, scen_reg, opt_ids, meta):
    summary = _scenario_summary_rows(scenarios, dff)
    top_projects = (
        dff[dff["project_id"].isin(opt_ids)]
        .sort_values("bcr", ascending=False)
        .head(12)
    )
    rows_html = ""
    for row in summary:
        rows_html += (
            f"<tr><td>{html.escape(str(row['Scenario']))}</td>"
            f"<td>{row['Projects Funded']}</td>"
            f"<td>{row['Total Investment']}</td>"
            f"<td>{row['Risk Reduction']}</td>"
            f"<td>{row['Portfolio BCR']}</td></tr>"
        )
    proj_html = ""
    for r in top_projects.itertuples(index=False):
        proj_html += (
            f"<tr><td>{html.escape(str(r.project_id))}</td>"
            f"<td>{html.escape(str(r.project_name))}</td>"
            f"<td>{html.escape(str(r.category))}</td>"
            f"<td>${r.cost_usd:,.0f}</td>"
            f"<td>{r.bcr:.2f}x</td>"
            f"<td>${r.risk_reduction_usd:,.0f}</td></tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Grid Investment Executive Brief</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: {BODY}; margin: 40px; }}
  h1 {{ color: {NAVY}; font-size: 24px; margin-bottom: 4px; }}
  .eye {{ color: {GOLD}; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; }}
  .meta {{ color: {MUTED}; font-size: 13px; margin-bottom: 24px; }}
  h2 {{ color: {NAVY}; font-size: 16px; border-bottom: 2px solid {GOLD}; padding-bottom: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin: 12px 0 24px; }}
  th {{ background: {NAVY}; color: {GOLD}; text-align: left; padding: 8px 10px; }}
  td {{ border-bottom: 1px solid {RULE}; padding: 7px 10px; }}
  tr:nth-child(even) td {{ background: #F8F9FC; }}
  .note {{ background: #FFFBF0; border-left: 4px solid {GOLD}; padding: 12px 14px;
            font-size: 12.5px; line-height: 1.6; }}
  .footer {{ margin-top: 28px; font-size: 11px; color: {MUTED}; }}
</style></head><body>
<p class="eye">Grid Investment Prioritization Engine · Executive Brief</p>
<h1>Portfolio Decision Support Report</h1>
<p class="meta">Generated {date.today().isoformat()} · Budget ${meta['budget_m']:.0f}M ·
Objective: {html.escape(meta['objective'])} · {meta['project_count']} projects in scope</p>
<h2>Scenario Comparison</h2>
<table><thead><tr><th>Scenario</th><th>Projects</th><th>Investment</th>
<th>Risk Reduction</th><th>BCR</th></tr></thead><tbody>{rows_html}</tbody></table>
<h2>Optimized Portfolio — Top Projects</h2>
<table><thead><tr><th>ID</th><th>Project</th><th>Category</th>
<th>Cost</th><th>BCR</th><th>Risk Reduction</th></tr></thead><tbody>{proj_html}</tbody></table>
<div class="note"><strong>Key insight:</strong> The optimized portfolio delivers
${scen_opt['risk_red']/1e6:.1f}M in risk reduction at ${scen_opt['cost']/1e6:.1f}M invested
(BCR {scen_opt['bcr']:.2f}x), versus ${scen_reg['risk_red']/1e6:.1f}M under the regulatory minimum.
Methodology aligns with CPUC Risk-Based Decision-Making (RBDM) expected-value framework.</div>
<p class="footer">Built by Sherriff Abdul-Hamid · poverty360.org · Simulated SoCal utility portfolio.
Decision support only — not a regulatory filing.</p>
</body></html>"""


def _build_report_pdf(dff, scenarios, scen_opt, scen_reg, opt_ids, meta):
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    w, h = letter
    margin = 0.75 * inch
    c = canvas.Canvas(buf, pagesize=letter)
    y = h - margin

    c.setFillColor(rl_colors.HexColor(NAVY))
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, y, "Grid Investment Executive Brief")
    y -= 22
    c.setFillColor(rl_colors.HexColor(MUTED))
    c.setFont("Helvetica", 10)
    c.drawString(
        margin, y,
        f"Generated {date.today().isoformat()} · Budget ${meta['budget_m']:.0f}M · "
        f"{meta['project_count']} projects in scope",
    )
    y -= 28

    c.setFillColor(rl_colors.HexColor(NAVY))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Scenario Comparison")
    y -= 16
    c.setFont("Helvetica", 9.5)
    for s in scenarios:
        c.drawString(
            margin, y,
            f"{s['name']}: {len(s['ids'])} projects · "
            f"${s['cost']/1e6:.1f}M invested · "
            f"${s['risk_red']/1e6:.1f}M risk reduction · BCR {s['bcr']:.2f}x",
        )
        y -= 13
        if y < margin + 80:
            c.showPage()
            y = h - margin

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(rl_colors.HexColor(NAVY))
    c.drawString(margin, y, "Optimized Portfolio — Top Projects")
    y -= 16
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "ID")
    c.drawString(margin + 55, y, "Project")
    c.drawString(margin + 260, y, "BCR")
    c.drawString(w - margin - 90, y, "Cost")
    y -= 12
    c.setFont("Helvetica", 9)
    top_projects = (
        dff[dff["project_id"].isin(opt_ids)]
        .sort_values("bcr", ascending=False)
        .head(14)
    )
    for r in top_projects.itertuples(index=False):
        c.drawString(margin, y, str(r.project_id))
        c.drawString(margin + 55, y, str(r.project_name)[:34])
        c.drawString(margin + 260, y, f"{r.bcr:.2f}x")
        c.drawRightString(w - margin, y, f"${r.cost_usd/1e6:.1f}M")
        y -= 11
        if y < margin + 40:
            break

    y -= 10
    c.setFillColor(rl_colors.HexColor(MUTED))
    c.setFont("Helvetica", 8.5)
    c.drawString(
        margin, margin - 2,
        "Decision support brief · Sherriff Abdul-Hamid · poverty360.org · "
        "Simulated portfolio aligned to CPUC RBDM framework.",
    )
    c.save()
    return buf.getvalue()


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = generate_projects()

CATEGORIES = sorted(df["category"].unique())
REGIONS    = sorted(df["region"].unique())
CAT_COLORS = {
    "Wildfire Mitigation":    RED,
    "Grid Hardening":         NAVY,
    "Predictive Maintenance": GREEN,
    "Undergrounding":         "#6a1b9a",
    "Substation Upgrades":    TEAL,
    "Vegetation Management":  "#558b2f",
    "Climate Resilience":     "#00838f",
}


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Investment Engine")
    st.markdown("**Benefit-Cost Optimization · CPUC RBDM**")
    st.markdown("---")

    with st.expander("How to use this app", expanded=False):
        st.markdown(
            """
1. Set **filters** and **capital budget** in this sidebar.
2. Review tabs for portfolio analytics, scenarios, BCA, and Monte Carlo.
3. Download the **Executive Portfolio Brief** from the gold report card below the hero banner.
"""
        )

    st.markdown("**Filter Projects**")
    cat_filter = st.multiselect("Category", CATEGORIES, default=CATEGORIES)
    reg_filter = st.multiselect("Region",   REGIONS,    default=REGIONS)
    reg_req    = st.checkbox("Show regulatory-required only", value=False)

    st.markdown("---")
    st.markdown("**Discount Rate (NPV)**")
    discount_rate = st.slider("Rate (%)", 3.0, 12.0, 7.0, 0.5) / 100

    st.markdown("**Minimum BCR Threshold**")
    min_bcr = st.slider("Min BCR", 0.5, 4.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("**Scenario Settings**")
    scenario_budget = st.slider(
        "Capital Budget ($M)",
        min_value=5.0, max_value=200.0, value=50.0, step=5.0,
    ) * 1e6
    prioritize_by = st.radio(
        "Optimization Objective",
        list(OBJ_MAP.keys()),
    )
    include_reg = st.checkbox("Always include regulatory-required projects", value=True)

    st.markdown("---")
    st.caption(
        "Built by [Sherriff Abdul-Hamid](https://poverty360.org)  \n"
        "Data: Simulated SoCal utility projects  \n"
        "Method: Expected Value Optimization"
    )


# ── FILTERS ───────────────────────────────────────────────────────────────────
mask = (
    df["category"].isin(cat_filter) &
    df["region"].isin(reg_filter) &
    (df["bcr"] >= min_bcr)
)
if reg_req:
    mask &= df["regulatory_required"]
dff = df[mask].copy()

scenarios, opt_sel, opt_ids, scen_opt, scen_reg = build_scenarios(
    dff, scenario_budget, prioritize_by, include_reg,
)
report_meta = {
    "budget_m": scenario_budget / 1e6,
    "objective": prioritize_by,
    "project_count": len(dff),
}


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-eye">Utility Infrastructure · Portfolio Analytics</div>
  <div class="hero-title">Grid Investment Prioritization Engine</div>
  <div class="hero-sub">
    Expected-value optimization for utility capital decisions — ranking investments by
    risk reduction per dollar to maximize grid resilience within budget constraints.
    Aligned to the CPUC Risk-Based Decision-Making (RBDM) framework.
  </div>
</div>
""", unsafe_allow_html=True)

# ── EXECUTIVE REPORT DOWNLOAD ─────────────────────────────────────────────────
st.markdown('<div class="report-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="report-title">📄 Executive Portfolio Brief</div>'
    '<div class="report-sub">One-page decision brief with scenario comparison and '
    'top optimized projects — ready for stakeholder review.</div>'
    '<div class="report-loc">↓ Download location: use the buttons below this card</div>',
    unsafe_allow_html=True,
)
dl1, dl2, dl3 = st.columns(3)
report_slug = f"grid_investment_brief_{int(scenario_budget/1e6)}M"
with dl1:
    try:
        pdf_bytes = _build_report_pdf(
            dff, scenarios, scen_opt, scen_reg, opt_ids, report_meta,
        )
        st.download_button(
            "Download report (.pdf)",
            data=pdf_bytes,
            file_name=f"{report_slug}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except ModuleNotFoundError:
        st.caption("PDF requires reportlab on deploy.")
with dl2:
    html_bytes = _build_report_html(
        dff, scenarios, scen_opt, scen_reg, opt_ids, report_meta,
    ).encode("utf-8")
    st.download_button(
        "Download report (.html)",
        data=html_bytes,
        file_name=f"{report_slug}.html",
        mime="text/html",
        use_container_width=True,
    )
with dl3:
    csv_bytes = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download portfolio (.csv)",
        data=csv_bytes,
        file_name="grid_investment_portfolio.csv",
        mime="text/csv",
        use_container_width=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f'<div class="sec-lbl">Active scenario</div>'
    f'<div class="sec-sub">Budget <strong>${scenario_budget/1e6:.0f}M</strong> · '
    f'{len(opt_ids)} projects selected · '
    f'<strong>${scen_opt["risk_red"]/1e6:.1f}M</strong> risk reduction · '
    f'BCR <strong>{scen_opt["bcr"]:.2f}x</strong></div>',
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📋  Portfolio Overview",
    "⚙️  Scenario Builder",
    "📐  Benefit-Cost Analysis",
    "🎲  Sensitivity & Uncertainty",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:

    total_cost      = dff["cost_usd"].sum()
    total_risk_red  = dff["risk_reduction_usd"].sum()
    total_benefit   = dff["total_benefit_usd"].sum()
    portfolio_bcr   = total_benefit / total_cost if total_cost > 0 else 0
    total_customers = dff["customers_protected"].sum()
    reg_cost        = dff[dff["regulatory_required"]]["cost_usd"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value">{len(dff)}</p>
            <p class="kpi-label">Projects in Portfolio</p></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card kpi-amber">
            <p class="kpi-value">${total_cost/1e6:.0f}M</p>
            <p class="kpi-label">Total Investment Required</p></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card kpi-green">
            <p class="kpi-value">${total_risk_red/1e6:.0f}M</p>
            <p class="kpi-label">Total Risk Reduction</p></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card kpi-green">
            <p class="kpi-value">{portfolio_bcr:.2f}x</p>
            <p class="kpi-label">Portfolio BCR</p></div>""",
            unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value">{total_customers/1e3:.0f}K</p>
            <p class="kpi-label">Customers Protected</p></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_scatter, col_cat = st.columns([3, 2])

    with col_scatter:
        st.markdown('<div class="section-header">Investment Priority Matrix — BCR vs. Risk Reduction</div>',
                    unsafe_allow_html=True)
        st.caption("Bubble size = project cost. Top-right quadrant = highest priority.")

        fig_scatter = px.scatter(
            dff, x="risk_reduction_usd", y="bcr",
            size="cost_usd", color="category",
            color_discrete_map=CAT_COLORS,
            hover_name="project_name",
            hover_data={
                "project_id":      True,
                "region":          True,
                "cost_usd":        ":,.0f",
                "bcr":             ":.2f",
                "risk_reduction_usd": ":,.0f",
                "duration_months": True,
            },
            size_max=30, height=420,
            labels={
                "risk_reduction_usd": "Risk Reduction ($)",
                "bcr":                "Benefit-Cost Ratio",
                "category":           "Category",
            },
        )
        # Add quadrant lines
        med_risk = dff["risk_reduction_usd"].median()
        med_bcr  = dff["bcr"].median()
        fig_scatter.add_hline(y=med_bcr, line_dash="dot",
                              line_color="#999", annotation_text="Median BCR")
        fig_scatter.add_vline(x=med_risk, line_dash="dot",
                              line_color="#999", annotation_text="Median Risk Red.")
        fig_scatter.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            paper_bgcolor="white", plot_bgcolor="#FAFBFC",
            font=dict(family="Source Sans 3, sans-serif", color=BODY),
            xaxis=dict(gridcolor=RULE),
            yaxis=dict(gridcolor=RULE),
            legend=dict(orientation="h", yanchor="bottom", y=-0.30,
                        xanchor="left", x=0),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_cat:
        st.markdown('<div class="section-header">Portfolio by Category</div>',
                    unsafe_allow_html=True)

        cat_summary = dff.groupby("category").agg(
            Projects=("project_id", "count"),
            Total_Cost=("cost_usd", "sum"),
            Avg_BCR=("bcr", "mean"),
            Total_Risk_Red=("risk_reduction_usd", "sum"),
        ).reset_index()
        cat_summary["Total_Cost_M"]    = (cat_summary["Total_Cost"] / 1e6).round(1)
        cat_summary["Total_Risk_Red_M"]= (cat_summary["Total_Risk_Red"] / 1e6).round(1)
        cat_summary["Avg_BCR"]         = cat_summary["Avg_BCR"].round(2)

        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(
            name="Cost ($M)", x=cat_summary["category"],
            y=cat_summary["Total_Cost_M"],
            marker_color=[CAT_COLORS[c] for c in cat_summary["category"]],
            opacity=0.55, yaxis="y",
        ))
        fig_cat.add_trace(go.Scatter(
            name="Avg BCR", x=cat_summary["category"],
            y=cat_summary["Avg_BCR"],
            mode="markers+lines",
            marker=dict(size=10, color=NAVY),
            line=dict(color=NAVY, width=2),
            yaxis="y2",
        ))
        fig_cat.update_layout(
            xaxis=dict(tickangle=-35),
            yaxis=dict(title="Total Cost ($M)", gridcolor="#eee"),
            yaxis2=dict(title="Avg BCR", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)"),
            height=260, margin=dict(l=10, r=40, t=10, b=80),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", y=1.12),
            barmode="group",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown('<div class="section-header">Cost per Customer by Region</div>',
                    unsafe_allow_html=True)
        reg_cpc = (dff.groupby("region")["cost_per_customer"]
                   .mean().sort_values().reset_index())
        fig_cpc = go.Figure(go.Bar(
            x=reg_cpc["cost_per_customer"], y=reg_cpc["region"],
            orientation="h", marker_color=NAVY,
            text=[f"${v:.0f}" for v in reg_cpc["cost_per_customer"]],
            textposition="outside",
        ))
        fig_cpc.update_layout(
            xaxis_title="Avg Cost per Customer ($)",
            height=200, margin=dict(l=10, r=60, t=10, b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_cpc, use_container_width=True)

    # ── TOP PROJECTS TABLE ────────────────────────────────────
    st.markdown('<div class="section-header">Top 20 Projects by Benefit-Cost Ratio</div>',
                unsafe_allow_html=True)
    top20 = (dff.nlargest(20, "bcr")
             [["project_id","project_name","category","region",
               "cost_usd","risk_reduction_usd","bcr","npv_usd",
               "duration_months","regulatory_required","customers_protected"]]
             .copy())
    top20["cost_usd"]            = top20["cost_usd"].map("${:,.0f}".format)
    top20["risk_reduction_usd"]  = top20["risk_reduction_usd"].map("${:,.0f}".format)
    top20["npv_usd"]             = top20["npv_usd"].map("${:,.0f}".format)
    top20["customers_protected"] = top20["customers_protected"].map("{:,}".format)
    top20["regulatory_required"] = top20["regulatory_required"].map({True:"✅ Yes", False:"No"})
    top20 = top20.rename(columns={
        "project_id":          "ID",
        "project_name":        "Project",
        "category":            "Category",
        "region":              "Region",
        "cost_usd":            "Cost",
        "risk_reduction_usd":  "Risk Reduction",
        "bcr":                 "BCR",
        "npv_usd":             "NPV",
        "duration_months":     "Duration (Mo)",
        "regulatory_required": "Reg. Required",
        "customers_protected": "Customers",
    })
    st.dataframe(top20, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — SCENARIO BUILDER
# ══════════════════════════════════════════════════════════════
with tab2:

    st.markdown(
        '<div class="sec-lbl">Scenario builder</div>'
        '<div class="sec-sub">Compare three investment scenarios using the budget and objective '
        'set in the sidebar: <strong>Do Nothing</strong>, <strong>Regulatory Minimum</strong>, '
        'and <strong>Optimized Portfolio</strong>.</div>',
        unsafe_allow_html=True,
    )

    col_b, col_p = st.columns([1, 2])
    with col_b:
        st.markdown(
            f'<div class="formula-box">'
            f'<strong>Active settings</strong><br>'
            f'Capital budget: ${scenario_budget/1e6:.0f}M<br>'
            f'Objective: {prioritize_by}<br>'
            f'Include regulatory required: {"Yes" if include_reg else "No"}<br>'
            f'Projects in optimized portfolio: {len(opt_ids)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── COMPARISON CHARTS ─────────────────────────────────────
    with col_p:
        scen_names  = [s["name"] for s in scenarios]
        scen_colors = [MUTED, AMBER, NAVY]

        fig_comp = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Investment Cost ($M)", "Risk Reduction ($M)", "Avg BCR"),
        )
        for i, (key, row_i) in enumerate([("cost", 1), ("risk_red", 2), ("bcr", 3)]):
            vals = [s[key] for s in scenarios]
            if key != "bcr":
                vals = [v / 1e6 for v in vals]
            fig_comp.add_trace(
                go.Bar(x=scen_names, y=vals, marker_color=scen_colors,
                       showlegend=False,
                       text=[f"{v:.1f}" for v in vals], textposition="outside"),
                row=1, col=row_i,
            )
        fig_comp.update_layout(
            height=280, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="white", plot_bgcolor="#FAFBFC",
            font=dict(family="Source Sans 3, sans-serif", color=BODY),
        )
        for i in range(1, 4):
            fig_comp.update_xaxes(tickangle=-20, row=1, col=i)
            fig_comp.update_yaxes(gridcolor="#eee", row=1, col=i)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # ── SCENARIO DETAIL COMPARISON ────────────────────────────
    st.markdown('<div class="section-header">Scenario Comparison Summary</div>',
                unsafe_allow_html=True)

    comp_rows = _scenario_summary_rows(scenarios, dff)
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    reg_pct = (
        (scen_opt["risk_red"] / scen_reg["risk_red"] - 1) * 100
        if scen_reg["risk_red"] > 0 else 0
    )
    cost_pct = (
        (scen_opt["cost"] / scen_reg["cost"] - 1) * 100
        if scen_reg["cost"] > 0 else 0
    )
    st.markdown(
        f'<div class="insight-box">💡 <strong>Key Insight:</strong> '
        f'The optimized portfolio funds <strong>{len(scen_opt["ids"])} projects</strong> '
        f'for <strong>${scen_opt["cost"]/1e6:.1f}M</strong>, delivering '
        f'<strong>${scen_opt["risk_red"]/1e6:.1f}M</strong> in risk reduction '
        f'(BCR: <strong>{scen_opt["bcr"]:.2f}x</strong>). '
        f'This is <strong>{reg_pct:.0f}% more risk reduction</strong> than the regulatory minimum at '
        f'<strong>{cost_pct:.0f}% higher cost</strong>.'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── SELECTED PROJECTS LIST ────────────────────────────────
    st.markdown('<div class="section-header">Optimized Portfolio — Selected Projects</div>',
                unsafe_allow_html=True)
    if opt_ids:
        opt_display = (dff[dff["project_id"].isin(opt_ids)]
                       .sort_values("bcr", ascending=False)
                       [["project_id","project_name","category","region",
                         "cost_usd","bcr","risk_reduction_usd",
                         "duration_months","regulatory_required"]]
                       .copy())
        opt_display["cost_usd"]           = opt_display["cost_usd"].map("${:,.0f}".format)
        opt_display["risk_reduction_usd"] = opt_display["risk_reduction_usd"].map("${:,.0f}".format)
        opt_display["regulatory_required"]= opt_display["regulatory_required"].map(
            {True: "✅ Required", False: "Optional"})
        opt_display = opt_display.rename(columns={
            "project_id":          "ID",
            "project_name":        "Project",
            "category":            "Category",
            "region":              "Region",
            "cost_usd":            "Cost",
            "bcr":                 "BCR",
            "risk_reduction_usd":  "Risk Reduction",
            "duration_months":     "Duration (Mo)",
            "regulatory_required": "Status",
        })
        st.dataframe(opt_display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — BENEFIT-COST ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab3:

    st.markdown(
        "Detailed benefit-cost analysis aligned to the "
        "**CPUC Risk-Based Decision-Making (RBDM)** framework."
    )

    st.markdown(
        '<div class="formula-box">'
        'Expected Value (EV) = P(Failure) × Consequence Cost ($)<br>'
        'Benefit-Cost Ratio (BCR) = Total Benefits ($) ÷ Project Cost ($)<br>'
        'Net Present Value (NPV) = Σ [Annual Benefit / (1 + r)ᵗ] − Cost<br>'
        'Risk Reduction per Dollar = Risk Reduction ($) ÷ Project Cost ($)'
        '</div>',
        unsafe_allow_html=True
    )

    col_bcr, col_npv = st.columns(2)

    with col_bcr:
        st.markdown('<div class="section-header">BCR Distribution by Category</div>',
                    unsafe_allow_html=True)
        fig_box = go.Figure()
        for cat in sorted(dff["category"].unique()):
            vals = dff[dff["category"] == cat]["bcr"].values
            fig_box.add_trace(go.Box(
                y=vals, name=cat,
                marker_color=CAT_COLORS.get(cat, "#999"),
                boxpoints="outliers", line_width=1.5,
            ))
        fig_box.add_hline(y=1.0, line_dash="dash", line_color="#d32f2f",
                          annotation_text="BCR = 1.0 (break-even)")
        fig_box.update_layout(
            yaxis_title="Benefit-Cost Ratio",
            height=350, showlegend=False,
            margin=dict(l=10, r=10, t=10, b=60),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(tickangle=-30),
            yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_npv:
        st.markdown('<div class="section-header">NPV vs. Cost — Investment Efficiency</div>',
                    unsafe_allow_html=True)
        fig_npv = px.scatter(
            dff, x="cost_usd", y="npv_usd",
            color="category", color_discrete_map=CAT_COLORS,
            hover_name="project_name",
            hover_data={"project_id": True, "bcr": ":.2f",
                        "cost_usd": ":,.0f", "npv_usd": ":,.0f"},
            height=350,
            labels={"cost_usd": "Project Cost ($)", "npv_usd": "NPV ($)",
                    "category": "Category"},
        )
        fig_npv.add_hline(y=0, line_dash="dash", line_color="#d32f2f",
                          annotation_text="NPV = 0")
        fig_npv.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
            showlegend=False,
        )
        st.plotly_chart(fig_npv, use_container_width=True)

    # ── EFFICIENT FRONTIER ────────────────────────────────────
    st.markdown('<div class="section-header">Efficient Frontier — Risk Reduction vs. Investment Cost</div>',
                unsafe_allow_html=True)
    st.caption("Each point represents the optimized portfolio at a given budget level.")

    budgets    = np.arange(5e6, 201e6, 5e6)
    frontier_x, frontier_y, frontier_n = [], [], []
    for b in budgets:
        ids  = greedy_optimize(dff, b, "risk_per_dollar")
        sel  = dff[dff["project_id"].isin(ids)]
        frontier_x.append(sel["cost_usd"].sum() / 1e6)
        frontier_y.append(sel["risk_reduction_usd"].sum() / 1e6)
        frontier_n.append(len(ids))

    fig_front = go.Figure()
    fig_front.add_trace(go.Scatter(
        x=frontier_x, y=frontier_y,
        mode="lines+markers",
        line=dict(color=NAVY, width=2.5),
        marker=dict(size=6, color=frontier_n,
                    colorscale="Blues", showscale=True,
                    colorbar=dict(title="# Projects")),
        text=[f"Budget: ${b:.0f}M<br>Projects: {n}" for b, n in zip(frontier_x, frontier_n)],
        hovertemplate="%{text}<br>Cost: $%{x:.1f}M<br>Risk Red: $%{y:.1f}M<extra></extra>",
        fill="tozeroy", fillcolor="rgba(31,56,100,0.07)",
        name="Efficient Frontier",
    ))
    # Mark current budget
    cur_cost = opt_sel["cost_usd"].sum() / 1e6
    cur_rr   = opt_sel["risk_reduction_usd"].sum() / 1e6
    fig_front.add_trace(go.Scatter(
        x=[cur_cost], y=[cur_rr], mode="markers",
        marker=dict(size=14, color=RED, symbol="star"),
        name=f"Selected Budget (${scenario_budget/1e6:.0f}M)",
    ))
    fig_front.update_layout(
        xaxis_title="Total Investment ($M)",
        yaxis_title="Total Risk Reduction ($M)",
        height=320,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_front, use_container_width=True)

    # ── BCR TABLE ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Full Benefit-Cost Register</div>',
                unsafe_allow_html=True)
    bca_df = dff.sort_values("bcr", ascending=False)[
        ["project_id","project_name","category","cost_usd","total_benefit_usd",
         "bcr","npv_usd","risk_reduction_usd","cobenefit_usd",
         "pf_reduction","co2_reduction_tonnes"]
    ].copy()
    for col in ["cost_usd","total_benefit_usd","risk_reduction_usd",
                "cobenefit_usd","npv_usd"]:
        bca_df[col] = bca_df[col].map("${:,.0f}".format)
    bca_df["pf_reduction"]      = bca_df["pf_reduction"].map("{:.1%}".format)
    bca_df["co2_reduction_tonnes"] = bca_df["co2_reduction_tonnes"].map("{:,}".format)
    bca_df = bca_df.rename(columns={
        "project_id":           "ID",
        "project_name":         "Project",
        "category":             "Category",
        "cost_usd":             "Cost",
        "total_benefit_usd":    "Total Benefit",
        "bcr":                  "BCR",
        "npv_usd":              "NPV",
        "risk_reduction_usd":   "Risk Reduction",
        "cobenefit_usd":        "Co-Benefits",
        "pf_reduction":         "P(F) Reduction",
        "co2_reduction_tonnes": "CO2e (tonnes)",
    })
    st.dataframe(bca_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — SENSITIVITY & UNCERTAINTY
# ══════════════════════════════════════════════════════════════
with tab4:

    st.markdown(
        "Monte Carlo simulation of portfolio outcomes under uncertainty. "
        "Each simulation applies random noise to cost, risk reduction, and BCR "
        "estimates to model real-world variability in project performance."
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        mc_projects = st.multiselect(
            "Select projects for Monte Carlo analysis",
            options=dff.sort_values("bcr", ascending=False)["project_id"].tolist(),
            default=dff.nlargest(10, "bcr")["project_id"].tolist(),
            max_selections=20,
        )
    with col_sel2:
        n_sims = st.selectbox("Simulations", [500, 1000, 2000, 5000], index=1)

    if mc_projects:
        mc_results = run_monte_carlo(dff, mc_projects, n_sims=n_sims)
        mc_sel     = dff[dff["project_id"].isin(mc_projects)]

        base_bcr  = mc_sel["bcr"].mean()
        base_risk = mc_sel["risk_reduction_usd"].sum()
        p10_bcr   = mc_results["sim_bcr"].quantile(0.10)
        p90_bcr   = mc_results["sim_bcr"].quantile(0.90)
        p10_risk  = mc_results["sim_risk_reduction"].quantile(0.10)
        p90_risk  = mc_results["sim_risk_reduction"].quantile(0.90)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="kpi-card">
                <p class="kpi-value">{base_bcr:.2f}x</p>
                <p class="kpi-label">Base BCR (Point Estimate)</p></div>""",
                unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi-card kpi-amber">
                <p class="kpi-value">{p10_bcr:.2f}–{p90_bcr:.2f}x</p>
                <p class="kpi-label">BCR P10–P90 Range</p></div>""",
                unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card kpi-green">
                <p class="kpi-value">${base_risk/1e6:.1f}M</p>
                <p class="kpi-label">Base Risk Reduction</p></div>""",
                unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card kpi-amber">
                <p class="kpi-value">${p10_risk/1e6:.1f}–${p90_risk/1e6:.1f}M</p>
                <p class="kpi-label">Risk Red. P10–P90 Range</p></div>""",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown('<div class="section-header">BCR Distribution (Monte Carlo)</div>',
                        unsafe_allow_html=True)
            fig_mc_bcr = go.Figure()
            fig_mc_bcr.add_trace(go.Histogram(
                x=mc_results["sim_bcr"], nbinsx=50,
                marker_color=NAVY, opacity=0.75, name="Simulated BCR",
            ))
            fig_mc_bcr.add_vline(x=base_bcr, line_dash="solid",
                                 line_color="#d32f2f", line_width=2,
                                 annotation_text=f"Base: {base_bcr:.2f}x",
                                 annotation_position="top right")
            fig_mc_bcr.add_vline(x=1.0, line_dash="dash",
                                 line_color="#f57c00", line_width=1.5,
                                 annotation_text="BCR=1 (break-even)")
            prob_above_1 = (mc_results["sim_bcr"] >= 1.0).mean()
            fig_mc_bcr.update_layout(
                xaxis_title="Simulated BCR",
                yaxis_title="Frequency",
                height=300,
                margin=dict(l=10, r=10, t=10, b=40),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
                annotations=[dict(
                    x=0.98, y=0.97, xref="paper", yref="paper",
                    text=f"P(BCR ≥ 1.0) = {prob_above_1:.1%}",
                    showarrow=False, bgcolor="white",
                    bordercolor="#1F3864", borderwidth=1,
                    font=dict(size=12),
                )],
            )
            st.plotly_chart(fig_mc_bcr, use_container_width=True)

        with col_h2:
            st.markdown('<div class="section-header">Risk Reduction Distribution</div>',
                        unsafe_allow_html=True)
            fig_mc_rr = go.Figure()
            fig_mc_rr.add_trace(go.Histogram(
                x=mc_results["sim_risk_reduction"] / 1e6,
                nbinsx=50, marker_color="#388e3c", opacity=0.75,
                name="Simulated Risk Reduction ($M)",
            ))
            fig_mc_rr.add_vline(x=base_risk / 1e6, line_dash="solid",
                                line_color="#d32f2f", line_width=2,
                                annotation_text=f"Base: ${base_risk/1e6:.1f}M",
                                annotation_position="top left")
            fig_mc_rr.update_layout(
                xaxis_title="Risk Reduction ($M)",
                yaxis_title="Frequency",
                height=300,
                margin=dict(l=10, r=10, t=10, b=40),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
            )
            st.plotly_chart(fig_mc_rr, use_container_width=True)

        # ── SENSITIVITY TABLE ─────────────────────────────────
        st.markdown('<div class="section-header">Sensitivity Analysis — Discount Rate Impact on NPV</div>',
                    unsafe_allow_html=True)
        rates       = [0.03, 0.05, 0.07, 0.09, 0.11]
        sens_rows   = []
        for r_val in rates:
            sel_proj = dff[dff["project_id"].isin(mc_projects)]
            npv_total = 0
            for _, row in sel_proj.iterrows():
                years = row["duration_months"] / 12
                ann_b = row["total_benefit_usd"] / years
                npv_p = sum(ann_b / (1 + r_val) ** t
                            for t in range(1, int(years) + 2)) - row["cost_usd"]
                npv_total += npv_p
            sens_rows.append({
                "Discount Rate":     f"{r_val:.0%}",
                "Portfolio NPV":     f"${npv_total/1e6:.1f}M",
                "NPV Positive?":     "✅ Yes" if npv_total > 0 else "❌ No",
                "% Change from 7%":  "",
            })
        base_npv = float(sens_rows[2]["Portfolio NPV"].replace("$","").replace("M",""))
        for row in sens_rows:
            val = float(row["Portfolio NPV"].replace("$","").replace("M",""))
            row["% Change from 7%"] = f"{(val/base_npv - 1)*100:+.1f}%"
        st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

        st.markdown(
            f'<div class="insight-box">'
            f'💡 <strong>Uncertainty Summary:</strong> Across {n_sims:,} simulations, '
            f'the selected portfolio achieves a positive BCR (≥ 1.0) in '
            f'<strong>{prob_above_1:.1%}</strong> of scenarios. '
            f'The P10–P90 BCR range is <strong>{p10_bcr:.2f}x – {p90_bcr:.2f}x</strong>, '
            f'indicating <strong>'
            f'{"low" if (p90_bcr - p10_bcr) < 1.5 else "moderate"} estimation uncertainty'
            f'</strong>. Risk reduction estimates range from '
            f'<strong>${p10_risk/1e6:.1f}M to ${p90_risk/1e6:.1f}M</strong> '
            f'under conservative and optimistic assumptions respectively.'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Select at least one project above to run the simulation.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="byline">'
    f'<strong style="color:{GOLD};">Grid Investment Prioritization Engine</strong> · '
    f'Benefit-cost optimization for utility infrastructure capital decisions.<br>'
    f'Built by <a href="https://poverty360.org">Sherriff Abdul-Hamid</a> · '
    f'Download the executive brief from the report card above the tabs.<br>'
    f'Methodology: CPUC Risk-Based Decision-Making (RBDM) · Simulated SoCal utility portfolio.'
    f'</div>',
    unsafe_allow_html=True,
)
