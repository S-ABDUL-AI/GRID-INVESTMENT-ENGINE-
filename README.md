# 📊 Grid Investment Prioritization Engine

**Benefit-cost optimization and capital portfolio planning for utility infrastructure investment**

[![Open App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://grid-investment-prioritization-engine-hsyvnqi9iesad5met5pw7v.streamlit.app/)

---

## Business Problem

Utilities face a fundamental capital allocation problem: unfunded infrastructure risk
far exceeds available maintenance and investment budgets. The CPUC's Risk-Based
Decision-Making (RBDM) framework requires utilities to justify investment decisions
with quantified benefit-cost analyses — but most tools handle risk scoring and
capital optimization separately.

This engine answers three questions simultaneously:

1. Which investments deliver the highest risk reduction per dollar spent?
2. What is the optimal portfolio given a constrained capital budget?
3. How robust are these decisions under uncertainty in cost and benefit estimates?

---

## Financial Impact

On a simulated portfolio of 100 grid investment projects totalling $580M+ in
required capital, the optimization engine identifies a $50M portfolio that delivers
**$180M+ in risk reduction** — a BCR of **3.5x** — compared to $42M in risk
reduction from the regulatory-minimum portfolio at the same budget.

---

## Methodology

### Expected Value Framework
```
Expected Risk ($)        = P(Failure) × Consequence Cost ($)
Benefit-Cost Ratio (BCR) = Total Benefits ($) ÷ Project Cost ($)
Net Present Value (NPV)  = Σ [Annual Benefit ÷ (1 + r)ᵗ] − Cost
Risk per Dollar          = Risk Reduction ($) ÷ Project Cost ($)
```

### Portfolio Optimization
Greedy knapsack algorithm maximizing risk reduction per dollar within budget constraint,
with option to force-include regulatory-required projects. Three optimization objectives:
- Maximize risk reduction
- Maximize BCR
- Maximize customers protected

### Monte Carlo Uncertainty Analysis
2,000-simulation Monte Carlo with ±15% BCR noise and ±18% risk reduction noise,
producing P10–P90 confidence intervals and probability of positive BCR.

---

## Features

### Tab 1 — Portfolio Overview
- Investment priority matrix: BCR vs. risk reduction scatter with bubble sizing by cost
- Portfolio breakdown by project category with dual-axis BCR comparison
- Cost-per-customer efficiency by region
- Top 20 projects ranked by benefit-cost ratio

### Tab 2 — Scenario Builder
- Three-scenario comparison: Do Nothing, Regulatory Minimum, Optimized Portfolio
- Capital budget slider ($5M–$200M)
- Three optimization objectives selectable
- Insight box with automated narrative comparison
- Full selected project list for optimized portfolio

### Tab 3 — Benefit-Cost Analysis
- BCR distribution by category with box plots
- NPV vs. cost scatter identifying highest-efficiency investments
- Efficient frontier: risk reduction vs. investment cost across all budget levels
- Full benefit-cost register with NPV, co-benefits, and CO2 reduction

### Tab 4 — Sensitivity & Uncertainty
- Monte Carlo simulation (500–5,000 runs)
- BCR and risk reduction histograms with confidence intervals
- Discount rate sensitivity table (3%–11%)
- Automated uncertainty narrative

---

## Applications in Utility Infrastructure

| Use Case | How This Tool Addresses It |
|----------|---------------------------|
| Capital budget allocation | Greedy knapsack optimizer maximises risk reduction per dollar spent |
| Regulatory benefit-cost filings | Full BCR and NPV register aligned to CPUC RBDM framework |
| Portfolio scenario comparison | Three-scenario builder quantifies tradeoffs between investment strategies |
| Investment uncertainty management | Monte Carlo simulation produces P10-P90 confidence intervals on returns |
| Efficient frontier analysis | Identifies optimal portfolios at every budget level |
| Stakeholder communication | Automated narrative insight box explains scenario tradeoffs in plain language |

---

## Project Categories Modelled

| Category | Avg BCR | Description |
|----------|---------|-------------|
| Vegetation Management | 5.1x | Fuel load reduction and clearance programs |
| Predictive Maintenance | 4.2x | Sensor deployment and condition monitoring |
| Wildfire Mitigation | 3.8x | Covered conductor, PSPS automation, hardening |
| Grid Hardening | 2.9x | Transmission and distribution reinforcement |
| Substation Upgrades | 2.4x | Automation and protection system upgrades |
| Climate Resilience | 2.2x | Extreme heat and drought adaptation |
| Undergrounding | 1.8x | Overhead-to-underground conversion |

---

## Run Locally

```bash
git clone https://github.com/S-ABDUL-AI/grid-investment-engine.git
cd grid-investment-engine
pip install -r requirements.txt
streamlit run app.py
```

---

## Methodological Note

This project applies the same expected-value optimization framework used in
global health economics (GiveWell, USAID cost-effectiveness analyses) to utility
infrastructure investment decisions. The mathematical structure is identical:
identify the intervention that produces the greatest risk reduction per dollar,
subject to a budget constraint, accounting for uncertainty in both costs and outcomes.

This connection — between development economics methodology and utility asset
management — reflects the author's background in both fields.

---

## Related Energy Portfolio Tools

| Tool | Description | Live App |
|------|-------------|----------|
| California Grid Asset Risk Dashboard | Wildfire risk prediction and asset failure modeling for electric utility infrastructure | [Open App](https://california-grid-asset-risk-wildfire-vulnerability-dashboard-hm.streamlit.app/) |
| California Grid Demand Forecast | Time series load forecasting and EV/solar scenario planning for electric utility operations | [Open App](https://california-grid-demand-forecast-35hscpx2oxi3y7glsgkphj.streamlit.app/) |
| Oil Price Shock Transmission Dashboard | Interactive econometric tool exploring oil price shock transmission into Ghana public debt dynamics | [Open App](https://oil-price-shock-transmission-9xazdnfk5tnuaffap34ysp.streamlit.app/) |

---

## Author

**Sherriff Abdul-Hamid**
Data Scientist · Energy Economist · Public Infrastructure Analytics
[poverty360.org](https://poverty360.org) | [LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) | [GitHub](https://github.com/S-ABDUL-AI)
