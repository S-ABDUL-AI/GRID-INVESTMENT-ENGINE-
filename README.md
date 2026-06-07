# 📊 Grid Investment Prioritization Engine

**Benefit-Cost Optimization for Utility Infrastructure Capital Decisions**  
Built by [Sherriff Abdul-Hamid](https://poverty360.org) | Portfolio project for SCE Senior Data Science Manager

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

## Connection to SCE Priorities

| CPUC / SCE Requirement | This Tool |
|------------------------|-----------|
| Risk-Based Decision-Making (RBDM) | Expected value framework + BCR ranking |
| Wildfire Mitigation Plan filings | Wildfire category with regulatory flag |
| Benefit-cost analyses for investment decisions | Full BCA register with NPV, BCR |
| Capital budget optimization | Greedy knapsack with budget constraint |
| Uncertainty and sensitivity analysis | Monte Carlo + discount rate sensitivity |
| Executive stakeholder communication | Scenario comparison with narrative insight |

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
git clone https://github.com/S-ABDUL-AI/grid-investment-engine
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

## Author

**Sherriff Abdul-Hamid**  
Development Economist · Data Scientist · Public Infrastructure Analytics  
[poverty360.org](https://poverty360.org) | [LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)
