# Portfolio Stress Testing

This project evaluates how a diversified investment portfolio performs under extreme market conditions. It models the impact of real crisis periods (2008 Financial Crisis, 2020 COVID-19 crash) and simulated macro shocks on returns, volatility, and downside risk. All market data is automatically pulled using Yahoo Finance, so no external dataset upload is required.

---

## Key Features
- Automated data pull from Yahoo Finance (no CSVs needed)
- Historical stress testing on:
  - 2008 Financial Crisis window
  - 2020 COVID-19 crash window
- Monte Carlo shock simulations with:
  - ~35% market drop scenarios
  - Volatility spikes of ~25%
- Risk metrics:
  - Value at Risk (VaR) at 95% & 99%
  - Expected Shortfall (ES) at 95% & 99%
- Portfolio performance analytics:
  - Maximum drawdown
  - Annualized volatility
- Covariance modeling:
  - Sample covariance vs. Ledoit–Wolf shrinkage
  - Shows error reduction and improved prediction accuracy

---

## Assets Used
| Asset | Classification |
|-------|----------------|
| SPY | Equities (S&P 500) |
| AGG | Bonds (US Aggregate) |
| GLD | Gold (Commodity Hedge) |
| BTC-USD | Crypto (Alternative Asset) |

These create a diversified portfolio for stress analysis.

---

## How to Run
```bash
pip install -r requirements.txt
python stress_test.py

