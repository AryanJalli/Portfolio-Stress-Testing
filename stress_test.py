import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.covariance import LedoitWolf


# -----------------------------
# Configuration
# -----------------------------

TICKERS = ["SPY", "AGG", "GLD", "BTC-USD"]
WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1])  # Example diversified weights
START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

PLOTS_DIR = "plots"


# -----------------------------
# Data utilities
# -----------------------------

def download_price_data(tickers, start, end):
    """
    Download adjusted close prices for given tickers using yfinance.
    """
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    # Ensure DataFrame (when single ticker)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")


def compute_log_returns(price_df):
    """
    Compute daily log returns from price DataFrame.
    """
    return np.log(price_df / price_df.shift(1)).dropna()


def compute_portfolio_returns(returns_df, weights):
    """
    Compute portfolio returns as weighted sum of asset returns.
    """
    weights = np.array(weights)
    return returns_df.dot(weights)


def compute_drawdown(cumulative_returns):
    """
    Compute drawdown series and maximum drawdown.
    """
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1.0
    max_dd = drawdown.min()
    return drawdown, max_dd


def annualized_volatility(returns, trading_days=252):
    """
    Compute annualized volatility from daily returns.
    """
    return returns.std() * np.sqrt(trading_days)


# -----------------------------
# Risk measures: VaR and ES
# -----------------------------

def historical_var(returns, alpha=0.95):
    """
    Historical Value at Risk (VaR) at confidence level alpha.
    Returns a positive number representing loss.
    """
    # For losses: negative returns tail
    q = np.quantile(returns, 1 - alpha)
    return -q


def historical_es(returns, alpha=0.95):
    """
    Historical Expected Shortfall (ES) at confidence level alpha.
    Returns a positive number representing loss.
    """
    q = np.quantile(returns, 1 - alpha)
    tail_losses = returns[returns <= q]
    return -tail_losses.mean()


# -----------------------------
# Stress testing: historical crises
# -----------------------------

def stress_window(returns_df, start_date, end_date, weights):
    """
    Restrict returns to a window and compute portfolio metrics.
    """
    window_returns = returns_df.loc[start_date:end_date].dropna()
    port_returns = compute_portfolio_returns(window_returns, weights)
    cum = (1 + port_returns).cumprod()
    dd_series, max_dd = compute_drawdown(cum)
    vol = annualized_volatility(port_returns)

    return {
        "start": start_date,
        "end": end_date,
        "max_drawdown": float(max_dd),
        "annualized_vol": float(vol),
        "cum_returns": cum,
        "drawdown_series": dd_series,
        "portfolio_returns": port_returns,
    }


def plot_cumulative_and_drawdown(result_dict, label_prefix):
    """
    Save plots for cumulative returns and drawdown for a given scenario.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    cum = result_dict["cum_returns"]
    dd = result_dict["drawdown_series"]

    # Cumulative returns
    plt.figure()
    cum.plot()
    plt.title(f"{label_prefix} - Cumulative Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value (Starting at 1)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{label_prefix}_cumulative.png"))
    plt.close()

    # Drawdown
    plt.figure()
    dd.plot()
    plt.title(f"{label_prefix} - Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{label_prefix}_drawdown.png"))
    plt.close()


# -----------------------------
# Scenario simulation (macro shocks)
# -----------------------------

def monte_carlo_scenario(returns_df, weights, horizon_days=60,
                         mean_shock=-0.35, vol_multiplier=1.5, n_sims=5000):
    """
    Simple Monte Carlo stress test:
    - Scale mean returns down (mean_shock, e.g., -0.35 over horizon)
    - Increase vol via vol_multiplier
    - Simulate portfolio PnL distribution over given horizon.
    """
    # Historical estimates
    mu = returns_df.mean().values  # daily
    cov = returns_df.cov().values  # daily

    # Adjust mean and covariance for shock
    # Target a total drawdown ~mean_shock over horizon -> daily drift
    drift_adj = (1 + mean_shock) ** (1 / horizon_days) - 1
    mu_shocked = np.full_like(mu, drift_adj)

    cov_shocked = cov * (vol_multiplier ** 2)

    # Simulate multivariate normal returns
    rng = np.random.default_rng(seed=42)
    sims = rng.multivariate_normal(mu_shocked, cov_shocked,
                                   size=(n_sims, horizon_days))

    # sims.shape = (n_sims, horizon_days, n_assets)
    sims = sims.reshape(n_sims, horizon_days, len(mu))
    weights = np.array(weights)

    # Portfolio path per simulation
    port_daily = sims @ weights  # (n_sims, horizon_days)
    port_cum = np.cumprod(1 + port_daily, axis=1)
    # Total horizon return
    horizon_ret = port_cum[:, -1] - 1

    # Metrics
    var_95 = historical_var(horizon_ret, alpha=0.95)
    es_95 = historical_es(horizon_ret, alpha=0.95)
    var_99 = historical_var(horizon_ret, alpha=0.99)
    es_99 = historical_es(horizon_ret, alpha=0.99)

    return {
        "horizon_days": horizon_days,
        "mean_shock": mean_shock,
        "vol_multiplier": vol_multiplier,
        "horizon_returns": horizon_ret,
        "VaR_95": var_95,
        "ES_95": es_95,
        "VaR_99": var_99,
        "ES_99": es_99,
    }


def plot_scenario_distribution(result_dict, label_prefix="macro_scenario"):
    """
    Plot the distribution of simulated horizon returns.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    horizon_ret = result_dict["horizon_returns"]

    plt.figure()
    plt.hist(horizon_ret, bins=50, density=True)
    plt.title(f"{label_prefix} - Horizon Return Distribution")
    plt.xlabel("Return over Horizon")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{label_prefix}_distribution.png"))
    plt.close()


# -----------------------------
# Covariance shrinkage vs sample covariance
# -----------------------------

def compare_covariance_estimators(returns_df, weights, train_window=252, test_window=126):
    """
    Compare sample covariance vs Ledoit-Wolf shrinkage in predicting
    portfolio volatility over a test window.

    Approach:
    - Use rolling window of 'train_window' days to estimate covariances.
    - Predict next-day portfolio volatility.
    - Compare predicted vol to realized vol (using next-day return).
    - Compute RMSE for both methods.
    """
    returns = returns_df.copy()
    weights = np.array(weights)
    n = len(returns)

    sample_preds = []
    lw_preds = []
    realized = []

    for end_idx in range(train_window, n - 1):
        start_idx = end_idx - train_window
        train_slice = returns.iloc[start_idx:end_idx]

        # Next day return
        next_ret = returns.iloc[end_idx + 1].dot(weights)

        # Realized daily volatility ~ abs(next_ret)
        realized_vol = abs(next_ret)

        # Sample covariance
        sample_cov = train_slice.cov().values
        sample_vol = np.sqrt(weights @ sample_cov @ weights.T)

        # Ledoit-Wolf covariance
        lw = LedoitWolf().fit(train_slice.values)
        lw_cov = lw.covariance_
        lw_vol = np.sqrt(weights @ lw_cov @ weights.T)

        sample_preds.append(sample_vol)
        lw_preds.append(lw_vol)
        realized.append(realized_vol)

        # Limit the backtest length
        if len(realized) >= test_window:
            break

    sample_preds = np.array(sample_preds)
    lw_preds = np.array(lw_preds)
    realized = np.array(realized)

    sample_rmse = np.sqrt(((sample_preds - realized) ** 2).mean())
    lw_rmse = np.sqrt(((lw_preds - realized) ** 2).mean())

    improvement = (sample_rmse - lw_rmse) / sample_rmse * 100

    return {
        "sample_rmse": float(sample_rmse),
        "lw_rmse": float(lw_rmse),
        "improvement_pct": float(improvement),
    }


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    print("Downloading historical data...")
    prices = download_price_data(TICKERS, START_DATE, END_DATE)
    returns = compute_log_returns(prices)

    # Full-portfolio metrics
    port_returns = compute_portfolio_returns(returns, WEIGHTS)
    cum = (1 + port_returns).cumprod()
    dd_series, max_dd = compute_drawdown(cum)
    vol = annualized_volatility(port_returns)

    print("\n=== Full Sample Portfolio Metrics ===")
    print(f"Period: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"Annualized volatility: {vol:.2%}")
    print(f"Maximum drawdown: {max_dd:.2%}")

    # Save full-sample plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure()
    cum.plot()
    plt.title("Full Sample - Cumulative Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value (Starting at 1)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "full_sample_cumulative.png"))
    plt.close()

    plt.figure()
    dd_series.plot()
    plt.title("Full Sample - Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "full_sample_drawdown.png"))
    plt.close()

    # Historical stress tests
    print("\n=== Historical Stress Tests ===")

    # 2008 Financial Crisis approximate window
    crisis_2008 = stress_window(returns,
                                start_date="2007-10-01",
                                end_date="2009-06-30",
                                weights=WEIGHTS)
    print("\n2008 Financial Crisis Window:")
    print(f"Max drawdown: {crisis_2008['max_drawdown']:.2%}")
    print(f"Annualized volatility: {crisis_2008['annualized_vol']:.2%}")
    plot_cumulative_and_drawdown(crisis_2008, "crisis_2008")

    # 2020 COVID-19 Crash approximate window
    crisis_2020 = stress_window(returns,
                                start_date="2020-02-01",
                                end_date="2020-06-30",
                                weights=WEIGHTS)
    print("\n2020 COVID-19 Crash Window:")
    print(f"Max drawdown: {crisis_2020['max_drawdown']:.2%}")
    print(f"Annualized volatility: {crisis_2020['annualized_vol']:.2%}")
    plot_cumulative_and_drawdown(crisis_2020, "crisis_2020")

    # Macro shock scenario simulation
    print("\n=== Macro Scenario Simulation ===")
    scenario = monte_carlo_scenario(
        returns_df=returns,
        weights=WEIGHTS,
        horizon_days=60,
        mean_shock=-0.35,   # target ~35% drawdown over 60 days
        vol_multiplier=1.5,  # ~volatility spike
        n_sims=5000
    )

    print(f"Horizon: {scenario['horizon_days']} days")
    print(f"Mean shock (total): {scenario['mean_shock']:.2%}")
    print(f"VaR 95: {scenario['VaR_95']:.2%}")
    print(f"ES 95: {scenario['ES_95']:.2%}")
    print(f"VaR 99: {scenario['VaR_99']:.2%}")
    print(f"ES 99: {scenario['ES_99']:.2%}")

    plot_scenario_distribution(scenario)

    # VaR/ES on daily returns (historical simulation)
    print("\n=== Daily Risk Measures (Historical VaR/ES) ===")
    for alpha in [0.95, 0.99]:
        var = historical_var(port_returns, alpha=alpha)
        es = historical_es(port_returns, alpha=alpha)
        print(f"{int(alpha * 100)}% VaR (1-day): {var:.2%}")
        print(f"{int(alpha * 100)}% ES (1-day): {es:.2%}")

    # Covariance shrinkage vs sample covariance
    print("\n=== Covariance Modelling: Sample vs Ledoit-Wolf ===")
    cov_results = compare_covariance_estimators(returns, WEIGHTS)
    print(f"Sample cov RMSE: {cov_results['sample_rmse']:.6f}")
    print(f"Ledoit-Wolf RMSE: {cov_results['lw_rmse']:.6f}")
    print(f"Improvement (shrinkage): {cov_results['improvement_pct']:.2f}%")

    print("\nDone. Plots saved in ./plots")


if __name__ == "__main__":
    main()
