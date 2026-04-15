import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_vol    = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_vol

def portfolio_sharpe(weights, mean_returns, cov_matrix, rf):
    r, v = portfolio_performance(weights, mean_returns, cov_matrix)
    return (r - rf) / v

def _base_constraints():
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

def _base_bounds(n):
    return tuple((0, 1) for _ in range(n))

def compute_gmv(mean_returns, cov_matrix):
    n    = len(mean_returns)
    init = np.ones(n) / n
    result = minimize(
        lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
        init,
        method="SLSQP",
        bounds=_base_bounds(n),
        constraints=_base_constraints()
    )
    if not result.success:
        return None, "GMV optimization failed: " + result.message
    return result.x, None

def compute_tangency(mean_returns, cov_matrix, rf):
    n    = len(mean_returns)
    init = np.ones(n) / n
    result = minimize(
        lambda w: -portfolio_sharpe(w, mean_returns, cov_matrix, rf),
        init,
        method="SLSQP",
        bounds=_base_bounds(n),
        constraints=_base_constraints()
    )
    if not result.success:
        return None, "Tangency optimization failed: " + result.message
    return result.x, None

def compute_efficient_frontier(mean_returns, cov_matrix, n_points=50):
    n      = len(mean_returns)
    init   = np.ones(n) / n
    bounds = _base_bounds(n)

    gmv_weights, _ = compute_gmv(mean_returns, cov_matrix)
    if gmv_weights is None:
        return [], []

    gmv_return, _ = portfolio_performance(gmv_weights, mean_returns, cov_matrix)
    max_return     = mean_returns.max()
    target_returns = np.linspace(gmv_return, max_return, n_points)

    frontier_vols    = []
    frontier_returns = []

    for target in target_returns:
        constraints = _base_constraints() + [
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t}
        ]
        result = minimize(
            lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            frontier_vols.append(np.sqrt(result.fun))
            frontier_returns.append(target)

    return frontier_vols, frontier_returns

def compute_risk_contribution(weights, cov_matrix):
    weights      = np.array(weights)
    port_var     = np.dot(weights.T, np.dot(cov_matrix, weights))
    marginal     = np.dot(cov_matrix, weights)
    contribution = weights * marginal
    return contribution / port_var

def portfolio_metrics(weights, returns_df, rf=0.02):
    port_returns = returns_df.dot(weights)
    ann_return   = port_returns.mean() * TRADING_DAYS
    ann_vol      = port_returns.std()  * np.sqrt(TRADING_DAYS)
    sharpe       = (ann_return - rf) / ann_vol

    downside     = port_returns.copy()
    downside[downside > 0] = 0
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)
    sortino      = (ann_return - rf) / downside_vol

    wealth       = (1 + port_returns).cumprod()
    running_max  = wealth.cummax()
    drawdown     = (wealth - running_max) / running_max
    max_dd       = drawdown.min()

    return {
        "Annual Return":     ann_return,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio":      sharpe,
        "Sortino Ratio":     sortino,
        "Max Drawdown":      max_dd
    }
