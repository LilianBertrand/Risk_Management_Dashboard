import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_dashboard.metrics import historical_var, expected_shortfall, hhi_concentration, max_drawdown


def test_historical_var_is_negative_for_loss_tail():
    returns = pd.Series([-0.10, -0.05, 0.01, 0.02, 0.03])
    var_95 = historical_var(returns, 0.95)
    assert var_95 < 0


def test_expected_shortfall_less_or_equal_var():
    returns = pd.Series([-0.10, -0.05, 0.01, 0.02, 0.03])
    var = historical_var(returns, 0.95)
    es = expected_shortfall(returns, var)
    assert es <= var


def test_hhi_equal_weight_portfolio():
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(hhi_concentration(weights), 0.25)


def test_max_drawdown_negative_or_zero():
    returns = pd.Series([0.02, -0.01, -0.05, 0.03])
    max_dd, _ = max_drawdown(returns)
    assert max_dd <= 0
