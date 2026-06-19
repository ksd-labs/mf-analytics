"""
analytics/regime_model.py
==========================
Two-state Gaussian Hidden Markov Model for return regime classification.

The model identifies two latent states from the daily return series:
  - Bull regime:  higher mean return, typically lower volatility
  - Bear regime:  lower (often negative) mean return, higher volatility

States are labelled automatically by ordering the estimated means
(lowest mean = Bear). No pre-labelling or supervised training is required.

Why HMM?
  Market regimes are empirically real — the return distribution during
  bull markets is statistically different from bear markets. HMM is the
  standard unsupervised approach for detecting these latent states without
  imposing arbitrary threshold rules. It also produces a probabilistic
  current-state estimate rather than a hard classification.

Limitations:
  - Number of states (2) is chosen a priori. 3-state models (adding a
    "high-volatility" state) can be explored but require more data.
  - HMM assumes Markov property: current state depends only on previous
    state, not the full history. This is an approximation.
  - The model is applied to the FUND's own return series, not the broad
    market. Regime labels reflect fund-level dynamics. Upgrade path:
    replace with Nifty 500 TRI when available.
  - Current-state probabilities have uncertainty — present them with
    confidence bounds, not as certainties.

Dependencies:
  hmmlearn >= 0.3.0  (pip install hmmlearn)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from utils.constants import TRADING_DAYS_PER_YEAR


def get_regime_summary(
    returns:  pd.Series,
    n_states: int = 2,
    n_iter:   int = 1000,
    seed:     int = 42,
) -> Dict:
    """
    Fit a Gaussian HMM and return regime classifications and statistics.

    Args:
        returns:  Daily simple returns (decimal). Must have DatetimeIndex.
        n_states: Number of hidden states (2 recommended for first version).
        n_iter:   Maximum EM iterations for fitting.
        seed:     Random seed for initialisation.

    Returns dict with keys:
        is_valid              — bool
        labeled_states        — pd.Series of state labels ("Bull 🐂" / "Bear 🐻")
                                indexed by date
        regime_stats          — {label: {mean_ann_return, ann_vol, sharpe, pct_time}}
        current_regime        — str, label of the most recent state
        current_posteriors    — {label: probability} for today's state
        transition_matrix     — pd.DataFrame, labelled transition probabilities
        regime_blocks         — list of {label, start, end} for timeline chart
        expected_duration     — {label: expected days in regime before switching}
    """
    empty = {"is_valid": False, "error": ""}

    clean = returns.dropna()
    if len(clean) < 252:
        empty["error"] = "Minimum 252 trading days required for regime detection."
        return empty

    try:
        from hmmlearn import hmm
    except ImportError:
        empty["error"] = "hmmlearn not installed. Run: pip install hmmlearn"
        return empty

    try:
        r_arr = clean.values.reshape(-1, 1)
        dates = clean.index

        model = hmm.GaussianHMM(
            n_components   = n_states,
            covariance_type= "full",
            n_iter         = n_iter,
            random_state   = seed,
        )
        model.fit(r_arr)

        raw_states = model.predict(r_arr)
        posteriors = model.predict_proba(r_arr)

        # Label states by ascending mean return (lowest mean = Bear)
        means       = model.means_[:, 0]
        state_order = np.argsort(means)    # [bear_idx, bull_idx] for 2 states

        if n_states == 2:
            state_labels = {state_order[0]: "Bear 🐻", state_order[1]: "Bull 🐂"}
        else:
            state_labels = {state_order[i]: f"Regime {i+1}" for i in range(n_states)}

        # Date-indexed series of state labels
        labeled_states = pd.Series(
            [state_labels[s] for s in raw_states],
            index=dates,
        )

        # ── Regime statistics ─────────────────────────────────────────────
        regime_stats = {}
        for orig_idx, label in state_labels.items():
            mask          = (raw_states == orig_idx)
            state_returns = clean.values[mask]   # decimal daily returns
            mean_daily    = float(model.means_[orig_idx, 0])
            var_daily     = float(model.covars_[orig_idx, 0, 0])
            std_daily     = np.sqrt(var_daily)
            mean_ann      = mean_daily * TRADING_DAYS_PER_YEAR
            vol_ann       = std_daily  * np.sqrt(TRADING_DAYS_PER_YEAR)
            sharpe        = (mean_ann / vol_ann) if vol_ann > 0 else 0.0

            regime_stats[label] = {
                "mean_ann_return": mean_ann,
                "ann_vol":         vol_ann,
                "sharpe":          sharpe,
                "pct_time":        float(mask.mean()),
                "n_days":          int(mask.sum()),
            }

        # ── Current regime ────────────────────────────────────────────────
        current_raw   = int(raw_states[-1])
        current_label = state_labels[current_raw]
        current_post  = {state_labels[i]: float(posteriors[-1, i]) for i in range(n_states)}

        # ── Transition matrix ─────────────────────────────────────────────
        labels_ordered = [state_labels[state_order[i]] for i in range(n_states)]
        transmat_reordered = model.transmat_[np.ix_(state_order, state_order)]
        transition_df = pd.DataFrame(
            transmat_reordered,
            index   = [f"From: {l}" for l in labels_ordered],
            columns = [f"To: {l}"   for l in labels_ordered],
        )

        # Expected duration in each state = 1 / (1 - P_stay)
        expected_duration = {}
        for i, orig_idx in enumerate(state_order):
            p_stay = model.transmat_[orig_idx, orig_idx]
            dur    = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else np.inf
            expected_duration[labels_ordered[i]] = float(dur)

        # ── Regime blocks for timeline chart ──────────────────────────────
        regime_blocks = []
        if len(raw_states) > 0:
            state_arr  = np.array([state_labels[s] for s in raw_states])
            changes    = np.where(np.diff(raw_states) != 0)[0] + 1
            block_ends = np.concatenate([changes, [len(raw_states)]])
            block_starts = np.concatenate([[0], changes])
            for s, e in zip(block_starts, block_ends):
                regime_blocks.append({
                    "label": state_arr[s],
                    "start": dates[s],
                    "end":   dates[min(e, len(dates) - 1)],
                })

        return {
            "is_valid":           True,
            "labeled_states":     labeled_states,
            "regime_stats":       regime_stats,
            "current_regime":     current_label,
            "current_posteriors": current_post,
            "transition_matrix":  transition_df,
            "regime_blocks":      regime_blocks,
            "expected_duration":  expected_duration,
            "raw_states":         raw_states,
            "posteriors":         posteriors,
            "dates":              dates,
        }

    except Exception as e:
        return {"is_valid": False, "error": str(e)}
