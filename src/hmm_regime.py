import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_hmm(observations: np.ndarray, n_states: int = 3,
            n_iter: int = 2500, random_state: int = 42) -> GaussianHMM:
    """
    Fit Gaussian HMM to observations.

    Args:
        observations: 2D array (n_samples, n_features)
        n_states: Number of hidden states
        n_iter: Maximum iterations for EM
        random_state: For reproducibility

    Returns:
        Fitted GaussianHMM model
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=n_iter,
        random_state=random_state
    )
    model.fit(observations)
    return model

def analyze_states(model: GaussianHMM, returns: pd.Series,
                   states: np.ndarray) -> pd.DataFrame:
    """Analyze characteristics of each state."""
    results = []
    for state in range(model.n_components):
        mask = states == state
        state_returns = returns[mask]
        results.append({
            'state': state,
            'mean_return': state_returns.mean() * 252,  # Annualized
            'volatility': state_returns.std() * np.sqrt(252),
            'sharpe': (state_returns.mean() / state_returns.std()) * np.sqrt(252),
            'pct_time': mask.sum() / len(mask) * 100,
            'n_obs': mask.sum()
        })
    return pd.DataFrame(results)

def label_states(analysis: pd.DataFrame) -> dict:
    """Assign meaningful labels based on characteristics."""
    # Sort by volatility to identify regimes
    # Sort by mean returns to identify regimes
    sorted_df = analysis.sort_values('volatility')
    labels = {}

    if len(sorted_df) == 3:
        labels[sorted_df.iloc[0]['state']] = 'Calm/Bull'
        labels[sorted_df.iloc[1]['state']] = 'Transition'
        labels[sorted_df.iloc[2]['state']] = 'Crisis/Bear'
    elif len(sorted_df) == 2:
        labels[sorted_df.iloc[0]['state']] = 'Low Vol'
        labels[sorted_df.iloc[1]['state']] = 'High Vol'

    return labels
