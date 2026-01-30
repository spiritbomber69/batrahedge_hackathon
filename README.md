# batrahedge_hackathon

data/Equity_1min: Storage for raw high-frequency intraday CSV files.

notebooks/: Research environment for model tuning and exploratory data analysis.

src/: Core application logic:

data_loader.py: Handles complex timestamp parsing and OHLC daily aggregation.

features.py: Computes vectorized returns, rolling volatility, and technical indicators.

hmm_regime.py: Contains the HMM architecture, state labeling, and transition matrix logic.

strategy.py: Implements regime-specific trading rules for intraday execution

