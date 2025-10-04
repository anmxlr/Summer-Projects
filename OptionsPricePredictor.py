import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class BlackScholesModel:
    """
    Black-Scholes Option Pricing Model for European Options
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Initialize Black-Scholes parameters

        Parameters:
        -----------
        S : float - Current stock price
        K : float - Strike price
        T : float - Time to maturity (in years)
        r : float - Risk-free interest rate (annual)
        sigma : float - Volatility (annual)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def _calculate_d1_d2(self) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters for Black-Scholes formula"""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def call_option_price(self) -> float:
        """Calculate European call option price"""
        d1, d2 = self._calculate_d1_d2()
        call_price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call_price

    def put_option_price(self) -> float:
        """Calculate European put option price"""
        d1, d2 = self._calculate_d1_d2()
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return put_price

    def delta_call(self) -> float:
        """Calculate Delta for call option"""
        d1, _ = self._calculate_d1_d2()
        return norm.cdf(d1)

    def delta_put(self) -> float:
        """Calculate Delta for put option"""
        d1, _ = self._calculate_d1_d2()
        return norm.cdf(d1) - 1

    def gamma(self) -> float:
        """Calculate Gamma (same for call and put)"""
        d1, _ = self._calculate_d1_d2()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        """Calculate Vega (same for call and put)"""
        d1, _ = self._calculate_d1_d2()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100

    def theta_call(self) -> float:
        """Calculate Theta for call option"""
        d1, d2 = self._calculate_d1_d2()
        theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
                 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        return theta / 365  # Per day

    def theta_put(self) -> float:
        """Calculate Theta for put option"""
        d1, d2 = self._calculate_d1_d2()
        theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
                 + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        return theta / 365  # Per day

    def rho_call(self) -> float:
        """Calculate Rho for call option"""
        _, d2 = self._calculate_d1_d2()
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100

    def rho_put(self) -> float:
        """Calculate Rho for put option"""
        _, d2 = self._calculate_d1_d2()
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100

    def all_greeks_call(self) -> Dict[str, float]:
        """Return all Greeks for call option"""
        return {
            'Delta': self.delta_call(),
            'Gamma': self.gamma(),
            'Vega': self.vega(),
            'Theta': self.theta_call(),
            'Rho': self.rho_call()
        }

    def all_greeks_put(self) -> Dict[str, float]:
        """Return all Greeks for put option"""
        return {
            'Delta': self.delta_put(),
            'Gamma': self.gamma(),
            'Vega': self.vega(),
            'Theta': self.theta_put(),
            'Rho': self.rho_put()
        }


class BinomialModel:
    """
    Binomial Option Pricing Model for American and European Options
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, N: int = 100):
        """
        Initialize Binomial Model parameters

        Parameters:
        -----------
        S : float - Current stock price
        K : float - Strike price
        T : float - Time to maturity (in years)
        r : float - Risk-free interest rate (annual)
        sigma : float - Volatility (annual)
        N : int - Number of time steps
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N

        # Calculate binomial parameters
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor
        self.q = (np.exp(r * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability
        self.discount = np.exp(-r * self.dt)

    def _build_stock_tree(self) -> np.ndarray:
        """Build the stock price tree"""
        stock_tree = np.zeros((self.N + 1, self.N + 1))

        for i in range(self.N + 1):
            for j in range(i + 1):
                stock_tree[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)

        return stock_tree

    def european_call_price(self) -> float:
        """Calculate European call option price"""
        stock_tree = self._build_stock_tree()
        option_tree = np.zeros((self.N + 1, self.N + 1))

        # Calculate option values at maturity
        option_tree[:, self.N] = np.maximum(stock_tree[:, self.N] - self.K, 0)

        # Backward induction
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = self.discount * (
                        self.q * option_tree[j, i + 1] + (1 - self.q) * option_tree[j + 1, i + 1]
                )

        return option_tree[0, 0]

    def european_put_price(self) -> float:
        """Calculate European put option price"""
        stock_tree = self._build_stock_tree()
        option_tree = np.zeros((self.N + 1, self.N + 1))

        # Calculate option values at maturity
        option_tree[:, self.N] = np.maximum(self.K - stock_tree[:, self.N], 0)

        # Backward induction
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = self.discount * (
                        self.q * option_tree[j, i + 1] + (1 - self.q) * option_tree[j + 1, i + 1]
                )

        return option_tree[0, 0]

    def american_call_price(self) -> float:
        """Calculate American call option price"""
        stock_tree = self._build_stock_tree()
        option_tree = np.zeros((self.N + 1, self.N + 1))

        # Calculate option values at maturity
        option_tree[:, self.N] = np.maximum(stock_tree[:, self.N] - self.K, 0)

        # Backward induction with early exercise
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = self.discount * (
                        self.q * option_tree[j, i + 1] + (1 - self.q) * option_tree[j + 1, i + 1]
                )
                # Early exercise condition
                option_tree[j, i] = max(option_tree[j, i], stock_tree[j, i] - self.K)

        return option_tree[0, 0]

    def american_put_price(self) -> float:
        """Calculate American put option price"""
        stock_tree = self._build_stock_tree()
        option_tree = np.zeros((self.N + 1, self.N + 1))

        # Calculate option values at maturity
        option_tree[:, self.N] = np.maximum(self.K - stock_tree[:, self.N], 0)

        # Backward induction with early exercise
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = self.discount * (
                        self.q * option_tree[j, i + 1] + (1 - self.q) * option_tree[j + 1, i + 1]
                )
                # Early exercise condition
                option_tree[j, i] = max(option_tree[j, i], self.K - stock_tree[j, i])

        return option_tree[0, 0]

    def delta_call(self) -> float:
        """Calculate Delta for call option using binomial model"""
        stock_tree = self._build_stock_tree()

        # Price at two nodes at time step 1
        S_up = stock_tree[0, 1]
        S_down = stock_tree[1, 1]

        # Temporary binomial models for option prices
        bs_up = BinomialModel(S_up, self.K, self.T - self.dt, self.r, self.sigma, self.N - 1)
        bs_down = BinomialModel(S_down, self.K, self.T - self.dt, self.r, self.sigma, self.N - 1)

        C_up = max(S_up - self.K, 0) if self.N == 1 else bs_up.european_call_price()
        C_down = max(S_down - self.K, 0) if self.N == 1 else bs_down.european_call_price()

        delta = (C_up - C_down) / (S_up - S_down)
        return delta


class SensitivityAnalysis:
    """
    Perform sensitivity analysis on option prices
    """

    @staticmethod
    def volatility_sensitivity(S: float, K: float, T: float, r: float,
                               sigma_range: np.ndarray) -> pd.DataFrame:
        """Analyze sensitivity to volatility changes"""
        results = []

        for sigma in sigma_range:
            bs = BlackScholesModel(S, K, T, r, sigma)
            results.append({
                'Volatility': sigma,
                'Call_Price': bs.call_option_price(),
                'Put_Price': bs.put_option_price(),
                'Call_Delta': bs.delta_call(),
                'Put_Delta': bs.delta_put(),
                'Vega': bs.vega()
            })

        return pd.DataFrame(results)

    @staticmethod
    def interest_rate_sensitivity(S: float, K: float, T: float, sigma: float,
                                  r_range: np.ndarray) -> pd.DataFrame:
        """Analyze sensitivity to interest rate changes"""
        results = []

        for r in r_range:
            bs = BlackScholesModel(S, K, T, r, sigma)
            results.append({
                'Interest_Rate': r,
                'Call_Price': bs.call_option_price(),
                'Put_Price': bs.put_option_price(),
                'Call_Rho': bs.rho_call(),
                'Put_Rho': bs.rho_put()
            })

        return pd.DataFrame(results)

    @staticmethod
    def expiry_sensitivity(S: float, K: float, r: float, sigma: float,
                           T_range: np.ndarray) -> pd.DataFrame:
        """Analyze sensitivity to time to expiry changes"""
        results = []

        for T in T_range:
            if T <= 0:
                continue
            bs = BlackScholesModel(S, K, T, r, sigma)
            results.append({
                'Time_to_Expiry': T,
                'Call_Price': bs.call_option_price(),
                'Put_Price': bs.put_option_price(),
                'Call_Theta': bs.theta_call(),
                'Put_Theta': bs.theta_put()
            })

        return pd.DataFrame(results)

    @staticmethod
    def spot_price_sensitivity(K: float, T: float, r: float, sigma: float,
                               S_range: np.ndarray) -> pd.DataFrame:
        """Analyze sensitivity to spot price changes"""
        results = []

        for S in S_range:
            bs = BlackScholesModel(S, K, T, r, sigma)
            results.append({
                'Spot_Price': S,
                'Call_Price': bs.call_option_price(),
                'Put_Price': bs.put_option_price(),
                'Call_Delta': bs.delta_call(),
                'Put_Delta': bs.delta_put(),
                'Gamma': bs.gamma()
            })

        return pd.DataFrame(results)


def compare_models(S: float, K: float, T: float, r: float, sigma: float, N: int = 100):
    """
    Compare Black-Scholes and Binomial model results
    """
    # Black-Scholes Model
    bs = BlackScholesModel(S, K, T, r, sigma)

    # Binomial Model
    binom = BinomialModel(S, K, T, r, sigma, N)

    results = {
        'Model': ['Black-Scholes', 'Binomial'],
        'Call_Price': [bs.call_option_price(), binom.european_call_price()],
        'Put_Price': [bs.put_option_price(), binom.european_put_price()],
        'Call_Delta': [bs.delta_call(), binom.delta_call()],
        'Put_Delta': [bs.delta_put(), np.nan]  # Binomial put delta not implemented
    }

    return pd.DataFrame(results)


def short_maturity_delta_analysis(S: float, K: float, r: float, sigma: float,
                                  T_short_range: np.ndarray) -> pd.DataFrame:
    """
    Analyze Delta behavior for short-maturity options
    """
    results = []

    for T in T_short_range:
        if T <= 0:
            continue
        bs = BlackScholesModel(S, K, T, r, sigma)
        binom = BinomialModel(S, K, T, r, sigma, N=100)

        results.append({
            'Time_to_Expiry_Days': T * 365,
            'BS_Call_Delta': bs.delta_call(),
            'BS_Put_Delta': bs.delta_put(),
            'Binomial_Call_Delta': binom.delta_call(),
            'BS_Gamma': bs.gamma(),
            'Moneyness': S / K
        })

    return pd.DataFrame(results)


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("OPTION PRICING MODELS: BLACK-SCHOLES & BINOMIAL")
    print("=" * 80)

    # Define parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # Time to maturity (3 months)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.25  # Volatility (25%)
    N = 100  # Number of steps for binomial model

    print(f"\nInput Parameters:")
    print(f"  Stock Price (S): ${S}")
    print(f"  Strike Price (K): ${K}")
    print(f"  Time to Maturity (T): {T} years ({T * 365:.0f} days)")
    print(f"  Risk-free Rate (r): {r * 100}%")
    print(f"  Volatility (σ): {sigma * 100}%")

    # 1. Black-Scholes Model
    print("\n" + "=" * 80)
    print("1. BLACK-SCHOLES MODEL RESULTS")
    print("=" * 80)

    bs = BlackScholesModel(S, K, T, r, sigma)

    print(f"\nOption Prices:")
    print(f"  European Call: ${bs.call_option_price():.4f}")
    print(f"  European Put:  ${bs.put_option_price():.4f}")

    print(f"\nCall Option Greeks:")
    for greek, value in bs.all_greeks_call().items():
        print(f"  {greek}: {value:.6f}")

    print(f"\nPut Option Greeks:")
    for greek, value in bs.all_greeks_put().items():
        print(f"  {greek}: {value:.6f}")

    # 2. Binomial Model
    print("\n" + "=" * 80)
    print("2. BINOMIAL MODEL RESULTS")
    print("=" * 80)

    binom = BinomialModel(S, K, T, r, sigma, N)

    print(f"\nOption Prices (N={N} steps):")
    print(f"  European Call: ${binom.european_call_price():.4f}")
    print(f"  European Put:  ${binom.european_put_price():.4f}")
    print(f"  American Call: ${binom.american_call_price():.4f}")
    print(f"  American Put:  ${binom.american_put_price():.4f}")

    print(f"\nCall Option Delta:")
    print(f"  Delta: {binom.delta_call():.6f}")

    # 3. Model Comparison
    print("\n" + "=" * 80)
    print("3. MODEL COMPARISON")
    print("=" * 80)

    comparison = compare_models(S, K, T, r, sigma, N)
    print("\n" + comparison.to_string(index=False))

    # 4. Sensitivity Analysis
    print("\n" + "=" * 80)
    print("4. SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Volatility Sensitivity
    print("\n4.1 Volatility Sensitivity")
    print("-" * 80)
    sigma_range = np.linspace(0.10, 0.50, 9)
    vol_sensitivity = SensitivityAnalysis.volatility_sensitivity(S, K, T, r, sigma_range)
    print(vol_sensitivity.to_string(index=False))

    # Interest Rate Sensitivity
    print("\n4.2 Interest Rate Sensitivity")
    print("-" * 80)
    r_range = np.linspace(0.01, 0.10, 10)
    rate_sensitivity = SensitivityAnalysis.interest_rate_sensitivity(S, K, T, sigma, r_range)
    print(rate_sensitivity.to_string(index=False))

    # Time to Expiry Sensitivity
    print("\n4.3 Time to Expiry Sensitivity")
    print("-" * 80)
    T_range = np.linspace(0.05, 1.0, 10)
    expiry_sensitivity = SensitivityAnalysis.expiry_sensitivity(S, K, r, sigma, T_range)
    print(expiry_sensitivity.to_string(index=False))

    # 5. Short-Maturity Delta Analysis
    print("\n" + "=" * 80)
    print("5. SHORT-MATURITY DELTA ANALYSIS")
    print("=" * 80)

    T_short = np.array([1 / 365, 5 / 365, 10 / 365, 20 / 365, 30 / 365])  # 1, 5, 10, 20, 30 days
    delta_analysis = short_maturity_delta_analysis(S, K, r, sigma, T_short)
    print("\n" + delta_analysis.to_string(index=False))

    # 6. Accuracy Analysis
    print("\n" + "=" * 80)
    print("6. MODEL ACCURACY: CONVERGENCE ANALYSIS")
    print("=" * 80)

    print("\nBinomial Model Convergence (varying N):")
    print("-" * 80)

    N_values = [10, 25, 50, 100, 200, 500]
    convergence_results = []

    bs_call_price = bs.call_option_price()

    for n in N_values:
        binom_temp = BinomialModel(S, K, T, r, sigma, n)
        binom_call = binom_temp.european_call_price()
        error = abs(binom_call - bs_call_price)
        error_pct = (error / bs_call_price) * 100

        convergence_results.append({
            'N_Steps': n,
            'Binomial_Call_Price': binom_call,
            'BS_Call_Price': bs_call_price,
            'Absolute_Error': error,
            'Error_Percentage': error_pct
        })

    convergence_df = pd.DataFrame(convergence_results)
    print(convergence_df.to_string(index=False))

    # 7. At-the-Money, In-the-Money, Out-of-the-Money Analysis
    print("\n" + "=" * 80)
    print("7. MONEYNESS ANALYSIS")
    print("=" * 80)

    K_values = [90, 95, 100, 105, 110]  # Different strike prices
    moneyness_results = []

    for k in K_values:
        bs_temp = BlackScholesModel(S, k, T, r, sigma)

        if S > k:
            status = "ITM"
        elif S == k:
            status = "ATM"
        else:
            status = "OTM"

        moneyness_results.append({
            'Strike': k,
            'Moneyness': status,
            'S/K': S / k,
            'Call_Price': bs_temp.call_option_price(),
            'Put_Price': bs_temp.put_option_price(),
            'Call_Delta': bs_temp.delta_call(),
            'Put_Delta': bs_temp.delta_put(),
            'Gamma': bs_temp.gamma()
        })

    moneyness_df = pd.DataFrame(moneyness_results)
    print("\n" + moneyness_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
