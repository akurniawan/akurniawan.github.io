---
title: "Bitcoin Regime Analysis: From EDA to Hidden Markov Models"
date: 2024-12-25T00:00:00+07:00
draft: false
tags: ["cryptocurrency", "bitcoin", "regime-analysis"]
categories: ["quant", "research"]
description: "A comprehensive exploratory data analysis of Bitcoin price dynamics, examining volatility clustering, heavy tails, and persistence patterns to inform regime-switching models using Hidden Markov Models."
summary: "This post explores Bitcoin's price behavior through statistical analysis, revealing distinct volatility regimes and persistence patterns that can be captured using HMM-based regime switching models."
toc: true
---

As someone with background in Natural Language Processing (NLP), I've spent considerable time working with sequence modeling like Hidden Markov Models (HMMs). The concept of regime switching, where a system transitions between different underlying states, seems like a natural first project for me to take. It serves as my bridge from NLP to quantitative finance, applying the same probabilistic framework I used for language modeling to understand market dynamics.

Bitcoin's price data presented itself as the perfect natural experiment. The data is publicly available and can be retrieved easily from many exchanges. For this experiment in particular, I retrieved the data from Binance using [freqtrade](https://www.freqtrade.io/en/stable/). Cryptocurrency markets are notorious for their volatility clustering, sudden regime shifts, and non-stationary behavior, characteristics that align perfectly with the regime-switching framework I was already familiar with from NLP applications.

Just as words in a sentence depend on their grammatical context (noun vs. verb phrases), Bitcoin returns exhibit different statistical properties depending on the underlying market regime (bull vs. bear, high vs. low volatility). My experience with HMMs in NLP, where we model the hidden states that generate observable tokens, translates naturally to modeling the hidden market regimes that generate observable price movements.

Before going to the modeling part, as per usual, let's start with some analysis of the data.

## Basic Price and Return Characteristics

Price evolution. From February 2024 to late-July 2025 Bitcoin trended upward with two notable drawdown–rally cycles (see Figure 1). After a steady climb into late-Q4 2024, price accelerated into early-January 2025, corrected through March–April, and then advanced to fresh highs above $115–120k by July 2025. This path already hints at regime-like phases (risk-on acceleration, corrective consolidation, renewed trend).

{{< figure src="/images/posts/btc-regime-eda/btc_price.png" alt="Bitcoin price evolution from February 2024 to July 2025 showing upward trend with drawdown-rally cycles" caption="Figure 1: Bitcoin price evolution showing distinct regime-like phases" >}}

Return distribution. Hourly log-returns exhibit a sharp peak around zero with very heavy tails (see Figure 2 and Figure 3). The boxplot shows a dense core and numerous extreme outliers on both sides—typical for crypto. Empirically, hourly skewness is slightly negative (≈ −0.10) and kurtosis is strongly leptokurtic (≈ 9.44), implying more frequent large moves than a Gaussian would predict.

{{< figure src="/images/posts/btc-regime-eda/btc_histogram_returns.png" alt="Histogram of Bitcoin hourly returns showing sharp peak around zero with heavy tails" caption="Figure 2: Distribution of hourly returns showing non-Gaussian characteristics" >}}

{{< figure src="/images/posts/btc-regime-eda/btc_boxplot_returns.png" alt="Boxplot of Bitcoin hourly returns showing dense core and numerous outliers" caption="Figure 3: Boxplot revealing heavy-tailed distribution with extreme outliers" >}}
Models assuming normality will understate tail risk; fat-tail-aware likelihoods (e.g., Student-t) are more appropriate for volatility and regime work.

The non-Gaussian nature of returns varies with aggregation:

| Timeframe |    Skewness | Excess Kurtosis |      N |
| --------- | ----------: | --------------: | -----: |
| **1H**    | **−0.0973** |      **9.4406** | 12,890 |
| **1D**    | **+0.4935** |      **2.2668** |    537 |
| **1W**    | **+0.2572** |      **0.7631** |     77 |

Hourly returns are leptokurtic (heavy-tailed), daily still exhibit fat tails and positive skew, while weekly get closer to normal but remain non-Gaussian in the tails. This supports the use of fat-tailed likelihoods across horizons, especially intraday.

## Statistical Properties Across Timeframes

Normality checks.
- Hourly: The Q-Q plot (btc_qqplot_hourly_returns.png) shows an “S” shape with both tails deviating far above the normal reference—clear fat tails and some asymmetry.
- Weekly: Aggregation tempers extremes, and the weekly Q-Q plot (btc_qqplot_weekly_returns.png) hugs the line more closely in the center, yet tail deviations persist.
Implication. Tail risk survives temporal aggregation; the hourly distribution is especially non-Gaussian, while weekly returns are closer to—but still not—normal.

{{< figure src="/images/posts/btc-regime-eda/btc_qqplot_hourly_returns.png" alt="Q-Q plot of hourly returns showing S-shaped deviation from normal distribution" caption="Figure 4: Q-Q plot of hourly returns revealing fat tails and asymmetry" >}}

{{< figure src="/images/posts/btc-regime-eda/btc_qqplot_weekly_returns.png" alt="Q-Q plot of weekly returns showing closer fit to normal but persistent tail deviations" caption="Figure 5: Q-Q plot of weekly returns showing improved normality with residual tail effects" >}}

## Volatility Analysis

Clustering and persistence. Rolling standard deviation (black line in Figure 6) varies meaningfully over time, with volatility bursts around March–April 2025 and intermittent spikes elsewhere. The rolling mean (red) stays near zero, but dispersion shifts markedly—classic ARCH/GARCH behavior.

{{< figure src="/images/posts/btc-regime-eda/btc_return_stationarity.png" alt="Rolling standard deviation and mean of Bitcoin returns showing time-varying volatility" caption="Figure 6: Rolling volatility showing clustering and persistence patterns" >}}
Implication. Time-varying volatility is pronounced; volatility-aware models (GARCH/EGARCH/FIGARCH) or state-dependent variances in HMMs are warranted.

## Stationarity and Core Time-Series Properties

Stationarity of returns. The Augmented Dickey–Fuller test on hourly returns rejects a unit root with overwhelming confidence (e.g., test statistic ≈ −25.49, p ≪ 0.01; see also the stable rolling mean in Figure 6). Prices themselves are non-stationary (as expected), but returns are stationary around zero with time-varying variance.
Implication. It is appropriate to fit return-based models (including HMM emissions) under stationarity of the mean, while letting variance be regime- or time-dependent.

## Seasonality and Cyclical Patterns

Decomposition. A multiplicative decomposition of log price (Figure 7) shows:

{{< figure src="/images/posts/btc-regime-eda/btc_seasonality.png" alt="Seasonal decomposition of Bitcoin log price showing trend, seasonal, and residual components" caption="Figure 7: Seasonal decomposition revealing weak periodic structure" >}}
- a trend component that mirrors the medium-term advance and drawdowns;
- a seasonal component with a small-amplitude, high-frequency cycle (oscillations of roughly ±0.1% around 1.00);
- residuals that dominate the short-horizon dynamics.

The seasonal amplitude is very small relative to trend and noise, indicating weak deterministic seasonality in this hourly aggregation.
Implication. Intraday/short-cycle effects exist but are minor; signal-to-noise is low. Seasonality alone is unlikely to produce robust trading signals without additional filters.

Seasonal decomposition on log price with a ~10-day period yields:
- **r2_seasonal:** `8.7598e-06`
- **r2_residual:** `4.8789e-05`
- **SNR (seasonal vs. noise, as computed in your script):** `1.5077`

Both seasonal and residual components explain **< 0.01%** of total variance; the seasonal signal is extremely small relative to overall variability. The SNR≈1.51 (per your computation) indicates only a very weak periodic structure—consistent with Figure 7’s small oscillation amplitude.

## Serial Correlation and Memory Properties

Linear dependence in returns. The ACF and PACF of hourly returns (Figure 8 and Figure 9) are near zero beyond the first few lags, with a small but statistically significant lag-1 effect (visible in both plots). This is consistent with microstructure frictions or short-horizon mean-reversion/continuation effects that are economically small but detectable with large samples.

{{< figure src="/images/posts/btc-regime-eda/btc_acf.png" alt="Autocorrelation function of Bitcoin hourly returns showing weak linear dependence" caption="Figure 8: ACF of hourly returns revealing minimal linear autocorrelation" >}}

{{< figure src="/images/posts/btc-regime-eda/btc_pacf.png" alt="Partial autocorrelation function of Bitcoin hourly returns" caption="Figure 9: PACF of hourly returns showing lag-1 effect" >}}
Long-memory in volatility. While raw returns show little linear autocorrelation, the volatility clustering in Section C suggests persistence in squared or absolute returns (a hallmark of long memory in volatility even when returns themselves are nearly white noise).
Implication. Mean dynamics are weak at the hourly horizon, but variance dynamics are persistent—an important cue for regime modeling and risk sizing.

## Rolling Hurst Exponent (Persistence vs. Mean Reversion)

Computed rolling Hurst exponents for **returns** and **volatility** (absolute returns), classifying regimes as:
- **Mean reverting:** H < 0.5  
- **Random walk:** H ≈ 0.5  
- **Persistent (trend-following):** H > 0.5

### Sample (every 3 days)
```
Date        Returns H  Volatility H  Returns Type      Volatility Type
-----------------------------------------------------------------------
2025-03-24  0.2831     0.8180        Mean reverting    Strong persistent
2025-03-27  0.6501     0.4990        Strong persistent Random walk
2025-03-30  0.7408     0.7689        Strong persistent Strong persistent
2025-04-02  0.7101     0.5255        Strong persistent Weak persistent
2025-04-05  0.6872     0.9093        Strong persistent Strong persistent
2025-04-08  0.7218     0.4279        Strong persistent Mean reverting
...
2025-07-19  0.2264     0.8137        Mean reverting    Strong persistent
2025-07-22  0.4272     0.7366        Mean reverting    Strong persistent
```

### Weekly snapshot (Feb–Jul 2024 excerpt)
```
Date        Returns H  Volatility H  Returns Type      Volatility Type
-----------------------------------------------------------------------
2024-02-08  0.6094     0.6099        Strong persistent Strong persistent
2024-02-15  0.5624     0.5938        Strong persistent Strong persistent
...
2025-07-17  0.5901     0.5898        Strong persistent Strong persistent
```

- **Returns:** The majority of windows show **H > 0.5** (persistent), punctuated by shorter **mean-reverting** episodes—useful for state labeling and for toggling between momentum vs. reversion tactics.  
- **Volatility:** H is **consistently > 0.6**, indicating **strong persistence** in volatility (long memory), reinforcing the need for regime-dependent or long-memory volatility models ((F)GARCH/FIGARCH, HMM with state-specific variance).

## Conclusion

Bitcoin returns are non-Gaussian, heteroskedastic, and exhibit long-memory volatility. Prices evolve through distinct phases that align with bull/bear/regime narratives. These properties invalidate constant-parameter Gaussian models and directly motivate the use of regime-switching models with fat-tail-aware likelihoods and volatility-sensitive features.

Key EDA Takeaways (for downstream modeling):
1.	Price action already hints at regimes.
Bitcoin’s Feb 2024–Jul 2025 path alternated between accelerations, corrections, and consolidations — visually consistent with “risk-on / correction / trend” phases.
2.	Return distributions are far from Gaussian.
- Hourly returns are strongly leptokurtic (excess kurtosis ≈ 9.4), with extreme outliers in both tails.
- Daily returns remain fat-tailed and positively skewed, while weekly returns are closer to Normal but still heavy-tailed.
3.	Volatility is time-varying and clustered. Rolling volatility shows pronounced bursts (March–April 2025 drawdown) and intermittent spikes elsewhere. This confirms ARCH/GARCH-like persistence and supports state-dependent variance in HMMs.
4.	Returns are stationary in the mean, but not homoskedastic. ADF tests reject a unit root in hourly returns, with rolling mean ≈ 0. Prices are non-stationary, but returns are mean-stationary with heteroskedastic variance — exactly the setting for regime models.
5.	Seasonality is negligible. Seasonal decomposition shows extremely weak periodic structure (<0.01% variance explained, SNR ≈ 1.5). Intraday/short-cycle patterns exist but are economically minor relative to trend and noise.
6.	Linear dependence is weak, but variance persistence is strong.
- ACF/PACF of returns ≈ 0 beyond lag-1 → no linear predictability.
- Volatility clustering and autocorrelation in |r|/r² remain → volatility has memory even when returns do not.
7.	Hurst exponent reveals persistence/long memory.
- Returns often show H > 0.5 (persistent / trending), punctuated by short mean-reverting windows.
- Volatility consistently shows H > 0.6–0.9 → strong long memory.