---
title: "Bitcoin Regime Analysis: From EDA to Hidden Markov Models (Part 2)"
date: 2024-12-25T00:00:00+07:00
draft: false
tags: ["cryptocurrency", "bitcoin", "regime-analysis"]
categories: ["quant", "research"]
description: "Implementation of Hidden Markov Models for Bitcoin regime detection, comparing multiple model specifications from basic returns-only to advanced fractional differentiation with volume features."
summary: "This post implements and compares various HMM specifications for Bitcoin regime detection, demonstrating how feature engineering and fractional differentiation improve regime identification and persistence."
toc: true
---

Building on the exploratory data analysis from [Part 1](), let's continue by implementing Hidden Markov Models (HMMs) to identify and characterize Bitcoin's market regimes. The analysis progresses through increasingly sophisticated model specifications, from basic returns-only models to advanced variants incorporating volatility, volume, and fractional differentiation.

## Hidden Markov Model Framework

### Theoretical Foundation

**HMM fundamentals for financial time series.** Hidden Markov Models provide a probabilistic framework for modeling time series with unobserved, or "hidden," state dynamics. In financial applications, these hidden states are typically interpreted as market regimes—distinct periods characterized by different return distributions, volatility levels, or volume activity. Each observation (e.g., log-returns) is assumed to be generated from a probability distribution conditioned on the hidden regime. The dynamics of regime switching are governed by a Markov chain, where the probability of moving from one regime to another depends only on the current regime.

**Gaussian HMM specification.** In this experiment, we employ a Gaussian HMM, where the emission distribution of returns (and additional features such as volatility or volume) is modeled as a multivariate Gaussian. The parameters estimated for each state include:

- Mean return (expected value within regime)
- Variance-covariance structure (volatility and cross-feature correlation)
- Transition probabilities between regimes

This specification is well-suited for continuous-valued financial features and enables interpretable characterization of regimes (e.g., "high-volatility negative-return regime" vs. "low-volatility trending regime").

**Model selection criteria (AIC/BIC).** Because the number of hidden regimes is not known a priori, we rely on information criteria such as Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) to balance model fit with complexity. AIC and BIC penalize overfitting by adding penalties for additional parameters, with BIC typically imposing a harsher penalty for larger models. Throughout the analysis, models are compared across different numbers of regimes (3–7), and the final selection considers both information criteria and economic interpretability of the regimes.

### Model Variants and Feature Engineering

#### Model 1: Returns Only (RO)

The baseline model considers only log-returns as the observable feature.

**Basic regime identification.** Figure 1 overlays HMM-detected regimes on the Bitcoin price series. Distinct colored markers denote inferred hidden states, aligning with periods of trending rallies, drawdowns, and sideways consolidations. We can see from Figure 1 below that Regime 0 and Regime 2 capture wicks at different ranges. Regime 0 captures smaller wicks while Regime 2 captures larger wicks. Regime 1, on the other hand, marks a calm period that dominates most of the candles. This is also supported by the returns analysis in Figure 2.

{{< figure src="/images/posts/btc-regime/hmm_regime_ro_3.png" alt="Bitcoin price with HMM-detected regimes overlaid showing distinct colored markers for different market states" caption="Figure 1: HMM regime detection on Bitcoin price series using returns-only model" >}}

The distributional properties of returns by regime reveal heavy tails and significant differences in volatility across regimes. Certain states correspond to calm, near-zero return periods, while others capture fat-tailed distributions with higher variance. This establishes the foundation for multi-regime characterization.

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_3.png" alt="Distribution of returns by HMM regime showing different volatility and tail characteristics across states" caption="Figure 2: Return distributions by regime for the returns-only model" >}}

**Limitations of returns-only models with higher K.** Expanding beyond 3 states reveals significant limitations when using only returns as features. Figure 3 demonstrates that in the 5-state returns-only model, most of the price action is dominated by a single regime (Regime 0), indicating poor regime separation and suggesting that returns alone lack sufficient discriminatory power for higher-dimensional state spaces.

{{< figure src="/images/posts/btc-regime/hmm_regime_ro_5.png" alt="HMM regime detection with 5 states showing poor regime separation with most candles dominated by Regime 0" caption="Figure 3: 5-state returns-only model showing limited regime differentiation" >}}

**Distributional overlap problem.** Figure 4 reveals a more fundamental issue: the return distributions for Regimes 0 and 1 are nearly identical, making them virtually indistinguishable. Additionally, Regime 2 appears to be poorly detected, with minimal representation in the sample. This distributional overlap confirms that returns-only models struggle to capture the nuanced differences between market states when K > 3, necessitating the inclusion of additional features for meaningful regime identification.

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_5.png" alt="Return distributions by regime showing significant overlap between Regimes 0 and 1, with poor detection of Regime 2" caption="Figure 4: Return distributions for 5-state returns-only model showing distributional overlap" >}}

#### Model 2: Returns + Volatility (ROV)

The second model augments returns with volatility estimates derived from FIGARCH. Extra experiments in this section is to also compare the number of states from 3 to include 5, 6, and 7.

**FIGARCH volatility forecasting.** Unlike GARCH, which models short memory in volatility, FIGARCH (Fractionally Integrated GARCH) captures long memory effects, allowing shocks to volatility to persist over extended horizons. This is particularly relevant in crypto markets, where volatility clustering can be long-lived.

**Bivariate regime analysis.** Figure 3 shows price with regime overlays for the returns–volatility specification, while Figure 4 shows return distributions by regime. Incorporating volatility produces more economically meaningful regimes:

- High-volatility, negative-return states during corrections
- Low-volatility, positive-return states during bull trends
- Transitional regimes with moderate volatility and flat returns

{{< figure src="/images/posts/btc-regime/hmm_regime_ro_vol_5.png" alt="Bitcoin price with HMM regimes incorporating both returns and volatility features" caption="Figure 3: HMM regime detection using returns and volatility features" >}}

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_5.png" alt="Return distributions by regime for the returns-volatility model showing improved separation" caption="Figure 4: Return distributions by regime for the returns-volatility model" >}}

**5, 6, and 7 regime comparisons.** Figures 5-7 illustrate how increasing the number of regimes refines state granularity. For example, splitting a broad "high volatility" state into two sub-states differentiates between panic-driven crashes vs. slower grinding corrections. Information criteria are later used to evaluate whether these additional regimes improve model quality.

{{< image-gallery >}}
{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_5.png" alt="HMM regime detection with 5 states showing moderate granularity" caption="Figure 5: 5-regime HMM model" >}}

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_6.png" alt="HMM regime detection with 6 states showing increased granularity" caption="Figure 6: 6-regime HMM model" >}}

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_7.png" alt="HMM regime detection with 7 states showing maximum granularity" caption="Figure 7: 7-regime HMM model" >}}
{{< /image-gallery >}}

**ROV Model Summary:** The addition of FIGARCH volatility features addresses a critical limitation revealed in the returns-only analysis: the inability to meaningfully expand beyond 3 states due to poor regime separation and distributional overlap. While the ROV model produces incremental rather than dramatic improvements in regime identification, it successfully enables the use of higher-dimensional state spaces (K=5,6,7) that were problematic with returns-only features. The volatility component provides the additional discriminatory power needed to distinguish between regimes that appear similar in return space but differ in their volatility characteristics. This is particularly evident in the improved separation between high-volatility stress periods and moderate-volatility consolidation phases, which were indistinguishable in the returns-only model. The results demonstrate that while Bitcoin's regime behavior is primarily driven by return dynamics, volatility features serve as essential complementary information for achieving meaningful regime granularity beyond the basic 3-state model.

#### Model 3: Returns + Volatility + Volume (ROVV)
The third specification incorporates trading volume, which provides additional information about market participation intensity. By capturing how aggressively buyers and sellers engage, volume adds a behavioral dimension that pure return–volatility models overlook.

**Volume-return interaction terms.** Periods of strong positive returns accompanied by surging volume often indicate genuine momentum phases with broad market participation. Conversely, negative returns paired with abnormal volume surges tend to reflect capitulation events, where selling pressure is both intense and widespread. Without volume, such events may be indistinguishable from ordinary corrections.

**Relative volume (RVOL) calculation.** To account for changing liquidity conditions over time, raw volume is normalized into relative volume (RVOL), defined as current volume divided by its rolling average. This adjustment highlights abnormal participation spikes while discounting structural changes in baseline trading activity. RVOL thus allows the HMM to distinguish quiet drift from high-participation breakout even when returns and volatility appear similar.

Trivariate regime analysis.
Figures 1 and 2 (hmm_regime_ro_vol_volume.png, hmm_returns_ro_vol_volume.png) demonstrate that including volume produces sharper differentiation between regimes:

“Quiet” consolidation phases characterized by muted returns, low volatility, and below-average volume.

Trend-following regimes where strong positive drift coincides with sustained RVOL > 1.

Stress/capitulation regimes combining negative returns with abnormal surges in volume, often clustering around sharp liquidations.

**5, 6, and 7 regime comparisons.** Extending the state space reveals further structure:

**K = 5:**
{{< image-gallery >}}
{{< figure src="/images/posts/btc-regime/hmm_regime_ro_vol_volume_5.png" alt="HMM regime detection incorporating returns, volatility, and volume features" caption="Figure 8: 5-regime HMM model with volume features" >}}

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_volume_5.png" alt="Return distributions by regime for the trivariate model showing enhanced separation" caption="Figure 8b: Return distributions for 5-regime ROVV model" >}}
{{< /image-gallery >}}
The broad stress regime splits into two distinct modes: (i) gradual drawdowns with moderate participation, and (ii) crash-like episodes with both heavy volume and fat-tailed negative returns. Similarly, bullish phases separate into low-volume drift vs. high-volume breakout rallies.

**K = 6:**
{{< image-gallery >}}
{{< figure src="/images/posts/btc-regime/hmm_regime_ro_vol_volume_6.png" alt="6-regime HMM model with volume features showing increased granularity" caption="Figure 9: 6-regime HMM model with volume features" >}}

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_volume_6.png" alt="Return distributions by regime for 6-state model showing emergence of low-vol flat state" caption="Figure 9b: Return distributions for 6-regime ROVV model" >}}
{{< /image-gallery >}}

An additional sideways regime emerges with very low variance and muted volume, consistent with accumulation or distribution phases. This refinement prevents flat markets from being mistakenly lumped with quiet bullish trends. On the downside, capitulation states become more clearly isolated, helping to distinguish “panic” from ordinary corrections.

**K = 7:**
{{< image-gallery >}}
{{< figure src="/images/posts/btc-regime/hmm_regime_ro_vol_volume_7.png" alt="7-regime HMM model with volume features showing maximum granularity" caption="Figure 10: 7-regime HMM model with volume features" >}}

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_volume_7.png" alt="Return distributions by regime for 7-state model showing concentrated stress states" caption="Figure 10b: Return distributions for 7-regime ROVV model" >}}
{{< /image-gallery >}}

The 7-state model provides the cleanest bull vs. bear separation. In the return distribution plots, State 1 clusters around persistently positive means with relatively narrow tails, consistent with trending bullish rallies, while State 5 centers around negative means with heavier downside tails, consistent with bearish phases. Importantly, volume ensures these states are not just mirror images: bullish states exhibit high participation on rallies, whereas bearish states show crowding during sell-offs. This separation is absent in returns-only models and less clear in ROV.

**ROVV Model Summary:** The inclusion of volume features resolves two critical limitations observed in the previous models:

1. **Distributional overlap** — where returns-only models produced indistinguishable states, RVOL introduces clear participation-driven separation.

2. **Bull vs. bear asymmetry** — the 7-state ROVV model reveals distinct upward vs. downward regimes (States 1 vs. 5) with supporting evidence from volume, improving interpretability and potential trading applications.

In practical terms, ROVV regimes align more closely with market intuition: high-volume upswings correspond to momentum opportunities, while high-volume downswings flag stress/liquidation phases. By enriching the model with volume, we achieve both stronger statistical separation and more actionable economic interpretation than in RO or ROV.

#### Model 4: Fractional Differentiation + Volatility + Volume (FD-ROVV)

The final specification combines fractionally differentiated returns with volatility and volume. Fractional differencing (López de Prado, 2018) strikes a balance between preserving memory in the series and achieving approximate stationarity. Unlike full differencing, which destroys long-memory dependencies, fractional differencing retains information about persistent structures—an important property for regime models where transitions often evolve gradually rather than abruptly.

**Memory preservation vs. stationarity trade-off.** Figures 11-12 show the 3-state FD-ROVV model. Compared with the raw return ROVV equivalent, state transitions appear smoother, with fewer flickers between regimes. The FD transformation dampens noise-driven jumps that otherwise lead to spurious state switches.

{{< image-gallery >}}
{{< figure src="/images/posts/btc-regime/hmm_regime_ro_vol_volume_fd_3.png" alt="HMM regime detection using fractionally differentiated returns with 3 states" caption="Figure 11: 3-regime HMM model with fractional differentiation" >}}

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_volume_fd_3.png" alt="Return distributions by regime for fractional differentiation model with 3 states" caption="Figure 11b: Return distributions for 3-regime FD model" >}}
{{< /image-gallery >}}

**Regime characterization (K=3):**

- One calm, near-zero regime (sideways or accumulation)
- One upward-drifting regime, associated with higher volume
- One high-volatility negative regime, clustering around drawdowns
Already at this simple level, FD helps achieve persistence in state labeling without sacrificing sensitivity to major moves.

**5-state expansion.** Figures 13-14 illustrate the additional granularity when K=5.

{{< image-gallery >}}
{{< figure src="/images/posts/btc-regime/hmm_regime_ro_vol_volume_fd_5.png" alt="HMM regime detection using fractionally differentiated returns with 5 states" caption="Figure 13: 5-regime HMM model with fractional differentiation" >}}

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_volume_fd_5.png" alt="Return distributions by regime for fractional differentiation model with 5 states" caption="Figure 13b: Return distributions for 5-regime FD model" >}}
{{< /image-gallery >}}

On the downside, the broad “stress” regime from K=3 now splits into:

- Capitulation states — extreme negative returns with surging RVOL
- Grinding correction states — negative drift with moderate volume

On the upside, trending states separate into low-volume drift and high-participation breakout rallies.

A “quiet accumulation” regime persists, with returns tightly centered around zero and muted RVOL.

This decomposition is far cleaner than in the non-FD ROVV model, where higher K sometimes led to overlapping or unstable regimes.

**7-state expansion.** Figures 15-16 show the richest FD-ROVV structure. The benefits of fractional differencing are clearest here: regimes are both stable and economically interpretable.

{{< image-gallery >}}
{{< figure src="/images/posts/btc-regime/hmm_regime_ro_vol_volume_fd_7.png" alt="HMM regime detection using fractionally differentiated returns with 7 states" caption="Figure 15: 7-regime HMM model with fractional differentiation" >}}

{{< figure src="/images/posts/btc-regime/hmm_returns_ro_vol_volume_fd_7.png" alt="Return distributions by regime for fractional differentiation model with 7 states" caption="Figure 15b: Return distributions for 7-regime FD model" >}}
{{< /image-gallery >}}

Clear bull vs. bear differentiation: State 1 clusters around sustained positive returns (bullish rallies with high participation), while State 5 centers on heavy negative returns (bearish sell-offs). Unlike in raw return models, these states are persistent and non-overlapping.

Sideways refinements: At least two low-variance regimes emerge—one representing quiet accumulation and another reflecting range-bound chop.

Stress stratification: On the downside, capitulation shocks are isolated from longer but less severe drawdowns.

Momentum granularity: The bullish complex splits into “gentle trend” vs. “momentum thrust,” consistent with differences in both RVOL and volatility.

**Enhanced regime detection.** Across K=3, 5, and 7, the return distributions show less overlap than in ROVV. States that previously shared nearly identical centers are now separated by persistence and volatility structure. The boxplots highlight this: FD states have tighter clustering within regimes and heavier tails concentrated in specific states, rather than spread thinly across several.

**FD-ROVV Model Summary:** Fractional differentiation improves regime switching in two key ways:

1. **State persistence** — FD reduces spurious regime flips, producing cleaner, more stable regime paths.

2. **Economic interpretability** — By preserving long-memory effects, FD enables clear separation between capitulation vs. correction on the downside and drift vs. breakout on the upside.

The result is a regime framework that scales well as K increases: unlike RO and ROVV, where higher-K models risk degeneracy or overlap, the FD-ROVV model delivers both statistical and economic clarity. In particular, the 7-state specification reveals distinct bull and bear phases that align closely with market intuition, making FD-ROVV the strongest candidate among the tested variants for robust, regime-aware trading strategies.

## Conclusion

The progression from basic returns-only models to sophisticated fractional differentiation variants demonstrates how feature engineering and advanced preprocessing techniques can significantly improve regime detection quality. The incorporation of volatility and volume features provides more economically meaningful regime characterizations, while fractional differentiation enhances regime persistence and reduces noise-induced state switching.

Key findings from the HMM analysis include:

1. **Feature engineering matters.** Adding volatility and volume features produces more interpretable regimes that align with observable market behavior.

2. **Regime granularity trade-offs.** While more regimes provide finer granularity, they also increase model complexity and potential overfitting.

3. **Fractional differentiation benefits.** FD variants show improved regime persistence and better separation between distinct market states.

4. **Model selection criteria.** AIC and BIC provide quantitative guidance for balancing model fit with complexity, though economic interpretability remains crucial.

These results establish a robust foundation for regime-dependent trading strategies and risk management approaches in cryptocurrency markets.