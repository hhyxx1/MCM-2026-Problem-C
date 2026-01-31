# 2026 MCM Problem C: The Fairness-Engagement Equilibrium Model
**Based on Pareto Optimization & Global History Inversion Reform Plan**

---

## Phase 1: Data Archeology & Global Scan

**Core Task:** Do not look at individual cases only; perform a full physical examination of Seasons 1-34.

### 1. Data Cleaning
* **Standardization:** Unify judge scores from all seasons (whether 30-point or 40-point systems) into a percentage ($J\%$).
* **Withdrawal Handling:** Exclude N/A and 0-point data to ensure the denominator is accurate.

### 2. Key Feature: Popularity Bias Index (PBI)
* **Formula:** $PBI_i = Rank_{Judge}(i) - Rank_{Final}(i)$

#### 【Patch 1: Partner Impact】
* Calculate the historical average PBI for each Professional Dancer.
* Identify "Star Makers" (e.g., Derek Hough) to serve as correction coefficients for subsequent simulations.

#### 【Patch 1B: Celebrity Covariates】
Extract and standardize the following features from the dataset to explain "Judge Scores" and "Fan Votes" separately (checking for same-direction influence):
* **Age:** $Age_i$ (can use segmentation or splines).
* **Industry:** $Industry_i$ (one-hot; e.g., Athlete/Actor/Musician/Reality/Politician, etc.).
* **Region:** State/Country or Region (US vs Non-US; or clustered by area).
* **Season & Week:** Season/Week fixed effects (controlling for rule and overall trend changes).
* **Implementation:** Convert wide tables to long format $(i,w)$ panel data to facilitate mixed-effects/hierarchical models.

### 3. 【Patch 2: Global Scan】
* **Action:** Do not just pick seasons with good data. Write code to loop through Season 1 to Season 34.
* **Output:** Draw a "Chronological Heatmap" showing the trend of "Judge-Audience Divergence" over time. Prove that divergence increases with the development of social media (this is the necessity for reform).

---

## Phase 2: Inverse Inference & Validation

**Core Task:** Accurately reconstruct the "crime scene," estimate the fan vote share for each contestant in every week they competed, and prove the model is not guessing blindly.

### 1. Bayesian Inverse Inference
* **Output Definition:** For each remaining contestant $i$ in week $w$, estimate fan vote share $f(i,w)$ (constraint $f(i,w) \ge 0$ and $\sum_i f(i,w) = 1$), providing point estimates (mean/median) + 95% Credible Intervals.
* **Modeling:**
    * Under the given season's aggregation rule $g$ (Rank or Percentage), define Combined Score as $Score(i,w) = g(J\%(i,w), f(i,w))$.
    * Use historical elimination information to form inequality constraints: If the eliminated set for the week is $E_w$ (size $k$), require $E_w$ to be in the Bottom-$k$ of that week's Score.
    * Equivalently, for any $e \in E_w, s \notin E_w$, ensure $Score(s,w) > Score(e,w)$.
    * Use MCMC/Hit-and-Run to sample within the feasible region under these constraints to get the posterior distribution of $f(i,w)$.
* **Special Week Handling (Must make the review bulletproof):**
    * **Multi-elimination weeks:** Directly set $E_w$ as the Bottom-$k$ set for constraints, and use Bottom-$k$ match metrics in consistency checks.
    * **No elimination/Vote Accumulation weeks:** If the show sets "votes accumulate to next week," treat the two weeks as one block (accumulate Score then apply Bottom-$k$ constraint); if accumulation mechanism is unconfirmed, do not add elimination constraints for that week, using it only as a trend prior (avoid forced fitting).

### 2. 【Patch 3: Certainty & Consistency】
**Two Mandatory Indicators:**

**Indicator A: Certainty** — Regarding the $f(i,w)$ estimation for each (contestant $i$, week $w$).
* **Measurement Suggestions** (Provide at least two, showing "variability by person/week"):
    * **95% CI Width:** $CIW(i,w) = q_{97.5\%} - q_{2.5\%}$.
    * **Coefficient of Variation (CV) or Posterior Entropy:** Measure whether the "vote momentum" for the week is highly uncertain.
* **Output:** Provide $CIW(i,w)$ for each contestant/week, and summarize "The most uncertain Week/Person" (demonstrating uncertainty is not constant).

**Indicator B: Consistency** — Whether the model can replicate weekly elimination results.
* **Consistency Layers (Provide at least two):**
    1.  **Exact-Match Rate:** Use posterior mean/median to form predicted elimination set $\hat{E}_w$, calculate the average of $I(\hat{E}_w = E_w)$ (count only weeks with eliminations).
    2.  **Posterior Consistency:** $P_w = Prob(E_w \text{ is Bottom-}k \mid posterior)$, estimated via posterior sampling frequency; Overall consistency is $\bar{P} = (1/W) * \sum_w P_w$.
    * For multi-elimination weeks, provide set metrics (Jaccard/F1) to avoid the controversy of "hitting only one person counts as correct."
* **Target Reporting:** Report the mean/distribution of Exact-Match and $\bar{P}$, grouped by season. High consistency is the cornerstone of trust for the subsequent "Parallel Universe Rule Comparison."

---

## Phase 3: Omni-Simulator & Case Studies

**Core Task:** Use parallel universes to verify rules; must cover the four major controversial cases.

### 1. Simulator Architecture
* **Switch Rules:** Support Rank, Percentage, New_Strategy.
* **Switch Mechanisms:** Judges' Save (On/Off).

### 2. 【Patch 4B: Rank vs Percentage Cross-Season Comparison & "Fan Bias" Measurement】
* **Requirement:** Apply both aggregation methods to every season and answer "whether one method favors fans more."
* **Execution:** For each Season, run the full season using Rank and Percentage respectively (using $f(i,w)$ from Phase 2 or its posterior samples) to get two "Parallel Universe Elimination Lines."
* **Comparison Output (Summary Table/Chart across seasons):**
    * **Weekly Difference:** $D_{season} = \#\{ w : E_w^{rank} \neq E_w^{pct} \}$, noting which week the difference occurred.
    * **Final Standing Difference:** Kendall tau / Top-3 overlap (Did the Champion/Top-3 change under the two methods).
* **"Fan Bias" Quantification (Choose at least 1-2 to explain clearly):**
    * **Fan-Favor Index (FFI):** Correlation between final ranking and fan ranking (Spearman); compare $FFI_{rank}$ vs $FFI_{pct}$.
    * **Fan-Elasticity:** Add small perturbations to $f(i,w)$ (or resample from posterior) to see elimination reversal probability; if reversals are frequent and follow fan changes, it indicates favoring fans.
* **Conclusion:** Use these metrics to give a clear conclusion: which method favors fans more.

### 3. 【Patch 4: Historical Case Studies】
**Must establish an independent section** to specifically test the following 4 scenarios:

* **Case 1 (Season 2): Jerry Rice** — Test: If "Judges' Save" existed then, which week would he leave? (Expectation: Eliminated by judges in Week 4-5).
* **Case 2 (Season 4): Billy Ray Cyrus** — Test: If switched to Rank system, would he still be 5th?
* **Case 3 (Season 11): Bristol Palin** — Test: Can your new strategy stop her from entering the Top 3?
* **Case 4 (Season 27): Bobby Bones** — Test: In the season he won, if there was a "Safety Mechanism," would the champion change?
* **Value:** This is the most intuitive persuasion. You tell the judges: "Our new rules can correct historical errors."

---

## Phase 4: Pareto Optimization

**Core Task:** Find the optimal balance point between "Fairness" and "Engagement."

### 1. Define Dual Objectives
* **Objective J (Meritocracy):** Correlation between final ranking and judge ranking.
* **Objective F (Engagement):** Correlation between final ranking and fan ranking.

### 2. Draw Pareto Frontier
In the $(J, F)$ coordinate system, mark:
* 🔴 **Current Rule** (Usually in the non-optimal zone).
* 🔵 **Rule with "Judges' Save"** ($J$ increases, $F$ decreases slightly).
* ⭐ **New Recommended Rule** (Located at the "Knee Point" of the frontier).

### Supplement: Pro Dancer & Celebrity Effects Model
**Requirement:** Use data (including estimated fan votes) to quantify the impact of "Pro Dancers" and "Celebrity Features" (Age, Industry, Region, etc.) on results, and answer whether they influence Judge Scores and Fan Votes in the same direction.

1.  **Panel Data Construction:** Observation unit is $(contestant\ i, week\ w)$, containing $J\%(i,w)$, Phase 2 estimated $f(i,w)$, partner Pro, and features like Age/Industry/Region.
2.  **Judge Score Model (Merit channel):** Mixed effects regression
    $$J\%(i,w) = \alpha + \beta_{age}^J \cdot Age_i + \beta_{ind}^J \cdot Industry_i + \beta_{reg}^J \cdot Region_i + b_{pro}^J[partner_i] + b_{celebrity}^J[i] + b_{season}^J[season] + \tau_w + \epsilon$$
    * Where $b_{pro}^J$ measures the systematic lift/drag of the Pro Dancer on judge scores; $\tau_w$ controls overall weekly difficulty.
3.  **Fan Vote Model (Popularity channel):** Logit or Dirichlet/Logistic-Normal hierarchical model on $f(i,w)$:
    $$logit(f(i,w)) = \alpha' + \beta_{age}^F \cdot Age_i + \beta_{ind}^F \cdot Industry_i + \beta_{reg}^F \cdot Region_i + b_{pro}^F[partner_i] + b_{celebrity}^F[i] + b_{season}^F[season] + \tau'_w + \eta \cdot J\%(i,w) + u(i,w)$$
    * Here $\eta$ captures "does dancing better bring more fan votes"; $u(i,w)$ can be a random walk to depict popularity changes over weeks.
4.  **Impact Quantification & Comparison (Must answer the question):**
    * Compare the sign/magnitude of $\beta^J$ and $\beta^F$ for each feature (Do they influence judges and fans in the same direction?).
    * **Variance Decomposition:** Percentage of variance explained by Pro dancer random effects (calculated separately for Judge Scores and Fan Votes).
    * **Optional:** Map model outputs to marginal effects on "Promotion Probability/Final Rank" to answer "How far can these factors take a contestant."

---

## Phase 5: Strategy Recommendation & Memo

### 0. The "Choose One" Conclusion (Rank vs Percentage)
* Use Phase 3 metrics to provide a clear judgment on "whether one method favors fans more" and recommend one based on this.
* **Criterion Example:** If Method A has higher FFI, larger Fan-Elasticity, and more frequent "Fan overwhelms Judge" events, then Method A favors fans more; if the producer prioritizes suppressing extreme canvassing, select the other method.
* **Final:** Provide a clear recommendation in the main text after summarizing across seasons.

### 1. Final Recommendation: Dynamic Log-Weighting
* **Formula:** $Score = (0.5 + 0.05 t) \cdot J\% + (0.5 - 0.05 t) \cdot \log(F\%)$
* **Supporting Mechanism:** Suggest introducing "Judges' Save" only for the Bottom 2.

### 2. Memo to Producer
* **Summarize in non-technical language:**
* **Verifiable Statement of New Rules:** In historical season replays, it should significantly reduce the frequency of "Fans overwhelming Judges causing ranking anomalies" (report your defined extreme event statistics) without significantly lowering fan engagement (e.g., FFI remains close to current rules).

---

## Phase 6: Report Structure & Deliverables

**Hard Deliverables (Required by the problem statement):**
* **Summary Sheet:** One-page overview (Core findings, Recommended rules, Key metrics table).
* **Table of Contents:** With page numbers.
* **References:** Data citations and external sources.
* **1-2 Page Memo:** Suggestions and risk warnings for DWTS producers.
* **AI Use Report:** A separate page explaining LLM usage and human verification responsibility.

**Formatting Suggestions (Max 25 Pages):** Only place the most critical charts in the main text; put large tables/season-by-season comparisons in the appendix or replace with summary charts.

**(Writing Reminder):** Quantitative statements in the Memo should directly cite statistical results from the main text (e.g., difference in weeks, FFI, frequency of extreme events), avoiding untraceable "90%/99%" slogans.

---

## Phase 7: AI Use Report

**Core Task:** Meet compliance requirements (Page 6).

**Mandatory Single Page:**
* **Declaration:** This paper utilized Large Language Models (LLM) for assistance.
* **Specific Uses:**
    * **Ideation:** Used for brainstorming the Pareto optimization framework.
    * **Coding Support:** Assisted in generating Python code snippets for MCMC sampling and Monte Carlo simulations.
    * **Refinement:** Used to check for logical gaps in the paper (e.g., supplementing Consistency metrics).
* **Confirmation:** All mathematical derivations, code execution, and final conclusions were verified and are the responsibility of the human team members.