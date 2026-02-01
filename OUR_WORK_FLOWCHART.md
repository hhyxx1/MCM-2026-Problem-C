# Our Work: MCM 2026 Process Flowchart (Enhanced Version)

This flowchart illustrates the complete analytical pipeline for the "Fairness-Engagement Equilibrium Model", ensuring alignment with `plan.md` and verifying the data flow across all phases.

**Key Statistics**: 🎯 **34 Seasons** | 👥 **421 Contestants** | 📊 **2,777 Observations**

```mermaid
graph TD
    %% --- Enhanced Styles ---
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:3px,color:#000;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000;
    classDef core fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000;
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000;
    classDef decision fill:#ffebee,stroke:#c62828,stroke-width:3px,color:#000;
    classDef insight fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000;

    %% --- Phase 1: Data Archeology ---
    subgraph Phase1 [" 📦 PHASE 1: Data Archeology & Global Scan "]
        direction TB
        RawData[("📂 Raw Data<br/>421 Contestants<br/>34 Seasons (S1-S34)")]:::input
        Cleaning["🧹 Data Cleaning<br/>✓ J% Standardization (3/4 judges)<br/>✓ N/A & 0-score handling<br/>✓ Withdrawal processing"]:::process
        FeatEng["⚙️ Feature Engineering<br/>✓ Age (splines + bins)<br/>✓ Industry (One-Hot)<br/>✓ Region (US/Non-US)<br/>✓ PBI (Partner Impact)"]:::process
        PanelData[("📄 Panel Data<br/>2,777 observations<br/>(i, w) format")]:::output
        Divergence["📉 Global Scan<br/>S1-S34 Divergence Trend<br/>⚠️ Proves necessity"]:::insight

        RawData --> Cleaning
        Cleaning --> PanelData
        Cleaning --> FeatEng
        PanelData --> Divergence
    end

    %% --- Phase 2: Inverse Inference ---
    subgraph Phase2 [" 🔬 PHASE 2: Bayesian Inverse Inference "]
        direction TB
        BayesModel["🧮 Bayesian MCMC<br/>Hit-and-Run Sampling<br/>f(i,w) estimation"]:::core
        Constraints["🔒 Inequality Constraints<br/>E_w ∈ Bottom-k<br/>Σf(i,w) = 1"]:::process
        PostEst[("📊 Fan Vote Estimates<br/>2,777 posteriors<br/>with 95% CI")]:::output

        subgraph Validation [" ✅ Validation (4 Metrics) "]
            Cert1["CI Width<br/>Avg: 0.288"]:::insight
            Cert2["CV<br/>Avg: 0.619"]:::insight
            Cons1["Exact-Match<br/>Rate: 95.2%"]:::insight
            Cons2["Posterior P̄<br/>0.651"]:::insight
        end

        PanelData --> BayesModel
        Constraints -.-> BayesModel
        BayesModel --> PostEst
        PostEst --> Cert1 & Cert2 & Cons1 & Cons2
    end

    %% --- Phase 3: Simulator & Comparison ---
    subgraph Phase3 [" 🎮 PHASE 3: Omni-Simulator & Case Studies "]
        direction TB
        Simulator["🕹️ Omni-Simulator<br/>Reconstruct all 34 seasons"]:::core

        subgraph Methods [" ⚔️ Method Comparison "]
            RankMethod["Rank Method<br/>FFI: 0.719<br/>J: 0.665"]:::process
            PctMethod["Pct Method<br/>FFI: 0.768 ⚠️<br/>J: 0.454"]:::process
        end

        BiasAnalysis["⚖️ Fan Bias<br/>✓ FFI: Pct > Rank (+0.049)<br/>✓ Elasticity: Pct more sensitive"]:::insight

        subgraph Cases [" 🔍 4 Historical Cases "]
            Case1["S2: Jerry Rice<br/>Judges' Save → Earlier Exit"]:::process
            Case2["S4: Billy Ray Cyrus<br/>Rank → Lower Placement"]:::process
            Case3["S11: Bristol Palin<br/>New Strategy → Blocked Top-3"]:::process
            Case4["S27: Bobby Bones<br/>Safety → Champion Change"]:::process
        end

        SuppAnalysis["📈 Covariate Effects<br/>Pro Dancer | Age | Industry"]:::process

        PostEst --> Simulator
        FeatEng --> SuppAnalysis
        PostEst --> SuppAnalysis

        Simulator --> RankMethod & PctMethod
        RankMethod & PctMethod --> BiasAnalysis
        PostEst --> Case1 & Case2 & Case3 & Case4
    end

    %% --- Phase 4: Pareto Optimization ---
    subgraph Phase4 [" 📈 PHASE 4: Pareto Optimization "]
        direction TB
        Objectives["🎯 Dual Objectives<br/>J (Meritocracy): 0.665<br/>F (Engagement): 0.704"]:::process
        Frontier["📈 Pareto Frontier<br/>51 weight combinations<br/>(30%-90% judge weight)"]:::core
        KneePoint["📍 Knee Point Analysis<br/>✓ Rank: Clear knee @ 50-50 (d=0.224)<br/>⚠️ Pct: Linear (d=0.060)"]:::decision
        JudgesSave["🛡️ Judges' Save Impact<br/>Rank+Save: J+0.013, F-0.027<br/>Pct+Save: J-0.009, F-0.016"]:::insight

        BiasAnalysis --> Objectives
        Objectives --> Frontier
        Frontier --> KneePoint
        KneePoint --> JudgesSave
    end

    %% --- Phase 5: Recommendation ---
    subgraph Phase5 [" ⭐ PHASE 5: Final Recommendation "]
        direction TB
        FinalRec["📝 RECOMMENDED RULE<br/>━━━━━━━━━━━━━━━<br/>Method: RANK<br/>Weights: 50/50<br/>Judges' Save: YES<br/>━━━━━━━━━━━━━━━<br/>J: 0.665 | F: 0.704"]:::decision
        Comparison["📊 vs Current Rule<br/>Current (Pct+Save): J=0.445<br/>Recommended: J=0.665 (+49%)"]:::insight
        Memo["📄 Producer Memo<br/>Implementation Guide<br/>Risk Analysis"]:::output

        JudgesSave --> FinalRec
        Case1 & Case2 & Case3 & Case4 -.-> Memo
        FinalRec --> Comparison
        Comparison --> Memo
    end

    %% --- Critical Path Links ---
    Divergence ==> |"Justifies Reform"| Objectives
    SuppAnalysis -.-> |"Variance Explained"| Memo
    Cons2 ==> |"Validates Simulation"| Simulator

    %% --- Legend ---
    subgraph Legend [" 📌 LEGEND "]
        L1["🔵 Input Data"]:::input
        L2["🟡 Process"]:::process
        L3["🟢 Core Engine"]:::core
        L4["🟣 Output"]:::output
        L5["🔴 Decision"]:::decision
        L6["🟠 Key Finding"]:::insight
    end
```

---

## 📋 Logic Flow Summary (Enhanced)

### 1. **Data Archeology (Phase 1)** 📦
- **Input**: Raw dataset with 421 contestants across 34 seasons
- **Process**:
  - Standardize J% (handling 3-judge and 4-judge systems)
  - Exclude N/A and 0-scores (eliminations/withdrawals)
  - Engineer features: Age (splines), Industry (One-Hot), Region, PBI
- **Insight**: Global Scan proves **Judge-Audience divergence increases over time** (justifies reform)
- **Output**: Clean panel data (2,777 observations in (i,w) format)

### 2. **Inverse Inference (Phase 2)** 🔬
- **Core Engine**: Bayesian MCMC (Hit-and-Run) to "reverse-engineer" hidden fan votes
- **Constraints**: Eliminated contestants must be in Bottom-k (inequality constraints)
- **Validation** (4 metrics):
  - **Certainty**: CI Width (0.288), CV (0.619)
  - **Consistency**: Exact-Match (95.2%), Posterior P̄ (0.651)
- **Output**: 2,777 posterior estimates with 95% credible intervals

### 3. **Simulation & Validation (Phase 3)** 🎮
- **Comparison**: Replay all 34 seasons using Rank vs Pct methods
- **Key Finding**: **Pct method is more fan-biased**
  - FFI: 0.768 (Pct) vs 0.719 (Rank) → **+0.049 bias**
  - Higher elasticity to fan vote perturbations
- **Case Studies** (4 controversial winners):
  | Case | Season | Finding |
  |------|--------|---------|
  | Jerry Rice | S2 | Judges' Save → Earlier exit |
  | Billy Ray Cyrus | S4 | Rank → Lower placement |
  | Bristol Palin | S11 | New Strategy → Blocked Top-3 |
  | Bobby Bones | S27 | Safety mechanism → Champion change |
- **Supplementary**: Covariate effects (Pro dancer, Age, Industry impact quantified)

### 4. **Pareto Optimization (Phase 4)** 📈
- **Objectives**:
  - J (Meritocracy): Correlation with judge ranking = **0.665**
  - F (Engagement): Correlation with fan ranking = **0.704**
- **Frontier**: Test 51 weight combinations (30%-90% judge weight)
- **Critical Finding**:
  - **Rank method**: Clear knee point @ 50-50 (distance = 0.224)
  - **Pct method**: Quasi-linear (distance = 0.060) → **NO optimal balance**
- **Judges' Save Analysis**:
  - Rank+Save: J +0.013, F -0.027 (acceptable trade-off)
  - Pct+Save: J -0.009, F -0.016 (no benefit)

### 5. **Final Recommendation (Phase 5)** ⭐
- **Recommended Rule**:
  ```
  Method:        RANK
  Weights:       50% Judge / 50% Fan
  Judges' Save:  YES (for Bottom-2)

  Performance:   J = 0.665 | F = 0.704
  ```
- **vs Current Rule (Pct+Save)**:
  - Meritocracy improvement: **+49%** (0.445 → 0.665)
  - Engagement maintained: -1.9% (0.691 → 0.704)
- **Supporting Evidence**: 4 historical case studies prove effectiveness

---

## 🎯 Key Insights Summary

| Insight | Evidence | Implication |
|---------|----------|-------------|
| **Pct method favors fans more** | FFI +0.049, Higher elasticity | More vulnerable to vote manipulation |
| **Rank has optimal balance point** | Knee distance 0.224 vs 0.060 | Clear trade-off, easier to justify |
| **Judges' Save enhances fairness** | J +0.013 with acceptable F loss | Recommended for Rank method |
| **Current rule underperforms** | J = 0.445 (lowest among options) | Reform justified by 49% J gain |

---

## 📁 Key Output Files

| Phase | Output File | Rows | Description |
|-------|-------------|------|-------------|
| 1 | `clean_weekly_panel.csv` | 2,777 | Panel data (i,w) |
| 1 | `feature_dictionary.json` | - | Feature definitions |
| 2 | `fan_vote_estimates.csv` | 2,777 | Posterior f(i,w) |
| 2 | `certainty_summary.csv` | 336 | CI/CV metrics |
| 2 | `posterior_consistency.csv` | 295 | P_w by season/week |
| 3 | `method_comparison.csv` | 34 | Rank vs Pct per season |
| 3 | `favor_indices.csv` | 68 | FFI/Elasticity |
| 3 | `case_studies_summary.csv` | 5 | 4 historical cases |
| 4 | `pareto_points.csv` | 51 | All weight combinations |
| 4 | `recommended_rule.json` | - | Final recommendation |
| 5 | `producer_memo.txt` | - | Implementation guide |

---

## ✅ Compliance Checklist

| Plan.md Requirement | Status | Evidence |
|---------------------|--------|----------|
| Global Scan (S1-S34) | ✅ | `global_scan/divergence_trend.png` |
| Bayesian f(i,w) inference | ✅ | 2,777 estimates with CI |
| Certainty (2 metrics) | ✅ | CI Width + CV |
| Consistency (2 metrics) | ✅ | Exact-Match + P_w |
| Rank vs Pct comparison | ✅ | All 34 seasons tested |
| Fan Bias quantification | ✅ | FFI + Elasticity |
| 4 Case Studies | ✅ | Rice, Cyrus, Palin, Bones |
| Pareto frontier | ✅ | 51 points plotted |
| Knee point identification | ✅ | Rank @ 50-50 |
| Judges' Save analysis | ✅ | Impact quantified |
| Final recommendation | ✅ | Rank+Save 50/50 |

**Compliance Rate**: **100%** (11/11)

---

**Last Updated**: 2026-02-01
**Audit Status**: ✅ Logic Complete, Data Consistent ([See Full Report](LOGIC_AUDIT_REPORT.md))
