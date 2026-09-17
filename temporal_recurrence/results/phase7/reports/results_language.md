# Claim Language & Rhetorical Guardrail Audit

This document establishes the binding rhetorical rules and vocabulary mappings for the manuscript. All text in the title, abstract, introduction, results, discussion, and conclusion must strictly follow the approved phrasing to prevent overclaiming, maintain precision, and ensure high scientific rigor.

---

## 1. Mandatory Phrase Substitutions

| Danger Level | Disallowed / Dangerous Phrasing (BAD) | Mandatory Scientific Phrasing (GOOD) | Rationale & Justification |
|:---:|---|---|---|
| **CRITICAL** | "TGN forgets history." / "Temporal GNNs inevitably forget." | "Continuously updated recurrent state representations exhibit reduced historical recoverability under conflicting temporal dynamics." | Avoids anthropomorphic/inevitability claims; limits scope to tested recurrent mechanisms under controlled conflicting regimes. |
| **CRITICAL** | "Recency bias is the central cause." | "Temporal memory interference under conflicting dynamics." / "Historical overwriting during conflicting regimes." | "Recency bias" is vague and can be conflated with static recency heuristics. The mechanism is active interference from conflicting updates. |
| **HIGH** | "Information is permanently destroyed." | "Historical information becomes less linearly decodable from the recurrent state and exhibits limited immediate reconstruction upon renewed online exposure without gradient adaptation." | The representation probe and re-exposure experiments show inertia and decodability decline, not mathematical proof of total erasure. |
| **HIGH** | "Retrieval beats temporal GNNs." / "Historical retrieval is a new SOTA model." | "Addressable historical memory recovers predictive information that continuous recurrent state compression fails to retain." | Historical Retrieval is a non-parametric diagnostic probe, not a learned deep architecture. It does not beat Current-Only on hard synthetic data. |
| **HIGH** | "Memory scaling solves / fails to solve forgetting." | "Within the tested capacity range ($d_m \in [16, 256]$), increasing recurrent state dimension does not substantially eliminate long-duration degradation." | Prevents unwarranted extrapolation to infinite-dimensional or differently regularized memory spaces. |
| **MEDIUM** | "Proves temporal memory interference." | "Provides convergent empirical evidence consistent with temporal memory interference." | Empirical benchmarking and probing provide strong convergent evidence, not deductive mathematical proof. |
| **MEDIUM** | "The specificity control proves degradation only occurs when history is required." | "The specificity control shows that regime transitions themselves disrupt continuous recurrent representations, even when the current snapshot is highly predictive." | Rigorous adherence to data: TGN achieves only $0.5427$ AP on the current-sufficient control ($\lambda_A=0.85$, Current-Only $0.9506$). |
| **MEDIUM** | "EdgeBank fails on dynamic graphs." | "EdgeBank relies on exact edge repetition; when latent structural recurrence occurs with fresh edge instances, structural historical retrieval retains predictive advantage over exact memorization." | Acknowledges EdgeBank's strong performance on exact recurrence while clarifying where structural retrieval provides distinct value. |
| **MEDIUM** | "CollegeMsg proves real-world forgetting." | "CollegeMsg provides qualitative supporting evidence across $n=4$ identified recurrence episodes, where exact interaction recurrence dominates." | Accurately states the statistical unit ($n=4$ episodes) without making ungrounded population-level generalizations. |

---

## 2. Core Terminology Definitions

1. **Temporal Memory Interference**: The degradation in the utility of a continuously updated recurrent state representation for predicting dynamics under a recurring regime $A$, caused by intervening updates from a conflicting regime $B$.
2. **Historical Overwriting under Conflicting Dynamics**: The displacement or distortion of latent historical regime information in recurrent node memory vectors as new, contradictory message vectors are integrated over time.
3. **Historical Recoverability**: The empirical fraction of predictive information from an earlier temporal regime that can be extracted to forecast future recurring events, measured via link prediction metrics (Average Precision) relative to informed baselines.
4. **Non-Parametric Diagnostic Probe**: A baseline (e.g., Historical Retrieval or Random Retrieval) deployed solely to measure the presence and extractability of information in raw graph snapshots, rather than as a proposed deployable machine learning architecture.

---

## 3. Strict Prohibitions Checklist

- [x] **No Claim of Universal Impossibility**: Never state that finite-memory GNNs "cannot" represent recurrent regimes in principle.
- [x] **No Universal Superiority Claims**: Never claim that snapshot retrieval is universally superior to continuous temporal models across all graph tasks.
- [x] **No Concealment of Negative / Nuanced Controls**:
  - Current-Only ($0.7635$) > Historical Retrieval ($0.7273$) on hard synthetic data must be explicitly reported and explained.
  - TGN failure on the current-sufficient benchmark ($0.5427$ vs. $0.9506$) must be framed as evidence of transition-induced representational disruption.
  - EdgeBank dominance on CollegeMsg ($0.8763$ vs. $0.7002$) must be highlighted as proof of exact edge recurrence in real email/message logs.
