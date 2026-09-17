# Phase 11: Final Claim-by-Claim Integrity Audit

**Status**: 100% Verified  
**Auditor**: Automated Scientific Consistency Suite

---

## 1. Audit Matrix of Major Scientific Claims

| # | Scientific Claim | Status | Source Evidence | Scope / Approved Language |
| :---: | :--- | :---: | :--- | :--- |
| **1** | Continuous recurrent representations degrade with distractor duration ($T_B$). | **SUPPORTED** | `table_a_baseline_reproduction.csv` | Scoped: "duration-dependent degradation under conflicting distractor regimes." |
| **2** | Capacity scaling ($d_m$) does not resolve temporal interference. | **SUPPORTED** | `fig3_capacity_scaling.pdf` & Table E | Scoped: "$d_m=64 \to 256$ increases parameters by $412\%$ with negligible AP gain ($+0.0076$)." |
| **3** | Historical predictive structure remains in uncorrupted graph history. | **SUPPORTED** | `table_a_baseline_reproduction.csv` | Historical Retrieval Probe achieves $0.7588$ vs. Continuous TGN $0.6523$. |
| **4** | Exact vs. structural recurrence impose distinct demands. | **SUPPORTED** | `table_f_exact_vs_structural.csv` | EdgeBank collapses under Condition B ($0.5774$), where structural retrieval retains $+0.0351$ AP advantage. |
| **5** | Addressable episodic memory provides historical access at onset. | **SUPPORTED** | `table_g_reexposure_dynamics.csv` | MA-TGN restores evaluated link prediction at recurrence onset ($k_A=0$) without adaptation delay. |
| **6** | MA-TGN improves over Continuous TGN in real-world recurrence. | **SUPPORTED** | `table_h_collegemsg_episodes.csv` & `table_i_bitcoin_otc_episodes.csv` | Across $n=4$ episodes: CollegeMsg ($0.7135$ vs. $0.6552$), Bitcoin-OTC ($0.6913$ vs. $0.5670$). |
| **7** | EdgeBank is superior when exact pairwise edges repeat. | **SUPPORTED** | `table_h_collegemsg_episodes.csv` & `table_i_bitcoin_otc_episodes.csv` | Explicitly reported: EdgeBank achieves $0.8763$ on CollegeMsg and $0.7753$ on Bitcoin-OTC. |
| **8** | Learned attention necessity claim. | **REMOVED / WEAKENED** | `table_addressing_ablation.csv` | Honestly acknowledged that cosine key retrieval ($0.6519$) matches learned attention ($0.6518$). |
| **9** | Complete elimination of forgetting claim. | **REMOVED** | Phase 10.5 Audit | Forbidden hyperbole completely excised from all manuscript text. |
| **10** | Mathematical contraction theorem claim. | **REMOVED** | Phase 10.5 Audit | Unproven theoretical conjecture completely removed; framed purely as empirical observation. |

---

## 2. Summary
- Total Major Claims Audited: **10**
- Unsupported Claims Removed: **3**
- Claims Scoped / Delimited: **4**
- Fully Verified Claims: **7**
- Discrepancies: **0**
