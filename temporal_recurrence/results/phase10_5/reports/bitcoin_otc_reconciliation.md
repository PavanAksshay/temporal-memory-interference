# Phase 10.5: SNAP Bitcoin-OTC Reconciliation Report

**Dataset**: SNAP Bitcoin-OTC (Who-trusts-whom dynamic network, 5,881 nodes, 35,592 transactions).  
**Protocol**: 4 Macro Recurrence Episodes across financial trust cycles.  
**Source Artifact**: `results/phase10/processed/table_i_bitcoin_otc_episodes.csv`

---

## 1. Episode-by-Episode Performance Matrix

| Episode | Current-Only | EdgeBank | Hist. Retrieval | Continuous TGN | MA-TGN (Proposed) | $\Delta\text{AP}$ (MA-TGN $-$ TGN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Episode 1** | 0.6420 | 0.7812 | 0.6840 | 0.5750 | 0.7020 | **+0.1270** |
| **Episode 2** | 0.5980 | 0.7430 | 0.6350 | 0.5340 | 0.6510 | **+0.1170** |
| **Episode 3** | 0.6710 | 0.8120 | 0.7100 | 0.6010 | 0.7280 | **+0.1270** |
| **Episode 4** | 0.6250 | 0.7650 | 0.6620 | 0.5580 | 0.6840 | **+0.1260** |
| **Mean** | **0.6340** | **0.7753** | **0.6728** | **0.5670** | **0.6913** | **+0.1243** |
| **Std** | 0.0306 | 0.0290 | 0.0319 | 0.0282 | 0.0323 | 0.0049 |
| **Median** | 0.6335 | 0.7731 | 0.6730 | 0.5665 | 0.6930 | +0.1265 |

---

## 2. Key Scientific Findings on Bitcoin-OTC

1. **Substantial Neural Gain over Continuous TGN**: MA-TGN improves over Continuous TGN by an average of **$\Delta\text{AP} = +0.1243$** ($+12.43\%$, range: $+0.1170$ to $+0.1270$), demonstrating that episodic memory recovers dormant trust relations that continuous recurrent state updates have overwritten.
2. **MA-TGN vs. Historical Retrieval**: MA-TGN outperforms Historical Retrieval Probes ($0.6913$ vs. $0.6728$, $+1.85\%$), showing that neural node memory combined with episodic retrieval provides superior structural adaptation over pure graph common-neighbor matching.
3. **EdgeBank Memorization**: EdgeBank achieves $0.7753$, again demonstrating that exact pair memorization is extremely potent in networks with repeated entity interactions.
