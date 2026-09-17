# Results Language and Framing Audit

To maintain scientific integrity and prevent reviewer objections regarding overclaiming, all text in the paper must adhere to the audited language mappings below.

---

## 1. Core Language Transformations

| # | Topic / Result | BAD (Overclaiming / Vulnerable) | GOOD (Defensible / Scientifically Precise) | Rationale |
|---|---|---|---|---|
| **1** | Continuous TGN Recurrence Behavior | *"TGN cannot remember historical information."* / *"TGN suffers from catastrophic forgetting."* | *"Under the tested recurring-dynamics benchmark, the continuously updated TGN representation loses access to historically predictive information as the conflicting regime lengthens."* | Restricts the claim to tested architectures, benchmarks, and the explicit distractor-duration relationship. |
| **2** | Historical Retrieval Superiority | *"Historical retrieval outperforms temporal GNNs."* | *"Addressable historical retrieval recovers predictive information that continuous recurrent state representations fail to retain under the tested recurring dynamics."* | Prevents claiming universal superiority over all GNNs on arbitrary dynamic benchmarks. |
| **3** | Recurrent Capacity / Dimension Scaling | *"Memory dimension does not matter for temporal GNNs."* | *"Within the tested 16–256 dimensional range, increasing recurrent state dimension does not substantially eliminate degradation at long distractor durations."* | Bounded to tested finite dimensions without implying asymptotic impossibility theorems. |
| **4** | Theoretical Generality of Forgetting | *"This proves a fundamental law of catastrophic forgetting in temporal graphs."* | *"The results demonstrate temporal memory interference / historical overwriting under conflicting dynamics in continuous recurrent graph models."* | Retires sweeping theoretical claims in favor of empirical characterization. |
| **5** | Historical Oracle Status | *"The Historical Oracle is the theoretical upper bound for all models."* | *"The Historical Oracle represents the benchmark oracle under the specified historical information definition."* | Acknowledges that the oracle is defined by the generative Dynamic SBM affinity matrix. |
| **6** | EdgeBank vs. Structural Retrieval | *"EdgeBank fails because it only memorizes edges."* | *"Exact historical edge memorization (EdgeBank) explains a substantial fraction (~65%) of the historical gain, but structural snapshot retrieval captures additional latent community structure in evolving synthetic networks."* | Accurately credits EdgeBank's strong performance while delineating structural retrieval's incremental advantage. |
| **7** | Real-World Generalization (SNAP CollegeMsg) | *"We prove that real-world social networks always require historical retrieval."* | *"Across n=4 independently discovered recurrence episodes in SNAP CollegeMsg, addressable historical storage provides consistent empirical advantages over continuous recurrent tracking, with exact edge recurrence accounting for the majority of the gain."* | Clarifies that n=4 is qualitative supporting evidence, not population-level statistical proof. |
| **8** | Mechanism Diagnosis | *"TGN fails because the optimization algorithm is broken."* | *"Ablation analyses (memory reset, random shuffle, and TGN-NoMemory) confirm that performance collapse stems from active state corruption induced by conflicting dynamics rather than optimization failure or static architectural bias."* | Directly supported by our diagnostic intervention experiments. |
| **9** | Terminology Choice | *"Recency bias in temporal GNNs."* | *"Temporal memory interference / Historical overwriting under conflicting regimes."* | Avoids anthropomorphic terminology and aligns with representation learning literature. |
| **10** | Retrieval Scope | *"We propose a new state-of-the-art neural retrieval architecture."* | *"We employ non-parametric snapshot retrieval as a diagnostic probe to isolate the recoverability of historical information."* | Prevents reviewers from demanding complex learned transformer retrieval architectures. |

---

## 2. Mandatory Terminology Standards

- **Primary Phenomenon**: Use *Temporal Memory Interference* or *Historical Overwriting under Conflicting Dynamics*.
- **Task Framing**: Use *Historical Information Recoverability under Regime Recurrence*.
- **Recurrent State Model**: Refer to *Continuously updated recurrent node representations* (e.g., TGN with GRU memory).
- **Non-Parametric Memory**: Refer to *Addressable historical memory* (or *structural snapshot retrieval* and *exact edge storage*).
- **Oracle**: Refer to *Benchmark historical oracle*.
