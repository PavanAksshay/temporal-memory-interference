# Audit 1: Canonical TGN Implementation Audit

This document audits the implementation in `src/models/tgn.py`, `src/models/memory.py`, and `src/models/time_encoder.py` against the canonical TGN formulation (Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs*, arXiv:2006.10637, 2020).

## 1. Component-by-Component Comparison

| Component | Canonical TGN Formulation | Repository Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Time Encoding** | $\phi_d(\Delta t) = \cos(\mathbf{\omega}_d \Delta t + \mathbf{b}_d)$ harmonic basis | `TimeEncoder` with learnable frequency $\mathbf{w}$ and cosine harmonic projection | **FAITHFUL** |
| **Node Memory Initialization** | $\mathbf{s}_i(0) = \mathbf{0}$ | `MemoryBank.reset_memory()` initializes buffer to zeros | **FAITHFUL** |
| **Message Construction** | $\mathbf{m}_i(t) = \text{MLP}([\mathbf{s}_i(t^-), \mathbf{s}_j(t^-), \Delta t, \mathbf{e}_{ij}])$ | `MessageFunction` concatenates node representations, time encoding, and edge features through a 2-layer MLP | **FAITHFUL** |
| **Message Aggregation** | Mean or Last interaction aggregation per batch | Mean aggregation via `torch.Tensor.index_add_` normalized by node interaction count | **FAITHFUL** |
| **Memory Update Timing** | Updates occur **after** computing edge predictions for the current batch (preventing target leakage) | Strictly executed after link logit prediction in both training and evaluation rollouts | **FAITHFUL** |
| **Memory Updater** | $\mathbf{s}_i(t) = \text{GRUCell}(\mathbf{\bar{m}}_i(t), \mathbf{s}_i(t^-))$ | `MemoryUpdater` uses standard `nn.GRUCell(message_dim, memory_dim)` | **FAITHFUL** |
| **Node Embedding Computation** | $\mathbf{h}_i(t) = [\mathbf{x}_i, \mathbf{s}_i(t)]$ | `TGN.compute_node_representations` combines `nn.Embedding` with dynamic memory buffer | **FAITHFUL** |
| **Link Decoder** | $\hat{y}_{ij}(t) = \sigma(\text{MLP}([\mathbf{h}_i(t), \mathbf{h}_j(t)]))$ | 2-layer MLP with ReLU and sigmoid / logit outputs | **FAITHFUL** |
| **Memory Detachment** | Memory states are detached across batches during BPTT | `updated_memory.detach()` applied at every update | **FAITHFUL** |
| **Batching & Negative Sampling** | Uniform random 1:1 negative edge sampling per timestep | `extract_events_from_sequence` generates exactly 1 negative non-edge per positive interaction | **FAITHFUL** |

## 2. Conclusion
The implementation faithfully represents the canonical continuous-time Temporal Graph Network. No anomalous or non-standard architectural choices are present.
