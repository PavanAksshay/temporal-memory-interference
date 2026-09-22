import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/pavanaksshay/se_research/temporal_recurrence')

from src.generator.dsbm import DynamicSBMGenerator
from src.generator.regimes import RegimeConfig

csv_dir = '/Users/pavanaksshay/se_research/temporal_recurrence/data/csv'
real_dir = '/Users/pavanaksshay/se_research/temporal_recurrence/data/real'
os.makedirs(csv_dir, exist_ok=True)

print("--- 1. Converting Real Datasets to CSV ---")

# 1. CollegeMsg
collegemsg_txt = os.path.join(real_dir, 'CollegeMsg.txt')
collegemsg_csv_real = os.path.join(real_dir, 'CollegeMsg.csv')
collegemsg_csv_all = os.path.join(csv_dir, 'CollegeMsg.csv')

df_college = pd.read_csv(collegemsg_txt, sep=r'\s+', names=['src', 'dst', 'timestamp'])
df_college.to_csv(collegemsg_csv_real, index=False)
df_college.to_csv(collegemsg_csv_all, index=False)
print(f"Exported CollegeMsg.csv: {len(df_college):,} rows")

# 2. BitcoinOTC
bitcoin_txt = os.path.join(real_dir, 'BitcoinOTC.txt')
bitcoin_csv_real = os.path.join(real_dir, 'BitcoinOTC.csv')
bitcoin_csv_all = os.path.join(csv_dir, 'BitcoinOTC.csv')

df_bitcoin = pd.read_csv(bitcoin_txt, sep=r'\s+', names=['src', 'dst', 'timestamp'])
df_bitcoin.to_csv(bitcoin_csv_real, index=False)
df_bitcoin.to_csv(bitcoin_csv_all, index=False)
print(f"Exported BitcoinOTC.csv: {len(df_bitcoin):,} rows")

print("\n--- 2. Generating Synthetic DSBM Datasets to CSV ---")

def export_sequence_to_csv(seq, filename):
    filepath = os.path.join(csv_dir, filename)
    rows = []
    T = seq.snapshots.shape[0]
    for t in range(T):
        rec = seq.scheduler.get_record(t)
        adj = seq.snapshots[t]
        srcs, dsts = np.where(np.triu(adj, k=1) == 1)
        for u, v in zip(srcs, dsts):
            rows.append({
                'timestep': t,
                'src': int(u),
                'dst': int(v),
                'regime': rec.regime,
                'episode_index': rec.episode_index,
                'time_since_transition': rec.time_since_transition
            })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Exported {filename}: {len(df):,} edges across {T} timesteps")
    return filepath

# Standard benchmark generator
base_configs = {
    "A": RegimeConfig(
        name="A",
        target_density=0.10,
        persistence=0.35,
        within_comm_multiplier=4.0,
        partition_seed=101
    ),
    "B": RegimeConfig(
        name="B",
        target_density=0.10,
        persistence=0.15,
        within_comm_multiplier=4.0,
        partition_seed=202
    ),
    "C": RegimeConfig(
        name="C",
        target_density=0.10,
        persistence=0.25,
        within_comm_multiplier=4.0,
        partition_seed=303
    )
}

gen = DynamicSBMGenerator(num_nodes=300, num_communities=3, regime_configs=base_configs)

for TB in [25, 50, 100, 200]:
    seq_spec = [("A", 100), ("B", TB), ("A", 50)]
    seq = gen.generate(seq_spec, seed=42)
    export_sequence_to_csv(seq, f"synthetic_dsbm_TB{TB}.csv")

# Condition A (High persistence / exact edge recurrence: lambda_A = 0.70)
configs_condA = {
    "A": RegimeConfig(name="A", target_density=0.10, persistence=0.70, within_comm_multiplier=4.0, partition_seed=101),
    "B": RegimeConfig(name="B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202)
}
gen_A = DynamicSBMGenerator(num_nodes=300, num_communities=3, regime_configs=configs_condA)
seq_condA = gen_A.generate([("A", 100), ("B", 100), ("A", 50)], seed=42)
export_sequence_to_csv(seq_condA, "synthetic_dsbm_condition_A_exact.csv")

# Condition B (Low persistence / structural community recurrence: lambda_A = 0.05)
configs_condB = {
    "A": RegimeConfig(name="A", target_density=0.10, persistence=0.05, within_comm_multiplier=4.0, partition_seed=101),
    "B": RegimeConfig(name="B", target_density=0.10, persistence=0.15, within_comm_multiplier=4.0, partition_seed=202)
}
gen_B = DynamicSBMGenerator(num_nodes=300, num_communities=3, regime_configs=configs_condB)
seq_condB = gen_B.generate([("A", 100), ("B", 100), ("A", 50)], seed=42)
export_sequence_to_csv(seq_condB, "synthetic_dsbm_condition_B_structural.csv")

# Control C (Novel regime C: A -> B -> C)
seq_controlC = gen.generate([("A", 100), ("B", 100), ("C", 50)], seed=42)
export_sequence_to_csv(seq_controlC, "synthetic_dsbm_control_C_novel.csv")

print("\n--- All CSV datasets generated successfully! ---")
