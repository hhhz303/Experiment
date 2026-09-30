# ============================================================
# 诊断 Seed 3：Test 到 Train 的 Maximum Tanimoto Similarity
# 只做诊断，不重训模型，不改 scaffold_split 实现
# 本脚本只回答 H1 / H2：
#   H1: Seed 3 整体是否更 OOD？
#   H2: Seed 3 的 OOD 是否主要集中于 negative molecules？
# （H3/H4 需要另开脚本，此文件不处理）
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

# 必须与训练时完全相同版本的 scaffold_split
from compare_model_GCN_SAGE_GAT import scaffold_split


# ---------- 1. Morgan 指纹（新版 API） ----------
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_gen.GetFingerprint(mol)


# ---------- 2. 每个 test molecule 到 train set 的 max similarity ----------
def max_sim_to_train(test_smiles, train_smiles):
    """
    返回长度 = len(test_smiles) 的 numpy array。
    无效 SMILES 保留 np.nan，保证与 label 索引对齐。
    """
    train_fps = []
    for smi in train_smiles:
        fp = get_morgan_fp(smi)
        if fp is not None:
            train_fps.append(fp)

    max_sims = []
    for smi in test_smiles:
        fp = get_morgan_fp(smi)
        if fp is None:
            max_sims.append(np.nan)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_sims.append(max(sims))
    return np.array(max_sims)


# ---------- 3. 统计函数 ----------
def similarity_stats(sims):
    sims = sims[~np.isnan(sims)]
    if len(sims) == 0:
        return {}
    return {
        "N":      len(sims),
        "Mean":   float(sims.mean()),
        "Median": float(np.median(sims)),
        "Q1":     float(np.percentile(sims, 25)),
        "Q3":     float(np.percentile(sims, 75)),
        "Min":    float(sims.min()),
        "Max":    float(sims.max()),
        "%<0.3":  float((sims < 0.3).mean() * 100),
        "%<0.4":  float((sims < 0.4).mean() * 100),
        "%<0.5":  float((sims < 0.5).mean() * 100),
    }


# ---------- 4. 主流程 ----------
def analyze_split_similarity(smiles_list, labels, seeds):
    all_sims     = {}
    all_sims_pos = {}
    all_sims_neg = {}
    stats_rows   = []
    detail_rows  = []

    for seed in seeds:
        print(f"\n[相似度分析] 重放 seed={seed} 的 scaffold split ...")
        (train_smiles, _), (_, _), (test_smiles, test_labels) = scaffold_split(
            smiles_list, labels, seed=seed
        )

        # sanity check：与训练时日志对比
        n_pos = int(sum(test_labels))
        n_neg = len(test_labels) - n_pos
        print(f"  test: n={len(test_smiles)}, pos={n_pos}, neg={n_neg}")

        sims = max_sim_to_train(test_smiles, train_smiles)
        labels_arr = np.array(test_labels)

        all_sims[seed]     = sims
        all_sims_pos[seed] = sims[labels_arr == 1]
        all_sims_neg[seed] = sims[labels_arr == 0]

        pos_valid = all_sims_pos[seed][~np.isnan(all_sims_pos[seed])]
        neg_valid = all_sims_neg[seed][~np.isnan(all_sims_neg[seed])]

        row = {"Seed": seed}
        row.update(similarity_stats(sims))

        # --- 正负类对称统计 ---
        row["Median_pos"] = float(np.median(pos_valid)) if len(pos_valid) else np.nan
        row["Q1_pos"]     = float(np.percentile(pos_valid, 25)) if len(pos_valid) else np.nan
        row["%<0.4_pos"]  = float((pos_valid < 0.4).mean() * 100) if len(pos_valid) else np.nan

        row["Median_neg"] = float(np.median(neg_valid)) if len(neg_valid) else np.nan
        row["Q1_neg"]     = float(np.percentile(neg_valid, 25)) if len(neg_valid) else np.nan
        row["%<0.4_neg"]  = float((neg_valid < 0.4).mean() * 100) if len(neg_valid) else np.nan

        stats_rows.append(row)

        # --- 保存 molecule-level 数据，为 H3 预留 ---
        for smi, lab, sim in zip(test_smiles, test_labels, sims):
            detail_rows.append({
                "Seed":          seed,
                "SMILES":        smi,
                "Label":         int(lab),
                "MaxSimilarity": float(sim) if not np.isnan(sim) else np.nan,
            })

    stats_df  = pd.DataFrame(stats_rows).set_index("Seed")
    detail_df = pd.DataFrame(detail_rows)
    return stats_df, detail_df, all_sims, all_sims_pos, all_sims_neg


# ---------- 5. 入口 ----------
if __name__ == "__main__":
    if not os.path.exists("BBBP.csv"):
        raise FileNotFoundError("BBBP.csv 不存在，请先运行 compare_model_GCN_SAGE_GAT.py")

    df = pd.read_csv("BBBP.csv")
    smiles_list = df["smiles"].tolist()
    labels = df["p_np"].tolist()
    seeds = [0, 1, 2, 3, 4]

    print("\n" + "=" * 60)
    print("Seed-level Test-vs-Train Maximum Tanimoto Similarity 分析")
    print("=" * 60)

    stats_df, detail_df, all_sims, all_sims_pos, all_sims_neg = \
        analyze_split_similarity(smiles_list, labels, seeds)

    print("\n========== 各 Seed 相似度统计 ==========")
    print(stats_df.round(4).to_string())

    stats_df.to_csv("similarity_stats_per_seed.csv")
    detail_df.to_csv("similarity_per_molecule.csv", index=False)
    print("\n统计表已保存到 similarity_stats_per_seed.csv")
    print("分子级数据已保存到 similarity_per_molecule.csv")

    # 固定 bins，让所有 seed 的 histogram 可以严格比较
    common_bins = np.linspace(0, 1, 31)

    # ---------- 图 1：5 个 seed 整体分布 ----------
    plt.figure(figsize=(9, 5))
    for seed in seeds:
        sims = all_sims[seed]
        sims = sims[~np.isnan(sims)]
        plt.hist(sims, bins=common_bins, alpha=0.30, density=True, label=f"Seed {seed}")
    plt.xlabel("Max Tanimoto similarity to training set")
    plt.ylabel("Density")
    plt.title("Test-to-Train Maximum Similarity Distribution per Seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("similarity_hist_per_seed.png", dpi=150)
    print("分布图已保存到 similarity_hist_per_seed.png")

    # ---------- 图 2：Seed 3 vs 其他 seed ----------
    plt.figure(figsize=(7, 5))
    other_sims = np.concatenate([all_sims[s] for s in seeds if s != 3])
    other_sims = other_sims[~np.isnan(other_sims)]
    seed3_sims = all_sims[3][~np.isnan(all_sims[3])]

    plt.violinplot([other_sims, seed3_sims], showmedians=True)
    plt.xticks([1, 2], ["Seeds 0/1/2/4", "Seed 3"])
    plt.ylabel("Max Tanimoto similarity")
    plt.title("Seed 3 vs Other Seeds: Test-to-Train Similarity")
    plt.tight_layout()
    plt.savefig("similarity_seed3_vs_others.png", dpi=150)
    print("对比图已保存到 similarity_seed3_vs_others.png")

    # ---------- 图 3：Positive / Negative 分开看 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for seed in seeds:
        sims = all_sims_pos[seed][~np.isnan(all_sims_pos[seed])]
        axes[0].hist(sims, bins=common_bins, alpha=0.30, density=True, label=f"Seed {seed}")
    axes[0].set_title("Positive test molecules")
    axes[0].set_xlabel("Max Tanimoto similarity")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    for seed in seeds:
        sims = all_sims_neg[seed][~np.isnan(all_sims_neg[seed])]
        axes[1].hist(sims, bins=common_bins, alpha=0.30, density=True, label=f"Seed {seed}")
    axes[1].set_title("Negative test molecules")
    axes[1].set_xlabel("Max Tanimoto similarity")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("similarity_hist_by_label.png", dpi=150)
    print("正负样本拆分图已保存到 similarity_hist_by_label.png")