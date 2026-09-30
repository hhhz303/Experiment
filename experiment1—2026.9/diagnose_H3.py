# ============================================================
# 验证 H3：Test molecule 与训练集的化学相似度越低，模型预测越差。
# 修改点（相对上一版）：
#   (1) 加入 reproduction gate：对比复现 Test AUC 与原始 AUC
#   (2) 检查重复 SMILES，若存在则警告
#   (3) 加入 per-seed Spearman，控制 split-level confounding
# ============================================================

import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from rdkit import Chem

from compare_model_GCN_SAGE_GAT import (
    scaffold_split,
    MoleculeDataset,
    GCNModel,
    GraphSAGEModel,
    GATModel,
)
from torch_geometric.loader import DataLoader


# ============================================================
# 0. 原始 AUC（来自之前 5-seed 实验的日志，用作复现对照）
# ============================================================
ORIGINAL_AUC = {
    0: {"GCN": 0.9167, "GraphSAGE": 0.9186, "GAT": 0.8938},
    1: {"GCN": 0.9077, "GraphSAGE": 0.8899, "GAT": 0.8941},
    2: {"GCN": 0.9090, "GraphSAGE": 0.9187, "GAT": 0.9123},
    3: {"GCN": 0.6952, "GraphSAGE": 0.5681, "GAT": 0.5271},
    4: {"GCN": 0.8963, "GraphSAGE": 0.8968, "GAT": 0.8972},
}


# ============================================================
# 1. 训练 + 预测函数
# ============================================================
def train_and_predict(model, train_loader, val_loader, test_loader,
                      device, epochs=100, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_model_state = None
    counter = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1)
            loss = criterion(out, batch.y.view(-1).float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch).view(-1)
                preds = torch.sigmoid(out)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch.y.view(-1).cpu().numpy())
        val_auc = roc_auc_score(val_labels, val_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch).view(-1)
            preds = torch.sigmoid(out)
            test_probs.extend(preds.cpu().numpy())
            test_labels.extend(batch.y.view(-1).cpu().numpy())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    test_auc = roc_auc_score(test_labels, test_probs)

    return test_probs, test_labels, best_val_auc, test_auc


# ============================================================
# 2. 重新训练并保存逐分子预测 + 复现验证
# ============================================================
def run_predictions(smiles_list, labels, seeds, device):
    HIDDEN = 64
    NUM_LAYERS = 3
    GAT_HEADS = 2

    model_builders = {
        "GCN":       lambda in_ch: GCNModel(in_ch, hidden_channels=HIDDEN, num_layers=NUM_LAYERS),
        "GraphSAGE": lambda in_ch: GraphSAGEModel(in_ch, hidden_channels=HIDDEN, num_layers=NUM_LAYERS),
        "GAT":       lambda in_ch: GATModel(in_ch, hidden_channels=HIDDEN, num_layers=NUM_LAYERS, heads=GAT_HEADS),
    }

    all_rows = []
    repro_rows = []

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n========== Seed {seed} ==========")

        (train_smiles, train_labels), (val_smiles, val_labels), (test_smiles, test_labels) = \
            scaffold_split(smiles_list, labels, seed=seed)

        # ---- 过滤无效 SMILES，与 mol_to_graph 保持一致 ----
        valid_test_smiles, valid_test_labels = [], []
        for smi, lab in zip(test_smiles, test_labels):
            if Chem.MolFromSmiles(smi) is None:
                continue
            valid_test_smiles.append(smi)
            valid_test_labels.append(lab)

        train_dataset = MoleculeDataset(train_smiles, train_labels)
        val_dataset   = MoleculeDataset(val_smiles, val_labels)
        test_dataset  = MoleculeDataset(valid_test_smiles, valid_test_labels)

        in_channels = train_dataset[0].x.shape[1]

        for model_idx, (name, builder) in enumerate(model_builders.items()):
            model_seed = seed * 100 + model_idx
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)
            random.seed(model_seed)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

            model = builder(in_channels).to(device)

            test_probs, test_labels_arr, best_val_auc, test_auc = train_and_predict(
                model, train_loader, val_loader, test_loader, device,
                epochs=100, patience=20
            )

            orig_auc = ORIGINAL_AUC[seed][name]
            repro_rows.append({
                "Seed": seed,
                "Model": name,
                "Original_AUC": orig_auc,
                "Reproduced_AUC": test_auc,
                "Delta": test_auc - orig_auc,
            })

            print(f"  {name:10s} | Val AUC: {best_val_auc:.4f} | "
                  f"Test AUC: {test_auc:.4f} (orig {orig_auc:.4f}, "
                  f"Δ={test_auc-orig_auc:+.4f})")

            for smi, lab, p in zip(valid_test_smiles, valid_test_labels, test_probs):
                all_rows.append({
                    "Seed": seed,
                    "Model": name,
                    "SMILES": smi,
                    "Label": int(lab),
                    "Prob": float(p),
                })

    df_long = pd.DataFrame(all_rows)
    repro_df = pd.DataFrame(repro_rows)
    return df_long, repro_df


# ============================================================
# 3. Pivot 成宽表（如检测到重复 SMILES 会警告）
# ============================================================
def pivot_predictions(df_long):
    df_wide = df_long.pivot_table(
        index=["Seed", "SMILES", "Label"],
        columns="Model",
        values="Prob",
        aggfunc="mean",   # 如果有重复，会取平均
    ).reset_index()

    rename_map = {"GCN": "GCN_Prob", "GraphSAGE": "SAGE_Prob", "GAT": "GAT_Prob"}
    df_wide = df_wide.rename(columns=rename_map)

    for col in ["GCN_Prob", "SAGE_Prob", "GAT_Prob"]:
        if col not in df_wide.columns:
            df_wide[col] = np.nan

    return df_wide


# ============================================================
# 4. 合并 similarity
# ============================================================
def merge_with_similarity(df_wide, sim_csv="similarity_per_molecule.csv"):
    if not os.path.exists(sim_csv):
        raise FileNotFoundError(
            f"{sim_csv} 不存在，请先运行 diagnose_seed3_similarity.py"
        )
    sim_df = pd.read_csv(sim_csv)
    merged = pd.merge(
        df_wide,
        sim_df[["Seed", "SMILES", "MaxSimilarity"]],
        on=["Seed", "SMILES"],
        how="inner"
    )
    return merged


# ============================================================
# 5. BCE Loss
# ============================================================
def bce_loss(prob, label, eps=1e-7):
    p = np.clip(prob, eps, 1 - eps)
    return -(label * np.log(p) + (1 - label) * np.log(1 - p))


# ============================================================
# 6. H3 分析
# ============================================================
def run_H3_analysis(df):
    models = ["GCN", "SAGE", "GAT"]
    model_to_col = {"GCN": "GCN_Prob", "SAGE": "SAGE_Prob", "GAT": "GAT_Prob"}

    # ---- 6.1 BCE Loss ----
    for m in models:
        col = model_to_col[m]
        df[f"{m}_Loss"] = df.apply(lambda r: bce_loss(r[col], r["Label"]), axis=1)

    # ---- 6.2 Pooled Spearman ----
    pooled_rows = []
    for m in models:
        loss_col = f"{m}_Loss"
        rho, p_val = spearmanr(df["MaxSimilarity"], df[loss_col], nan_policy="omit")
        pooled_rows.append({
            "Model": m,
            "Pooled_rho": rho,
            "Pooled_p": p_val,
            "N": int(df[[loss_col, "MaxSimilarity"]].dropna().shape[0]),
        })
    pooled_df = pd.DataFrame(pooled_rows)

    # ---- 6.3 Per-seed Spearman（关键：控制 split-level confounding） ----
    per_seed_rows = []
    for m in models:
        loss_col = f"{m}_Loss"
        for seed in sorted(df["Seed"].unique()):
            sub = df[df["Seed"] == seed]
            rho, p_val = spearmanr(sub["MaxSimilarity"], sub[loss_col], nan_policy="omit")
            per_seed_rows.append({
                "Model": m,
                "Seed": seed,
                "rho": rho,
                "p_value": p_val,
                "N": len(sub),
            })
    per_seed_df = pd.DataFrame(per_seed_rows)

    # ---- 6.4 Similarity 分箱 ----
    bins = [0, 0.3, 0.5, 1.01]
    bin_labels = ["Low (<0.3)", "Medium (0.3-0.5)", "High (>=0.5)"]
    df["SimBin"] = pd.cut(df["MaxSimilarity"], bins=bins,
                          labels=bin_labels, include_lowest=True)

    bin_rows = []
    for m in models:
        loss_col = f"{m}_Loss"
        for b in bin_labels:
            sub = df[df["SimBin"] == b]
            if len(sub) == 0:
                continue
            bin_rows.append({
                "Model": m,
                "SimBin": b,
                "N": len(sub),
                "Mean_Loss": float(sub[loss_col].mean()),
                "Median_Loss": float(sub[loss_col].median()),
                "Accuracy": float(
                    ((sub[model_to_col[m]] > 0.5).astype(int) == sub["Label"]).mean()
                ),
            })
    bin_df = pd.DataFrame(bin_rows)

    # ---- 6.5 散点图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, m in zip(axes, models):
        loss_col = f"{m}_Loss"
        ax.scatter(df["MaxSimilarity"], df[loss_col], alpha=0.25, s=10, color="gray")
        df3 = df[df["Seed"] == 3]
        ax.scatter(df3["MaxSimilarity"], df3[loss_col],
                   alpha=0.6, s=20, color="red", label="Seed 3")
        rho_pooled = pooled_df.loc[pooled_df["Model"] == m, "Pooled_rho"].values[0]
        ax.set_xlabel("Max Tanimoto Similarity")
        ax.set_ylabel(f"{m} BCE Loss")
        ax.set_title(f"{m} (pooled ρ={rho_pooled:.3f})")
        ax.legend()
    plt.tight_layout()
    plt.savefig("H3_scatter_similarity_vs_loss.png", dpi=150)
    print("散点图: H3_scatter_similarity_vs_loss.png")

    # ---- 6.6 ErrorCount ----
    df["ErrorCount"] = 0
    for m in models:
        pred_label = (df[model_to_col[m]] > 0.5).astype(int)
        df["ErrorCount"] += (pred_label != df["Label"]).astype(int)

    error_stats = df.groupby("ErrorCount")["MaxSimilarity"].agg(
        ["count", "median", "mean"]
    ).reset_index()

    plt.figure(figsize=(7, 5))
    unique_ec = sorted(df["ErrorCount"].unique())
    data_to_plot = [df[df["ErrorCount"] == k]["MaxSimilarity"].dropna() for k in unique_ec]
    plt.boxplot(data_to_plot)  # 不传 labels 参数
    plt.xticks(ticks=range(1, len(unique_ec) + 1),
               labels=[f"E={k}" for k in unique_ec])  # 用 xticks 手动设置
    plt.xlabel("Error Count (# models misclassified)")
    plt.ylabel("Max Tanimoto Similarity")
    plt.title("Error Count vs Chemical Similarity")
    plt.tight_layout()
    plt.savefig("H3_errorcount_vs_similarity.png", dpi=150)
    print("ErrorCount 图: H3_errorcount_vs_similarity.png")

    df3 = df[df["Seed"] == 3]
    error_stats_seed3 = df3.groupby("ErrorCount")["MaxSimilarity"].agg(
        ["count", "median", "mean"]
    ).reset_index()

    # ---- 6.7 保存 ----
    pooled_df.to_csv("H3_pooled_spearman.csv", index=False)
    per_seed_df.to_csv("H3_per_seed_spearman.csv", index=False)
    bin_df.to_csv("H3_bin_stats.csv", index=False)
    error_stats.to_csv("H3_errorcount_stats_all.csv", index=False)
    error_stats_seed3.to_csv("H3_errorcount_stats_seed3.csv", index=False)
    df.to_csv("H3_data.csv", index=False)

    print("\n========== Pooled Spearman ==========")
    print(pooled_df.round(4).to_string(index=False))

    print("\n========== Per-seed Spearman ==========")
    print(per_seed_df.round(4).to_string(index=False))

    print("\n========== Similarity Bin Analysis ==========")
    print(bin_df.round(4).to_string(index=False))

    print("\n========== Error Count vs Similarity (all seeds) ==========")
    print(error_stats.round(4).to_string(index=False))

    print("\n========== Error Count vs Similarity (Seed 3) ==========")
    print(error_stats_seed3.round(4).to_string(index=False))

    return pooled_df, per_seed_df, bin_df, error_stats, error_stats_seed3


# ============================================================
# 7. 入口
# ============================================================
if __name__ == "__main__":
    if not os.path.exists("BBBP.csv"):
        raise FileNotFoundError("BBBP.csv 不存在")

    df_raw = pd.read_csv("BBBP.csv")
    smiles_list = df_raw["smiles"].tolist()
    labels = df_raw["p_np"].tolist()

    # ---- 重复 SMILES 检查 ----
    n_dup = df_raw.duplicated(subset=["smiles"]).sum()
    print(f"[Sanity] 重复 SMILES 数量: {n_dup}")
    if n_dup > 0:
        print("[WARN] BBBP 存在重复 SMILES，pivot_table 会合并它们。")
        print("       若结果异常，考虑改用 SampleID 作为 merge key。")

    seeds = [0, 1, 2, 3, 4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Step 1: 重新训练并保存预测 ----
    if not os.path.exists("predictions_per_molecule.csv"):
        print("\n[Step 1] 重新训练 5 seeds x 3 models ...")
        pred_long, repro_df = run_predictions(smiles_list, labels, seeds, device)
        pred_long.to_csv("predictions_per_molecule.csv", index=False)
        repro_df.to_csv("H3_reproduction_check.csv", index=False)
        print("\n预测与复现检查已保存")
    else:
        print("[Step 1] 检测到 predictions_per_molecule.csv，直接加载 ...")
        pred_long = pd.read_csv("predictions_per_molecule.csv")
        repro_df = None

    # ---- Step 2: 复现检查 ----
    if repro_df is not None:
        print("\n========== Reproduction Check（复现对照） ==========")
        print(repro_df.round(4).to_string(index=False))
        max_delta = repro_df["Delta"].abs().max()
        print(f"\n最大 |Δ| = {max_delta:.4f}")
        if max_delta > 0.05:
            print("[WARN] 部分模型 Test AUC 与原始实验差异 > 0.05。")
            print("       H3 分析的是'重新训练的实验'，而非原实验。")
            print("       若需要严格对齐，请加载原实验 checkpoint 做 inference。")
        else:
            print("[OK] 复现结果与原始实验一致（|Δ| < 0.05）。")

    # ---- Step 3: Pivot + merge ----
    df_wide = pivot_predictions(pred_long)
    df_H3 = merge_with_similarity(df_wide)
    print(f"\n[Step 3] H3 数据表: {df_H3.shape}")

    # ---- Step 4: H3 分析 ----
    print("\n[Step 4] H3 分析 ...")
    run_H3_analysis(df_H3)

    print("\n全部完成！")