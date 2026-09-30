import os
import time
import copy
import random
from collections import defaultdict
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# 1.分子图构建工具


ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Se']
HYBRIDIZATION_TYPES = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
CHIRAL_TYPES = [
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

def one_hot_encoding(value, choices):
    """通用 one-hot 编码，最后一位留给“其他”类别。"""
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def atom_features(atom):
    """生成原子的特征向量。"""
    features = []
    features += one_hot_encoding(atom.GetSymbol(), ATOM_TYPES)
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    features += one_hot_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    features += one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPES)
    features += one_hot_encoding(atom.GetChiralTag(), CHIRAL_TYPES)
    features.append(1 if atom.IsInRing() else 0)
    features.append(1 if atom.GetIsAromatic() else 0)
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    features.append(atom.GetValence(Chem.ValenceType.IMPLICIT))
    return features


def bond_features(bond):
    """生成化学键的特征向量。"""
    features = []
    features += one_hot_encoding(bond.GetBondType(), BOND_TYPES)
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    features.append(1 if bond.GetStereo() != Chem.BondStereo.STEREONONE else 0)
    return features


def mol_to_graph(smiles, label=None):
    """将 SMILES 转换为 PyG 的 Data 对象。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 节点特征
    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features_list, dtype=torch.float)

    # 边索引和边特征
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 8), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ============================================================
# 2. Scaffold Split（骨架划分）
# ============================================================


def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def scaffold_split(smiles_list, labels,
                   frac_train=0.8, frac_val=0.1, frac_test=0.1,
                   seed=42):
    """
    保证：
      1. 同一骨架的分子不跨集合（整组进/整组出）
      2. val / test 非空
      3. val / test 同时包含 0 和 1（只要全局两类都有）
      4. 不同 seed 产生不同的划分
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-8
    n = len(smiles_list)

    # 关键：使用 seed 控制随机性
    random.seed(seed)

    # 1. 骨架分组
    sc2idx = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        sc = generate_scaffold(smi)
        if sc is not None:
            sc2idx[sc].append(i)

    groups = list(sc2idx.values())
    random.shuffle(groups)              # 让不同 seed 产生不同顺序


    # 2. 目标规模
    n_val = max(int(n * frac_val), 1)
    n_test = max(int(n * frac_test), 1)

    # 3. 主分配：先填 val，再填 test，train 兜底
    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(val_idx) + len(g) <= n_val:
            val_idx.extend(g)
        elif len(test_idx) + len(g) <= n_test:
            test_idx.extend(g)
        else:
            train_idx.extend(g)

    # 4. 修复：把“整个骨架组”从 train 调到 val/test
    def _move_group(src, dst, want_label=None):
        if not src:
            return False
        local = defaultdict(list)
        for i in src:
            local[generate_scaffold(smiles_list[i])].append(i)
        cand = list(local.values())
        if want_label is not None:
            hit = [g for g in cand if any(labels[i] == want_label for i in g)]
            if hit:
                cand = hit
        cand.sort(key=len)
        for g in cand:
            for i in g:
                src.remove(i)
                dst.append(i)
            return True
        return False

    if not val_idx:
        _move_group(train_idx, val_idx)
    if not test_idx:
        _move_group(train_idx, test_idx)

    # 5. 修复：val/test 单一类别 → 用“整组”换
    def _balance(target_idx, source_idx):
        if not target_idx:
            return
        seen = {labels[i] for i in target_idx}
        if len(seen) >= 2:
            return
        dom = next(iter(seen))
        need = 1 - dom

        src_groups = defaultdict(list)
        for i in source_idx:
            src_groups[generate_scaffold(smiles_list[i])].append(i)
        donors = [g for g in src_groups.values()
                  if any(labels[i] == need for i in g)]
        if not donors:
            print(f"[warn] train 中也不存在类别 {need}，无法平衡")
            return

        tgt_groups = defaultdict(list)
        for i in target_idx:
            tgt_groups[generate_scaffold(smiles_list[i])].append(i)
        pure_dom = [g for g in tgt_groups.values()
                    if all(labels[i] == dom for i in g)]
        eject_pool = pure_dom if pure_dom else list(tgt_groups.values())
        eject = min(eject_pool, key=len)
        donor = min(donors, key=len)

        for i in eject:
            target_idx.remove(i)
            source_idx.append(i)
        for i in donor:
            source_idx.remove(i)
            target_idx.append(i)

    _balance(val_idx, train_idx)
    _balance(test_idx, train_idx)

    # 在 _balance 之后加：
    def _ensure_min_neg(target_idx, source_idx, min_neg=20):
        """确保 target_idx 至少有 min_neg 个负类"""
        neg_count = sum(1 for i in target_idx if labels[i] == 0)
        if neg_count >= min_neg:
            return
        # 从 train 里换更多的负类骨架组进来
        need = min_neg - neg_count
        src_groups = defaultdict(list)
        for i in source_idx:
            src_groups[generate_scaffold(smiles_list[i])].append(i)
        neg_groups = [g for g in src_groups.values() if any(labels[i] == 0 for i in g)]
        neg_groups.sort(key=len)
        for g in neg_groups:
            if need <= 0:
                break
            for i in g:
                source_idx.remove(i)
                target_idx.append(i)
            need -= sum(1 for i in g if labels[i] == 0)

    _ensure_min_neg(test_idx, train_idx, min_neg=20)
    _ensure_min_neg(val_idx, train_idx, min_neg=20)

    # 6. 输出
    def pack(idx_list):
        return ([smiles_list[i] for i in idx_list],
                [labels[i] for i in idx_list])

    (tr_s, tr_l) = pack(train_idx)
    (va_s, va_l) = pack(val_idx)
    (te_s, te_l) = pack(test_idx)

    print(f"[seed={seed}] 骨架组数: {len(groups)}")
    print(f"[seed={seed}] train: n={len(tr_s)}, pos={sum(tr_l)}, neg={len(tr_l) - sum(tr_l)}")
    print(f"[seed={seed}] val:   n={len(va_s)}, pos={sum(va_l)}, neg={len(va_l) - sum(va_l)}")
    print(f"[seed={seed}] test:  n={len(te_s)}, pos={sum(te_l)}, neg={len(te_l) - sum(te_l)}")
    pos_ratio = sum(te_l) / len(te_l) if len(te_l) > 0 else 0.0
    print(f"[seed={seed}] test 正类比例 (AP baseline): {pos_ratio:.4f}")

    return (tr_s, tr_l), (va_s, va_l), (te_s, te_l)

# ============================================================
# 3. 自定义 Dataset
# ============================================================

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels):
        self.data_list = []
        for smi, lab in zip(smiles_list, labels):
            data = mol_to_graph(smi, lab)
            if data is not None:
                self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ============================================================
# 4. 模型定义
# ============================================================


class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.fc = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.5)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)  # 所有的层都加Dropout
        x = global_mean_pool(x, batch)  # 图级池化，把属于同一个分子的原子特征求平均 [800原子, 128] -> [32分子, 128]，从节点级任务（原子）上升到图级任务（分子）
        return self.fc(x)    # 全连接层（线性分类头），把[32, 128]的图级表示映射到[32, 1]的预测值，用于计算交叉熵损失

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # 注意：arr='max'是GraphSAGE论文验证的最强聚合器
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='max'))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='max'))
        self.fc = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # 隐藏层：多头注意力拼接 (concat=True)
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
        # 最后一层：多头平均(concat=False), 输出单头
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False))
        self.fc = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


# ============================================================
# 5. 通用训练与评估
# ============================================================

def train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs=100, patience=20):
    """
    统一的训练与评估框架，适用于GCN、GraphSAGE、GAT
    返回：
        best_val_auc: 最佳验证集 AUC
        test_auc: 测试集AUC
        test_prc_auc: 测试集 PRC-AUC
        train_time: 训练总耗时（秒）
    """
    # 1.基础组件定义
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # 二分类任务损失函数（输出Logits，与BCEWithLogitLoss搭配）
    criterion = nn.BCEWithLogitsLoss()

    # 2.初始化早停相关变量
    best_val_auc = 0.0
    best_model_state = None
    counter = 0

    # 3.记录开始时间
    start_time = time.time()

    # 4.训练主循环
    for epoch in range(epochs):
        #--------训练阶段------------
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch).view(-1)  # 形状从[batch, 1]变为[batch]
            loss = criterion(out, batch.y.view(-1).float())

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        ave_train_loss = total_train_loss / len(train_loader)


        #-------------验证阶段------------------
        model.eval()
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch).view(-1)
                preds = torch.sigmoid(out)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(batch.y.view(-1).cpu().numpy())

        val_auc = roc_auc_score(all_val_labels, all_val_preds)

        # ------------早停逻辑-----------------
        # 监控验证集 AUC
        if val_auc > best_val_auc:
            best_val_auc =val_auc
            # 深拷贝当前最佳模型权重，防止被后续训练覆盖
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, best_val_auc {best_val_auc:.4f}")
                break

        # 每 10 个epoch 打印一次日志
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}  |  Train Loss: {ave_train_loss:.4f}  |  Val AUC: {val_auc:.4f}")

    # 5.记录结束时间
    train_time = time.time() - start_time

    # 6.回滚到验证集最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    #-----------测试阶段------------------
    model.eval()
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch).view(-1)
            preds = torch.sigmoid(out)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(batch.y.view(-1).cpu().numpy())

    test_auc = roc_auc_score(all_test_labels, all_test_preds)
    test_prc_auc = average_precision_score(all_test_labels, all_test_preds)

    return best_val_auc, test_auc, test_prc_auc, train_time




# ============================================================
# 6. 主流程
# ============================================================
if __name__ == "__main__":
    # 下载数据（若已存在则跳过）
    if not os.path.exists("BBBP.csv"):
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        urlretrieve(url, "BBBP.csv")

    df = pd.read_csv("BBBP.csv")
    smiles_list = df["smiles"].tolist()
    labels = df["p_np"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = [0, 1, 2, 3, 4]

    # 根据诊断调整后的超参数（可自行修改）
    HIDDEN = 64
    NUM_LAYERS = 3
    GAT_HEADS = 2

    # 模型构造器：接受 in_channels，返回模型实例
    model_builders = {
        "GCN":      lambda in_ch: GCNModel(in_ch, hidden_channels=HIDDEN, num_layers=NUM_LAYERS),
        "GraphSAGE":lambda in_ch: GraphSAGEModel(in_ch, hidden_channels=HIDDEN, num_layers=NUM_LAYERS),
        "GAT":      lambda in_ch: GATModel(in_ch, hidden_channels=HIDDEN, num_layers=NUM_LAYERS, heads=GAT_HEADS),
    }

    # 用于保存每个 seed 下每个模型的测试指标
    all_results = {name: {"test_auc": [], "test_prc_auc": []} for name in model_builders}

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)


        print(f"\n{'='*25} Seed {seed} {'='*25}")

        # 1. 按 seed 重新划分数据
        (train_smiles, train_labels), (val_smiles, val_labels), (test_smiles, test_labels) = \
            scaffold_split(smiles_list, labels, seed=seed)

        # 2. 构建 Dataset 和 DataLoader
        train_dataset = MoleculeDataset(train_smiles, train_labels)
        val_dataset   = MoleculeDataset(val_smiles, val_labels)
        test_dataset  = MoleculeDataset(test_smiles, test_labels)


        in_channels = train_dataset[0].x.shape[1]

        # 3. 每个模型独立实例化、训练、评估
        for model_idx, (name, builder) in enumerate(model_builders.items()):
            # 每个模型使用一个由 seed 和 model_idx 组合的确定性种子
            model_seed = seed * 100 + model_idx
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)
            random.seed(model_seed)

            print(f"\n--- Seed {seed} | 训练 {name} ---")

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

            model = builder(in_channels).to(device)

            best_val_auc, test_auc, test_prc_auc, train_time = train_and_evaluate(
                model, train_loader, val_loader, test_loader, device,
                epochs=100, patience=20
            )

            all_results[name]["test_auc"].append(test_auc)
            all_results[name]["test_prc_auc"].append(test_prc_auc)

            param_count = sum(p.numel() for p in model.parameters())
            print(f"    Best Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f} | "
                  f"Test PRC-AUC: {test_prc_auc:.4f} | Params: {param_count} | Time: {train_time:.1f}s")

    # 4. 汇总结果：mean ± std
    print("\n" + "="*60)
    print("多 Seed 实验结果汇总 (mean ± std over {} seeds)".format(len(seeds)))
    print("="*60)

    summary_rows = []
    for name in model_builders:
        aucs = np.array(all_results[name]["test_auc"])
        prcs = np.array(all_results[name]["test_prc_auc"])
        summary_rows.append({
            "Model": name,
            "Test ROC-AUC": f"{aucs.mean():.4f} ± {aucs.std():.4f}",
            "Test PRC-AUC": f"{prcs.mean():.4f} ± {prcs.std():.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("multi_seed_summary.csv", index=False)
    print("\n结果已保存到 multi_seed_summary.csv")

