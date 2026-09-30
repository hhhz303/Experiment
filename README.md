# Experiment —— 关于 molecular GNN 在 scaffold distribution shift 下泛化行为的实证研究

多随机种子 scaffold split 实验发现，三个 GNN 架构在 Seed 3 上同时出现显著性能下降。进一步的化学空间分析表明，Seed 3 测试集相对于训练集具有明显更低的 maximum Morgan-Tanimoto similarity，且该分布偏移同时存在于正负样本中，支持整体 chemical-space shift 而非类别特异性 shift 的解释。进一步的 molecule-level 分析发现，在合并多个 scaffold splits 后，较低的 test-to-train similarity 与更高的 BCE loss、更低的分类准确率以及更高的跨模型共同错误数呈一致关联；然而该关系在单个 split 内并不稳定，尤其在 Seed 3 内未观察到显著的单分子 similarity-loss 单调关系。因此，chemical-space novelty 更适合作为 scaffold split 层面的泛化难度指标，而不足以单独解释具体分子的预测失败。
