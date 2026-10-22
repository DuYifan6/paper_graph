# 正文公式对应代码

本版本将主要预测模型及配套控制统一到同一实现，依据 Main Text R2 的式 (2)–(20)、§3.4 和 Appendix D.4。原公开代码基线为提交 `b08ebbf10f73ca3689d2ab6f5d8a201299d47487`。版本标记为 `papergraph-v2-undirected-union-formula-loss`。

**本版本改变了网络、模型层结构和罚项的实际计算，必须重新训练并据新结果更新论文。它不能作为证明文中现有 R²、p 值或经济收益正确的复现记录。** 原始数据没有附在公开仓库或本代码包中；验证使用矩阵恒等式及合成数据，没有模拟或替换原论文数值。

## 已统一的实现

| 对应正文 | 本版本实现 |
|---|---|
| §2.4，式 (2)–(5) | 五分钟收益裁剪、等权市场成分去除、股票内标准化、正 Pearson 相关、逐行 Top-s、无向 union；union 后不再截断，因此最终度可大于 s |
| Graph-1/2/3 | Graph-1 保留正相关权重，不应用二元阈值；Graph-2 为全日二元图；Graph-3 只取每个历史日最后 24 个五分钟收益 |
| 式 (14)–(16) | 单层 128 隐藏单元 LSTM、原始边权的邻居加权均值、单层 sigmoid 门控、单层 ReLU 消息变换、残差相加 |
| §3.3.1 | 线性预测头；移除旧实现额外的 MLP 隐藏层 |
| 式 (18)–(20) | 同一有效股票子图内重算度；罚项为 `2/|V| × yᵀL_sym y`，不再对 N×40 存储槽取 mean；明确零度节点约定 |
| ZeroGraph/IdentityGraph/no-Laplacian | 同一 LSTM 与预测头；ZeroGraph 注册 GraphMix 后绕过；IdentityGraph 仅自身表示；no-Laplacian 保留经验图、去掉罚项 |
| 5m-NoGraph | 每只股票自己的 20×48 收益历史，使用与新版 GraphMix 精确匹配的有效参数预算；没有跨股票聚合或闲置填充参数 |
| §3.4 | 拟合截至 2024-06-05，2024-06-06 至年底验证，2025-01-02 至 2025-11-17 评价；默认要求 211 个评价日期 |
| 训练 | Adam，学习率 5e-4，weight decay 1e-4，最多 20 epochs，patience 4，梯度裁剪 1.0；checkpoint 只按 pooled validation MSE 选择；没有 ranking loss 或组合收益选模 |
| RandomGraph | 最终无向图上做无向双边交换；精确保度、对称、无自环、无重复边；默认 5 个模型种子各 20 个独立图序列，共 105 次训练 |
| 推断 | 先在每个模型种子内平均 20 个随机图的日损失，再与经验图作差，最后跨 5 个模型种子平均；以交易日为单位进行 lag=2 Newey–West / t(T−1) 检验 |
| Hybrid14:55 预测 | 前 19 个序列位置保留原历史输入，仅最新位置用用户提供的 Hybrid 张量；图为 19 个完整历史日加信号日截至 14:55 的收益 |
| GCN-LSTM | 两个 64 单元 GCN 层、LSTM，与主模型共用同一新版 Graph-2 图缓存，按验证 MSE 选 checkpoint |

IdentityGraph 与 no-Laplacian 的训练目标为 MSE。经济预测入口生成预测，不把预测准确率解释为成交可行性。

## 正文内部仍需同步的两项

正文的简化层结构公式与原参数计数不能同时成立。按单层 128 单元 LSTM、式 (16) 的两次 256→128 变换和线性预测头计算，实测如下。

| 模型 | 注册参数 | 有梯度路径参数 |
|---|---:|---:|
| Graph-2 / IdentityGraph / RandomGraph / no-Laplacian | 137,089 | 137,089 |
| ZeroGraph | 137,089 | 71,297 |
| 新版 5m-NoGraph | 137,089 | 137,089 |

其中 GraphMix 为 65,792，线性预测头为 129。正文 §3.3.3 和 Table D2 的旧数 170,113 / 87,809 / 82,304 对应旧 MLP 实现，不能继续用来描述本版本。没有增加无用参数来凑旧总数。可以同步写为：

> GraphMix contains 65,792 trainable parameters. Graph-2 has 137,089 registered and active parameters. ZeroGraph retains 137,089 registered parameters and activates 71,297 parameters because GraphMix is bypassed.

对孤立节点，本版本采用 `D^{-1/2}(D-A)D^{-1/2}` 的零度扩展，令其逆度及 Laplacian 对角为零，使孤立节点没有边罚项。这符合正文“孤立节点不贡献边罚项”的说明；式 (18) 的 I 应注明仅在正度节点上成立。可补充：

> For zero-degree nodes, the inverse square root of the degree and the corresponding diagonal entry of the normalized Laplacian are defined as zero.

归一化罚项平滑的是度归一化预测；应相应修正把它直接解释为原始预测差的句子。

## 安装与输入

使用 Python 3.10 或以上版本，安装 `requirements.txt`。所有路径由命令行显式指定，程序不会访问旧脚本中硬编码的 D 盘目录。

共同缓存沿用原文件格式：

- NPZ 为 `X_np[T,N,9]`、`y_np[T,N]`、`mask_np[T,N]`。X 已经按 t−1 对齐、填补和标准化；目标单位为百分点。
- 本地可信 metadata pickle 包含 `all_dates` 和 `ts_codes_order`，可带 `train_start_idx`。若含 `feature_names` 或 `selected_features`，程序验证九变量集合并按名称显式重排 X 和 Hybrid 张量；若没有列名，输出记录 `feature_order_verified=false`，需作者核实实际列映射。
- 返回缓存为 `returns[R,N,48]`、`available[R,N]`、`dates[R]`、`stocks[N]`，每行对应该原始交易日。日期后缀必须与共同面板完全一致，可以前缀添加面板前的历史交易日。程序会计算前缀偏移，图始终只用目标日前的数据。
- Hybrid NPZ 为 `X_latest_hybrid_np[T,N,9]` 和 `mask_hybrid_np[T,N]`。本包使用已有的冻结张量，不猜测资金流或多空比公式，不重新估计预处理参数。

共同缓存的预处理来源和真实列映射仍须由作者提供。本包加载已有输入，不足以认证历史证券覆盖或原缓存没有泄漏。

### 准备五分钟收益

每只股票 CSV 需含 `trade_time`、`close`，保留既有 49 个对齐价格形成 48 个日内收益的约定。最后两个价格时间必须为 14:55、15:00。程序检查重复时间和股票/日期顺序，并输出缺失覆盖审计。

```powershell
python run.py build-returns --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --minute-dir C:/data/min5_aligned --returns C:/data/returns_v2.npz
```

首个模型目标日之前的原始交易日不一定全在模型面板中。例如原始样本有 2022-01-04，而模型日期从 2022-01-05 开始，可用含 `date` 列的 CSV 显式提供前缀交易日，避免第一段历史窗口少一天：

```powershell
python run.py build-returns --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --minute-dir C:/data/min5_aligned --returns C:/data/returns_v2.npz --prehistory-dates C:/data/pre_panel_exchange_dates.csv
```

程序不会猜测交易日历，也不会把不足 W 日的窗口强行当完整窗口。

## 运行主要实验

以下示例中路径须换成实际输入；输出目录应使用新的空目录，防止混入旧结果。

```powershell
python run.py ablation --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --returns C:/data/returns_v2.npz --out C:/runs/main_v2 --save-details
python run.py randomgraph --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --returns C:/data/returns_v2.npz --out C:/runs/topology_v2
python run.py five-seeds --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --returns C:/data/returns_v2.npz --out C:/runs/initialization_v2
python run.py robustness --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --returns C:/data/returns_v2.npz --out C:/runs/robustness_v2
```

旧文件名的入口仍可使用，例如：

```powershell
python graph/graph2.py --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --returns C:/data/returns_v2.npz --out C:/runs/graph2_v2
python random/build_random_graphs.py --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --returns C:/data/returns_v2.npz --out C:/runs/topology_v2
python random/RandomGraph.py --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --returns C:/data/returns_v2.npz --out C:/runs/topology_v2
```

`build_random_graphs.py` 只构建并审计随机图，`RandomGraph.py` 执行五种子训练与统计。随机图序列按模型 seed、图 seed、日期分别确定，不改变神经网络初始化 RNG。每日期缓存含校验指纹，旧有向固定 Top-40 缓存不能混入。

若某日固定度序列的参照集合退化（空图、单边或完整图），它会保留同一图并明确标记 `degenerate_fixed_degree_reference`。其他图必须达到目标有效交换次数，达不到就报错；不会把交换失败悄悄当成功随机化。

### Hybrid14:55 模型预测

```powershell
python backtest/Hybrid1455_PositiveGraph2.py --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --returns C:/data/returns_v2.npz --hybrid-cache C:/data/hybrid1455_latest_tensor.npz --out C:/runs/hybrid_graph_v2
python backtest/Hybrid1455_ZeroGraph.py --common C:/data/common_tensor_cache.npz --meta C:/data/common_tensor_meta.pkl --returns C:/data/returns_v2.npz --hybrid-cache C:/data/hybrid1455_latest_tensor.npz --out C:/runs/hybrid_zero_v2
```

这两个入口负责同构模型的训练与预测；不自动生成或认证 Table 9/F2 的实际成交回测。历史资格、执行数量、容量和 carry-forward 仍需真实数据及独立执行账本。旧公开的回测与输入重建脚本没有在本包中冒充已完成验证的实现。

### 扩展证券池

使用另外构建的 expanded common/meta/returns 输入运行 `run_nostatic_st.py`，程序分别拟合 ZeroGraph 和 Graph-2。此入口不会凭空恢复缺失证券，也不会把当前证券名单当作完整 PIT 历史。

## 输出与结果同步

每次拟合保存 checkpoint、逐 epoch 拟合/验证损失、逐日评价损失、实际参数计数、全部输入指纹和配置。`--save-details` 额外保存验证及测试的股票日预测。`test_monthly.csv`、`development_quarterly.csv` 和 summary 分开报告 pooled R² 与观察数加权的期间 R²；开发期结果明确是开发期诊断。

拓扑比较保存 100 次随机图运行清单、5 行模型种子对照、各 seed 日度差、最终 pooled 日度差、效应均值、HAC 标准误、95% 区间和单双侧 p 值。不把 100 次训练当作额外的独立市场样本。

应重新运行所有受影响的 Graph-1/2/3、ZeroGraph、消融、RandomGraph、多种子和 GCN-LSTM 对照，并重新生成图表。不能沿用旧权重或把代码修改记录当作新实验结果。真实数据下的完整复算、静态 ST 及历史覆盖、经济执行结果在本次交付中未被验证。

## 验证

```powershell
python -m pytest tests -q
```

测试包括完整 5×20 随机图合成训练流程、精确保度与对称性、无向 union 后超过 s 的度、罚项与显式矩阵公式、孤立/无效节点处理、所有模型训练选模、时间截止、修改测试期目标不改变 checkpoint、匹配股票日和目标检查，以及输入列名和原始历史前缀对齐。
