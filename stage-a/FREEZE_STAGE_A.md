# Stage A 独立确认:新成员实例(认证级)——冻结协议

日期:2026-10-19(授权),2026-10-20(冻结执行)。上游:perm0 筛选(v2 协议,verdict `meaningful_positive`,学校端已核验归档)。

## 0. 授权与级别

- **用户授权(聊天记录原文,2026-10-19 深夜):"我给授权,同时按认证级做,同样每半个小时检查一次"。**
- 授权内容:Stage A 独立确认在本机 RTX 4090D 运行;**认证级(certified)= 不复用任何旧嵌入**,RR/PP/PR 全部在本机重训(perm0 筛选的 RR/PP 复用为 exploratory_only,不可升级;认证级要求重训)。
- 预算:**cap 10 小时墙钟累计**(授权基于运行方预估:25 次训练,实测 15.5 分钟/次,约 6.5–7.2 h,含 1 次复现校验 run);实测单次 ~0.26 h。
- 执行修订:阈值、种子数、判定规则与 v2 完全一致,不改动;新增的只有实例、认证级 manifest 与 RR/PP 重训。

## 1. 新成员实例 perm3

- **生成规则(预注册)**:读真实 `data/het_g8p0/seg2region.csv`;`np.random.RandomState(20261019)`;对节点做一次均匀随机排列,按真实区域大小直方图(区域 id 0..2159 依 size 降序原序)顺序消费——即与存档 perm0/1/2 同族的全局重排;只改 `region_id` 列;`geo_id` 顺序不变。
- 该族自动满足:逐节点关联度 d_i=2+b_i 不变(b_i 来自 roadmap.rel);区域 id 集合、区域个数、大小多重集不变;路型群组与边界二元群组不动。**不要求**逐节点区域大小不变(perm0 实测也不保持,11,090/11,095 节点换区)。
- 生成后验证(不通过则停止):geo_id 0..N-1;区域 id 连续 0..2159;大小多重集逐位一致;与 real/perm0 的赋值数组不同;`a3_coeff_override.load_partition` 可加载。
- 数据目录:`data/het_g8p0_perm3/`(新 seg2region.csv + 共享输入字节级副本,含表头版 roadmap.geo 桩,说明文件同前)。

## 2. 臂与训练(认证级,零复用)

| 臂 | 内容 | 系数 | 训练 |
|---|---|---|---|
| RR | real | real | **重训** exact + clique,seeds 0–4(10 次) |
| PP | perm3 | perm3 | 重训 exact + clique,seeds 0–4(10 次) |
| PR | perm3 | real | 重训 clique(`induced_coeff_override`),seeds 0–4(5 次);PR-exact = PP-exact(恒等,运行前验证) |
| RP | real | perm3 | 不训练(同 v2,仅诊断) |

共 **25 次训练**,5000 episodes/次,`--device cuda`,隔离 `--output-dir`。运行顺序:RR_exact → RR_clique → PP_exact → PP_clique → PR_clique(各 s0–s4)。

## 3. 认证级 manifest(每次 run 记录)

1. 代码哈希:A3_ROOT 全部 9 个 py 文件(train_graph/a3_coeff_override/a3_output_paths/model/config/preprocess/utils/a3_exact + dataset.py 桩,桩永不执行)。
2. 数据哈希:该 run 读取的全部数据文件(共享输入 ×2 目录 + perm3 seg2region + PR 的 real seg2region)。
3. 环境:python/torch/cuda/GPU/驱动。
4. 算子哈希:运行前以相同参数做确定性 build-only 探针,记录 nnz/行和误差/min/max 及稀疏矩阵字节 sha256;并与 `train_graph.py --build-only` 的打印值交叉核对。
5. 初始化参考:按 pinned 模块复制构造序列(setup_seed → RoadModel → PairHGNN 交换 → RNG 快照恢复 → weight_init),对 state_dict 求 sha256;探针跑两遍证明确定性;同 seed 各臂构造序列相同(断言)。端到端位级复现校验见 §5。
6. 输出哈希:emb + graph_model 的 sha256/大小;日志 sha256;返回码与墙钟。

## 4. 读出与判据(与 v2 逐字一致)

- 探针:Ridge(alpha=1.0)+ QuantileTransformer(normal),KFold(5, shuffle, random_state=0),9,384 条有标签道路;每臂重新拟合。
- E_arm,s = MAE(clique) − MAE(exact);PR 的 exact 用 PP-exact 嵌入。
- 主指标 Δ_content = E_RR − E_PR(5 seed 配对均值,95% t 区间,df=4);次要 Δ_coeff、Δ_total、加和校验。
- 判定:mean ≥ +0.02 且 CI 下界 > 0 → `meaningful_positive`;mean ≤ −0.02 且 CI 上界 < 0 → `meaningful_negative`;CI 完全在 ±0.01 内 → `negligible`;否则 `inconclusive`。
- 停止规则:不通过/反转/不确定 → 停止独立选题并入 A3;不追加 seed/实例追结果;通过 → Stage B(第二个新实例或新城市)。

## 5. 认证加强:位级复现校验

训练全部完成后,用相同参数在独立输出目录重跑 **RR_clique seed0** 一次:embedding 的 sha256 必须与原 run 完全一致(cudnn.deterministic + 固定种子 + 同环境)。一致 → 初始化参考与算子哈希的可复现性得到端到端证明;不一致 → 全部 25 次结果降级为"确定性存疑",如实报告。预算 +0.27h,仍在 cap 内。

## 6. 边界

- Stage A 通过≠独立确认完成:单城市单实例,Stage B 才是第二个实例;本实验相对 perm0 筛选是独立的(新实例 + 零复用),但对"城市级外推"仍非独立样本。
- 认证级 manifest 覆盖本机运行;不追溯认证 perm0 存档(其 exploratory_only 状态不变,仅用于历史对比)。
- 路径分解路径依赖;RP 不训练,不宣称唯一因果占比。
