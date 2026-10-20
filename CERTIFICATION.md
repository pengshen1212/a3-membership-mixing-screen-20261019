# CERTIFICATION.md — 本链条 "certified" 的定义(2026-10-20)

应学校端核验要求,本文件显式定义本仓库 Stage A / Stage B1 所声称的认证级别。**核心结论:certified = manifest 完备 + 数值等价复现,不是 bitwise 可复现。**

## 1. 认证包含什么(每 run 均记录于 stage_a_manifest.json / stage_b1_manifest.json)

- **代码哈希**:A3_ROOT 全部 9 个 py 文件(含 dataset.py 桩与表头版 roadmap.geo 桩的说明,见各 freeze 文档与 CHAIN_RESULTS.md)
- **数据哈希**:该 run 实际读取的全部输入(共享输入 + 分区 seg2region + PR 臂的 real seg2region);perm3/perm4 seg2region 由生成报告绑定(生成 seed、算法、SHA256)
- **环境**:python / torch 2.4.1+cu121 / CUDA / GPU / 主机
- **算子**:训练前 operator probe 与训练入口 `--build-only` 逐位互证(nnz/行和/稀疏字节哈希;Stage A 曾以诊断副本进程内证实字节级一致)
- **初始化**:pinned 模块复刻构造序列的 state_dict 哈希,5 seed 双跑确定性验证
- **输出**:embedding / graph_model / 日志哈希、返回码、墙钟
- **复用绑定**(仅 B1):RR 十个嵌入对 Stage A manifest 的哈希绑定,启动前 preflight 与评估时双重核验

## 2. 复现性判据:数值等价(用户批准的修订,2026-10-20"选2吧")

- 原判据"位级一致"实测失败(GPU 原子操作累加顺序非确定;嵌入相对 Frobenius 0.334、行余弦均值 0.959 = 运行级混沌)。
- 修订后判据(计算前预注册):相同参数重跑一次,**冻结探针 MAE 漂移 ≤ 0.005 m/s**(0.02 阈值的 25%)。
- 实测:Stage A(RR_clique s0 重跑)**0.00087**;Stage B1(PP_clique s0 重跑)**0.00022**。两者均判 `benign_nondeterminism`,即"运行级混沌、估计量级稳定"。
- 证据:stage-a/outputs/stage_a_determinism_assessment.json(含嵌入级统计的如实报告);stage-b1/outputs/stage_b1_result.json 的 repro_numerical_equivalence 块。bitwise 结果保留记录,不作判据。

## 3. 文件口径(勘误后)

- **stage_a_result.json(v2,当前)**:含 determinism 块(bitwise_identical=false + assessment_verdict=benign_nondeterminism + mae_drift)与 certified_definition;无 determinism_suspect 标志。v1(bitwise flag 口径)存档为 stage_a_result_v1_bitwise_flag.json,git 历史 3b5b570 亦有原版。
- **stage_b1_result.json**:原生成即一致(repro_numerical_equivalence.benign=true,无 flag)。
- **账本权威性**:stage_a_runs.json 原始账本有已知重复(pre-fix save_all append 缺陷,50 条 = 25 run × running/finished);**权威 per-run 记录 = stage_a_manifest.json**;去重视图 stage_a_runs_unique.json(unique_runs=25,全部 returncode=0)。B1 账本为 upsert 修复版,无此问题。
- **勘误**:FREEZE_STAGE_A.md §7(11,090 为 perm0 数;perm3 实际 10,973)。

## 4. 边界(不可由此声称)

- 不声称 bitwise 可复现;不声称跨硬件/跨驱动可复现(manifest 记录了本机环境)。
- certified 复用(RR@B1)的效力限于本机 Stage A 链条;不追溯认证 perm0 筛选的 exploratory_only 复用。
- 三实例同城(Porto):实例级复现,非城市级外推(B2 待数据)。
- 路径分解指定路径,RP 未训练,无唯一因果占比声明。
