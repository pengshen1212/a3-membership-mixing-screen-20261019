# A3 内容效应实验链 — 最终结果(2026-10-20)

本仓库原有内容:perm0 筛选(2026-10-19,学校端已核验,commit f8420bf + MANIFEST 修复 54f02cc)。本次新增:**Stage A(perm3,认证级)与 Stage B1(perm4,认证级 + RR 认证复用)的完整结果**,均在同一运行机(RTX 4090D,torch 2.4.1+cu121)按预注册冻结协议执行。

## 结论:三实例全部 meaningful_positive

主指标 Δ_content = E_RR − E_PR(clique vs exact 读出对比的内容路径效应),主子集 = 全部 9,384 条有标签道路,判据 mean ≥ +0.02 且 95% CI 下界 > 0:

| 实例 | 实验 | Δ_content 均值 (m/s) | 95% CI | Δ_coeff | Δ_total | 级别 |
|---|---|---:|---|---:|---:|---|
| perm0 | 筛选(10-19,见根目录) | +0.0387 | [+0.0183, +0.0591] | +0.0075 | +0.0462 | exploratory_only |
| perm3 | **Stage A**(stage-a/) | **+0.0566** | **[+0.0367, +0.0765]** | +0.0090 | +0.0656 | certified(零复用) |
| perm4 | **Stage B1**(stage-b1/) | **+0.0441** | **[+0.0315, +0.0566]** | +0.0147 | +0.0588 | certified + RR 认证复用 |

三个实例方向一致、CI 重叠、均值稳定在 0.039–0.057:区域成员内容对 A3 结构评估的贡献是实例稳定的,不是单次置换的偶然。每实验逐 seed 满足 Δ_content + Δ_coeff = Δ_total(误差 0)。

## 执行与认证要点

- **Stage A**:26 runs(25 正式 + 1 复现校验),6.726h(cap 10h),全部 returncode=0。perm3 为新生成实例(seed 20261019,与 perm0 同族:全局重排、区域大小多重集逐位一致;生成与算子验证见 outputs/perm3_generation.json、operator_validation_stage_a.json,算子与训练入口逐位互证)。
- **复现判据修订(用户批准,2026-10-20"选2吧")**:位级一致 → 数值等价(MAE 漂移 ≤ 0.005 m/s,预注册于 scripts/assess_determinism.py)。Stage A 实测漂移 **0.00087**(seed 标准差的 5.4%),判 benign_nondeterminism;嵌入级发散如实报告(相对 Frobenius 0.334)= 运行级 GPU 原子序混沌、估计量级稳定。见 outputs/stage_a_determinism_assessment.json 与 FREEZE_STAGE_B1.md §7。
- **Stage B1**:16 runs(15 正式 + 1 复现校验),~4.2h(cap 5h),全部 returncode=0。perm4(seed 20261020,10,970 节点换区,与 real/perm0/perm3 均不同);**RR 十个嵌入认证复用 Stage A 产物**(启动前 preflight 与评估时双重哈希绑定,rr_reuse_hash_verified=true);B1 自身复现校验 MAE 漂移 **0.00022**,benign。
- 授权链(聊天记录,均在案):perm0 15 GPU-h(10-19)→ Stage A 认证级 10h(10-19"我给授权,同时按认证级做")→ 链式衔接(10-20"如果通过后你自动衔接上b1和b2,你做决定")→ 判据修订选项 2(10-20"选2吧")。

## 目录

- `stage-a/`:FREEZE_STAGE_A.md、configs/stage_a_freeze.json、scripts/(生成/算子/init/运行/评估/确定性评估)、outputs/(结果、账本、manifest、确定性评估、26 日志、26 嵌入含复现对照)
- `stage-b1/`:FREEZE_STAGE_B1.md(含 §7 修订)、configs/stage_b1_freeze.json、scripts/、outputs/(结果、账本、manifest、preflight、16 日志、16 嵌入含复现对照)
- `stage-b2/`:B2_DATA_REQUIREMENTS.md——新城市为城市级外推的必要步骤,**待数据提供后另行预注册**,不在本链

## 诚实声明

- 三个实例均在 Porto 同一路网:这是**同城实例级复现**,不构成城市级外推(B2 待数据)。
- 路径分解依赖指定路径(RP 未训练),不宣称唯一因果占比;Δ_coeff 在实例间有波动(0.0075–0.0147)。
- **认证口径(2026-10-20 勘误后,见根目录 CERTIFICATION.md)**:certified = manifest 完备 + 数值等价复现(Stage A 漂移 0.00087、B1 0.00022,均 ≤ 0.005 判 benign),不是 bitwise 可复现。stage_a_result.json 为 v2 口径(含 determinism 块与 certified_definition,无 suspect 标志;v1 存档为 stage_a_result_v1_bitwise_flag.json)。
- Stage A 原始账本(stage_a_runs.json)有已知重复(pre-fix 缺陷,50 条);**权威 per-run 记录 = stage_a_manifest.json**,去重视图 stage_a_runs_unique.json(unique_runs=25);B1 账本无此问题。FREEZE_STAGE_A.md §7 勘误:11,090 为 perm0 换区数,perm3 实际 10,973。
- ridge 探针存在 ill-conditioning 警告(rcond≈5e-8),跨臂、跨实验一致存在,不影响对比有效性;结果均值距阈值均有裕量(非边缘判定)。
- graph_model.pt(40 × 75.2MB)按"可选、较大"保留运行机,未上传;嵌入足以独立复核全部 MAE 与判定。
- 各目录 MANIFEST.sha256 为该目录全量校验清单(根目录原 MANIFEST 只覆盖 perm0 筛选部分,未改动);`.gitattributes(* -text)` 保证任意平台检出字节一致。
