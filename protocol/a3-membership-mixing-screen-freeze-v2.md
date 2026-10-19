# A3 同 perm0 系数覆盖筛选：冻结协议 v2

日期：2026-10-19。取代 [同大小类控制冻结设计](./a3-membership-mixing-screen-freeze.md) 作为主筛选取舍；原配置与原记录保留。对应复核：[a3-membership-mixing-screen-review.md](./a3-membership-mixing-screen-review.md)。执行框：`experiments/a3-membership-mixing/`。

> 同日执行修订：修复了新训练覆盖存档 `graph_model.pt` 和失败后重启重置累计预算的问题。科学指标、种子、阈值与 15 h 上限不变。`--output-dir` 将每个种子的新 checkpoint/嵌入写入 `experiments/a3-membership-mixing/outputs/pr_training/seed{seed}/`；评估器只从这个新位置读取 PR-clique。执行器在任何子进程前独占建立并写入账本；已有账本即拒绝启动，含失败/中断记录，不能自动清零续跑。旧 RR/PP 产物保持原位。当前复用缺口仍未补齐。

**状态：`authorized_pending_school_run`。用户在 2026-10-19 授权 15 GPU-hours，并确认 `exploratory_only` 复用等级；实验将在学校端 4090D 运行。本地未开始训练、未接 API；只做了 CPU 算子/入口验证。** 旧配置 `configs/screen_freeze.json` 保留，`authorized=false`。

## 1. 为什么改设计

复核指出旧设计的第一问题是 PR 与 PP 的成员不是同一次干预：PR 是同大小类重分区，PP 是原全局 perm0，`E_PR − E_PP` 同时包含成员差异与系数差异。因此旧设计不能拆解原 perm0 效应。

v2 改为**同一份 perm0 内容**，只切换逐节点系数：

- `RR = M(real content, real coefficients)`
- `PR = M(perm0 content, real coefficients)` ← 新的内容对照
- `PP = M(perm0 content, perm0 coefficients)` ← 原效应
- `RP = M(real content, perm0 coefficients)` ← 不训练；仅诊断

主路径分解：`Δ_total = E_RR − E_PP = Δ_content + Δ_coeff`，其中
`Δ_content = E_RR − E_PR`（内容改变、系数保持真实），
`Δ_coeff = E_PR − E_PP`（系数改变、内容保持 perm0）。
这是**指定路径**的分解；训练非线性，反向路径可以不同，RP 未训练，因此不宣称唯一“内容占比”。

## 2. 省 5 次训练的恒等关系

exact 的系数只依赖 `d_i = 2 + b_i`，而 perm0 区域置换保留每个节点的关联度，所以 exact 的 `q_r = q_t = q_b = 1/d_i` 在 R 与 P 中相同：

`M(P,R)_exact = M(P,P)_exact`，`M(R,P)_exact = M(R,R)_exact`。

因此 **PR-exact 不需要训练**，直接复用 PP-exact 的 5 个种子嵌入；只需新增 **5 次 PR-clique 训练**。clique 的系数 `s_i/c_i, t_i/c_i, 2/c_i` 依赖区域大小，PR-clique 与 PP-clique 不同。

## 3. 算子与训练入口验证（已做，无训练）

新模块：`路网超图/code/DST/a3_coeff_override.py`；训练入口新 variant：`induced_coeff_override`，参数 `--coeff-seg2region`、`--override-weighting`、`--build-only`。

真实 11,095 节点静态构图，固定随机四列输入，独立关联矩阵参考：

| 检查 | exact | clique |
|---|---:|---:|
| RR 与参考最大绝对误差 | 1.11e-16 | 1.39e-17 |
| PP 与参考最大绝对误差 | 1.11e-16 | 3.47e-18 |
| PR 与 PP 的随机输入最大差 | **0** | 0.8359 |
| RP 与 RR 的随机输入最大差 | **0** | 0.7913 |
| 系数行和误差 | ≤6.7e-16 | ≤8.9e-16 |
| 最小系数 | >0 | >0 |

训练入口 `--build-only`（perm0 内容 + real 系数，clique，使用存档训练环境 `C:/Users/Peng1/anaconda3/envs/jgrm_data/python.exe`）：2,199 条边，`nnz=40,567,671`，行和误差 `6.66e-16`，与独立验证一致。输出见 `outputs/operator_validation.json` 与 `outputs/train_entry_build_only.txt`。

这些是 CPU 静态算子验证，不是任务效应、旧 GPU 运行认证或独立确认。

## 4. 读出与主指标

- 固定探针：Ridge(alpha=1.0) + QuantileTransformer(normal)，`KFold(5, shuffle, random_state=0)`，匹配非 NaN 速度标签；每个臂重新拟合，方案相同但不共享拟合权重。
- 每臂/seed：`E = MAE(clique_self) − MAE(exact)`；同时报告各臂绝对 MAE。
- **主子集：全部有标签道路（9,384）**。同一份 perm0 内容改变的是全图区域消息，不再受 3,647 可动节点限制；3,482 有标签的旧“可动子集”只作连续性次要报告。
- 主指标：`Δ_content` 的 5 个共享 seed 配对均值，95% t 区间（df=4）。
- 次要：`Δ_coeff`、`Δ_total`、逐 seed、旧可动子集与补集。三者在每条路径上应满足 `Δ_content + Δ_coeff = Δ_total`（逐 seed 核对）。

## 5. 双侧判据（修正负向误读）

| 类别 | 条件 | 解释与动作 |
|---|---|---|
| `meaningful_positive` | mean ≥ +0.02 且 CI 下界 > 0 | 内容在预期方向有实际量级贡献；只进入预注册的独立确认，不据此立新方法 |
| `meaningful_negative` | mean ≤ −0.02 且 CI 上界 < 0 | 内容使对比朝相反方向改变；**预期正向解释失败，但内容有作用**；归档/并入，除非预先注册过反向假设（没有） |
| `negligible` | 95% CI 完全在 [−0.01, +0.01] 内 | 在预注册精度下有界的可忽略证据；归档/停止独立选题 |
| `inconclusive` | 其他（含跨 0、区间过宽、0.01–0.02 带） | 证据不足；停止，不得写成“内容无作用”，也不得加 seed/实例/模型追结果 |

0.02 约为存档全道路 perm0 总效应（+0.0462）的 43%。0.01 是**可忽略带**，不是“非显著=等效”；只有整个 CI 落在带内才判 negligible。

## 6. 复用资格（执行检查，不是文字声明）

`scripts/reuse_audit.py` 的结果（`outputs/reuse_audit.json`）：

- `reuse_level = exploratory_only`，`hard_blocked = 0`。
- 20 个 RR/PP 嵌入存在、variant/seed/shape/有限性一致。
- 初版审计有 8 项认证缺口；加入独立输出模块后重跑为 9 项，详见 `reuse_audit.json.certification_gaps`。`train_graph.py` 因新增 override variant 与输出参数而不同于存档哈希（编辑前观测为 `76547b…`）；若干依赖模块无存档哈希，且无 per-run manifest。产物到分区身份与运行环境/GPU 的额外未核实项见 `unverified_fields`；它们没有被升级为已认证。
- 存档训练环境存在（`jgrm_data`，torch 2.4.1+cu121，CUDA 可用）；这只是版本探测，不是旧运行环境的追溯认证。
- **这不是 certified。** 若要求 certified，必须重训 RR/PP，成本不在本预算内；本协议只允许 `exploratory_only` 且必须由用户明确确认该等级。

`scripts/run_screen_perm0.py` 在执行前检查：
1. `authorized=true` 且 `user_acknowledged_exploratory=true`；
2. `reuse_level ∈ {exploratory_only, certified}` 且无 hard block；
3. `train_graph.py`、`a3_coeff_override.py`、`a3_output_paths.py`、真实/perm0 `seg2region.csv` 的 code/data pins；
4. 独立输出目录中的 5 个目标嵌入及 checkpoint 均不存在；账本通过独占创建认领，已有账本则拒绝启动；
5. 每次运行前累计墙钟 + 2.5 h ≤ 15 h；每次 subprocess 的 timeout = 剩余预算（到点终止并记录 `cap_exceeded_mid_run`）；失败不重试；逐次写 `outputs/screen_runs.json`（返回码、耗时、累计、日志、产物存在性）。

每次启动前先落盘 `running` 记录。进程返回 0 但嵌入不存在时记 `output_missing` 并停止。失败/中断账本保留，不支持自动重启；如需继续必须另行核算原消耗并修订执行决定，不能删除账本来获得新一轮预算。

## 7. 成本与停止

- 新训练：5 次 clique（perm0 内容 + real 系数，seeds 0–4），5000 episodes。
- 训练解释器由学校端 `A3_PYTHON` 指定（需与存档环境一致：torch 2.4.1+cu121、CUDA 可用；本地参考路径为 `C:/Users/Peng1/anaconda3/envs/jgrm_data/python.exe`）。`run_screen_perm0.py` 优先读 `A3_PYTHON`，否则读配置（打包配置中为 null），最后回退到 `sys.executable`。
- 按存档本地日志约 2.5 h/次，约 **12.5 GPU-hours**（3060 估计）；4090D 同层同 variant 实测约 **0.48 h/次，5 次约 2.4 h**；硬上限仍为 **15 GPU-hours**（墙钟累计代理），两种平台都能在 cap 内完成。
- 0 次 API；评估为 CPU ridge probe。
- 触顶、运行失败、产物缺失、pins 不符、复用未确认 → 停止并报告不完整单元；不延长、不换 seed、不调阈值。
- 执行器 `--dry-run` 已验证：5 条命令、12.5 h 保守估算、15 h 上限；授权后不再因 `authorized=false` 拒绝。评估器在 5 个 PR-clique 嵌入缺失时退出码 2。

## 8. 结果解释边界

- 一个 perm0 实例 + 5 个训练 seed 不是独立结构/城市样本；本筛选是门，不是独立确认。
- `Δ_coeff` 是**perm0 内容下**的系数效应；`Δ_content` 是**真实系数下**的内容效应；两者之和是指定路径的总效应，不是唯一因果份额。
- exact 臂对内容敏感但对系数不变；`PR-exact = PP-exact` 不表示内容对 exact 读出没有影响。
- 旧同大小类控制（3,647 节点、局部重分区）保留为次级受限对照；32.9% 不是固定系数干预的一般上限（`M(P,R)` 可作用于全图）。
- 不做新方法/新归一化声明；正向结果只解释现有结构评估。

## 9. 当前状态与启动条件

- 控制/算子/入口：已通过 CPU 验证；执行修订后 17 项无训练单测通过（新增 4 项产物/账本反例）。
- 复用：`exploratory_only`，用户已确认；certified 不可得。
- 预算：5 次/12.5 h 保守估算（3060），15 h 上限；4090D 预计约 2.4 h；执行器已实现，`--dry-run` 可检查命令与预算。
- 评估器：`scripts/eval_screen_perm0.py` 在 5 个 PR-clique 嵌入缺失时退出码 2，不编造结果。
- **授权已给（15 GPU-hours，2026-10-19），本地未训练、未接 API；学校端运行包与手册见 `release/a3-membership-mixing-screen-20261019.zip`。** 学校端仍需 `reuse_audit.py` 通过（exploratory_only 或 certified，无 hard block）和 code/data pins 匹配；`run_screen_perm0.py` 会在任何训练前检查。
