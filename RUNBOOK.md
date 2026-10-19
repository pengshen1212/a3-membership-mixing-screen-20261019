# A3 同 perm0 筛选：学校端运行手册

授权：用户于 2026-10-19 授权 **15 GPU-hours**，并确认 `exploratory_only` 复用等级。配置已置 `authorized=true`、`user_acknowledged_exploratory=true`。**不要修改阈值、种子、判定规则或预算上限。**

## 0. 环境

- Linux，Python 环境含 `torch 2.4.1+cu121`、CUDA、`dhg`；使用学校端与存档一致的训练环境（torch 2.4.1+cu121、CUDA、dhg）；具体解释器由 `A3_PYTHON` 指定。
- 路径变量（按实际修改）：
  ```bash
  export A3_ROOT=/root/autodl-tmp/dst/code/DST
  export A3_PYTHON=/root/miniconda3/envs/jgrm_data/bin/python   # 必须能 import torch,dhg
  ```

## 1. 安装代码与数据

```bash
# 备份学校端原文件（不要跳过）
mkdir -p $A3_ROOT/backup_20261019
cp $A3_ROOT/train_graph.py $A3_ROOT/model.py $A3_ROOT/config.py \
   $A3_ROOT/preprocess.py $A3_ROOT/utils.py $A3_ROOT/a3_exact.py \
   $A3_ROOT/backup_20261019/ 2>/dev/null || true

# 必需的新/改代码（code pins 只检查这三个）：
cp a3_code/train_graph.py a3_code/a3_coeff_override.py a3_code/a3_output_paths.py $A3_ROOT/
# 可选基线文件：仅当学校端的 model/config/preprocess/utils/a3_exact 与打包版本不同才替换
# cp a3_code/model.py a3_code/config.py a3_code/preprocess.py a3_code/utils.py a3_code/a3_exact.py $A3_ROOT/
cp a3_meta/*.txt $A3_ROOT/

# RR/PP 存档嵌入 + 共享输入；若学校端已有同内容数据，可跳过并用 reuse_audit 核对
cp -r a3_data/het_g8p0 $A3_ROOT/data/
cp -r a3_data/het_g8p0_perm0 $A3_ROOT/data/
```

注意：`a3_data` 只含本次需要的文件（两个数据集的 seg2region、共享输入、各 10 个 RR/PP 嵌入），不覆盖其他数据集。不要删除 `$A3_ROOT/data/het_g8p0` 里的其他文件。

## 2. 复用与命令检查（不训练）

```bash
cd hypergraph/experiments/a3-membership-mixing
export A3_ROOT A3_PYTHON
$A3_PYTHON scripts/reuse_audit.py
$A3_PYTHON scripts/run_screen_perm0.py --dry-run
```

期望：

- `reuse_audit.json` 的 `reuse_level` 为 `exploratory_only`（或 `certified`），`hard_blocked_reasons` 为空。若为 `blocked`，停止并回传 `reuse_audit.json`，不要绕过。
- dry-run 显示 **5 条** `induced_coeff_override` 命令，`n_runs=5`、`cap_hours=15`，并显示 `A3_PYTHON` 路径。
- 如果 code/data pins 不符：说明学校端代码/分区不是打包版本，停止并回传错误，不要改 pin。

## 3. 正式运行（5 次，4090D 预计约 2.4 h）

```bash
$A3_PYTHON scripts/run_screen_perm0.py
```

- 输出：`outputs/pr_training/seed0..4/`（`graph_model.pt` + `emb_induced_coeff_override_s*.pt`）；日志 `outputs/screen_runs/seed*.log`；账本 `outputs/screen_runs.json`。
- 运行器在任何子进程前建立账本；已有账本会拒绝启动（失败/中断也不能自动清零续跑）。若需重来，先把旧账本和 `outputs/pr_training` 移到别处并保留，再请示；不要删除证据。
- 15 h 为硬上限；每次子进程 timeout = 剩余预算；失败不重试。

## 4. 评估

```bash
$A3_PYTHON scripts/eval_screen_perm0.py
```

- 读取存档 RR/PP + 新建 PR-clique；PR-exact 直接复用 PP-exact（exact 恒等）。
- 输出 `outputs/screen_result_perm0.json`，按预注册规则给出 `meaningful_positive` / `meaningful_negative` / `negligible` / `inconclusive`。
- 不训练 RP；残差不是纯权重效应。

## 5. 回传清单

1. `outputs/screen_result_perm0.json`
2. `outputs/screen_runs.json`
3. `outputs/screen_runs/seed*.log`
4. `outputs/pr_training/seed*/emb_induced_coeff_override_s*.pt`（5 个，约 56 MB）
5. `outputs/reuse_audit.json`（学校端生成的）
6. `outputs/pr_training/seed*/graph_model.pt`（可选，较大；至少保留到结果确认）

## 6. 事实边界

- 一个 perm0 实例 + 5 个 seed 不是独立确认；本筛选是“是否继续投入”的门。
- `Δ_content` 与 `Δ_coeff` 是同一路径的两步分解；非线性训练下反向路径可不同，RP 未训练，不宣称唯一因果占比。
- 旧 RR/PP 复用为 `exploratory_only`：入口/部分代码与产物身份未完成 certified 认证；结果按探索性报告。
- 无 API 调用；全部为本地 GPU 训练 + CPU ridge 读出。
