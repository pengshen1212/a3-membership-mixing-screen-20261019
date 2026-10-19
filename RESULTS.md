# A3 同 perm0 系数覆盖筛选 — 本机 4090D 运行结果

运行日期:2026-10-19。按包内 `RUNBOOK.md`(授权 15 GPU-hours,`exploratory_only` 复用)在本机 RTX 4090D 完成 5 次 PR-clique 训练与预注册评估。

## 结果(预注册四类判定)

**verdict: `meaningful_positive`(正向有意义)**

| 预注册估计量 | 均值 (m/s) | 95% CI |
|---|---|---|
| **Δ_content(主指标,内容效应)** | **+0.0387** | **[+0.0183, +0.0591]** |
| Δ_coefficient(系数效应) | +0.0075 | [+0.0005, +0.0146] |
| Δ_total(路径总效应) | +0.0462 | [+0.0274, +0.0651] |

- 主子集:全部有标签道路 9,384 条;判据 mean ≥ +0.02 且 CI 下界 > 0,满足。
- 次要子集(3,482 可动道路):Δ_content = +0.0831 [+0.0405, +0.1257],与包内事后分层参考 +0.0795 [+0.0387, +0.1204] 一致。
- 逐 seed 加和 Δ_content + Δ_coeff = Δ_total 最大误差 0.0;Δ_total 均值 +0.0462 与存档 perm0 全道路总效应一致。
- 按停止规则:内容在预期方向有实际量级贡献,只进入预注册的独立确认设计,不据此立新方法。

## 文件

| 文件 | 说明 |
|---|---|
| `screen_result_perm0.json` | 完整评估结果(含 design/阈值/逐 seed MAE) |
| `screen_runs.json` | 运行账本(5 run 全部 returncode=0,累计 1.294 h) |
| `reuse_audit.json` | 复用审计:`exploratory_only`,0 hard block,9 项认证缺口(协议已知) |
| `logs/seed0-4.log` | 5 次训练日志 |
| `embeddings/emb_induced_coeff_override_s{0..4}.pt` | 新产 PR-clique 嵌入(各 ~11.4MB) |
| `protocol/` | 问题卡、v2 冻结协议(预注册判据)、筛选记录 |
| `configs/screen_freeze_perm0.json` | 冻结配置(authorized=true) |
| `RUNBOOK.md` | 学校端运行手册(原样保留) |

未上传(按手册"可选,较大"):`outputs/pr_training/seed*/graph_model.pt`(5 × 75.2MB),保留在运行机直到结果确认。原始数据目录(`data/het_g8p0*`)不在回传清单内。

## 运行环境

- Windows 10 (10.0.26100),RTX 4090D 24GB,驱动 551.68(CUDA 12.4)
- Python 3.11 venv:torch **2.4.1+cu121**(wheel 官方 sha256 校验通过)、dhg 0.9.5、torch_geometric 2.6.1、numpy 1.26.4、pandas 2.2.3、scipy 1.14.1、scikit-learn 1.5.2
- 训练前验证:17 项无训练单测通过;code/data pins(3 代码 + 2 分区)逐哈希匹配;build-only 冒烟与协议存档验证完全一致(nnz=40,567,671,行和误差 6.66e-16,2,199 边)
- 时间线:18:43 启动,每 seed ≈15.5 min,20:01 全部完成;总耗时 1.294 h(15 h 上限的 8.6%)

## 本机适配说明(未触碰任何 pin 文件)

1. `dataset.py` 桩:包内未携带 DST 原版,`preprocess.py` 顶层 import 需要;实际永不执行(road_feature.csv 已打包且被 pin)。
2. 表头版 `roadmap.geo` 桩:`train_graph.py` 经 `get_edge_attr(only_index=True)` 读一次但内容即弃;本筛选取数全部经哈希 pin 文件。
3. 环境变量 `DST_DATASET=het_g8p0`:`config.py` import 时需读默认数据集,train_graph 随后按实际数据集覆盖维度。

## 证据完整性边界

- 账本、日志、JSON 为运行机原样文件,未改写;其中含运行机本地绝对路径(Windows 用户目录),仅作路径记录,不代表仓库结构。
- `MANIFEST.sha256` 为本上传目录新生成的校验清单(原包 MANIFEST 只覆盖包内 78 个原始文件)。
- 结果为探索性(`exploratory_only` 复用):1 个 perm0 实例 × 5 seed,不是独立确认;`Δ_coeff` 是 perm0 内容下的系数效应,不宣称唯一因果占比(RP 未训练)。
