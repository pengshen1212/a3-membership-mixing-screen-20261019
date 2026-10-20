# B2 北京数据包（a3-b2-beijing-20261020）

真实北京路网数据，供远端 B2 使用；**未生成/伪造任何数据**。

## 文件映射

| 文件 | 来源 | 说明 |
|---|---|---|
| `roadmap.geo` | beijing_tst | 真实 OSM 坐标，40,306 行 |
| `roadmap.rel` | bj_win_early | 101,024 有向边；与 beijing_tst 的 rel 字节一致 |
| `road_feature.csv` | bj_win_early | 40,306 路段特征 |
| `seg2region.csv` | bj_win_early | 8,656 区域；与 a3_bjext 的 1-num_hyperedge=8656 一致 |
| `win_speed.csv` | bj_win_early | 32,261 有标签；speed 0.085–34.99 |
| `w_by_hop_sparse.npz` | bj_win_early | hop 先验，40,306×40,306，nnz 12,100,907 |
| `edge_attr.csv` | beijing_tst | 101,024 行，与 rel 对齐 |

## 资格摘要

- 8,656 区域（size 1–5,386）、14 路型、88 边界二元组、总超边 **8,758**。
- seed/region 连续性、road_id、geo_id、rel/edge_attr 行数、hop 行数均通过（见 `qualification.json`）。
- 阈值：Porto 的 ±0.02/±0.01 m/s 不能直接沿用；北京速度尺度与标签数不同，B2 预注册时需重审。

## 重要：预算冲突

- 存档北京 40k 训练约 **2.85 h/run**（4090D，5000 episodes）。
- B2 的 25 正式 + 1 复现 = **约 74 h**，不是 B2 文档里的 7 h；**10 h cap 会在约第 3 次 run 后停止**。
- 开跑前必须二选一：
  1. 新授权约 **80 h** 的完整 B2；
  2. 预注册缩减设计（3 seeds = 15+1 runs ≈46 h；2 seeds ≈31 h），并如实报告 CI 变宽；
  3. 换更小的第二城市/子网，但需重新做数据资格。
- 不要沿用 10 h cap 直接启动；否则只会得到不完整单元格。

## 安装

```bash
export A3_ROOT=/path/to/code/DST
cp -r bj_b2 $A3_ROOT/data/bj_b2
# build-only 冒烟（可选，不训练）：
$A3_PYTHON train_graph.py --dataset bj_b2 --ver b2_smoke --variant induced_coeff_override   --coeff-seg2region data/bj_b2/seg2region.csv --override-weighting clique --build-only --device cpu
```

## 未包含

- 711 MB 原始轨迹（hop 已预计算；如需重算 hop 再单独传）。
- 北京 exact/clique 旧嵌入（B2 要求 RR/PP/PR 全部重训）。
- `dataset.py`：远端已有修改；若仍用本机 stub，需确认其不执行（本包已提供 edge_attr/features/hop，训练不依赖 geo 重算）。

## 完整性

`MANIFEST.sha256` 覆盖本目录全部文件；`qualification.json` 含每个文件 SHA-256 与资格检查。

## 代码前置（必须，否则启动即失败）

`a3_coeff_override.load_partition` 当前默认 `n_expected=11095`（Porto）。北京是 40,306，训练入口会在这里抛 `ValueError`。远端 B2 代码需：

- 在 `train_graph.py` 的 override 分支传 `n_expected=len(seg2region)`（本包为 40,306），或把 `load_partition` 的默认改为从 `seg2region` 推断；
- 北京 clique 算子实测 **nnz=480,176,828**（40,306×40,306，行和误差 4.4e-16）；必须走已有的 >20k 行分块路径，24GB 4090D 可容纳但显存/时间显著高于 Porto；
- 若远端尚未做这项修改，先做 build-only 冒烟，确认通过后再进入 B2。

## 预算再确认

北京 40k 单次约 2.85h；B2 25+1 次约 **74h**。B2 文档的 10h cap 只适用于 Porto 规模，**不能直接启动**；先决定 80h 完整预算或缩减 seeds 的预注册设计。
