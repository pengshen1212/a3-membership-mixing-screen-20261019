# Stage B1:第二个新成员实例 perm4——冻结协议(条件授权)

日期:2026-10-20 冻结(Stage A 运行中,结果未知时预注册)。上游:Stage A(perm3,认证级)。

## 0. 授权与触发条件

- **用户授权(聊天原文,2026-10-20 凌晨):"可以,a如果通过后你自动衔接上b1和b2,你做决定"。**
- 本协议为**条件授权**:仅当 `outputs/../a3-membership-mixing-stage-a/outputs/stage_a_result.json` 的 `verdict == meaningful_positive` 时,B1 才允许启动。Stage A 为 `meaningful_negative`/`negligible`/`inconclusive` 或未完成/复现校验失败 → B1 永不启动,按 Stage A 停止规则处理。
- 运行方决定(用户委托):B1 采用 **RR 认证复用** 设计——Stage A 在本机以完整 manifest 重训了 RR 臂,其嵌入哈希可被绑定核验,构成协议意义上的 certified 复用;PP/PR 针对新实例 perm4 重训。
- 预算:**cap 5 小时墙钟**(15 次训练 + 1 次复现校验,实测 ~0.26h/次,估 4.2h)。

## 1. 新实例 perm4

- 生成规则与 perm3 同族同算法(全局节点均匀重排,消费真实区域大小直方图,区域 id 升序),**生成 seed = 20261020**(≠ perm3 的 20261019)。
- 验证(不通过则拒绝启动):geo_id 0..N-1;区域 id 连续;大小多重集逐位一致;与 real/perm0/perm3 赋值数组均不同;pinned loader 可加载、2,199 边;生成报告落盘(含 seg2region SHA256)。
- 数据目录 `data/het_g8p0_perm4/`:新 seg2region.csv + 共享输入字节级副本(哈希与 het_g8p0 逐一相同)+ 表头版 roadmap.geo 桩(说明文件同前)。

## 2. 臂与训练

| 臂 | 内容 | 系数 | 来源 |
|---|---|---|---|
| RR | real | real | **认证复用 Stage A 产物**(emb sha256 必须与 stage_a_manifest 逐一匹配,启动前核验) |
| PP | perm4 | perm4 | 重训 induced_exact + induced_clique_self,seeds 0–4(10 次) |
| PR | perm4 | real | 重训 induced_coeff_override(clique),seeds 0–4(5 次);PR-exact = PP-exact(恒等,启动前对 perm4 实例重验) |
| RP | real | perm4 | 不训练 |

共 **15 次训练**;顺序 PP_exact → PP_clique → PR_clique(各 s0–s4)。**复现校验**:PP_clique s0 在独立输出目录重跑一次,嵌入 sha256 必须与原 run 逐位一致(维持 Stage A 同级认证标准)。

## 3. 认证级 manifest(与 Stage A 同标准)

代码/数据哈希(perm4 seg 以生成报告绑定)、环境、算子探针(nnz/行和/sha256,复用 operator_probe 逻辑对 perm4 重跑 exact 恒等 + clique 内容差)、init 参考(同 Stage A 的 init_reference.json,构造序列跨实例相同——启动时断言文件存在且 all_deterministic)、逐 run 输出/日志哈希、RR 复用哈希绑定记录。

## 4. 读出与判据(与 v2/Stage A 逐字一致)

同 `stage_a_freeze.json` 的 probe/estimands/thresholds:9,384 有标签道路;Δ_content 主指标;四类判定;Δ_content + Δ_coeff = Δ_total 逐 seed 校验;次要子集换为 perm4 movable。

## 5. 结果解释

- B1 通过 + Stage A 通过 = **同城第二实例复现**;仍非城市级外推(B2/新城市才解决)。
- 评估器须同时报告 perm3(Stage A)与 perm4(B1)的 Δ_content/Δ_coeff/Δ_total 对照表。
- B1 不通过/反转/不确定:独立选题停止,perm3+perm4 结果一并并入 A3 论文;不追加实例/seed。

## 6. 边界

- RR 认证复用的效力限于"本机 Stage A 运行"这一链条;不追溯认证 perm0 筛选的 exploratory 复用。
- 路径分解仍为指定路径;RP 不训练;不宣称唯一因果占比。
- B2(新城市)不在本协议内:数据资格是硬前置,见 `../a3-membership-mixing-stage-b2/B2_DATA_REQUIREMENTS.md`。

## 7. 修订记录(2026-10-20,用户批准选项 2)

用户聊天原文:"选2吧"。复现判据由**位级一致**修订为**数值等价**:

- **门**:重跑与原 run 在冻结探针下的 MAE 漂移 ≤ 0.005 m/s(0.02 阈值的 25%)。
- **Stage A 实测**(评估文件 `../a3-membership-mixing-stage-a/outputs/stage_a_determinism_assessment.json`):漂移 **0.00087 m/s** = seed 标准差(0.0161)的 5.4%,判 `benign_nondeterminism`;嵌入级发散如实报告(相对 Frobenius 0.334、行余弦均值 0.959)——运行级混沌、估计量级稳定,正是本修订要容纳的模式。
- **触发条件**随之改为:Stage A `meaningful_positive` **且** determinism 评估为 benign;原 bitwise 要求作废(保留在 Stage A manifest 中作历史记录)。
- **B1 自身复现校验**同样按数值等价评判:`run_stage_b1.py` 记录哈希+嵌入统计,`eval_stage_b1.py` 计算 MAE 漂移,>0.005 才标 `determinism_suspect`。
