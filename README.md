# VETO: Evidence-Constrained Harness Selection for Chart Reasoning

本项目以 [krafton-ai/WHALE](https://github.com/krafton-ai/WHALE) 的公开代码为 baseline 和 infrastructure。WHALE 的交替优化框架、RSFT、Meta-Harness、环境与训练器均不是本项目贡献。

必须引用：Haechan Kim, Yoonho Lee, Gisang Lee, Chelsea Finn, Kangwook Lee. **WHALE: A Simple Recipe for Joint Harness-Weight Optimization.** [arXiv:2609.00196v1](https://arxiv.org/abs/2609.00196v1), 2026.

**VETO — Visual Evidence Tests for Optimization** 研究一个具体问题：选择下一轮训练使用的答题策略时，要求模型同时答对原图和答案变化图，是否能改善训练后的图表推理表现？方法效果仍待验证。

当前执行[2026-09-15精简协议](ours/fast_chart_protocol_20260915.md)：图表领域、Qwen3.5-4B、种子42/43/44、一次完整训练—搜索—续训交替。18个逻辑条件共享同一种子的共同前缀和候选测量，分支进行真实续训。原51组矩阵、Chess、CLEVR和2B扩展暂停。

真实视觉训练—搜索—续训闭环已完成，固定结构化h0与LR=1e-6。seed42原三种规则均选择h3；第二阶段开发分数为单图447/512、两图同时正确212/256，尚无VETO独立收益证据。VETO v2选择h1后在独立R512上低于h3。VETO v3进一步采用不确定性感知安全审计；新的H128/C256档案将h1、h2判为FAIL、h3判为UNCERTAIN，WHALE、VETO及匹配点估计门最终都选择h0。由于没有不同选择，V3按预注册规则停止，没有执行后续RSFT或扩大到其他种子。见[V3完整诊断](results/veto-v3-seed42-complete-diagnostic-20260917-v1.json)与[当前状态](PROJECT_STATUS.md)。

公开快照位于[GitHub whale](https://github.com/kevin-yihang-chen/whale)，独立发布目录是`Documents/whale`。本工作区保留完整本地历史及实验数据；公开快照只包含审阅后的代码、文档、聚合结果和论文稿，不包含模型、数据、原始轨迹或凭据。

## 方法与实验

唯一方法模块是可关闭的接受门。V3的E1把配对正确率分解为边际能力与单侧脆弱性；E2使用多重比较调整后的配对区间，将候选判为PASS、FAIL或UNCERTAIN；E3只在PASS候选和始终保留的incoming中沿用WHALE普通H准确率、turn数排序。E4仍复用原生RSFT成功轨迹训练。本次V3未产生不同选择，因此没有执行E4分叉；该机制仍是已测试但未验证的方法假设。

| 条件 | 用途 | 种子数 |
|---|---|---:|
| weight-only | 固定共同策略训练 | 3 |
| harness-only | 固定初始权重优化策略 | 3 |
| WHALE | 普通准确率选择后续训 | 3 |
| VETO | 配对接受门选择后续训 | 3 |
| 单图准确率门 | 使用同一接受数据的简单对照 | 3 |
| 反事实数据增强 | 同预算下以50%配对样本替换第二阶段训练输入 | 3 |

派生数据集PlotQA-EvidencePairs包含大小比较、单值读取和两值差值，按源图表分组保留W/H/C/V/T/R角色。答案解析与评分由所有条件共享的适配器控制，候选不能修改评分器。C用于优化；封存T不参与提示词、学习率或检查点选择。外部ChartQA使用原始开放答案任务的固定512题子集，派生任务成绩和子集成绩不会标为官方完整benchmark成绩。

全部种子、失败结果、逐样本答案及实际成本均保留；按源图表计算区间。共同计算在账本只计一次，方法比较中明确分摊。继续或停止扩大实验依据冻结协议，不能从工程验收直接推断方法有效。

## 复现与证据入口

- [当前执行命令与模型交接](ours/fast_chart_runbook.md)
- [冻结的精简实验协议](ours/fast_chart_protocol_20260915.md)
- [运行状态、实际结果与资源账本](PROJECT_STATUS.md)
- [来源及修改记录](CHANGES.md)
- [完整实验历史](EXPERIMENTS.md)
- [方法正文源码](ours/paper/main.tex)与[补充材料](ours/paper/supplement.tex)
- [审稿证据核对清单](ours/paper/reviewer_checklist.md)
- [实现及公式对应](ours/README.md)
- [VETO v3协议](ours/veto_v3_protocol_20260916.md)
- [VETO v3完整负向诊断](results/veto-v3-seed42-complete-diagnostic-20260917-v1.json)
- [下一版受控路线建议](ours/veto_v4_route_adjustment_20260917.md)

正文与补充材料已可编译；未知结果保持明确占位。当前构建命令与证据索引见runbook，编译成功不表示论文实验证据已齐全。

总预算100GPU小时，包含已有视觉支出；新增存储上限400GiB；项目GLM API累计上限45元。每次提交前核查预算，失败和加载成本也计入。任何清理先提供具体路径、引用和保留清单并获得明确授权；旧B组数据不获清理授权。作业配置状态邮件通知。公开仓库只同步经过审阅的提交。

## 研究历史

下列文档保留最初提案与复现调查，当前执行范围以9月15日精简协议为准。

- [代码解剖](notes/whale_anatomy.md)
- [10 个研究切入点](notes/attack_surface.md)
- [最初 thesis 与证伪实验](notes/thesis.md)
- [相关工作碰撞检查](notes/novelty_screen.md)
- [Claude、GLM与本地proposer的接线和预算](notes/proposer_options.md)
- [本次更新前的README历史快照](ours/history/README_20260915_before_compact_scope.md)

上游完整 checkout 在 `upstream/WHALE/`，固定于 `fbe125eb7abea7f760c99ab9acc1a6261e708fc6`。顶层 [LICENSE](LICENSE) 与 [NOTICE](NOTICE) 是上游原文件的完整副本，上游副本也保留。没有修改上游文件；未来新增实现只放 `ours/`，不改任何 `verl/`，对上游修改必须提供明确 patch 和修改说明。

用户已授权实现精简计划，覆盖最初仅文档review状态。当前受限预算实验不称为原作者完整规模复现；原冻结计划和历史结果仍保留。

研究代码保持一个可关闭的接受模块；视觉数据、环境和训练接线是共享基础设施，须单独记账。已有 CAT / CPL / PAPO / CFPO 等工作覆盖了视觉捷径诊断和反事实训练的大量思想。只做领域迁移或拼接既有损失，增量不够；本项目不预设提升比例，也不以故意弱化 baseline 或增加无用途脚本制造贡献。

## 获取代码与版本回溯

```bash
git clone --recurse-submodules https://github.com/kevin-yihang-chen/whale.git
cd whale
```

`upstream/WHALE`是固定commit的submodule。实验工作区保留更完整的本地历史、原始证据和运行产物；公开仓库只包含审阅后的代码、文档、聚合结果和论文草稿。环境、凭据、缓存、下载数据、原始轨迹与checkpoint不纳入版本库。
