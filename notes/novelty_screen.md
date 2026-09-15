# 候选增量的相关工作碰撞初筛

2026-09-09。只依据论文、官方项目和正式论文站点；下面区分摘要与方法阅读。有限初筛不能证明没有撞题。WHALE 全文及研究执行链已审读；不能把其余所有论文都说成全文审读。

当前候选：**VETO 在视觉 harness—权重交替中以配对图像正确性约束接受；先检验交替是否放大视觉捷径，再判断 gate 的独立价值。** 多分支试训 lookahead 因用户更新的轻量目标不再是当前方法。

同日补查：[CHILL-Harness v1](https://arxiv.org/html/2607.25825v1)的摘要及方法中
success-preserving objective已有反事实workflow效应学习与成功率退化约束；不能认领
“首个反事实harness学习/受约束harness优化”。[Don't Blink v1](https://arxiv.org/html/2604.04207v1)
的validation policy已有visual-engagement veto用于选择性预测。二者与当前配对图像
接受位置不完全同义，但进一步限制广义“视觉证据veto”的命名和机制claim。未完成
这两篇官方源码审计，不能据有限阅读宣布严格等价或新颖性通过。

## 1. 最危险的近邻

| 原始来源 / 阅读深度 | 已有内容 | 对候选的实际约束 |
|---|---|---|
| [WHALE v1](https://arxiv.org/html/2609.00196v1)，正文/附录/公开执行链 | RSFT 与可执行 harness 交替、schedule/adaptive、任务准确率收益 | 整个框架属于 baseline；需要具体视觉失效与干预证据 |
| [CAT](https://aclanthology.org/2024.findings-emnlp.205/)，正式摘要，2024 | 反事实部分输入评估；已观察增加 demonstrations 时准确率上升但 attentiveness 下降；数据增强可改善 | “准确率掩盖输入忽略”本身绝非新发现。必须识别 harness—RSFT 的交替放大，并击败普通增强 |
| [AutoDesign](https://arxiv.org/html/2608.13560v1)，摘要、§2、§3.2 接受机制、§7，2026-08 | 固定模型的多模态 meta-harness；训练质量提高且 dev 不退化才接受更新；gate 数据不向 proposer 暴露 | “视觉 harness + 非退化 gate”也不是新机制。VETO 不能认领通用接受约束；需要配对证据与权重反馈的必要性 |
| [CPL](https://aclanthology.org/2022.emnlp-main.224/)，正式摘要，2022 | counterfactual generation 与 contrastive prompt learning，改善 vision-language 泛化 | 不认领首个视觉反事实 prompt 优化；只加对比项/改名不可行 |
| [PAPO](https://arxiv.org/abs/2507.06448)，当前摘要；另查 [ICLR 原文](https://openreview.net/pdf?id=izbBqTL8vb) 的摘要，2025/2026 | 用感知相关 KL 目标改进多模态 RLVR | “小视觉正则 + GRPO”已被覆盖；WHALE 用 RSFT，不能混淆训练路径 |
| [CFPO v1](https://arxiv.org/html/2606.23206v1)，摘要与§3.1–3.4，2026-06 | 干预高显著视觉特征、通过预测分布差异做反事实 policy regularization，接 GRPO/DAPO | 空白/抑制图像后加视觉依赖 reward/loss 是高碰撞路线；我们只有接受位置与反馈机制可能留下空间 |
| [DeFacto v4](https://arxiv.org/abs/2509.20912v4)，摘要，2026-05 修订 | 视觉证据构造/反事实样本，GRPO 下的回答/结构/证据一致性 reward，并有人工评测集 | “超越答案准确率的视觉一致性”不新；仅添加一个 grounded reward 不能算独立增量 |
| [CF-VLM v1](https://arxiv.org/abs/2506.17267v1)，摘要，2025-06 | counterfactual fine-tuning，保持对齐、事实表征稳定性和对关键编辑的敏感性 | 正确答案变化/无关编辑不变的训练思想不新；仅换到 WHALE 不够 |
| [CounterAlign](https://arxiv.org/abs/2608.21740)，摘要，2026-08 | 由专家轨迹重标指令构造反事实监督，学习 VLA 离线 reward | 不因迁移到机器人就认为反事实 reward 新颖；本次不扩展昂贵机器人训练 |

AutoDesign §3.2 是尤其直接的碰撞：它已有 harness 的非退化接受条件。因此，本项目 E2–E3 **只能作为已有选择原则的任务化应用**，不能独立宣称新优化范式。若没有 F2/F3 的交替特有证据与 F4 的必要性，VETO 也是普通组合，这个增量不够。

## 2. 当前 verdict：高风险，先允许证伪，不批准投稿级 claim

只在以下证据同时出现时，才可能形成“失效机制 + 有效干预”的独立论文：

1. 配对测试在有效且独立的视觉编辑上成立；不是文件名/文本泄漏、错误 oracle、人工挑选坏例子或一般输入扰动。
2. 相比相同预算的固定 harness 训练和 harness-only，交替具有额外的、可重复的负面反馈；共同 h0 下的成对续训能定位权重传播。
3. 普通独立复评、同配对数据的准确率打分、counterfactual augmentation、通用非退化 gate 和随机限速均不能解释方法收益。
4. 完整计入视觉 audit、训练、proposer、服务和失败成本后，两个域两个规模仍有实际收益，并在外部真实数据上验证边界。

这不是保证新颖性：最近邻多项目前只读到摘要，进入实现前还需逐项实读方法与官方代码的兼容性，并沿 AutoDesign 的接受机制引文核查。若存在同样的优化对象、成对接受信号和权重反馈解释，先报告碰撞并重写 thesis；不能靠新缩写、换数据集或增加脚本继续包装。

## 3. 共适应方向的其他背景与已排除宽 claim

| 来源 / 阅读深度 | 不应再认领的宽思想 |
|---|---|
| [Meta-Harness](https://arxiv.org/abs/2603.28052v1)，摘要/范围 | frozen-model harness 搜索、轨迹 archive、跨 held-out 模型迁移；不能声称首次 harness 迁移 |
| [Fast-Slow Thinking](https://arxiv.org/abs/2605.12484v2)，摘要/范围 | context/weights 双时间尺度、持续学习/plasticity；WHALE 中的 prompt-only FST 控制另行声明 |
| [P²O](https://arxiv.org/html/2603.21877v3)，方法§3与实验设置结构 | prompt evolution 提供困难题学习信号、context distillation 减轻推理 prompt 依赖；不能声称首次内化提示收益 |
| [Harness-1](https://arxiv.org/html/2606.02373v1)，Intro 与 trainability 要求 | harness 对可训练性、外置状态和轨迹分布的作用 |
| [LEGO-RL](https://arxiv.org/abs/2608.17393v1)，摘要/方法支柱 | 原生 harness 与训练一致性、raw token 与执行可靠性；修复基础设施不是本论文贡献 |
| [Learning to Reweight Examples](https://arxiv.org/abs/1803.09050v3)，摘要 | validation 梯度反馈与样本重加权；普通 lookahead/gradient alignment 不是新算法 |
| [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580v4)，摘要 | teacher 根据 student 更新后表现改进训练来源 |
| [ScaleBiO](https://aclanthology.org/2025.acl-long.1543/)，正式摘要 | scalable bilevel LLM 数据重加权；不能用 bilevel 这个名称制造理论贡献 |
| [System Prompt Optimization with Meta-Learning](https://arxiv.org/abs/2505.09666)，摘要 | system/user prompt 的双层优化；泛称 meta prompt 不新 |

检索覆盖：harness trainability、post-update selection、joint policy/prompt、visual harness optimization、counterfactual prompt、visual grounding reward、PAPO/CFPO、CAT、multimodal meta-harness。尚未完成系统性的全部引文追踪。本文件旨在限制 claim，不是给 proposal 出具新颖性认证。
