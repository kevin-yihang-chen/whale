# WHALE attack surface：10 个可检验的研究切入点

2026-09-09；版本与代码定位见 [whale_anatomy.md](whale_anatomy.md)。这是研究机会清单，不是宣称原作有十项错误。区分：已有结果、尚未覆盖的边界、我们待检验的假设。没有实验数值。

## 1. 当前成功率是否是下一轮训练 harness 的正确目标？

**证据：** [Math acceptance](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/math_reasoning/meta_harness/meta_harness_retool.py#L376)、[Chess acceptance](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/chess_puzzles/meta_harness/chess_puzzle_benchmark.py#L164) 均按当前 frozen weights 的成功率选，再交给下一轮 RSFT。没有试训后的效用评估。

**缺口：** 同样正确的轨迹可能具有不同的可学习性；更高当前成功率不保证更新后提升更大。原作承认耦合，却没有用 matched branch intervention 建立这一具体排名关系。

**证伪：** 同一 checkpoint、相同数据次序/预算下，从同一 archive 对多个 harness 真实分叉试训；在共同、独立且冻结的评测 harness 上比较学习增益。若当前排名已足够预测长期排序，或低当前分候选的优势经独立复评消失，停止这条增量。

**贡献门槛：** 必须产生可重复的“部署质量与训练效用分离”证据及预算匹配的有效选择方法；仅增加 lookahead 或把它称为 bilevel，不足以投稿。

## 2. 切换准则已有消融，但缺少噪声和信号偏移校准

**已有：** 原作 §6.2 研究多种 fixed schedule、stagewise 和 adaptive；不能写“未做 schedule 消融”。

**缺口：** [trainer patience](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/math_reasoning/verl/trainer/ppo/ray_trainer.py#L1632) 是滑窗训练 reward 的严格新高；[MH patience](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/chess_puzzles/meta_harness/meta_harness_chess_puzzle.py#L557) 是反复搜索后的 archive 最大值。这两个信号的噪声、winner selection bias 和非平稳性不同。

**证伪/诊断：** 匹配算力下改 rollout 重复数、harness训练样本量和采样温度，检验 phase 长度/赢家是否只因观测噪声改变。先和强 fixed / 已修复 adaptive 比。仅调 patience 常数增益小，独立贡献弱，不选为主线。

## 3. 目标模型 rollouts 不能代表完整成本

**证据：** [MH 选择](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/search_qa/meta_harness/meta_harness_search_r1.py#L355) 以成功率优先、turn 数平分；proposer 是另外的 Claude session，包含读日志、重试与代码执行。不是完全没考虑成本。

**缺口：** proposer、judge、SFT backward、服务切换、候选失败、输入上下文长度没有统一预算约束。token 数也不是不同 GPU/模型的等价成本。

**证伪：** 完整成本向量下重画质量—预算曲线，给 WHALE 相同硬件时间和 API 额度。若新方法优势只存在于不计 probe/proposer 的口径，效率 claim 不成立。把成本算清本身是评估改进，除非改变决策并带来稳健收益，否则贡献不足。

## 4. 反复在小 harness 训练集上取最大值，缺少独立 acceptance 验证

**证据：** 默认候选一次/题、固定256题，`val.json` 被缓存；新候选从先前同集合的失败/答案中产生。[Chess sweep](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/chess_puzzles/meta_harness/chess_puzzle_benchmark.py#L56)。这是 harness training set，文件名 val 不能改变角色。

**缺口：** 没有独立验证能区分真实候选提升、采样误差和对搜索集的适应。

**证伪：** 赢家/runner-up 在从未向 proposer 展示的样本上等量复评；与同等预算的重复评测贪心法比较。若新方法仅靠多一份 holdout 消除了 winner's curse，须归因于评估协议，不是学习效用目标。

## 5. 共适应后的权重收益能否离开训练 harness 保留？【辅助线】

**证据：** [外层传递](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/run/lib/alternate.sh#L158) 总把 winner 作为下一 phase 的训练 harness；[hook](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/math_reasoning/verl/utils/meta_harness_hook.py#L228) 把相应 prompt/反馈带入训练。没有公开的完整 checkpoint × training-harness × evaluation-harness transfer tensor 实验清单。

**缺口：** 需要分离新 harness 当场提供的帮助、模型只适应该接口、以及可跨接口复用的权重提升。原 [Meta-Harness](https://arxiv.org/abs/2603.28052) 已研究 frozen-model harness transfer，所以不能泛称“第一次测试 harness 迁移”。

**证伪：** 真实分叉后固定 weights，交叉换回 h0、incoming、其他冻结候选；若收益仅在配对对角线上，不能宣称可迁移学习增益。跨模型规模/seed 的 held-out 接口是辅助检验，不借迁移评测宣称拥有第二个新算法。

## 6. 独立种子、测试点选择与误差条的可靠性

**证据：** 三域 MH 配置/默认只有 seed42；Chess [load_results key](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/chess_puzzles/meta_harness/chess_puzzle_benchmark.py#L107) 不包含 seed。论文 best-test 指标和每题8次采样不等于独立训练重复；在已检查正文/附录中未见每条件三个独立训练 seed 的 mean±std 表。

**缺口：** 训练、proposer、采样、checkpoint selection 的方差没有完整分离；不能从配置推断论文实际只跑过一次。

**证伪：** 三个完整独立 run、固定测试清单、事前 checkpoint 选择，展示所有 seed 和 sample std；额外给题目聚类的不确定性。若效果小于运行间方差或只靠 best checkpoint，则停止强 superiority claim。仅补种子不算方法贡献。

## 7. 初始 h0 的脆弱性与搜索空间扩大的混淆

**证据：** [Search h0](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/search_qa/environments/search_r1/base_harness.py) top1、2 turns；[Math h0](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/math_reasoning/environments/retool/base_harness.py#L42) 以文本方式包 print；Chess parser/重试也是可优化表面。

**缺口：** 部分提升可能来自修复弱接口，或较大的推理预算，而非新的联合学习机制。

**证伪：** 加一个独立开发、未看正式test的强手工 h0，对齐 assistant/context/tool预算；prompt-only 与 full-harness 分开。若增量仅在刻意弱 h0 成立，写入 limitations，不能宣称普遍优化优势。

## 8. 域与模型规模没有完整交叉

**证据：** 公开 [launchers/config](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/docs/reproducing.md) 中 SearchQA/Math 为2B，Chess为4B。

**缺口：** 域差异与模型规模差异无法由这个配置区分；也未由相同域内的2B×4B矩阵建立规模稳定性。

**证伪：** 至少 Math/Chess × Qwen3.5-2B/4B 全交叉，各三个种子。若只有一个格子有效就缩小 claim；不以跨域的两个不同规模冒充“每域两个规模”。单纯加规模贡献弱。

## 9. RSFT 成功轨迹的覆盖度与训练权重未被因果分离

**证据：** [filter](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/math_reasoning/verl/trainer/ppo/ray_trainer.py#L1561) 保留所有成功轨迹；[loss](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/math_reasoning/verl/workers/actor/dp_actor.py#L716) 按token计算。一次采样中同题多次成功会产生多份训练样本，没有按unique题目去重的设计。

**缺口：** 接受数量、样本覆盖、生成token长度、已会题重复与能力增益的因果关系未隔离。**RSFT不是GRPO，不能直接套用“全组成功导致advantage为零”的解释。**

**证伪：** matched rollout budget 是主比较；另做 equal accepted-token/equal SFT-update、同题配对子集、uniform archive replay 的机制对照。若优势完全由更多监督token或常规重采样解释，主线必须降格，不包装为新 harness 原理。

## 10. 视觉任务中的交替反馈是否放大对非视觉捷径的依赖？【主线】

**证据：** [三个 h0](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/docs/data.md) 输入是文本QA、数学文本、FEN/ASCII棋盘；多模态模型架构不等于做了视觉实验。

**缺口：** 原作没有验证：harness 的当前答题成功率提高时，对图像中答案相关变化的正确响应能否保留；RSFT 是否把接口诱导的捷径进一步写入权重。单独的视觉忽略现象已有 [CAT](https://aclanthology.org/2024.findings-emnlp.205/) 等工作，不能认领。候选新问题是 executable harness 搜索与成功轨迹训练之间的反馈放大，以及干预应该放在哪个 phase。

**证伪：** 先对普通 WHALE 做每个 phase 前后的真实图像配对审计；若没有相对固定 harness / harness-only 的额外退化，交替放大叙事不成立。进一步用等预算、相同配对数据的普通增强/重评分打败候选 gate，即否定 gate 的独立价值。新机制不能只在故意制造的偏置或单个渲染器上成立。主线候选名 VETO，具体 gate、证伪门和局限见 thesis。

## 选择结论

根据用户更新的“轻量模块 + 视觉任务”目标，当前选 **10 为主线、5 为辅助线**。辅助线检验视觉可靠性收益在换回冻结 h0 后能否保留，不单独宣称新迁移算法。1 的真实分叉试训只保留为备选研究机会，因开销与当前目标不符，不进入本次实现计划。3/4/6/7/8/9 是成本、数据与归因对照；2 暂不发展新调度算法。公开 release 的缺文件与编排问题单列在 anatomy，修复它们不计科学贡献。

下一步只 review [thesis.md](thesis.md)。目前是有明确证伪条件的候选增量，尚没有支持证据。
