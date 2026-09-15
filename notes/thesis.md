# Thesis 待审稿：VETO

> 当前执行修订：用户要求在现有空间内优先完成单一图表域/4B/seed42视觉初筛；51组矩阵及Chess续训暂停，见[当前视觉优先计划](../ours/visual_priority_plan_20260910.md)。旧224041训练遗漏像素，已撤回其视觉训练机制解释；修正后224347完成真实图像训练，224358核实全参数变化，固定h0开发成绩前后均117/128、53/64对。真实GLM h1已完成旧theta1上的搜索/接受，不能跨权重复用。修正后权重的策略选择与续训仍待完成；没有VETO有效性证据。下文旧矩阵与F0顺序不作为当前队列。

> 2026-09-10 用户批准[新的执行协议](../ours/veto_execution_protocol_20260910.md)：收束 Chess seed42 联合闭环，暂缓其余 Chess 种子/FST，正式视觉矩阵改为51个条件运行。下文原F0的十二项前置及60-run矩阵保留为历史设计；当前执行以前述修订为准。现象、机制和公平对照的科学门槛仍需真实验证。

2026-09-09；状态 **OFFLINE_PROTOTYPE_IMPLEMENTED / SCIENTIFIC_CLAIM_UNVALIDATED**。本文件是事前研究假设与实验计划，不是已有结果。本次新Goal要求代码产出，已实现独立E1–E3模块及离线验证，覆盖初始版本的仅文档暂停状态；GPU工程采样与原生训练调试已有记录，但没有正式方法对比或投稿级贡献成立的证据。原F0仍约束真实联合实验，独立算术/合同测试不计为F0通过。当前执行状态见[PROJECT_STATUS](../PROJECT_STATUS.md)，初始要求与原版本保存在Git的initial-review快照中。

## 1. 一句话 claim 与边界

**待证伪 claim：在图像决定答案的任务中，按答题准确率进行的 harness—权重交替可能放大视觉捷径；在 harness 接受环节加入配对图像约束，能以相同总成本改善未见视觉变化上的可靠性。**

工作名：**VETO — Visual Evidence Tests for Optimization**。暂拟标题：*VETO: Visual Evidence Tests for Harness–Weight Co-adaptation*。不保证缩写独占；定稿前再检索命名。这里的“visual evidence”指模型对有已知答案变化的图像干预作出正确响应，不等于 attention 热图漂亮，也不等于证明内部推理因果忠实。

主线是 [attack surface #10](attack_surface.md)：交替反馈的视觉可靠性与接受决策。辅助线是 #5：收益能否在冻结、共同的评测 harness 下保留。WHALE、RSFT、Meta-Harness、可执行代码搜索和通用 VLM 适配均属于 baseline/infrastructure。

**若成立，对 WHALE 的扩展：** 原作在文本工具域观察到联合优化的准确率收益；本研究检验这一收益在视觉证据变化时是否伴随可靠性损失，并给出接受条件。WHALE 并没有宣称所有视觉条件下都可靠，所以不能写成推翻原作。必须进一步证明损失与交替反馈有关，不能只复述一般 VLM 的语言偏置。

**现阶段判断：** 单纯“WHALE + VLM + 视觉 loss/新名字”，这个增量不够。VETO 只是一个值得小规模证伪的候选；目前不能承诺支撑顶会投稿。最近邻见 [novelty_screen.md](novelty_screen.md)。

## 2. 先写证伪实验：任何一关失败都收缩或停止 claim

下表中的阈值是拟采用的决策标准，**不是实验结果**。在 pilot 前冻结；只能因评测器缺陷修改，修改留记录，不能看完结果再放宽。

| 顺序 | 真实实验/必要检查 | 什么结果会推翻主张 | 后续动作 |
|---|---|---|---|
| F0 原 baseline | 先恢复并跑完原文本域 weight-only / harness-only / WHALE-FST / WHALE 的小预算流程，核验训练权重进入下一 MH | 缺失组件无法溯源、权重未切换、四条件未跑通 | 报告复现前置问题或实际失败；不推进真实联合实验。独立模块可离线验证，小 subset 不声称复现原论文数字 |
| F1 视觉信息有效 | 同问题、两张有效图像、两个不同正确答案；检查 oracle、渲染、文件名/元数据；另测去图像与保答案的样式变化 | 答案从问题或文件名可直接恢复，视觉编辑不改变真值，或 oracle 错 | 修复协议；不把无效配对所得提升当证据 |
| F2 现象是否存在 | 对普通 WHALE 在每个权重/MH phase 前后测普通准确率 A 与配对正确率 P；同样测固定 harness 训练和 frozen-weight harness 搜索 | 普通 A 增长时 P 没有可重复退化；或退化在 fixed-harness 中同样明显，没有交替放大证据 | 停止“交替反馈放大”叙事，不靠挑坏 seed 继续 |
| F3 放大机制 | 在预先指定的 phase，从同一 checkpoint 对 incoming 与选中 harness 做一次成对续训，保持数据顺序、训练预算；共同 h0 上评估，另做 accepted-token 匹配诊断 | 差异只来自不同接受 token 数、推理预算、一次性 prompt 效果或普通训练遗忘 | 不声称捷径经交替被内化；辅助线失败须明确披露 |
| F4 模块的独立价值 | VETO 对比用同一配对数据做普通准确率重评分、普通 RSFT 数据增强，以及独立复评贪心；相同总资源上限 | 简单对照相当或更好，或只有多用 audit 数据/算力才赢 | 这个增量不够，停止方法论文路线 |
| F5 泛化与成本 | 两视觉域 × 两模型规模 × 三 seeds；固定测试集和最终 checkpoint，完整核算 proposer/audit/训练/服务成本 | 仅人工偏置集、一个规模、一个 renderer 或 best run 有效；去掉超额成本后优势消失 | 限定为负面/边界发现；不提交泛化/效率强 claim |

F2 的 pilot go 标准：三个独立 seeds 中至少两个在预登记的相邻 phase 出现 A 不降、P 下降至少 2 个百分点，并在新样本复评中保持方向；同时得到相对固定 harness 的交互效应证据。2pp 是项目的实际意义门槛，绝非已观察结果。若区间太宽，结论是 **inconclusive**，不能当支持；是否增加样本先按预算记录决定。

F3 是机制诊断的一次受控分叉，**不把多分支试训加入正式算法**。先比较共同 rollout/资源预算，再以相同接受 token 数作解释性对照；二者不能混为同一个公平性口径。

最终方法 go 标准：四个 domain×size 格子上，P 的预登记整体平均改善至少 2pp，三个训练 seeds 的配对改善方向一致，普通 A 的整体均值下降不超过 1pp；逐格完整报告，任一格明显变差必须解释并缩小 claim。还须胜过 F4 简单对照。该标准是投入下一阶段的门槛，不是三 seeds 就能证明普遍性或保证统计显著。

## 3. 方法草案：只新增一个接受模块

### 3.1 数据角色与视觉干预

分开记录 W（权重训练）、H（harness 搜索样本）、C（配对 audit、也是优化数据）、V（开发/超参选择）和 T（最终测试）。同一源场景及其所有编辑版本必须在同一 split；跨 split 的题目、图像、源数据和模板近重复均检查。T 的干预类型/场景组合不参与 gate 或 proposer 开发。

一个配对样本包含相同问题 q、两张有效图像 x 与 x′，以及可靠 verifier 给出的不同答案 y 与 y′。改变图中任务相关内容，比如改柱高使比较答案改变；不以随机噪声/空白图片作为主要证据。文件名、路径、元数据和工具返回不能泄露答案或 pair ID。另有只改样式而答案不变的配对，用来排除“只是更容易改变回答”，它是评测控制，不增加第二个训练模块。

C 的分数参与反复选择，**因此 C 不是独立验证集**。原始 audit 答案不进入 proposer 日志，只向接受器提供分数与完整审计记录；即使隐藏原始答案，重复选择仍会适应 C。pilot/开发决策使用 V，正式性能结论使用未参与开发的 T。所有对照获得等量 C 数据或等额替代预算，不能让 ours 独占额外监督。

### 3.2 自己的记号与决策

记 w 为当前权重，g 为可执行 harness，S(w,g;x,q) 为完整执行得到的答案，v 为冻结的域 verifier。方法草案的方程编号只用于后续 Method 对应，不来自 WHALE 的公式排版。

**E1 — 成对视觉正确性。**

\[
b_i(w,g)=v(S(w,g;x_i,q_i),y_i)\,v(S(w,g;x'_i,q_i),y'_i),\qquad
P_C(w,g)=|C|^{-1}\sum_{i\in C}b_i(w,g).
\]

两个答案都必须正确；仅输出不同不计成功。主诊断用固定解码设置，另报随机解码稳健性。P 是可验证的行为指标，不能由它直接声称网络“理解”或因果证明。

**E2 — 相对 incoming harness 的可接受集合。**

\[
\mathcal G_{\mathrm{ok}}(w)=
\{g\in\mathcal G:\ P_C(w,g)-P_C(w,g_{\mathrm{in}})\ge -\varepsilon\}
\cup\{g_{\mathrm{in}}\}.
\]

本 MH phase 中权重固定，g_in 是进入该 phase 的 harness。默认草案 ε=0，开发集上做预登记的 ε 敏感性；不从 T 选 ε。incoming 始终可用，因此没有合格改进时保留 incoming。不能把这个经验门宣称为总体非退化保证；下一权重 phase 本身仍可能降低 P。

**E3 — WHALE 原来的排序，仅限制候选集合。**

\[
g_{\mathrm{next}}=\operatorname*{arg\,max}_{g\in\mathcal G_{\mathrm{ok}}(w)}
\big(A_H(w,g),-\mathrm{turns}_H(w,g)\big)
\quad\text{（字典序）}.
\]

valid archive 中按准确率优先、turn 数次优选择，配对不合格者不能成为下一训练 harness。不能先按普通分数做 Pareto 剪枝再 gate：被剪掉的候选可能是唯一可靠改进；需保存全部 valid archive 的原始分数。所有普通验证/搜索空间限制继续应用。

**E4 — 权重更新复用原 RSFT。**

接受 g_next 后，在它生成的成功轨迹上调用 WHALE 的原有 RSFT 更新 w；不加自称新颖的 loss、不改变 reward、不改 verl。E4 在论文中标明是复用的 baseline 运算，不能把它计为贡献。

这是 constrained empirical selection 的应用，不是新优化理论。[AutoDesign §3.2](https://arxiv.org/html/2608.13560v1) 已有多模态 harness 的非退化接受 gate，所以 E2–E3 本身不能认领为新机制。科学价值必须来自特定反馈失效、配对信号与干预位置的必要性和独立测试收益。近邻如果已覆盖同样的权重反馈机制，应终止或重写。

### 3.3 与代码和方法图的对应（独立模块已实现，真实VLM接线未完成）

| 计划放在 ours/ 的对象 | Method 对应 | 职责/消融 |
|---|---|---|
| VisualPairEvaluator | E1，图中 Paired Visual Audit | 统一 verifier，返回每个 pair 两边结果；反转答案、无关样式、无图控制均留可审计记录 |
| EvidenceConstrainedAcceptance | E2–E3，图中 Acceptance Gate | 单一方法模块；mode=off 完全恢复基线选择，mode=paired 打开；ε 为显式配置 |
| WHALEAcceptanceAdapter | E3 与原 MH 的连接 | 在 ours/ 包装函数式 MH 入口；不是假装已有可继承的统一类；覆盖初始选择、普通选择、early-stop 以及恢复路径 |
| VisualTaskAdapter | 图中共享环境 | image payload、图像处理、工具权限、token masks 的 train/eval 一致性；所有条件共享，不算创新 |
| RunLedger / 结果汇总 | 实验协议，不属于 Method 新组件 | 保存身份 hash、种子、完整成本、逐题结果与消融配置；有明确复现用途才写脚本 |

原 Chess MH 的 [选择位置](https://github.com/krafton-ai/WHALE/blob/fbe125eb7abea7f760c99ab9acc1a6261e708fc6/domains/chess_puzzles/meta_harness/meta_harness_chess_puzzle.py#L531) 有成功率 early-stop 分支绕过 pick_accepted；只替换一个函数会漏 gate。新视觉域会使用统一接受路径，并以 mode=off 回归基线行为。不能为了“改动看起来少”藏掉这些接线。

同一 phase 内 w 固定，C/解码固定后可按 model weights + harness code + image/question split + decode 配置 hash 缓存 audit。跨 phase 权重变化必须失效。候选全部真实评测；没有跑出的缓存不能填假分数。cache miss、候选失败与拒绝成本都计入。

~~~mermaid
flowchart LR
  W[当前权重 w] --> R[WHALE RSFT]
  G[已接受 harness] --> R
  R --> W2[更新后的权重]
  W2 --> M[WHALE proposer 与候选执行]
  M --> A[普通任务分数 A]
  M --> C[配对视觉审计 P]
  A --> V[VETO 接受约束]
  C --> V
  V --> G2[下一训练 harness]
  G2 --> R
~~~

此图是方法假设的原创数据流。VETO 是一个接受模块；数据构造、logger 和图像环境不能凑成多个“创新组件”。

模块可以保持轻量，但从文本 release 接到真实 VLM 的总适配工作尚未估定；图像 payload、训练 mask、checkpoint 服务和缺失依赖都可能需要实质工程。不能把“方法模块小”误报成“整个项目只改几行就能跑”。

## 4. 再写支持实验与完整矩阵

### 4.1 先保留原文本复现关，再进入视觉域

阶段 2 已选择 Chess 小 subset：有确定性规则 verifier，不需检索服务。原缺失的 LLM client 已由明确归属的本地兼容接口接通，原生分离式训练正在验证实际权重更新与checkpoint传递；工程成本和失败保存在执行日志，不能据此宣称数值复现。Math 是备选，其 env/recipe/patch 前置缺失见此前preflight记录。不以跳到新视觉环境绕过原 baseline 复现。

四条件必须为 weight-only、harness-only、WHALE 中定义的 Fast-Slow（prompt-only MH + RSFT）和 WHALE。此 FST 控制不等于重现原 FST 论文的所有算法。各三个独立 seeds。results/repro.md 将列原论文来源值、我方真实值、数据/预算/指标差异和失败原因；未运行前不填表。subset 只做运行与趋势核查，全量数字不同不能随意宣称复现成功。

### 4.2 两个视觉域、两个规模

建议先做**图表数值推理**与**物体属性/空间组合推理**，避免导航/自动驾驶模拟器成为主工程。

- 图表域参考 [PlotQA 官方资源](https://github.com/NiteshMethani/PlotQA)。先核查图像对应原始数值、重渲染条件与许可；当前没有验证其能直接提供所需成对编辑。自行构造的数值图表必须标为新的受控 split，不能冒充原 benchmark。
- 空间域参考 [CLEVR 官方资源](https://cs.stanford.edu/people/jcjohns/clevr/)，利用其场景/问题程序和公开生成器验证属性、数量、关系变化。旧 renderer 兼容、渲染成本、数据偏置须先核验；不能假定原数据就是 counterfactual pairs。
- 原始图片必须承载答案所需信息，不能把 scene graph/CSV 真值放进模型上下文。搜索空间允许合法的 prompt、crop/zoom、调用次序、反馈格式；禁止读数据源真值、联网查题、改 verifier 或把测试集放入工具。
- 两个规模优先沿用 WHALE 的 Qwen3.5-2B/4B 家族；视觉加载、冻结/训练参数范围、显存和 token 对齐需先 smoke。若必须换成别的家族，事前更新矩阵，所有方法同换；不声称还是原模型复现。
- 受控合成域只够识别机制。进入投稿写作前还需一个真实图像/真实图表的外部评测，以及未见干预/模板迁移。若只能在自造偏置上成立，CV 投稿增量不足。

| 主条件 | 更新 weights | 搜索范围 | VETO |
|---|---|---|---|
| Weight-only | 原 RSFT | 固定强 h0 | 关 |
| Harness-only | 冻结 | full executable harness | 关 |
| WHALE-Fast-Slow | 原 RSFT | 严格 prompt-only | 关 |
| WHALE | 原 RSFT | full executable harness | 关 |
| Ours | 原 RSFT | 同 WHALE | 开 |

矩阵为 2 域 × 2 规模 × 5 条件 × 3 seeds = **60 个主训练/搜索 run**，这是拟运行数量，不是已完成数量；消融/诊断/复现另计。预训练 h0 作为参照，不把弱模型混入表格当主要对手。

最近邻 PAPO/CFPO 使用不同的 policy objective，不能直接把 KL 粘到 WHALE RSFT 就声称复现它们。若要比较，应单独用官方兼容训练路线、同任务/模型/预算并说明算法差异；至少须深入核查最近邻和强数据增强对照后才能作“优于视觉可靠性方法”的 claim。若此项超过资源，缩小比较范围，不用年代久远的弱 baseline 代替。

### 4.3 消融与决定归因的对照

| 对照 | 拿掉/替换什么 | 要区分的解释 |
|---|---|---|
| Gate off | 唯一方法模块关闭，其余相同 | VETO 的增量；off 必须行为等价于修复后的 WHALE |
| Marginal-score selection | 同样的配对图像，按单张平均正确率排序 | 是 pair 约束有用，还是单纯多了视觉数据 |
| Counterfactual augmentation | 同等配对数据进入普通 RSFT/H 搜索 | 接受干预是否胜过已有的数据增强思想 |
| Independent reevaluation | 把同额预算用于额外普通样本/重复解码 | 是否只是降低了候选选择噪声 |
| Generic non-regression gate | 同额 audit 预算，采用普通 held-out 准确率非退化条件 | 配对信号是否优于 AutoDesign 式通用接受原则 |
| Soft pair score | A + λP；λ 仅由 V 决定 | 硬约束是否必要，普通多目标打分是否足够 |
| Matched random rejection | 按同接受频率随机拒绝候选 | 是否只是减慢 harness 变化 |
| Frozen-weight gate | harness-only 中也打开 gate | 是否确有交替作用，而非一般 robust harness selection |
| No image / label-preserving edits | 只在诊断评测变输入 | 排除文本泄漏、只会改变回答及样式依赖 |
| Fixed common h0 | 在同一权重下替换评测 harness | 分开即时接口收益与留在权重中的收益 |

ε、audit 样本量和调用预算只做有目的的敏感性实验；不展开无依据的大网格。没有去掉某组件的真实 run，不填该消融行。性能提升归因报实际差值与区间，**不预设或承诺“90% 来自模块”**；存在交互时尤其不能把单项差值当可加的百分比。

### 4.4 统计与成本

三个独立 seeds 从最早的科学比较开始使用；调试 import/连通性不是性能 run。主结果报告所有 seeds、mean ± sample std（分母 n−1）、逐题/逐 pair 结果。mean@8 与八个训练 seed 不混淆；同一 pair 两张图在 bootstrap 中成组抽样，训练 seed 是最高重复单位。三 seeds 的区间可能很宽，如实报告。

固定最终预算 checkpoint，V 用于开发/超参，T 只作最终评价；不存在从 T 找最优 checkpoint。失败 run 的原因、消耗和重跑策略写入 ledger，不静默删除。

主公平性是同硬件时间与 API 资源上限；另报 target rollouts、视觉/文本输入输出 token、proposer/judge 费用、SFT backward、服务加载、audit、CPU 渲染、失败重试。Pareto 曲线用多个预登记预算点；不能只让 ours 跑到某个最优点。队列等待和训练运行分别报告，避免把偶然排队当算法加速。

GPU 计划在审批和 F0 前置恢复后，依据当时账户配额/节点状态比较单卡与多卡的总完成时间及 GPU-hours；此时不预订资源、不承诺具体小时数。

### 4.5 应该让读者理解机制的图

主分析图是 **A–P 相图**：横轴普通准确率 A，纵轴独立测试配对正确率 P，同一 seed 的 successive phases 以箭头连接，权重更新与 MH 搜索使用不同箭头样式，并叠加固定 harness 对照。

它直接展示“右移是否同时下移”“退化发生在搜索还是训练”“gate 是否改变轨迹”。图上每个点必须来自真实 checkpoint 与固定评测集。第二张可用 checkpoint×evaluation-harness 的 P 差值热图检查对角共适应。t-SNE/attention 只在能回答具体问题时使用，不作为因果证据或工程工作量装饰。

## 5. 预先承认的 limitations

1. 需要低噪声的有效成对视觉样本；自然图像编辑困难，错误 oracle 可能反而筛掉好 harness。
2. 经验 gate 会过拟合 C；它不保证未见分布可靠，也不阻止权重 phase 本身遗忘。
3. 可能拒绝暂时 P 较差、长期却更有训练价值的候选；并非普遍最优的共适应策略。
4. audit 有真实成本；小模型、短训练、昂贵视觉工具下可能得不偿失。
5. 纯文本可解的任务、本来已经可靠的 harness、或一般数据增强已足够时，预期几乎无收益。
6. 受控图表/CLEVR 不能支撑自动驾驶、机器人安全或通用因果理解的结论。
7. 三 seeds 是最低起点，无法消除 proposer/API 非确定性，也不等于强统计证明。

## 6. 写作定位与阶段门

如果 F2–F5 成立，Intro 的逻辑应是：WHALE 观察到文本工具任务的联合准确率收益；在我们测试的视觉变化条件下，仅此指标不足；我们定位交替反馈的失效，再检验接受约束。**现在不写“我们发现/显著提高”的结果句。**

Related Work 单列 WHALE，明确写 *we build on their released code*；并清楚区分 CAT 的已有诊断、AutoDesign 的已有接受原则、CPL 的 counterfactual prompt learning、PAPO/CFPO/DeFacto 的训练目标和本候选待验证的交替反馈问题。方法用本文件自己的记号/图，原有 RSFT 明示为复用。顶层 README、未来 paper/ 均引用 WHALE，LICENSE 与 NOTICE 原样保留。

**科学审核对象：** 这一“视觉交替反馈 + 单一接受约束”的窄假设及停止条件。新Goal下先落实原 WHALE 复现前置与独立模块，前置不通过则不进入真实联合实验。复现通过才做视觉 F1/F2 的最小诊断；诊断不支持就结束此路线。新文件放 ours/，不改任何 verl/。当前实现状态与公式对应见ours/README.md；尚未接通VLM训练/搜索，没有生成占位性能数字。工程图表的pixel oracle不替代F1的模型可读性检查。
