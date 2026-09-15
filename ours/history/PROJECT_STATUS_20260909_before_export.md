# Historical project status snapshot

Captured during export job222703. Relative links below originally resolved from the repository root.
This archive contains superseded live snapshots; consult ../../PROJECT_STATUS.md for current state.

# Project status

## 当前：h1后的第二阶段原生训练完成，正在检查参数导出（2026-09-09）

**222640 COMPLETED/exit0，23:27:52–23:34:00 HKT。** 完整复用222292的64条审计轨迹，
原生success-filter保留11条、30392训练token；8+3两个mini-batch完成2次优化器更新。
actor约141.72秒，保存约35.32秒，更新后NCCL返回约0.875秒；20,700,225,459字节
checkpoint已落盘，SHA`f42e2219040b036e0b926fdd5c2ad0d0d0c7082d715c0177cc069c3433b68ada`。
梯度范数8.16329是原生记录的两步均值；loss1.2237049是两mini-batch token均值之和，
不能与上一次单mini-batch的loss直接比较。关闭时有DataLoader worker Killed日志，
发生于完整指标/save/sync之后；Slurm确认exit0，该现象仍保留在归档中。

本次保持原生Torch fused128/原优化器与所有有效token，子类在原NCCL finalize后释放
闲置CuPy缓存。真实GPU fixture通过；实际初始和更新后同步各释放6GiB+512字节，
活跃池计数保持0。最终Torch分配峰值75.803GiB，reserved峰值77.352GiB，不能以早先
69.7GiB瞬时快照代替峰值。124项来源冻结；完整预检、3项原生集成检查、2项清理边界
检查通过，基础95通过/47依赖跳过。上游未修改。

前两次恢复222512/222615已归档为反向OOM，无最终更新计数或checkpoint。成功恢复
耗0.204444GPUh，已知项目累计9.064167GPUh；本轮无新增GLM或轨迹生成调用。
原69次采样仍计入222292，缓存返回吞吐量不作性能指标。

下一步用原生merger导出canonical BF16，逐项检查与本轮实际输入的FP32更新及落盘
变化，再从新vLLM worker读取能区分新旧模型的参数值。导出入口已拒绝实际失败训练
结果，须绑定完成证据后才能运行。当前参数变化/新权重推理尚未核验；无heldout或
VETO效果结论。既有h1=6/32、h0=0/32仅是MH优化集结果，F0–F5仍未通过。

证据：[完成结果](results/alternation-recovery-222640/result.json)、
[原生指标](results/alternation-recovery-222640/metrics.json)、
[真实缓存释放](results/alternation-recovery-222640/transport-memory.json)、
[冻结计划](results/alternation-recovery-plan-20260909-v3.json)。

## 前一阶段：h1在线批次11/64审计通过，222292前向OOM

**最新终态：222292在22:33:45 HKT以FAILED/exit1结束，运行1992秒/2H800，计
1.106667GPUh；项目已知累计8.519167GPUh。** 64条新在线轨迹完整，69次调用全部返回，
452201输出token、零请求错误。原生请求/配置/token/mask重放及独立棋盘审计PASS：
11条accepted、30392 loss token，覆盖003YF的5条与003IX的6条；是2个独立训练题，
不能与上一批不同题目的2条成功做因果提升比较。失败轨迹42次预算停止、7次wrong_move、
4次malformed。只裁无效回复尾部16384→6321，4096提示区域与所有有效token保留。

OOM发生在Qwen3.5的lm_head线性前向：需要4.82GiB，实际剩3.68GiB。无最终actor
指标或checkpoint，无法确认错误前是否已有mini-batch更新；已核验完整优化器阶段为0，
不把未知内部步数写成确定0。完整批次可复用，下一修复不重复采样。原仓库已有torch
分块输出后端通过小型混合Qwen的FP32/BF16两项CPU检查：损失各自相同，55个梯度
张量的聚合相对L2差异分别约1.17e-7/0.00410。BF16不是逐位等价，正式四条件必须
共享同一后端。尚未完成真实4B/FSDP恢复验证，未再次提交GPU作业。

证据：[完整失败归档](results/alternation-rsft-222292/result.json)、
[原始Slurm终态](results/alternation-rsft-222292/slurm-complete.txt)、
[64条独立批次审计](results/alternation-batch-audit-222292.json)、
[分块后端CPU数值检查](results/native-fused-rsft-cpu-validation-20260909.json)。

以下保留本轮授权、选择与运行中的历史快照：

- 用户明确回复“我授权”后，登录节点GLM调用获准并完成。77.31秒、7次新增API
  请求，保守记账新增0.039819元；项目累计0.087076元、held0、可预留44.912924元，
  均非官方账单。CLI显示的美元估价不作为智谱实付费用。
- h1已通过原loader及源码/行为检查，未人工修改候选。它同时改提示词、parser和
  side-to-move前缀。说明声称last-tag-wins，实际代码拒绝不同标签；可以从未结束
  thinking中解析标签。上述偏差保留，不能把文字说明当作已实现机制。
  日志显示proposer读了h0和汇总分数，没有读取可用的逐条失败轨迹。
- 222212正常COMPLETED/exit0，21:35:19–21:52:22 HKT、2H800、1023秒/0.568333GPUh。
  同checkpoint/同32题MH/同seed42和8129预算，h0缓存不重复生成。h1为6/32（18.75%），
  h0为0/32；原规则选择h1。34次调用/251088输出token，平均7846.5；原runner重放、
  独立棋盘/token/模板/计量审计全部PASS，两个worker的实际权重核验也通过。
  6条成功均为单次调用：4条正常结束thinking，2条从截断thinking中被原规则解析；
  不把后两条改写为完整最终答案。其余22条预算停止、3条wrong_move、1条malformed。
- 下一训练批按既有工程顺序选择train的第8至15行，共8个新题/n8；初始提示长度
  h0最多808、h1最多930 token。未按成功与否选题，未评测test；最终harness已选择h1。
  `alternation_training.py`要求完整搜索和双方审计，再冻结下一步实际Hydra配置，
  沿用新在线记录/无效尾部裁剪/原RSFT。2项原生CPU检查通过；最终计划已绑定h1、
  完整双方审计与88个源码/证据哈希，实际完整Hydra预检PASS。
  222292于22:00:33 HKT在g4的2H800启动，完成初始化并进入原在线循环；22:07:48
  快照登记64次调用、返回8次/63083 token、无错误，剩余消耗未知。尚无完整batch、
  新的优化器更新或checkpoint结果。运行来源哈希全部保持一致。
- Fast–Slow补充`prompt_subspace.py`与可溯源的重建说明，保留原AST校验；另在原
  loader执行前要求提示词是纯字符串，堵住原AST屏蔽整个表达式导致的随机状态修改。
  4项CPU检查通过，包含原proposer指令装配、原搜索循环接受提示词/拒绝代码变化。
  这是尚未接入正式阶段的WHALE-FST对照组件，不是恢复了作者未发布的原说明。
- 本轮MH运行依赖保持冻结，新增文件未改变222212。已结算GPU成本7.412500GPUh，
  旧221576亚秒占用未知。下一训练采用原一actor/一rollout布局；2H100预测9月11日
  18:57，2H800可立即。预计40–65分钟、90分钟/3GPUh硬上限，GPU配额剩175765分钟。
  最新基础测试95通过/34依赖跳过。正式四条件/三种子和VLM主矩阵仍未完成，F0–F5不变。
- 固定已保存回复的解析诊断：h0的35次回复在两parser下全部同动作；h1的34次中
  33次相同，另一条被h1判为歧义。6条成功回复在旧parser下也会读出相同正确走法。
  不能将本次改善归因于新parser；提示词和观察前缀还需真正重新采样的消融来区分。
- 22:27:57 HKT查询：同一222292仍RUNNING，累计69次调用开始、56次完成、
  376769输出token、0错误，13次未完成；尚无完整batch或新checkpoint。
  `audit_alternation_training.py`已独立补齐h1的请求/配置/原生token重放与棋盘续着审计，
  4项原生CPU检查通过。已核对实际请求配置哈希等于冻结配置经原生迁移后的哈希。
  审计区分棋盘成功和反馈截断导致的零训练奖励；遇到未入事件日志的丢弃回复会拒绝
  通过，不悄悄忽略调用。首轮CPU检查停滞日志保留；关闭tokenizer并行且使用宿主
  执行后完成，停滞的具体原因未确立。测试曾发现审计器误用win标签，已按原solved
  标签修正并通过，不涉及训练代码修改。
  `verify_alternation_transition.py`用于下一checkpoint与本轮实际输入的全参数比较；
  真实小张量文件测试通过，实际4B新参数尚待生成。两模块对应共享E4验证，非新机制。
  本次新增文件没有改变222292绑定的88项来源。

证据：[授权记录](results/updated-mh-proposer-authorization-20260909.json)、
[候选行为与费用审查](results/updated-mh-h1-semantic-review-20260909.json)、
[完整候选结果与资源](results/updated-mh-candidate-222212/result.json)、
[下一批训练数据](results/next-native-rsft-data-20260909.json)、
[FST原生CPU检查](results/fst-prompt-subspace-native-tests-20260909.log)、
[下一步选择/配置检查](results/alternation-training-native-tests-20260909.log)、
[下一训练冻结计划](results/alternation-rsft-plan-20260909.json)、
[最终CPU预检](results/alternation-rsft-preflight-20260909.log)、
[资源选择](results/alternation-rsft-resource-check-20260909.json)、
[新训练实际分配](results/alternation-rsft-222292-slurm-start.txt)、
[22:07新训练快照](results/alternation-rsft-222292-progress-20260909.json)、
[固定回复解析诊断](results/updated-mh-fixed-response-parser-comparison-20260909.json)、
[h1原生审计CPU检查](results/alternation-batch-audit-native-tests-20260909.log)、
[下一参数交接CPU检查](results/alternation-transition-native-tests-20260909.log)。

## 前一快照：32题基线审计完成，首次GLM发送审批被拒绝（现已获授权）

- 222196正常COMPLETED/exit0，21:00:31–21:18:20 HKT，2张H800、1069秒，
  计0.593889GPUh。完整32题MH结果为0/32，35次调用、260128输出token；
  32次调用因length停止且未结束thinking。30条轨迹最终因assistant预算耗尽停止，
  2条因malformed停止；没有正确落子。不是网络错误或未加载更新权重的证据。
- 原runner精确回复重放、独立棋盘检查、token解码/提示模板和完整计量均PASS；
  两个实际GPU worker的变化参数核验也通过。审计未新增模型调用。
- 下一步为同checkpoint/同32题/同预算的一个GLM候选，再评测并按原规则选择，
  然后才绑定下一次新在线RSFT。不能用零分预先判定GLM质量：它负责提案，
  当前32题由本地Qwen3.5-4B更新checkpoint执行，尚未生成本轮h1。
- 自动审批拒绝登录节点GLM命令，要求用户明确允许把这批研究轨迹发送到智谱。
  命令未执行，proposer工作区未创建，账本无新增请求/预留。7个研究输入文件约
  1.96MB，标准答案为空、题目ID匿名、未发现已保存凭据；含本地路径元数据。
  train128/test64、私有ID映射、模型权重及其他项目不在拟发送研究目录中。
  CLI协议上下文及其工具读取结果也会进入请求；文件总字节数不等于API计费token。
- 已计量项目累计6.844167GPUh，旧221576亚秒占用未知；API保守记账0.047257元，
  剩余可预留44.952743元，非官方账单。当前无后续GPU作业。未通过F0–F5。
- 新增独立审计器与新在线训练尾部裁剪hook的原生CPU检查各1项通过；基础测试
  94通过/29依赖跳过。下一步训练完整Hydra配置仅为CPU占位验证，尚未训练。

证据：[完整基线结果](results/updated-mh-baseline-222196/result.json)、
[独立审计](results/updated-mh-baseline-222196/baseline-audit.json)、
[拟发送材料说明](notes/mh_proposer_transfer_review.md)、
[逐文件大小与SHA清单](results/updated-mh-proposer-transfer-review-20260909.json)。
控制器成功查询在本轮工具输出中确认上述状态；稍后保存全文时该job记录已过期，
结果JSON明确标明转录来源，没有伪称取得sacct账单或完整终态文本文件。

## 本阶段启动与运行快照：新权重上的32题MH评测222196（2026-09-09）

上一阶段有实际进展：222156原生更新、222165全参数导出核验、222176新worker权重抽查
及一次工程生成已完成；并非重复状态或空转。本轮推进该checkpoint上的原生MH搜索。

- 使用既有32题MH划分全部样本，固定seed42、8129 assistant token上限；128题train及
  64题test不进入proposer工作区。h0沿用原始harness，GLM只提出一个完整h1候选。
- 将原搜索循环分成GPU h0评测→登录节点GLM提案→GPU h1评测/原生选择。阶段快照
  可在后续搜索文件变化后继续核验；h0无重复生成。proposer入口要求完整baseline审计。
- 两个固定ID分片，每片16题/一GPU。保留原runner的全部输出，合并后调用原统计函数；
  原`example_id`全部为`redacted`，内部顺序映射单独保存且不送给proposer。
- 原native循环与32题runner的CPU集成检查3项通过：异质成功/失败/多轮输出分片等价，
  原暂停/恢复、不重复h0、确定平局/原early-stop，以及实际worker子类调用原load后读取
  参数。94项基础测试通过/27跳过。测试夹具中的分数不是模型结果。
- 在原评分不变的前提下，按固定顺序读入结果，消除普通目标平局时的目录顺序差异；
  原early-stop优先级保留。正式四条件仍为weight-only、harness-only、Fast–Slow、WHALE，
  原始模型只是额外参照，不替代FST控制。
- 排队实查：H100的1/2GPU分别预测次日06:04/20:42；H800的1/2GPU当时均预测立即。
  选择2张H800独立副本，预计18–25分钟而单卡35–45分钟；每次GPU阶段40分钟上限，
  最多1.333333GPUh。初次在此运行时执行H800上的完整4B MH，实际启动仍待验证。
- v2冻结计划绑定代码、数据、checkpoint及原有8个变化参数坐标；每个GPU worker在
  模型加载后直接验证这些数值。v2最终CPU预检通过；222196于21:00:31 HKT在g4启动。
  两个实际H800 worker均完成8坐标核验，21:15:38快照有25次完整返回、195096输出token、
  无记录错误，部分请求仍未完成，未知消耗不记0；尚无完整32题分数。
- 新增`audit_mh_phase.py`，用保存回复重放原runner，并独立检查棋盘和题目续着；
  多轮正确/篡改动作或奖励/格式错误的CPU检查通过，真实整批审计待完成。
  `compact_recorded_training_bootstrap.py`保留下一阶段新在线采样及记录，在原actor前
  仅裁无效回复尾部；CPU原分发检查通过，不使用缓存恢复manager。训练入口支持
  `RSFT_HARNESS_PATH`，实际完整Hydra配置已确认新checkpoint/指定harness/标量AdamW。
  此配置使用h0作为临时CPU占位，未代表MH最终选择，也未提交下一轮训练。
- 当前运行的全部冻结来源哈希核验一致；新增审计和训练适配不在该作业依赖中。
  已计量项目成本仍6.250278GPUh，另有222196待结算；API保守记账0.047257元，本轮
  尚无新增GLM调用。F0–F5未通过，无留出集结果或VETO收益。

证据：[v2计划](results/updated-mh-phase-plan-20260909-v2.json)、
[原生CPU集成检查](results/updated-mh-phase-native-tests-20260909-final2.log)、
[服务参数解析](results/updated-mh-server-arguments-20260909-cpu.log)、
[222196实际分配](results/updated-mh-222196-slurm-start.txt)、
[21:15进度及冻结哈希检查](results/updated-mh-222196-progress-20260909.json)、
[独立棋盘审计测试](results/updated-mh-audit-native-tests-20260909.log)、
[新在线训练分发检查](results/compact-recorded-training-native-tests-20260909.log)、
[下一步配置的CPU占位验证](results/next-rsft-cpu-config-provisional-20260909.json)。
代码对应原E4权重→E3 harness搜索的执行连接，不增加VETO方法公式或性能声明。

## 前一阶段：真实训练→规范导出→新权重推理已验证（2026-09-09）

- **原生训练222156完成1次优化器更新**：19:50:12–19:54:46 HKT，2张H100，
  274秒 / 0.152222 GPUh。复用221967完整审计的64条训练轨迹，仅原验证器接受的
  2条进入训练；loss token16258，SFT loss0.627909，梯度范数9.324229。
  新模型/API调用0，原69次采样请求不重复记账。checkpoint已保存，原NCCL同步返回。
- 恢复只裁全无效响应尾部16384→8144，保留提示4096和有效token/mask/位置。
  沿用原RSFT筛选、损失、梯度累积及优化器；原生完整CPU循环和损失/梯度测试通过。
  222151路径失败、222154占位字段失败及此前OOM的完整日志与成本均保留。
- **规范导出222165及全参数审计PASS**：723个独立活动张量，原生FP32更新非零，
  全部导出值精确等于原生BF16转换；原始checkpoint哈希在导出前后不变。
  675个原始BF16张量中有8,991,538个元素变化，证明训练变化保留到导出结果。
  48个原始FP32张量共3840个元素另受BF16舍入影响，不能把整个导出差异都归因于学习。
  正式性能对比必须统一序列化精度或增加仅精度转换的控制条件。
- 首次导出222164被严格图核验拒绝：HF旧格式保存将297个视觉参数名改到了语言前缀。
  `ours/`中的原merger子类改为保存规范参数名，并使用已核验的base生成配置，防止
  训练期间的EOS/pad补丁混入新推理条件。旧导出与失败证据保留，不重新训练/采样。
  CPU导出因当前QOS要求至少1张GPU，各占用1张4090但不执行GPU运算；
  222164/222165分别计150秒/0.041667和148秒/0.041111 GPUh。
- **新权重推理222176通过工程核验**：实际worker中8个已知改变的BF16 embedding
  值全部等于规范导出且不同于base；1次生成返回2 token，保存的token独立解码为`Yes`。
  提示要求`ready`，因此不记作指令遵循成功；这是权重加载和生成可执行证据，非性能评测。
  原生in-place NCCL接收端数值未抽查，不能用此新进程证明那一接收端的具体内容。
- 前次222175加载后因回调序列化被拒绝，生成调用0、206秒/0.057222GPUh；官方worker
  扩展/具名数值RPC通过原生CPU检查后修复。222176结果完整保存后，Python/vLLM子进程
  仍未退出，显式释放GPU；Slurm最终CANCELLED，222秒/0.061667GPUh，不能记为正常退出。
  检查成功与退出清理分别归档。源码已补显式`engine_core.shutdown(timeout=10)`；
  此后置清理修订尚未另跑GPU，不重复已完成的生成。旧冻结源码可从归档回放。
- 已计量GPU成本累计**6.250278 GPUh**，另有旧221576亚秒占用未知；本阶段作业均已释放。
  智谱保守记账仍0.047257元，本阶段无新增API调用。最终基础测试94通过/24因缺原生
  依赖跳过；规范导出2项、原生参数比较器3项、数值RPC1项及此前恢复检查有单独证据。
  **没有留出集准确率、VETO收益或完整交替复现；F0–F5未通过。** 上游未改，未push。

证据：[真实训练与资源记录](results/native-rsft-recovery-222156/result.json)、
[64条独立审计](results/native-batch-audit-221967.json)、
[规范导出完整归档](results/native-checkpoint-export-222165/result.json)、
[全参数变化](results/native-checkpoint-transition-222156.json)、
[精度归因](results/native-export-precision-attribution-222156.json)、
[修订vLLM冻结计划](results/updated-vllm-handoff-plan-20260909-v2.json)、
[v2 CPU预检](results/updated-vllm-handoff-preflight-20260909-v2.log)、
[222175失败及资源归档](results/updated-vllm-handoff-222175/result.json)、
[222176推理核验及退出清理](results/updated-vllm-handoff-222176/result.json)。
各归档的状态是当时快照；后续阶段的成功不会反写到旧失败或旧待验证记录。

下一步恢复“新checkpoint上的MH评测/选择→下一次原生RSFT”，完成一次真正的交替。
全部调试与候选选择使用既定训练/MH角色，64题test不参与；随后才扩到原域四条件、
三个独立种子的pilot。单步工程通过不是论文复现，也不能与原作者测试准确率比较。

## 较早同日：原生采样接受1条，Adam显存修复已验证；221967启动（2026-09-09）

- 221921于18:21:03 HKT退出，实际38分10秒、2张H100、1.272222 GPUh。完整64条
  在线轨迹已保存；原验证器接受1条，67次调用，返回usage累计520256输出token。
  8个预定训练题各8条覆盖一致；成功行002vV中的a1h1与参考一致、合法且将杀。
  该行由原解析器从未结束思考的回复中提取落子，不能称为完整最终答案；默认dump
  缺少原请求/逐轮token/mask，因此这里只做有限独立核验，不冒充完整轨迹重放。
- 失败发生在首次AdamW更新的foreach临时张量分配，已完成反向传播并进入step，
  但step未返回、没有checkpoint或已验证权重更新。原始dump、源码、计划和日志已归档。
  OOM前可能改变了内存中的优化器状态，不能把失败步骤描述为原子操作。
- 使用上游原生配置接口`override_optimizer_config={foreach:false}`。CPU ZeRO-1
  对照中3次更新的两路径参数一致；单H100合成显存检查221964通过，16秒/0.004444GPUh，
  覆盖4B活动参数形状、文本梯度及6GiB传输缓冲。此检查不含真实前向/反向，完整
  FSDP训练能否通过仍待验证。新增代码属于E4共享执行检查，不是科学机制。
- v5完整Hydra比较只含两项变更：foreach=false和已CPU验证的记录hook。源码、模型、
  数据和运行配置启动预检查通过。新作业**221967**于18:32:36 HKT启动，2张H100、
  16CPU/160GiB、90分钟/3GPUh硬上限，64条新的在线采样；启用原始请求及精确批次记录。
  18:36 HKT已进入Online RSFT，四个Ray worker均写出请求记录；64个请求开始已记录，
  尚无完整返回。真实生成量此时未知，不记为0；返回内容与完整张量批次记录仍待验证。
  18:49 HKT快照：69次请求开始、24次完整返回、0记录错误，已返回usage合计189838
  输出token，其余请求消耗未知。新增完整批次审计器，8项针对性测试通过；完整64条
  尚未落盘，因此实际整批重放及权重更新仍待验证。新增审计器未改运行中的冻结源码。
  19:00 HKT附近仍在运行，已记录40次完整返回、0错误。另完成小型Qwen3.5合成
  checkpoint的原生导出/重载检查：35个活动名字全部恢复为预期BF16数值，共享输出头
  正确恢复且单独序列化。此检查替代了processor/tokenizer导出，不代表真实4B导出通过。
  不修改运行中的冻结源码；后续监测同一作业，不因日志安静或观察超时重启。
- 本轮无新增智谱调用，项目保守API记账仍0.047257元。已计量GPU成本累计
  **4.462500 GPUh**，另有旧221576亚秒占用未知及当前221967待结算。没有留出集评测，
  没有完成的原生优化器更新，F0–F5未通过，无VETO收益证据；上游未改、未push。

证据：[失败与完整归档](results/native-rsft-pilot-221921/result.json)、
[有限棋盘/覆盖核验](results/native-rsft-pilot-221921/dump-review.json)、
[CPU优化器对照](results/native-optimizer-cpu-probe-20260909.json)、
[GPU显存检查](results/native-optimizer-probe-221964.json)、
[v5冻结计划](results/native-rsft-pilot-plan-20260909-v5.json)、
[完整配置差异](results/native-rsft-pilot-config-diff-20260909-v5.json)、
[新作业启动](results/native-rsft-pilot-221967-slurm-start.txt)、
[分布式请求开始记录](results/native-rsft-pilot-221967-recording-start.json)、
[批次审计器测试及18:49请求快照](results/native-batch-audit-validation-20260909.json)、
[合成checkpoint导出检查](results/native-checkpoint-export-fixture-20260909.json)。

## 较早同日：64条完整审计通过，原生单步训练221921运行中

- 采样作业221898正常完成，16:43:01–17:19:45 HKT、2张H100、1.224444 GPUh。
  64条轨迹解出2条，来自两个不同训练题；70次模型调用、520237输出token。
  独立棋盘、请求/输出token、预算、种子覆盖及文件哈希审计全部通过。
  这是训练启动可行性结果，不能与旧8题直接比较，也不是留出集准确率。
- 不扩采chunk1。原生RSFT作业221906于17:23:31 HKT启动，2张H100分别承担
  actor/rollout，16CPU/160GiB、90分钟/3 GPUh硬上限。对同8道训练题进行64条新的
  在线采样，只训练原trainer本步接受集；没有把bootstrap缓存充作在线训练调用。
- 221906通过实际启动检查并加载4B模型与优化器，但NCCL checkpoint engine因缺少
  CuPy未注册，生成前失败，87秒/0.048333 GPUh、0训练轨迹、0更新。源码和日志已归档。
  已按官方CUDA13 wheel补装CuPy14.2.0，其他依赖未变；原NCCL导入、库版本22809和
  pip依赖检查通过。启动门现在显式导入原模块，避免可选导入错误延迟到GPU阶段。
- 221907成功启动生成服务和AgentLoop，但首次权重同步失败：先前工程配置128MiB
  缓冲装不下2542796800字节的FP32 embedding，211秒/0.117222 GPUh、0训练轨迹。
  已恢复原launcher默认3072MiB；独立比较完整Hydra配置，唯一变化就是该缓冲参数。
  新增CPU检查按actor精度和实际checkpoint张量头验证容量，2项边界测试通过。
- 221912已通过首次原生权重同步并进入Online RSFT。之后源码检查发现64条任务同时
  提交、服务8个生成槽，而420秒通信超时包含排队时间，因此主动取消；没有观测到
  超时异常，不能记作模型失败。296秒/0.164444 GPUh计入成本；未保存完整训练批次，
  实际生成调用/token数未知，不记为0；没有优化器更新。
- v4冻结计划将通信等待改为4800秒，完整Hydra比较确认除此无变化；job的90分钟
  硬上限和token预算不变。**221921**于17:42:53 HKT启动，仍用64条新的在线轨迹。
  18:20 HKT Slurm实查仍在运行（37分26秒）；最近一次生成GPU负载31%。
  完整64条批次尚未落盘。
  实际梯度更新、checkpoint导出及新权重推理仍待运行证据，更新数为0。
  新增参数核验逐张量检查数值变化，精度转换或文件哈希变化不能单独认领更新。
- CPU模型图核验发现原始738个张量名与HF活动图724个名字存在明确差异：15个
  `mtp.*`由当前HF/vLLM实现忽略，`lm_head.weight`与embedding共享；723个共同名字
  的形状一致。已记录精确映射审查规则，实际导出尚待验证。当前严格比较器仍会拒绝
  不同键集合；作业结束后才依据实际导出适配，运行中冻结源码哈希全部保持一致。
- 新增原生记录子类及显式Ray hook，保留请求开始/结束、完整张量和mask、逐轮事件。
  7项记录器测试、hook导入和Hydra配置检查通过；当前作业未启用，真实分布式记录
  仍待验证。统计区分完整返回、错误、未完成和截断记录，不能把未知消耗记为0。
- seed46的8条原AgentLoop离线重放通过，接受1条包含格式重试的成功轨迹；奖励、
  mask和请求一致。新模型调用0；原生成token ID与重编码逐条相同为4/8，差异已记录。
- 最新基础环境86项测试通过、10项因原生依赖跳过；记录器7项在训练环境通过，
  既有原生8项的通过证据保持不变。本阶段对应E4训练及权重传递测量，E1–E3未变。
  没有新增智谱API调用。
  项目已计量GPU成本为3.185833 GPUh，含两次失败和一次主动取消；另有221576亚秒
  占用未知及当前作业待结算。未完成采样的消耗不会因重试被抹除。
  F0–F5尚未通过，无VETO性能增益证据；上游未改，未push。

证据：[完整采样审计](results/chess-bootstrap-221898-summary.json)、
[完整原始记录](results/chess-bootstrap-221898/result.json)、
[修订冻结训练计划](results/native-rsft-pilot-plan-20260909-v4.json)、
[CPU启动检查](results/native-rsft-pilot-preflight-20260909-v4.log)、
[首次失败记录](results/native-rsft-pilot-221906/result.json)、
[第二次失败记录](results/native-rsft-pilot-221907/result.json)、
[主动取消与未知生成计量](results/native-rsft-pilot-221912/result.json)、
[记录器CPU验证](results/native-training-trace-validation-20260909.json)、
[checkpoint模型图及映射审查](results/native-checkpoint-graph-review-20260909.json)、
[含重试成功轨迹的原生重放](results/native-agent-loop-replay-221898-seed46/result.json)、
[当前作业18:20状态](results/native-rsft-pilot-221921-slurm-1820.txt)。
继续监测同一作业221921；仅非空优化器更新、真实参数变化和新checkpoint推理通过后，
才推进下一阶段。单次更新不代表论文复现或留出性能提升。

## 较早同日：训练启动采样已出现成功，首批64条仍在运行

- 作业**221898**于16:43 HKT在gpucluster-g18启动，2张H100独立副本、1小时上限，
  最多2 GPUh。原h0和4B权重不变，32题从pilot训练集按ID事前固定；首批只用前8题，
  每题seeds42–49共64条。模型与数据哈希、种子分工、采样参数均已冻结。
- 17:03 HKT快照：seeds42/43/46/47已完成，共32条、2条完整成功。成功来自两个不同
  训练题；独立棋盘重放核验通过。不能与旧8题比较准确率，也不能把运行中结果当完整统计。
- 原训练AgentLoop对历史8题及新seed42的离线重放通过，分别接受0/1条，调用消息与
  原采样一致，奖励一致，环境消息loss mask为0。它会重新编码回复，原token序列并非
  全部逐ID保留（历史7/8、新批次3/8一致）；这是原实现行为，已单独记录，不能隐去。
- 原生4B RSFT单步配置与8题数据加载预检查通过：每题8次，共64条**新的在线采样**，
  只训练原trainer当步接受集。此成本另计，不把bootstrap缓存称为新训练调用。
  GPU梯度更新、内存峰值、checkpoint导出与实际权重切换仍未验证，训练作业尚未提交。
- 本次81项既有测试及3项独立UCI边界检查通过。新增逻辑对应E4执行/测量层，E1–E3
  未改。无新增智谱调用；原项目和上游未改，未push。F0–F5仍未通过。

证据：[冻结采样计划](results/chess-bootstrap-plan-20260909.json)、
[已完成两条成功的局部独立核验](results/chess-bootstrap-221898-partial-success-audit.json)、
[原训练AgentLoop重放](results/native-agent-loop-replay-221898-seed42/result.json)、
[原生RSFT预检查](results/native-rsft-pilot-dataset-preflight-20260909.json)。
后续继续跟踪**同一作业221898**，不得因观察超时重启。64条完成并独立审计后，再冻结
并提交单步原生训练；既然已出现成功，不自动扩采到256条。

## 当前诊断：原作者也报告初始模型几乎全部截断；先验证训练启动条件（2026-09-09）

- 重新核对[论文附录D.3](https://arxiv.org/html/2609.00196v1#A4.SS3)，作者报告的
  初始失败模式与当前一致；这不代表我们已复现其最终结果。
- 发布代码的16384是包含环境消息的response region上限，8129是独立的assistant
  累计预算。不能把两者混为一谈，或把翻倍预算称为已证实的复现修复。
- 对已有h0/h1输出做[离线终止审计](results/chess-termination-diagnosis-20260909.json)：
  16次均无结束思考token、无具体move标签，均截断。只改解析器不能恢复尚未生成的答案。
- 下一步收敛为[训练启动与诊断计划](ours/chess_reproduction_plan.md)：先在pilot训练集
  固定32题、每题8条轨迹，分64/256轨迹上限检查完整成功；有成功再验证真实RSFT更新
  及checkpoint切换。达到上限仍无成功时停止扩大采样，转向预先规定的诊断对照。
- 本次只有查阅、离线审计与计划更新；新增模型/API调用0，训练0，未提交GPU作业，未push。

## 当前：一轮原生GLM搜索完成，h0/h1均0/8，无性能提升（2026-09-09）

- 作业221733完成新的8题h0评测，仍为0/8，随后因继承登录节点的本地代理而无法
  连接proposer转发器。计算节点gpucluster-g18绕过代理直连智谱官方端点返回401，
  证明该路由可达；不能据此假定所有节点都能访问所有外网。失败占用12分29秒已保留。
- 修正代理后，221743通过GLM完成真实候选生成，CLI正常退出，新增7次官方API请求。
  项目累计保守记账0.047257元/45元，未结预留0；这不是账户余额或真实账单。
  原接口要求name，候选元数据写成slot，导入因此失败；原始响应和完整候选均已保存。
- 候选只改SYSTEM/USER提示语和观察文本空白格式，解析器、合法性函数、重试/轮数预算
  未变。严格归一化只补name=h1，拒绝冲突、越界路径或错误parent；原生加载检查通过。
- 作业221754已评测这份**字节不变**的h1，1张H100、10分钟。通过来源和配置核验
  后复用原生h0缓存及GLM提案，不重复基线或API调用。h0/h1各8次调用、65032输出token，
  均全部截断且0/8解出。h0为8次格式重试；h1为7次格式重试、1次合法错误。
- 原选择器接受h1是因为解题率和调用次数均打平，而文件遍历先遇到h1；**不是性能提升**。
  独立输入/输出token、棋盘与验证器重放通过。首次审计误设平分必选h0而失败，原记录
  保留；修正后只核验目标最优性及平分集合成员，不声称冻结了文件遍历顺序。
- 一个逻辑搜索迭代由三个作业完成，全部计入成本：0.426389 GPUh；项目已计量累计
  1.631389 GPUh，另有旧作业221576亚秒占用未知。当前无运行作业，新增API调用共7次。
- 另已按原转换器准备128训练/32搜索验证/64测试题，排除已查看的64题及64局面组；
  独立检查无跨split重叠，790步参考续着均合法，无模型调用。这些数据未进入当前8题搜索。
- 81项CPU测试通过。源码均在ours/，上游及旧项目未改；未训练、未通过F0–F5，未push。

证据：[网络诊断](results/native-search-network-diagnosis-20260909.json)、
[已生成候选的原生检查](results/prepared-proposal-validation-20260909.json)、
[本轮计划](results/native-search-plan-20260909-v3.json)、
[完整独立审计](results/native-search-221754-summary.json)、
[后续数据独立审计](results/chess-pilot-audit-20260909.json)。

下一步由上方训练启动计划细化：停止在同8道工程题重复解码；先在已隔离的pilot验证
非空RSFT更新、checkpoint导出和真实权重切换，再推进原域四条件/三seeds。正式比较前需对所有条件共同规定
可复现的平分顺序，不能让文件遍历差异被误当作VETO收益。VLM训练/搜索与F0–F5仍未通过。

## 较早同日：原默认4B也完成检查，8题仍无成功轨迹（2026-09-09）

- 作业221725在1张H100上完成，11分10秒、0.18611 GPUh。固定官方4B权重，同8道
  工程题与8129生成预算，8次均截断，0题解出；独立重放为8次格式重试。
  输入token、输出解码、原验证器判定与全部来源哈希核验通过。
- 切到原默认4B后这组题仍失败，原因尚未定位；这不能代表原论文数据上的性能。
  尚无成功过滤后的梯度更新；不提交空批次来认领RSFT通过。
- 原AgentLoop的实际文本处理函数在8题上与4B服务输入一致。4B准备时一次CPU检查
  错用processor纯文本消息接口而失败；改结构化文本后，与纯文本tokenizer完全一致。
- 本轮没有新增智谱调用，项目累计保守费用仍0.008267元/45元预算，未结预留0。
  所有已计量GPU占用累计1.205 GPUh，另有221576亚秒占用未知；当前没有运行中的作业。
- 下一步：停止重复同设置解码检查，恢复受预算约束的原Meta-Harness搜索入口及其评测
  接线，检查能否产生成功轨迹；之后才验证非空RSFT、checkpoint导出和真实服务切换。
  原域四条件/三seeds与F0–F5科学门仍未通过。

证据：[4B独立审计](results/vllm-runtime-221725-summary.json)、
[运行前计划](results/chess-vllm-4b-plan-20260909.json)、
[资源记录](results/vllm-runtime-221725-resources.json)、
[原AgentLoop格式检查](results/native-chat-template-4b-preflight-20260909.json)。
2B作业221663的完整源码快照在本地`476baed`；其冻结计划需使用该快照核验，后续
launcher增加可选计划参数，不能混用旧哈希。全部修改本地保留，未push。

## 较早同日：vLLM真实运行及独立审计通过，2B仍无RSFT成功轨迹

- **真实GPU结果**：作业221663用1张H100完成同8道工程题，7分52秒、0.13111 GPUh。
  原runner/harness/verifier未修改；vLLM0.20/torch2.11/Transformers5.5.2实际CUDA执行通过。
  8次生成共65032输出token，全部截断、0题解出。独立重放为5次格式重试、2次非法重试、
  1次合法错误；输入token与本地分词一致，输出token可逐字还原回答。
- **含义与下一步**：相同8题在HF和vLLM均无成功轨迹，但后端、版本、批处理及随机流不同，
  不能据此作严格性能归因。这8题不能支撑非空RSFT更新；先核查原默认Qwen3.5-4B。
  官方4B版本已固定为`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；11个文件下载完成，
  两个权重分片通过官方LFS哈希核验，内容身份`bba57bcc77d3f063…`。尚未调用4B模型。
- **两次失败均保留**：221641启动配置不满足vLLM约束，45秒、0次生成；221643开始
  1次生成后请求180秒超时，作业5分30秒，没有返回完整回答，实际生成token数未知。
  修正为模型支持的chunked prefill、420秒请求超时、每组4并发后完成。三个作业合计
  0.23528 GPUh；截至此处所有已计量作业合计约1.01889 GPUh，另有221576亚秒占用未知。
- **原训练配置**：`ours/run_native_rsft.sh`沿用上游launcher/trainer/actor，限定一次
  工程迭代。缺失legacy_data映射从同提交生成配置恢复，三领域副本逐字段一致；Ray worker
  使用相同显式tools恢复hook。完整Hydra解析、原生validate_config和8题原生数据加载通过。
  CPU预检查中的IPC限制及两处诊断错误已保留并修正；没有提交训练，没有权重更新。
- **Method与费用**：新代码对应E4共享训练执行及测量层；E1–E3未变，无新增科学公式。
  本轮智谱调用0，项目累计保守记账仍0.008267元，预算上限45元。76项CPU测试通过。
  F0–F5仍未通过；上游和旧项目未修改，未push。

证据：[完整vLLM审计](results/vllm-runtime-221663-summary.json)、
[运行前计划](results/chess-vllm-runtime-plan-20260909-v3.json)、
[原训练配置](results/native-rsft-resolved-20260909.yaml)、
[原生数据加载](results/native-rsft-dataset-preflight-20260909.json)、
[缺失配置来源](results/legacy-data-restoration-20260909.json)。

4B权重证据：[官方checkpoint清单](results/qwen35-4b-checkpoint-20260909.json)。

## 较早同日：GLM proposer文件工具已验证，训练导入通过，联合实验尚未就绪

- **用户已授权智谱国内官方API，首轮预算45元**。密钥位于仓库外的0600私有配置，
  不进入源码或Git。直接请求返回`glm-5.3-flash`；经原生WHALE wrapper实际完成
  Read→Edit→DONE，AST核验只有事先指定的USER_PROMPT句子发生变化。
  这是文件工具接线检查，正式harness搜索迭代仍为0。
- **预算与排错**：共8次真实API请求，保守费率记账约0.008267元，无未结预留。
  这不是智谱账单或账户余额。持久化额度入口按45元限制本项目请求，不启用模型回退。
  四次原生CLI检查在Read后超时；完整SSE经官方SDK离线解析通过，给原样响应补齐
  Content-Length后第五次成功。失败记录保留，离线协议fixture不计API或性能结果。
- **隔离与来源**：候选文件与公开WHALE固定提交逐字节一致。自动审批曾因担心私有
  代码外发拒绝工具检查；完成公开来源核验及Landlock越界读取拒绝测试后复审放行。
  子进程仅测试目录可写，私有用户文件不可读。真实密钥只由父进程的官方API relay使用。
- **完整解码诊断**：221597在2张H100上完成同8题固定分片，22分56秒、0.76444 GPUh。
  恢复temperature1/top_p1/top_k20/thinking开启及8129预算；8次均截断，共65032输出
  token，0题解出。独立核验为6次格式重试、1次非法重试、1次合法错误，仍无成功轨迹。
  native legal_rate显示1.0只检查终止illegal事件，不能当作本次逐步合法率。
- **隔离训练环境**：237个依赖锁定URL及哈希，pip check通过。明确将numpy<2约束改为
  numpy2.2.6，以满足vLLM0.20.0依赖；torch2.11/transformers5.5.2。Chess发布目录缺少
  verl.tools，通过ours显式加载同固定WHALE提交内Math工具包，schema与SearchQA副本
  哈希相同。原Chess训练入口、actor及agent loop导入通过；实际CUDA训练未验证。
- **研究状态**：E1–E3模块保持不变；新代码属于共享执行设施，不增加Method创新公式。
  没有权重训练或正式搜索，没有F0–F5科学通过或VETO收益。上游和旧项目未修改，未push。
- **本轮验证**：72项CPU测试通过；新的文档本地链接与可提交文件的密钥扫描通过。
  三个成功GPU检查共约0.78361 GPUh；另有早先221576亚秒失败占用未测。

下一步：验证新环境中的实际GPU推理与成功过滤更新、checkpoint导出及真实服务权重切换，
再接通有明确预算的正式harness搜索。历史221577计划绑定旧代码，须回到`60b836c`或
`a542169`复现；当前分片计划与当前源码相绑定，不能混用。

当前证据：[GLM工具检查](results/glm-proposer-tool-probe-20260909-attempt5.json)、
[预算记录](results/glm-budget-summary-20260909.json)、
[完整解码审计](results/chess-decode-221597-summary.json)、
[训练导入](results/training-imports-restored-20260909.json)。

## 较早同日里程碑：原Chess评测器接通本地Qwen3.5

本次新Goal明确要求实现与实验准备，因此继续独立项目whale-delta中的可逆代码工作。
没有重启或修改beyond-entropy；下方initial-review的暂停/待批准记录是历史状态。

- **已实现**：`ours/` 中的E1配对评测、继承式E2–E3接受器及off/paired/三个同数据
  对照；initial/ordinary/early-stop/resume的边界API；身份哈希、覆盖与恢复核验。
  adapter尚未接到真实VLM循环，不把四种边界单测写成四路径端到端验证。
- **已验证**：66项CPU测试通过；执行了固定上游选择函数体的关闭回归。生成8对工程
  图表、16PNG，保存图像的独立pixel oracle全部通过，人工查看一张。
- **真实输入检查**：固定缓存Qwen2.5-VL-3B、seed17、greedy，完成32次生成。
  有图16/16题、8/8对正确；无图8/16题、0/8对正确。原始输出和独立重评分已保存。
  这8对简单图表有明显天花板，只能验证接线，不能作为VETO收益或F1整体通过的证据。
- **Chess执行端**：`ours/local_completion.py` 和显式compat命名空间补充本地HF客户端，
  原runner、harness、verifier未修改。固定官方Qwen3.5-2B权重与Lichess来源版本；
  作业221577完成8次生成，25秒，约0.00694 GPUh。8次落子合法、0题解出，均未截断。
  此次greedy、关闭thinking、缩短预算，只证明首轮调用接线；不是正式baseline分数。
  样本没有成功轨迹，不能直接启动RSFT。客户端是本项目实现，不是恢复了缺失原包。
- **前置阻塞证据**：真实pip dry-run确认vendored verl缺包装元数据；原Chess client、
  math环境、SearchQA配置、FST skills等缺口仍在，live checkpoint→server handoff未验证。
  这是NOT_READY，不是基线性能差或VETO收益的证据。详细见EXPERIMENTS.md。
- **proposer选择**：用户询问Claude成本后提出本地/GLM替代。GLM-5.3-Flash官方API和
  open weights已查；准备GLM Claude Code环境配置与本地OpenAI-chat配置，未调用服务。
  后端改动属于共享设施，不是Method创新。未确定实际服务/账单路由和费用上限。
- **预算**：同200k输入+20k输出token假设，每proposer session Opus4.7约$1.50，GLM
  常规价约$0.04；3,900session的日程例子为$5850/$156，均非实测/上限。
  GLM半价于2026-09-09 24:00 UTC+8结束。详情与官方来源见notes/proposer_options.md。
- **失败与修复**：两次datasets/Arrow退出异常已保留；改同步行组读取后正常退出，
  原始64条记录和8题Parquet与失败尝试逐字节一致。221576因提交前检查未拦截错误而
  缺计划退出，无模型调用；已增加check-only入口，并在提交前明确检查返回码。
- **成本与结论**：API费用0；221552/221577成功作业共69秒，约0.01917 GPUh；
  另有221576失败作业，控制器记录0整秒，亚秒占用未测。
  训练0、harness搜索0。VETO假设仍未成立，F0–F5科学判据未通过。
  ALL通知已配置，邮件是否送达未独立验证。所有修改本地保留，无push。

下一步：恢复原解码设置，验证多步轨迹和RSFT成功样本；落实proposer入口与预算、
训练依赖及真实权重切换，再执行F0–F2。
本地视觉target跑通不代表本地proposer代码修改能力已验证。原baseline各条件统一使用相同proposer与接线；
更换模型须标为复现变体。不能拿测试fixture填性能/消融表，也不预设提升归因比例。

入口：[实现与公式](ours/README.md)、[工程与失败记录](EXPERIMENTS.md)、
[真实输入检查结果](results/visual-smoke-221552-summary.json)、
[Chess运行检查结果](results/chess-smoke-221577-summary.json)、
[provider与成本](notes/proposer_options.md)。

## initial-review历史快照（本次新Goal之前）

更新：2026-09-09。项目目录：Documents/whale-delta。独立于原 beyond-entropy 项目；该项目文件没有修改。

## 当前结论

阶段 0 已完成研究执行链审读及10项 attack surface；阶段 1 已形成待审核 thesis。当前候选 VETO 聚焦视觉任务的 harness 接受约束；新颖性和有效性均未成立。只有领域迁移/既有视觉 loss 拼接，增量不够。

原昂贵的跨 harness 真实分叉 lookahead 不进入正式算法计划。用户更新为轻量模块与视觉方向后，主线改为 attack #10，辅助为 #5；受控分叉仅在必要机制诊断中使用。

## 交付物

- [whale_anatomy.md](notes/whale_anatomy.md)：确切控制流、fixed/adaptive、RSFT reward/filter/loss、proposer 与搜索空间、硬编码/脆弱参数和发布缺口。
- [attack_surface.md](notes/attack_surface.md)：10项机会，标明原作已经做的消融和不应认领的贡献。
- [thesis.md](notes/thesis.md)：一句话假设、先证伪后支持、轻量模块、数学与代码映射、两域两规模三 seeds、消融/图/limitations/停止条件。
- [novelty_screen.md](notes/novelty_screen.md)：相关工作碰撞与阅读深度；初筛不是新颖性证明。
- [CHANGES.md](CHANGES.md)、LICENSE、NOTICE：固定来源与修改声明。

## 阶段门

| 阶段 | 状态 | 下一步 |
|---|---|---|
| 0 原作审读 | 文档完成；未宣称逐行审计所有无关 vendored 模块 | 保留 commit 与源码定位 |
| 1 Thesis | 待用户 review/批准 | 审核 VETO 假设和 F0–F5 停止门 |
| 2 Baseline | 未启动；静态发现复现前置缺口 | 批准后先恢复可溯源依赖/接线，再四条件原域复现 |
| 3 方法 | 未开始，未创建 ours/ | 原 baseline 通过且视觉诊断支持后才进入 |
| 4 实验 | 未开始，无任何本项目实验数字 | 两域×两规模×五条件×三 seeds 的主矩阵只是计划 |
| 5 论文 | 未开始，未创建 paper/ | 依据真实证据决定是否值得写，不保证投稿档次 |

本轮最后读取的 goal 状态为 paused；未更改该状态。原始“thesis 批准前不写实现”要求也仍有效。

## 已知复现前置问题

共享循环仅导出 checkpoint 变量，并未完成 HF 导出/实际评测服务权重切换；adaptive 权重停止参数未在 launcher 接通；math 缺 env/recipe/patch，Chess 缺外部 client；WHALE-FST 的 prompt-only skills/入口不完整。详细证据见 anatomy §6。它们是静态发现，**不是已运行失败的记录**。

不要因这些缺口给 WHALE 记低分；所有可确认等价的修复须对所有条件统一应用，并公开记录。无法确认等价或四条件无法复现时，按用户要求停止并报告。

## 工作范围与接续

上游原 checkout 固定 fbe125eb7abea7f760c99ab9acc1a6261e708fc6；未改源文件，未安装依赖、下载训练数据或提交 GPU，未 push。所有性能数值待真实运行。

批准之后先核查运行环境、最新资源和有效通知设置，再做 F0；不是批准就直接铺开60个主 run。新方法没有证据时，不新增大量无用途脚本，不把框架适配成本计为科学创新。

## 本地 Git

按用户要求启用本地 Git，分支 main，初始文档快照标签 initial-review。WHALE 以 submodule 固定上述 commit；父仓库无远端。提交身份沿用现有 beyond-entropy 项目的 Yihang Chen / GitHub noreply 邮箱，仅设置在本仓库。此次授权用于版本管理，thesis 仍待审核，未进入实现或实验阶段。
