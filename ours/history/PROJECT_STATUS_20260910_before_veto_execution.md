# Project status

更新于2026-09-10 10:58，香港时间。**seed42 的两项独立基线已完成各自阶段；WHALE 第一阶段及导出完成，联合第一轮保留h2（5/32），第二轮h4/h5均已核验6/32，h6正在评测。** 尚无留出集分数或 VETO 收益结论。

之前逐次更新中的运行状态已归档到[历史快照](ours/history/PROJECT_STATUS_20260910_before_first_joint.md)。[实验日志](EXPERIMENTS.md)保留失败、恢复修订及完整成本来源。

## 四条件 × 三种子进度

所有试验从相同的未训练共同BF16权重开始，训练128题、MH优化32题、test64题不相交；[协议](ours/controlled_pilot_protocol.md)和[划分记录](results/chess-pilot-manifest-20260909.json)已冻结。下表中的“完成”指该训练/搜索阶段及证据闭合，不代表最终准确率通过。

| 条件 | seed42 | seed43 | seed44 |
|---|---|---|---|
| weight-only | 两批及最终规范导出完成；保留预检题序偏差 | 两批计划及真实DataLoader预检通过，未提交 | 同左，未提交 |
| harness-only | 5轮×3候选完成，最终h1 | 独立计划通过，未提交 | 独立计划通过，未提交 |
| WHALE | 第一阶段及导出完成；搜索完成1/5轮，保留h2，h4/h5完成6/32，h6作业223909运行中 | 第一阶段计划通过，未提交 | 第一阶段计划通过，未提交 |
| WHALE-FST | 第一阶段计划通过，未提交 | 第一阶段计划通过，未提交 | 第一阶段计划通过，未提交 |

共同模型：`data/models/qwen3.5-4b-common-bf16-v1`，manifest SHA `260337087e4d739db98c2744cf429678cd2f87674e55cad9f22d3cfbe60c3d1a`。723个活动张量/4,539,265,536元素已完整检查；精度转换不计为学习收益。WHALE-FST的提示词搜索约束含明确标注的公开重建部分。

## 已完成且可核验的结果

**Harness-only，seed42。** 原生五轮均保留h1；全部16个候选评测经过32题回复/token/棋盘检查、实际worker权重与seed检查及Slurm终态核验。[完整搜索证明](results/controlled-harness-only-seed42-completion-20260910.json)。

| 候选 | h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | h13 | h14 | h15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 解出题数/32 | 1 | 8 | 5 | 0 | 6 | 7 | 8 | 6 | 7 | 7 | 8 | 6 | 6 | 8 | 8 | 4 |

[完整搜索轨迹图](results/controlled-harness-only-seed42-search-trajectory-20260910-v1.png)及对应CSV/PDF已生成：实际选择只在每轮结束后更新，保留全部16个评测。累计515次任务模型调用、3,991,282输出token、9.165556GPUh。512次题目评测来自反复使用同一MH32，不能当作512道独立题或留出泛化证据。三次提案格式中断及显式恢复修订全部保留；未重复付费提案或已审计的模型评测。h1固定回复在旧parser下动作相同，不能把分数变化归因于parser或声称完成了因果消融。

**Weight-only，seed42。** 原生222801完成两批128条新轨迹：3/64和0/64 accepted，共1次SFT更新、22,989个loss token；139次请求、1,039,118输出token、0错误。[训练终态](results/controlled-pilot-222801/result.json)及[完整轨迹审计](results/controlled-pilot-schedule-deviation-audit-222801.json)。原预检遗漏DataLoader初始化推进随机流，预写题序错误；实际原生配置/seed推导顺序、两批轨迹及保存sampler状态一致。偏差保留，没有改题或重采样。

规范导出223620于07:43:24 COMPLETED/exit0，329秒/1RTX4090/0.091389GPUh。全部723张量逐元素满足原生权重的精确BF16转换；相对共同起点，原生FP32有3,853,212,360元素变化，导出BF16有9,033,512元素变化。两个原生checkpoint的模型字节相同，对应第二批无更新；全部原生恢复状态仍保留。参数变化不证明准确率提高。[导出终态](results/controlled-checkpoint-export-223620/result.json)、[完整参数变化](results/controlled-weight-only-seed42-222801-step2-transition.json)。

## 当前运行和下一步

WHALE seed42第一阶段223622从独立共同theta0和原h0开始，07:46:21–08:23:08 COMPLETED/exit0，2207秒/2H800/1.226111GPUh。[冻结计划](results/controlled-whale-seed42-phase1-plan-20260910-v1.json)、[完整64条轨迹审计](results/controlled-joint-phase1-223622/batch-audit.json)、[训练终态](results/controlled-joint-phase1-223622/result.json)。68次请求全部返回、517,481输出token、0错误；6/64 accepted，来自00BQD和007fJ各3条；45,996个loss token，1次真实SFT更新。模型、调度器、CPU/CUDA/NumPy/Python RNG及正确sampler进度均已保存核验；原生不保存Adam，尚未执行GPU恢复。训练接受率与独立weight-only第一批的差异不能归因于联合搜索：本阶段尚未修改h0。

联合搜索第一轮已完整评测，原生选择器保留h2；[四次评测和选择快照](results/controlled-mh-223898-completion.json)核对全部原生/token/棋盘、worker和Slurm证据。这里反复使用同一MH32，不是128道独立题；完整五轮搜索仍未结束。

| 联合候选 | h0 | h1 | h2 | h3 |
|---|---|---|---|---|
| 解出题数/32 | 1 | 1 | 5 | 3 |
| 任务模型调用 | 32 | 32 | 32 | 35 |

累计本次联合MH评测131次调用、1,032,116输出token、2.513889GPUh。h2/h3分别1099秒/0.610556GPUh与1129秒/0.627222GPUh，均COMPLETED/exit0。第二轮h4作业223905已完成6/32，1125秒/2H800/0.625000GPUh，32次调用/258316输出token，其中30次长度截断；[完整终态](results/controlled-mh-223905-completion.json)通过。h5作业223908也已完成6/32，1098秒/2H800/0.610000GPUh，32次调用/256806输出token、29次长度截断；[完整终态](results/controlled-mh-223908-completion.json)通过。控制器1413742在2H800上评测h6作业223909，第二轮尚未选择。不同搜索中的同名harness不是同一个候选，未完成联合条件不能作为最终胜负比较。

[h2固定回复诊断](results/controlled-joint-h2-parser-comparison-20260910-v2.json)确认提示词和声明预算未变，31/32次解析动作与h0不同；5个成功轨迹中4个动作不同，3个成功仍包含长度截断。按原生任务轨迹匹配后，32个请求和输入token相同，仅3个输出token序列完全一致，因此不能把总分差全部归因于parser。附加请求比较的初次按日志行序配对错误已明确纠正；原生得分和固定回复解析统计未受影响。

1. 第一阶段及[规范导出223757](results/joint-checkpoint-export-223757/result.json)完成：244秒/1RTX4090/0.067778GPUh，723活动张量逐元素符合原生BF16转换，9,013,513个变化元素保留到导出。两个实际worker的8坐标和seed42通过，其中5坐标区别于共同theta0；这是抽查，不是所有服务参数的完整证明。[实际交接快照](results/controlled-mh-223824-start-snapshot.json)。原生恢复状态保留。
2. 完成同一theta1下余下四轮联合MH搜索，再由原生checkpoint真正恢复并执行第二批。Method对应原WHALE的 theta1=RSFT(theta0,h0;B1) → harness搜索 → theta2=RSFT(theta1,h1;B2)。不复用独立weight-only的训练前缀或harness-only的搜索结果。
3. 完成其余独立种子和条件，封存12个trial的最终身份或未完成原因，再按[留出协议](ours/controlled_heldout_protocol.md)执行固定64题评测。封存/四分片/汇总代码已通过CPU检查，当前没有正式cohort、test计划或test分数；test题目及答案仍未加载。
4. 视觉输入通道确认后接通共享视觉环境、原生训练与VETO接受模块，逐项执行F0–F5。没有预设收益幅度或中稿结论。

## 视觉接线的真实边界

[VisualEvidenceDataset](ours/visual_evidence_dataset.py)在ours/中继承原生dataset，修复提示替换丢图及长度过滤漏计图像展开；prompt环境进入缓存身份，图像字节不与答案一同传给模型。它是所有视觉对照共用的输入通道，**不是VETO E1–E3的创新**。

两项原生CPU检查使用现有一对工程图片：提示替换前后均保留120个图像token，不同像素仍进入原生SingleTurn请求和训练DataProto，mRoPE与回复掩码保持。无图控制、长度阈值、缓存身份和提示变更拒绝通过。回复和reward明确为合成夹具；未加载模型权重，没有视觉生成、actor前反向或完整视觉训练/搜索结果。[实现与证据记录](results/visual-native-interface-implementation-check-20260910.json)。

[共享视觉奖励](ours/visual_evidence_reward.py)已接通原生reward加载器、manager、worker和RSFT筛选循环的CPU检查。Method为E4的二元验证 `r_i=v(answer_i,y_i)` 及原生 `B+={tau_i:r_i>0.5}`；不改变损失，所有视觉条件共用。2项检查通过：脚本回复得分[1,0,1,0]时只选第0/2行，像素张量和回复掩码保持；全不合规时跳过更新。模型回复、actor操作和Ray传输是夹具，实际模型调用/优化器步数为0，不是视觉准确率。原生动态导入要求`pkg://`，首次缺少此前缀的失败及修正均保留。[奖励集成记录](results/visual-reward-integration-check-20260910.json)。

[共享多轮工具适配](ours/visual_evidence_tool_loop.py)继承原生ToolAgentLoop和ImageZoomInTool：crop使用模型实际看到的初始像素，最后一段assistant决定二元奖励，工具反馈/图像维持原生零loss mask。独立agent/tool YAML及1项原生CPU测试的6种情形全部通过：正确/错误crop回答、无效框、无图、无最终答案、直接回答；无图不能借metadata取回图片，原生工具错误分不进入任务奖励。10次脚本请求，0实际模型调用/优化器步数。实际actor更新、视觉模型服务及搜索编排仍待接通；下述接口已提供CPU候选派发和评测批次执行；不是VETO新机制。[最终工具集成记录](results/visual-tool-integration-check-20260910-v3.json)。

[可执行视觉harness接口](ours/visual_harness.py)已接到原生dataset和tool loop：观察格式、crop参数、反馈、答案解析及预算内追加请求共五个函数。默认h0在测试中保持现有请求/token/像素/mRoPE；候选代码SHA进入缓存和每行数据，数据/rollout身份不符或源文件变化会拒绝。prompt-only只改两条字面提示，非提示AST必须与参考相同；可变函数默认参数也会被拒绝，避免跨题携带状态。2项原生CPU检查通过，包含实际关键词dataset工厂、Hydra加载、160→128的真实crop、反馈/追加请求、原始与提交答案、有效操作哈希和有/无工具批次合并。模型回复仍是脚本，0实际模型调用/优化器步数；AST检查不等于OS隔离。实际视觉proposer、GPU服务启动/计划和VETO接受位置仍待接通，强视觉h0与正式视觉数据协议尚待建立。[候选执行记录](results/visual-harness-integration-check-20260910-v2.json)。

[原生配对评测入口](ours/visual_native_evaluation.py)已在CPU接通原生Manager分批、Worker生成入口、Hydra视觉agent、crop、DataProto归档与E1配对统计。配对Parquet只把像素/问题放进模型消息；生成前检查完整覆盖及图像/标签/问题与manifest一致。返回按样本身份合并，独立核对parser、reward和mask；缺一侧、不均匀worker批次、重复身份或错误reward均拒绝，后续批次中断不产生整集得分。真实生成次数、返回assistant token、保留loss token与原生turns分开记录，不改变E3排序。最终评测1项原生检查23.076秒通过，另两项harness检查通过；Manager/Worker构造器未执行，Ray RPC和模型回复为夹具，0实际模型/API调用或优化器步数。首次冷启动导入失败保留；实际模型身份/解码、服务启动和VETO接受尚未由此验证。[完整记录](results/visual-evaluation-integration-check-20260910.json)。

视觉依赖安装在独立、显式启用的`data/visual-runtime-overlay-v1`，没有改动冻结的Chess训练环境。共同模型与223620导出模型的视觉/无图、thinking开关共8种输入检查得到完全相同的处理器张量和mRoPE，证明此次资产规范化保持这些输入的语义，不代表视觉性能通过。

VETO仍是待验证假设：E1配对视觉正确率、E2固定入阶段参考的非退化约束、E3先任务分数后调用数；E4属于原WHALE的成功筛选训练。F0–F5均未通过。此前16张图的3B视觉工程检查及旧训练权重上的0/32→6/32，均与本轮受控试验分开保留。

## 成本、资源和验证

截至223908终态，已知项目累计25.849444GPUh（含已归档失败），当前h6尚未计入终态成本。10:48保守API项目记账0.613044CNY，82次已结、0未结，45元预算内可预留44.386956元；不是官方账单或账户余额。原生日志中的美元估价沿用上游假设，不能作为GLM实际支出。

10:48账户GPU配额实时查询：总3700h，已用787.416667h，剩2912.583333h（21.28%）；这是账户总量，不能混同项目成本。同一快照主目录剩120.26GiB，可支撑下一阶段，但不足以保留完整矩阵预估的后续约430GiB；共享存储/scratch路径问题仍待用户回复。未删除或压缩既有checkpoint。

基础检查125通过/99依赖跳过（随后新增的故障情形已由原生测试执行）；视觉dataset原生2项、奖励原生2项、多轮工具原生1项/6情形、候选派发原生2项、配对评测原生1项通过，完整导出与资产等价检查通过；16份既有冻结执行记录中155个实际代码/配置文件重哈希一致（同时纳入joint_source_sha256；此前153为主清单路径字符串，对应149个实际文件），此次未重新哈希大型权重。联合GPU恢复、完整多种子和真实留出验证仍待执行。通知已配置，实际送达未验证。上游固定`fbe125eb7abea7f760c99ab9acc1a6261e708fc6`且工作区干净；新增实现只在ours/，更改仅本地，未push，未修改beyond-entropy。
