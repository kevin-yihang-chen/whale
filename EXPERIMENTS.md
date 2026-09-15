# Active experimental protocol: compact VETO,2026-09-15

## Complete seed42 search, equivalent choices (2026-09-15 23:17 HKT)

All four actual H/C measurements are complete. H correct / C both-correct are
h0:107/128 and49/64; h1:111/128 and54/64; h2:111/128 and48/64;
h3:112/128 and53/64. WHALE, VETO and the marginal gate all choose h3;
independent reconstruction agrees with both final and resumed native boundaries.
This archive does not demonstrate a VETO-specific selection contribution.
`first-search-report-v1/result.json` preserves all1024 rescored H/C answers,
format summaries, q0 counts, decisions, hashes and0.688056GPUh of H/C allocations.
That cost is explicitly narrower than total method cost.

The prospectively fixed VETO stage2 is in actual-parent sampler preflight.
Candidate h3 filtering retained all2048 W inputs; the independent incoming-h0
reference is now being checked. No actual continuation result exists yet.
Any later increase with the shared h3 choice cannot by itself establish VETO's
independent value. Other seeds and independent endpoints remain incomplete.

One post-hoc C example is preserved in `illustrative-case-v1/case.json` with
original image bytes and all six h0/h1/h2 answers. The h2 narrative reads the
changed bar values but reverses operand order on image B. It therefore illustrates
the limitation that paired failure does not prove visual non-use. No tool calls
occurred in these six trajectories. This selected case is not a frequency estimate.

The v15 main/supplement PDFs compile and all six pages have been inspected.
They retain empty final method tables and acknowledge the equivalent choices;
the draft remains scientifically incomplete and its case layout needs revision.

## Partial real archive: h2 tradeoff, h3 running (2026-09-15 22:58 HKT)

The fixed incoming h0 scores107/128 H,49/64 paired C and110/128 marginal C.
h1 scores111/128,54/64,115/128; h2 scores111/128,48/64,107/128. Both gates
admit h1 and reject h2. Thus a real optimization candidate improves ordinary H
while worsening paired C, but this is neither an independent validation gain
nor evidence uniquely favoring VETO. The old2-point threshold is not claimed.
Job233759 evaluates h3; final selection still waits for the complete archive.
H/C h0/h1 answer-format diagnostics also retain all512 records: unparseable
H stays4→4, C changes4→1. Format and parseable-answer changes coexist; these
descriptive transitions do not establish visual causality. No protocol changed.
See the h2-tradeoff-review and format-diagnosis-h0-h1 receipts in results.

## First real candidate archive and bounded continuation (2026-09-15 22:39 HKT)

At the fixed selected seed42 LR1e-6 prefix, incoming h0 gives107/128 H correct,
49/64 C pairs both correct and110/128 C images correct. These are optimization
measurements, not independent method results. Three GLM candidates have been
generated once; the tool trace did not consume available H feedback, so the
pool is described as generic mutations. Preserve this limitation, every
candidate and its cost; do not replace candidates based on this outcome.

Job233589 evaluates h1. The bounded controller then evaluates remaining slots,
compares WHALE/VETO/marginal decisions through the original bridge and runs one
VETO continuation plus full followup. VETO was fixed before candidate results,
in `results/fast-chart-first-cycle-bounded-admission-20260915-v1.json`.
All five allocations together are capped at6 additionalGPUh in setup; no
formal matrix admission, cleanup authorization or independent test claim follows.
Sources, scorer, h0, LR, incoming reference and all existing checkpoints remain
unchanged. The previous subsection timestamps are historical snapshots.

## Shared calibration complete; no VETO comparison yet (2026-09-15 21:27 HKT)

All three real four-batch LR stages, image/gradient audits, full parameter
comparisons and V512 evaluations are complete. LR1e-7:443/512 ordinary,212/256
paired;1e-6:448/512,207/256;1e-5:280/512,89/256. The frozen ordinary-accuracy
rule selects1e-6. All2048 raw answers including the initial reference were
rescored with the shared host parser/verifier; no VETO effect is established.
Evidence: `results/fast-chart-complete-calibration-review-20260915-v1.json`.

Job233485 measures two complete V512 passes at batch16/32 on1H800, with a1GPUh
setup reservation. It also times input preparation so2048-image T/R forecasts
include size-dependent preparation. Eleven targeted tests pass. E1 input and
resource measurement only; E2–E4, h0, LR, scorer and sealed tests are unchanged.
Old unsubmitted throughput-v1 and pre-change source snapshots are retained.
Total charged13.07GPUh, held1GPUh; full expansion still requires the measured
forecast and any separate exact-path cleanup approval.

See `ours/fast_chart_protocol_20260915.md` and `ours/fast_chart_runbook.md` for
current settings and executable entrypoints.18 logical conditions, three seeds,
one4B model, one training–search–training cycle.100GPUh includes historical6.898056h;
400GiB new storage and45CNY cumulative proposer cap. Old51-condition and Chess/CLEVR/2B
expansion remains paused. Unknown results remain unknown.

Completed full-V h0 calibration cases are stored under
`data/fast-chart-20260915-v1/h0-calibration-v2/`. Structured h0 was selected by
ordinary V accuracy. The resumed LR controller under `lr-calibration-v3` is
comparing1e-7/1e-6/1e-5 on ordinary V accuracy only. The first two complete
measurements are443/512,212/256 and448/512,207/256 (ordinary, paired); both are
below the initial452/512,214/256. These are configuration-calibration results,
not VETO effects. Third-rate training233361 completed four batches,144 successful
trajectories and20 updates for1.02GPUh; followup233399 is running. No LR is selected early.
T/R/ChartQA inference has not occurred. New scoring results are
not pooled with the old candidate-parser protocol.

The remaining document preserves earlier protocols and records; it does not
activate their larger budget, deleted scope, or superseded scientific gates.

# Experiment and engineering log

## E97 — 首遍C与配对指标的实际信息量（2026-09-10）

h1首遍C已完整，重算E1为64/64、单图128/128；连同H128/128，相比h0的H115/128、
C47/64/111/128均提高，当前步骤没有目标退化现象。H完美分数仍不能绕过最终接受。
C原答复的h0解析回放为122/128、59/64对：6条格式差异影响5对，不能当新的模型测量。

pair_error_decomposition对应E1解释与marginal_gate必要对照。q0表示两边都错的比例，
恒等式为P=2M−1+q0，M是同一配对集的两边平均正确率，不是另一H集上的任务准确率。
3份完整C记录（h0首遍/重复、h1首遍）均q0=0，所以默认epsilon=0的两个门对当前档案
等价。3份V记录也q0=0，仅描述误差结构；不能将跨权重V记录用作真实C接受档案。
两项CPU算术检查通过，包含相同M但不同P的反例，明确指标并非普遍等价。

结论是当前工程证据不能触发scientific go，不能把新harness的分数提升计作VETO贡献；
不据此宣称方法普遍无效，不修改阈值或选择新的ordinary endpoint。当前作业重复继续，
真实接受/续训/固定h0复测待完成；没有开启正式矩阵。

证据：results/visual-search-h1-first-pass-components-20260910-v1.json、
results/visual-h1-C-parser-replay-20260910-v1.json、
results/observed-pair-error-decomposition-20260910-v1.json、
results/pair-error-decomposition-tests-20260910-v1.log、
results/visual-current-scientific-gate-assessment-20260910-v1.json。

## E96 — theta1固定h0的参考解析基准（2026-09-10）

在theta2尚未生成时，对已经保存的theta1/固定h0首次V128答复同时使用原h0解析和h1
参考解析：两者均113/128、49/64对，0条提交答案改变。只回放已见过的64对开发数据，
不是新模型测量、独立复评或选较高重复；旧第二次115/128、51/64的推理波动继续保留。
在未来theta2固定h0复测后对两边应用同一参考解析，用于识别格式漂移；主分数保持。

visual_answer_parser_replay增加固定h0的V入口与配对计数；旧H回放源码保存在
ours/history/visual_answer_parser_replay_20260910_v1.py，对应原报告09d1b2b6…源码SHA。
V回放命令的--incoming是参考解析器参数，此处传h1，不表示训练已经接受h1。
当前H/C评测仍在运行，没有绕过真实E2–E3接受和E4恢复。

证据：results/visual-theta1-fixed-h0-parser-reference-20260910-v1.json及同名log。

## E95 — h1完整H局部结果与答案解析依赖（2026-09-10）

224335首遍H128已完整：h1=128/128，h0=115/128；13个0→1、115个1→1。
逐个核对真值、原回复解析与批次文件哈希，H/C整体认证及作业终态仍待完成。
相同128次模型调用，生成token从256增至27,460。H是优化数据，不能标为独立泛化成绩。

新增visual_answer_parser_replay对应E1答案提取的诊断通道，对保存的真实回复执行两个
隔离解析器，不重新调用模型。h1自身解析128/128；同一回复用h0解析120/128，有8条
提交答案改变。此事后重放只定位分数对格式恢复的依赖；不能把8/13当成因果贡献比例，
也不能代替真正的prompt-only/FST比较。h1仍未被接受，C及重复遍继续按冻结方案运行。

在theta2尚不存在之前记录后续解释约束：固定h0前后复测必须排查格式漂移，原主评分
保持；共享参考解析的诊断须对前后原始回复一起应用并同时报告，不择优改写结果。
没有格式因素与推理噪声控制的下降，不认作视觉捷径进入权重的机制证据。

证据：results/visual-search-h1-complete-H-component-audit-20260910-v1.json、
results/visual-h1-H-parser-replay-20260910-v1.json；两者均不代表整个候选评测或搜索接受完成。

## E94 — 续训后固定h0复测的数值核验入口（2026-09-10）

当前224335的真实h1评测仍运行，未提前接受或训练。补齐续训后的导出/同h0复测入口，
使用完整真实续训回执与原生FP32变化审计后才允许prepare。真实theta1与theta2比较不
使用舍入后的BF16模型充当原生训练起点；导出存活的变化另报。底层仍复用原生merger、
固定h0视觉输入、原答案验证器和按源图表的逐题比较，未添加训练步骤或读取R/T。

2项小张量CPU检查通过：舍入残差不能使原生不变变成“训练已更新”；低于BF16分辨率
的真实更新保留FP32差异、服务差异仍为0。它们不替代全模型、GPU导出和完整配对验证。
新入口准备1H800/12CPU/1小时档位及32GiB工作估计+40GiB余量，当前未提交或预留费用。

证据：results/visual-resumed-followup-implementation-20260910-v1.json、
results/visual-resumed-followup-tests-20260910-v1.log。

## E93 — 真实h1的原生续训输入预览（2026-09-10）

使用已获GLM候选h1和原global_step1保存的数据加载状态运行CPU预览，原32个W样本与
下一8个样本顺序均一致，每个输入仍包含图像；候选SHA绑定真实proposal-ready回执。
角色明确为CPU_H1_INPUT_PREVIEW_NOT_ACCEPTED_OR_EXECUTABLE_TRAINING_PLAN，selected=false。
不生成轨迹、不执行actor加载RPC、不训练、不预留GPU或调用API。若h1最终被接受，真实
prepare仍须核对完整H/C接受回执并再次检查输入；本预览不能绕过该条件。

证据：results/visual-h1-resume-input-preview-20260910-v1.json及同名log。

## E92 — h1长回答的测量时间修订（2026-09-10）

224334完成34道H时，已测完成间隔均值7.030536秒，512图约3599.6秒，尚不含启动。
原1800秒分配不能完成全部协议，因此按吞吐提前取消；未按中途准确率决定取消。
终态CANCELLED，464秒/0.128888889GPUh已结，全部原始答复保留，无完整汇总可供选择。

新不可变计划v2将资源上限改为5400秒并使用新输出目录；同一h1 SHA、theta1、H/C、
批次/解码、首遍选择与重复规则逐项保持，完整512图从头测量。未将第一次34题拼接成
新运行，未缩短回答、删除重复或改写候选。底层Slurm包装器源码保持，提交命令显式
--time=01:30:00与新计划/预留相同。224335已提交，预留1.5GPUh；0新增API请求。
这是工程测量预算修订，后续公平成本包含两次分配，不标作算法自身的性能收益。

证据：results/visual-search-h1-time-limit-revision-20260910-v1.json、
results/visual-search-theta1-h1-plan-20260910-v2.json、
results/visual-search-theta1-h1-preflight-20260910-v2.log、
results/visual-search-theta1-h1-allocation-20260910-v2/submission.json。

## E91 — 已授权的首个真实视觉提案与同权重评测（2026-09-10）

用户明确回复“我授权发送”；授权回执绑定此前审核的8份输入、43,161字节及源码SHA。
只向官方GLM提供H128反馈、h0和规则，生成一个h1；未发送C/V/R/T、像素或模型权重。
原生提案72.2秒，6次实际API请求，本地估算新增0.084845元，累计已核算1.000592元；
另保留历史未结算预留6.38976元。原生CLI显示的美元估计不作为GLM费用依据。
单次调用内部有一次Edit失败，随后同次会话完成，原始错误和全部工具记录保留；
未进行新的付费提案重试。h1 SHA为16dec3907f6e67a4374738c8ab6eaad472bb327162fa9f4b307015183937a24d。

提案报告错误声称h0为82/128；实际反馈重算115/128。候选描述还声称解析支持加粗/括号
标签，实际实现并未覆盖所有这些样式。报告仅作为未验证的提案假设保留，评测使用
真实代码与共享验证器，不按模型文字报告改分或补改候选。

原生搜索恢复至WAITING_CANDIDATE_GPU_EVALUATION，H/C基线与h1身份逐项相同。
224334使用1H800/12CPU/最多30分钟，H128、C64对各测两遍；首遍选择、重复仅诊断。
0训练、0新模型检查点，预留0.5GPUh，终态由独立观察入口保存。51组矩阵继续暂停。
本节在完整评测结束后追加实际分数，当前不能声称候选或VETO胜出。

证据：results/visual-proposer-authorization-20260910-v1.json、
data/visual-search-bridge-theta1-20260910-v1/proposal-ready.json、
results/visual-proposer-narrative-audit-20260910-v1.json、
results/visual-search-theta1-h1-plan-20260910-v1.json、
results/visual-search-theta1-h1-allocation-20260910-v1/submission.json。

## E90 — 接受后视觉E4续训入口的CPU验收（2026-09-10）

GLM外发授权尚待用户回复；无重复调用/替代外发。独立完成视觉续训入口及三处观察：
原生FSDP actor恢复后完整值/scheduler/RNG检查、原生零worker sampler待继续状态、
原NCCL同步后vLLM接收端8个坐标。生成之前必须有step1恢复和传递证据；step2保存后
再核对接收值。native RSFT的成功过滤、损失、采样及Adam-reset约定保持。

测试5项/17.114秒及完整worker初始化1项/14.431秒通过。CPU小张量案例确实调用原生
checkpoint loader；GPU、Ray传递及模型回复为夹具/未调用，不产生论文性能结果。
包括改变旧receiver值时拒绝、复用原共享多模态记录、初始化幂等且不推进RNG、裸
accepted文件不能伪装成完成搜索。原搜索选择仍由E89的bridge管理，未手写替代候选。

实际W新配置预览v1通过后，增加显式W manifest/parquet身份检查及搜索回执冻结，
v2再通过完整CPU数据读取：原32个W子集一致，原生global_step1/data.pt恢复成功，
下一8题与E89预期完全相同且含图像。采用已有incoming h0；尚无最终接受的新策略。
配置预览明确标为CPU_CONFIGURATION_PREVIEW_NOT_EXECUTABLE_TRAINING_PLAN，不能
通过真实prepare/run的完整选择与血缘检查。真实GPU actor/receiver恢复和新更新待执行。

计划入口预留2H800/24CPU/1小时档位，但未预留费用或提交；正式准备时须再次检查实际
容量，并保留40GiB工作空间估计及40GiB余量。本轮0GPUh、0API、0新模型检查点。
17:55实查约189.97GiB可用，GPU台账仍5.184722已计/0预留，正式扩展关闭。B组保留。

证据：results/visual-training-resume-tests-20260910-v1.log、
results/visual-resume-hook-tests-20260910-v1.log、
results/visual-resume-input-preflight-20260910-v2.json、
results/visual-resume-config-preview-20260910-v1.json。

## E89 — 推理波动复核、真实H/C起点与原生视觉搜索边界（2026-09-10）

224157完成548秒/1H800/0.152222GPUh。初始模型复测117/128、53/64对，与原次相同；
训练后模型复测115/128、51/64对，较首测113/128、49/64有2个答案变化。实际输入/解码
多重集相同。不能挑选较高重复，首测bootstrap不涵盖推理或跨种子变异。其终态从当时
真实工具stdout恢复并保留来源行/哈希，未根据预期时间重建Slurm记录。

随后224196使用固定theta1、h0，在H128与C64对上各测两遍，串行batch1/max_num_seqs1，
关闭prefix cache/chunked prefill。512次实际生成、1024输出token；630秒/1H800/0.175GPUh。
首遍H115/128、C单图111/128及配对47/64；第二遍H117/128、C112/128及48/64。
H有2个、C有1个答案变化，repeatability_passed=false。没有用串行设置声称完全确定性；
首遍作为候选选择证据的规则在运行前冻结，重复只诊断，所有结果保留。H/C为优化数据，
未读取R/T。累计本协议GPU费用5.184722、无待结算预留，正式扩展关闭。

新增visual_search_bridge对应E1→E2–E3连接，真实h0已进入原run_evolve和初始接受。
原搜索的准确率/turns及early-stop保持；阶段完成后用全候选C档案约束下一训练harness。
初始/普通/提前停止/恢复均通过同一adapter，off直接调用原selector；C回执保留在公开H
提案目录之外。4项CPU检查中真实调用原普通/early-stop循环，模型和候选分数是显式夹具。
真实运行尚未越过提案边界；没有把CPU h1或其分数计作模型实验。

GLM真实提案动作在进程创建前被自动审批拒绝，理由为新视觉h0代码及H反馈外发范围未获
明确授权。无paid-proposal目录、无调用日志、API账本未增：保守已计0.915747元、
未结预留6.38976元、45元内可预留37.694493元（均不是官方账单/余额）。外发清单
results/visual-proposer-transfer-review-20260910-v1.json逐项列出8份输入43,161字节，
目的地为已配置智谱官方接口，限一个h1/12请求/300秒，无自动重试；等待具体范围确认。

独立E4 CPU恢复检查使用真实global_step_1/data.pt，原loader加载global_step=1及相同
pending state；下一批8道W与前批8道不重叠并保留图像。actor加载RPC仅记录，未执行
GPU恢复或参数更新。零worker的StatefulDataLoader使用扁平schema，不套用Chess的
多worker_snapshot结构。首次本地socket受限失败与允许socket后的通过日志均保留。

证据：data/visual-search-theta1-20260910-v2-h0/result.json、
results/visual-search-theta1-h0-allocation-20260910-v2/allocation-result.json、
data/visual-search-bridge-theta1-20260910-v1/initial-acceptance.json、
results/visual-search-bridge-tests-20260910-v1.log、results/visual-native-data-resume-20260910-v1.json。

## E88 — 首次视觉E4→E1交接完成；固定开发子集下降，重复性待复核（2026-09-10）

224121于16:36:25–16:44:48 COMPLETED/exit0，1H800/12CPU/503秒/0.139722GPUh。
native_visual_followup复用原生FSDP merger、规范序列化、完整活动参数比较和共享E1
评测，对应E4→E1交接；不改变损失、成功筛选或VETO接受规则。原生FP32改变
3,575,378,309/4,539,265,536元素（426/723张量），BF16导出改变53,104,103元素
（302/723张量）。全部导出逐元素等于原生BF16转换；9份原生model/extra/data恢复
文件哈希保持。实际worker的8个坐标均区别于初始模型并匹配导出，属于加载抽查。

| 同一固定harness与64对V图表 | 初始模型 | 一批RSFT后 |
|---|---:|---:|
| 单图正确 | 117/128 | 113/128 |
| 两边均正确 | 53/64 | 49/64 |
| 实际模型调用 | 128 | 128 |

4个答案从正确变为错误，其余124个相同。以64个不同源图表成组重采样20000次、
seed20260910，差值区间：单图[-6.25,-0.78125]个百分点；配对[-12.5,-1.5625]个百分点。
这些区间只描述当前固定开发子集，不能替代训练种子、独立复评池或封存测试。
普通准确率也下降，且没有harness搜索；不满足研究现象门槛，也不构成VETO效果。

实际请求的RGB像素/token/采样参数多重集合完全相同。新增audit_visual_followup_inputs
独立比较16对归档batch的128个样本，处理后pixel_values、image_grid_thw、images_seqlens、
提示token、位置编码、原始提示、标签与harness身份逐一相同。它是E1测量核验，不保证
内部GPU激活或跨运行确定性。4道改变的题所选答案token概率接近决策边界；现准备
两个固定模型各重复全套128题一次，保留全部重复结果，不择优、不新增训练或检查点。

比较器在真实基线自比较上返回零差，并拒绝篡改一个请求token的输入，记录
results/native-visual-followup-matching-check-20260910-v1.json。主要结果位于
data/native-visual-followup-20260910-v1/；处理后输入核验为
results/native-visual-followup-input-audit-20260910-v1.json。新推理权重9.64GiB，
16:41磁盘余量191.11GiB，无新增清理/API费用。台账累计4.857500GPUh，正式矩阵未启用。

重复性复核224157已于16:57:03启动：1H800/12CPU/96GiB RAM/30分钟上限，预留
0.5GPUh。源与模型检查通过，同一64对V图表、相同完整输入和greedy解码；初始与
训练模型各重复一次，不新增模型检查点。计划为
results/visual-followup-repeatability-plan-20260910-v1.json，结果目录
data/visual-followup-repeatability-20260910-v1/。本行仅记录提交，尚无重复结果。

## E87 — 首批真实视觉RSFT完成；47条成功轨迹、6次优化器更新，效果待评测（2026-09-10）

native_visual_training.py从既有原生RSFT配置继承actor/AdamW/成功过滤/检查点行为，
使用共同4B与已完成的direct条件。仅为执行验证：原生seed42从W2048随机取32道，
DataLoader选8道，每题新采样8次，共64轨迹，一批训练，lr1e-7；尚未做正式LR选择。
无工具、无开发题训练、无API；原生model/extra保存不含Adam，恢复仍待实测。
2H800分别用于actor/rollout；4GPU会改变actor分片或增加短答案采样副本，暂无吞吐
证据支持扩大。现有磁盘按70GiB工作/检查点加40GiB余量检查，无删除。

CPU预检v1发现推理配置合并覆盖了原生log_prob_micro_batch_size_per_gpu，未提交GPU。
恢复原值1后v2通过真实32道dataset/processor/StatefulDataLoader及原生配置校验。
v2说明文字把原生随机子集误写成前32条；实际记录的sample IDs正确。保留v2源码和
未提交回执，v3只修正描述，再次记录相同原生题序。失败日志/源码都在对应版本目录。

224008在v3下实际完成64次生成/128输出token，然后在记录检查中止；0优化器更新。
原因是原生AgentLoopWorker._postprocess在reward-worker handles存在时不保留输入
metadata；E1无handles路径此前会保留，导致CPU记录器夹具未覆盖这个真实训练分支。
visual_sample_id丢失，而harness SHA、最终答案和像素仍由原生extra字段保留。
FAILED/2H800/196秒/0.108889GPUh，全额计入；请求/回复和冻结源码保留，检查发生在
全batch归档之前，因此该失败没有可直接续用的完整DataProto。不得写成成功训练。

新增visual_training_identity.IdentifiedVisualHarnessAgentLoop只继承原loop并将输入
sample ID放进output.extra_fields。独立注册visual_training_agent防止父类注册覆盖，
对应E4来源连接，不改变Method或训练行为。新的原生reward-handle分支测试与像素/
mask序列化及错误reward拒绝测试合计2项16.752秒通过，模型/actor为明确夹具。
首次单独记录器检查1项14.239秒通过，不能当成完整训练分支已覆盖。

v4预检通过，实际32题集合和首批8题顺序与v3完全一致；224041于16:13:31启动，
2H800/24CPU/160GiB RAM/1小时上限。新生成64条轨迹，47条reward=1进入原生SFT，
完整rollout及accepted DataProto已保存；实际参数变化、恢复及独立效果仍待核验。
提交/源身份/空间预检位于data/native-visual-rsft-20260910-v4/。新audit入口重放
原始W像素、token、回复和成功子集，对应E4证据；不调用模型、不证明参数更新本身。

16:29更新：224041已于16:22:30 COMPLETED/exit0，2H800/539秒/0.299444GPUh。
原生报告6次优化器更新、94个assistant loss token，平均SFT loss 2.45533972；
梯度范数均有限，检查点与原生extra状态已保存。参数变化、实际恢复和训练后准确率
尚未核验。不能把47/64训练采样接受率标成独立测试分数。

独立批次核验v1在输入重放完成后因TensorDict迭代语义错误失败，日志与源码保留；
v2显式遍历keys，完整核对64次实际请求的W像素/token/回复和原生47条成功子集通过。
未改变训练记录或选择规则。结果为results/native-visual-rsft-batch-audit-224041-v2.json，
对应E4来源证据，不等于参数变化证明。实际终态与费用位于v4目录allocation-result.json。

协议台账目前已结4.717778GPUh、待结0GPUh，不含协议前成本；正式扩展关闭。
无额外删除、无GitHub推送，T仍封存。旧冻结实现和失败来源保持原样。

## E86 — 工具提示校准提前停止；不作为正式初始harness（2026-09-10）

手工定义evidence_first_harness，只改literal SYSTEM_PROMPT，要求先看全图、必要时
最多裁图一次后作答。原生AST验证确认五函数/解析器不变。工具仍可用，native轮数
和token预算不变，不是VETO贡献。计划在全部V256对上只按普通准确率/完成性检查。

223996于15:45:55启动。提示仍未解决反复工具调用；为优先推进直接作答的真实E4，
在15:59:04主动停止。保存完整batch的144张图中65正确、72题最后仍为工具调用；
这些是提前停止时的部分记录，不是整个V的分数。此为自适应工程决策，未伪称预登记
科学futility rule，也未按配对退化挑选弱h0。正式共同harness未选定。

CANCELLED/1H800/789秒/0.219167GPUh，费用已结算，所有结果保留。计划
results/visual-initial-harness-calibration-plan-20260910-v1.json，实际数据在同名data
目录；停止、终态和成本在data/visual-initial-harness-calibration-allocation-20260910-v1/。
停止后执行E87，不继续堆叠提示尝试；51组历史矩阵仍关闭。

## E85 — 64对真实开发图表完成；图像必要性信号与工具未作答问题（2026-09-10 15:39 HKT）

223990于15:21:11–15:37:38完成，COMPLETED/exit0，1H800/987秒/0.274167GPUh。
独立重算核对全部384条结果、配对身份、原奖励、生成费用与批次归档SHA；结果一致。

| 预登记诊断条件 | 正确图像 | 两边都正确 | 调用数 | 输出token |
|---|---:|---:|---:|---:|
| direct | 117/128 | 53/64 | 128 | 256 |
| blind | 64/128 | 0/64 | 128 | 256 |
| normalized_zoom | 6/128 | 0/64 | 384 | 62600 |

blind实际请求没有图像，全部128次均输出A，64对两侧答案相同。normalized_zoom的
256次裁图均返回有效像素；102题最后为工具调用，另20题未提交共同规则接受的明确
标签，6题标签有效且正确。归一化坐标修复未解决工具循环未完成的问题。当前低分
不能归因为训练损伤或VETO待解决的现象，也不能把此弱策略用作刻意降低的正式基线。
后续共同h0需先按普通开发准确率校准作答完成性，再进入受控权重/搜索比较。

summary入口ours/summarize_visual_development_screen.py对应E1的逐样本重算和匹配
条件差异，完成后才生成汇总；以源图表为单位、两侧和三条件共同重采样20000次，
seed20260910。区间是开发子集的描述，不代表跨训练种子的稳定性；边界处退化区间
不能证明总体确定性。分析源码身份保留，不新增Method机制或改变E2/E3。

证据：results/visual-development-screen-rescore-20260910-v1.json，原结果和终态在
data/visual-development-screen-20260910-v1/。这轮新增目录文件逻辑大小5752385301
字节（5.36GiB），分配块399131648字节（0.372GiB）；逻辑大小超过原4GiB工作量估计，
如实保留该估算偏差。15:38文件系统仍有234852712448字节（218.72GiB）可用，
没有触及40GiB余量，不依赖额外清理。0新checkpoint/优化器步数/API调用。

当前协议累计4.090278GPUh，0待结算预留，正式扩展关闭；不含协议之前的费用。
T未评测、Chess续训暂缓、51组仍为历史设计。上游未改、无额外删除或GitHub推送。

## E84 — 真实视觉初筛、零分诊断与64对开发验证（2026-09-10）

223978完成16张工程图、48次真实4B生成、5571输出token，严格单图0/16、配对0/8。
32次裁图均超出输入像素边界，25次未返回图片，7次返回图片；16题最后仍为工具调用。
归因于模型发出的0–1000归一化坐标与像素工具约定不一致，不能归因为模型不会读图。
完整失败分解保存在results/native-visual-first-screen-diagnosis-20260910-v1.json。
作业158秒/1H800/0.043889GPUh，原始结果和费用均保留。

新增NormalizedEvidenceZoomTool将坐标乘以初始图像宽/高后交给原生裁图，保持原反馈、
最小裁图大小和任务奖励。转换依据Qwen官方image_zoom_in_qwen3vl.py。2项原生CPU检查
验证非方形图片的实际crop像素及非法坐标拒绝，0.033秒通过。
对应共享视觉工具输入块，不属于VETO E2–E3接受创新。

223979在同16张图比较direct和normalized_zoom，模型/题目/greedy/三轮/1024输出token
上限一致。direct单图16/16、配对8/8，16次生成；normalized_zoom的32次裁图全部返回
有效像素，48次生成，但解释加明确末行A/B被原整段严格评分记为0/16、0/8。
事后解析保存回复得到两条件均16/16、8/8，不产生新调用。原始strict分数不覆盖。
完整结果data/visual-coordinate-diagnostic-20260910-v1/result.json，解析重放
results/visual-coordinate-answer-format-replay-20260910-v1.json。作业298秒/1H800/
0.082778GPUh，已结算。不得把解析改变或简单题满分作为VETO收益。

canonical_answer_harness只接受唯一且位于末行的独立A/B标签，拒绝未完成工具调用和
多行歧义标签，不读取真值；其余五函数与原提示保持一致。BlindVisualHarnessDataset
只在模型消息中移除图像，题目和评分标签不变。两项原生CPU检查48.380秒通过，日志
results/visual-development-interface-tests-20260910-v3.log；实际vision处理确认盲测
无图像token/像素，配对两侧文字输入相同。此前sandbox本地socket拒绝和误把raw_prompt
当input_ids的检查失败分别保留v1/v2日志，已按原生tokenization接口修正。

下一轮在运行前固定PlotQA派生V中前64对（128图），不按答案或模型表现选题，比较
direct、blind、normalized_zoom。三者共用新末行解析、共同初始4B和greedy配置；
工具条件不同，属于图像必要性/工具诊断，不是正式VETO对比或h0选择。计划
results/visual-development-screen-plan-20260910-v1.json，提交223990，15:21:11启动，
1H800/12CPU/96GiB RAM/1小时上限，0训练/新检查点/API调用。提交时剩219.10GiB，
预估4GiB输出并保留40GiB空间；尚无实测完整占用或开发分数。费用台账已结3.816111
GPUh、新预留1GPUh；正式扩展关闭。新入口对应E1观测与共享输入控制，未执行E2–E4。

当前范围是小规模视觉验证，51组仍为历史设计，不是待执行队列；Chess续训继续暂缓。
没有额外删除文件或推送GitHub，T仍封存，全部条件结果和失败记录保留。

## E83 — 用户要求直接视觉验证；当前范围缩至一次视觉初筛（2026-09-10）

用户再次强调无额外空间，并质疑为何等待Chess及安排51组。将当前执行范围改为
一次8对/16张图的真实4B视觉初筛；原51组（30主比较+6规模扩展+15替代对照）JSON
保留为历史设计，不再作为当前待执行清单。正式阶段仍被预算层阻止，后续规模须依据
视觉结果与实测存储重新确定，不能静默删种子或把未运行条件标记为已完成。

核对旧顺序控制器1945704身份与仅等待搜索的事件后发送SIGTERM，随后确认其已退出。
未取消当前最后候选223966，没有修改冻结Chess源码、候选、训练配置或checkpoint。
原Chess第二阶段续训与导出暂缓；搜索控制器仍可完成并整理当前搜索。
停止回执：results/veto-chess-milestone-priority-interruption-20260910-v1.json。

新增visual_priority_controller只调用原Milestones的检查/提交/计费方法，跳过其run中的
Chess续训顺序。两项新测试检查仅视觉路径和预检失败不提交，连同三项原预算/队列
失败检查合计5项通过。新控制器PID2224068于14:43启动，14:44源/输入预检PASS；
只等待账户唯一作业位置空出。1H800/40分钟上限，无新增API/优化器步数/模型权重，
工作空间预估2GiB、保留40GiB余量，沿用原pilot预算账本和状态通知，不自动重试。
本次不依赖额外磁盘或任何新清理，MIMIC保留。

当前修订：ours/visual_priority_plan_20260910.md及
results/veto-visual-priority-amendment-20260910-v1.json；实际事件保存在
data/veto-visual-priority-20260910-v1/。Method对应E1真实配对测量，尚未执行E2–E4
或验证VETO收益；16张工程图片的结果不能充当正式PlotQA或独立泛化成绩。

14:48首次sbatch被明确拒绝：Requested node configuration is not available，未创建作业。
原0.666667GPUh预留保持未绑定；旧控制器停止，不自动重试。查询Slurm分区后，
--test-only配12CPU通过（不创建作业/不占GPU）。保留原源码、模型和解码，仅显式
覆盖--cpus-per-task=12，于14:51提交223978，复用并绑定此前未使用的同一预留。
14:52实查RUNNING，1H800/12CPU/96GiB RAM/40分钟。冻结Ray配置的16个CPU调度
槽位是逻辑数量，实际Slurm分配为12CPU；agent workers仍2个，模型副本仍1个。
该资源差异和原拒绝都写入data/veto-visual-priority-cpu12-20260910-v1/amendment.json。
新增observe_visual_priority_job仅等待这一个作业并结算包括失败在内的终态费用，
不会提交下一任务；PID2252267。真实答题结果尚未生成，不能把RUNNING写成视觉验证通过。

## E82 — 用户授权A组并拒绝B项；限定清理完成（2026-09-10 14:26 HKT）

用户明确授权A1–A4，B项MIMIC原图不授权。执行前重新检查19个模型权重blob的
精确路径、inode、大小、symlink引用，验证配置/分词器独立副本；两份20,700,225,459
字节的Chess权重完整重算SHA256一致。step2采用先创建硬链接再原子替换的方式，
两个公开路径始终保留；现在共享inode2365560，未来原位改写前必须复制为独立文件。
独立data.pt及恢复状态保留，替换后再次完整核验唯一保留inode的SHA256。

仅删除提案的19个blob及19个symlink、pip http-v2中1480个缓存文件及空目录。
逐文件分配空间释放107000403968字节（99.651892GiB）；文件系统可用空间由
128263913472字节（119.455078GiB）增至235301502976字节（219.141602GiB）。
文件系统增量含目录块及并发写入影响。MIMIC目录身份未变，无B项删除操作；
40项保留元数据/恢复文件哈希、33项活动冻结源文件哈希通过，上游工作区仍干净。

完整授权、预检、逐路径操作与结果在results/veto-authorized-a-cleanup-20260910-v1/。
原提案approved=false属于执行前证据，保留原字节；最新授权和结果由新回执表示。
这是复现资产维护，保持E4权重字节与恢复输入，不引入Method模块或模型性能结果。
本次没有GPU/API分配，也没有GitHub推送。

清理前空间已满足近期顺序续训70GiB、导出50GiB、视觉服务42GiB的提交门槛。
完整正式矩阵约520.72GiB仍是整体保留估算；A组完成后需重新设计并实测空间使用，
不能将B项自动改列候选，也不能未经授权删除后续检查点。近期工程验收可继续。

## E81 — PlotQA完整像素核验与原生输入完成；清理范围待用户决定（2026-09-10）

data/plotqa-evidence-pairs-20260910-v1/rendered/result.json已完成：4544个源表、全部
6912张图通过保存RGB柱高重建与Decimal二元答案核对。六个角色manifest及完整像素
审计已固定SHA，观察仅限一张W训练图，其题目/图例/年份可读；未人工查看测试图或运行
测试模型。像素几何检查不证明所有文本可读或图像对模型必需。

visual_dataset_materialization.py复核分组、已完成manifest及像素审计，再将W/H/C/V
转换为原生bytes-only Parquet，分别2048/128/128/512张图。消息中仅图像与问题，真值
只进入reward_model；C仍为优化数据。T/R图像未被本次转换读取。实际转换成功，0模型
调用、0优化器步数。完整结果在data/plotqa-evidence-native-20260910-v1/result.json。

results/veto-execution-implementation-check-20260910-v1.json保留245个源码/配置/文档
的快照、27份已有证据哈希及三个活动冻结入口的源文件复核；源快照为独立tar.gz。
快照中的进度文字是生成时点，后续状态更新以PROJECT_STATUS为准。上游pin保持
fbe125eb7abea7f760c99ab9acc1a6261e708fc6且无改动；本地没有commit或push。

扩大存储调查后，A组约99.69GiB包括14个32B模型权重blob、5个7B权重blob、pip下载
缓存及18.309GiB完全相同checkpoint的硬链接去重。模型config/tokenizer/index/revision
已另存，具体路径和保留字节证明见缓存/去重提案。B项MIMIC原图files目录精确为
631661594624 allocated bytes/588.2807GiB，123个旧VCD/LOCE代码文件提到该数据；
父目录元数据和SHA清单保留方案已记录。已询问用户授权A、A+B或暂不清理；尚无回复，
没有删除、移动或替换既有文件。A组仍不足完整矩阵，B项是原始数据而非软件缓存。

## E80 — 批准VETO执行计划、预算控制与连续交接入口（2026-09-10）

用户明确批准4–6周计划。ours/veto_execution_protocol_20260910.md记录100/280/160/60
GPUh分阶段上限、总600GPUh、45元API上限、视觉种子42/43/44、科学继续/停止门槛和
存储限制。原冻结Chess协议保留；收束seed42联合条件后暂缓其余种子/FST，不打开test64。
research_schedule生成51个PLANNED_NOT_EXECUTED条件，结果字段空，无预设性能。

research_budget采用锁保护append-only预留/绑定/终态结算；先预留完整Slurm时限，失败
实际用量也结算，提交不明保留预留。非pilot阶段明确拒绝执行。3项预算测试及2项提案
恢复请求/流结束测试通过。顺序控制器veto_milestone_controller另有3项测试，覆盖
等待恢复进程退出、失败作业收费和不明提交不重试。PID1945704等待当前搜索，之后依次
核验完整搜索→原生第二批→逐轨迹/恢复/优化器审计→规范导出→真实视觉工程服务。
控制器尚未完成这些GPU阶段；它不启用正式矩阵。执行计划和日志保存于results/veto-milestone-*。

原搜索第四轮9个API请求中的6个为Internal Network Failure，300秒超时且没有候选。
joint_transport_recovery保留原失败、原计划和h0–h9结果，重新提出同一第四轮请求。
恢复预检首次把SSE [DONE]当JSON失败；修正仅跳过结束标记后完整核验通过，失败控制台
记录仍属已发生尝试。恢复10个请求/117秒完成；第五轮最多12请求，整阶段60请求上限
不变，失败请求的未确定费用预留不释放。PID1863343运行；h10的223942已审计8/32，
1125秒/2H800/0.625GPUh；h11的223944为6/32，1103秒/0.612778GPUh。h12=223951
正在运行。本修订新增分配才进入新100GPUh账本，先前费用继续保留在原结果中。

## E79 — 真实视觉服务入口及内核约束的候选回调（2026-09-10）

native_visual_service计划固定共同4B、原生结构化配置、16行工程图像、processor及greedy
解码。native_visual_agent_observation逐请求保留模型实际接收的PIL像素SHA、prompt
tokens、sampling params和返回tokens；模型Worker在实际load后记录8坐标与模型/seed。
这些是E1真实评测通路和E4共同输入的工程观测，不代表整份GPU内存权重证明。

isolated_visual_harness/visual_callback_worker/callback_confinement将五个候选函数放入
全新受限进程：Landlock ABI1只读白名单、seccomp禁止网络/写入/元数据更改/进程创建、
CPU/地址空间/超时上限。传入内容仅限当前函数可见输入，不能获得模型进程的完整环境。
内核独立测试拒绝read/write/truncate/chmod/socket/fork；全部五函数与原h0一致，
源修改及无限循环拒绝。第一次受限加载因trusted validator目录访问失败，改为限制前
导入可信validator后3项检查通过，原失败日志保留。v1实际服务CPU测试冷导入遗漏
prepare_worker导致verl.tools缺失；v2构造器检查通过；集成隔离后的v3四项原生检查
61.177秒通过。实际模型和Ray服务仍为测试替代物；0新视觉GPU调用、0参数更新。

native-visual-service-plan-20260910-v1.json固定1H800/40分钟/最大48次生成、16384
assistant tokens；工程图片和可信h0范围不变。Slurm入口保持离线，状态通知使用既有
用户邮箱要求。GPU验收由E80顺序控制器提交，尚未完成。新隔离仅修改ours/的可选
适配入口，上游submodule未改动，冻结Chess实现不变。

## E78 — 官方源下载、非有限数值排除与PlotQA派生配对（2026-09-10）

从PlotQA官方链接下载训练annotations.json，实际1157345635字节，SHA
be5cf2f9cac2a34a571db9b58f8189c079c852558fee29838aba2ecae74ab617。
首次1GiB下载上限触发拒绝；核实官方HEAD后按1280MiB上限在新v2目录下载成功。
CLEVR官方no-images ZIP为89363837字节，SHA
d4435dceab30022b64f7d34b3ec217162bfc1c2950c79999784750ced22674c4。
保留来源、许可证、字节身份和失败；没有下载/评测Chess test。CLEVR原图尚未获取。

初次严格ijson流扫描遇到发布文件中的NaN中止。新增逐对象有界解析并显式排除非有限
记录，原字节不替换、不猜数。完整157070条含15条非有限记录。v1错误假设竖向柱图
x为数字坐标而排除全部柱图，结果明确保留0 eligible；检查真实schema后将字符串类别
和数字tick坐标逐一对齐，v2获得41027个源表。解析截断/尾逗号/尾内容均使全扫描失败，
不当成普通科学样本排除。4项相关检查通过。

PlotQA-EvidencePairs为明确命名的派生二元数值比较任务。源表进一步按数值与类别/系列
名称规范分组，忽略标题及显示顺序，固定split seed20260910。W2048/H128只使用原始
数值；C64/V256/T1024/R1024交换两个单元产生答案相反的一对重渲染图，不声称最小
单因素干预。R独立于V且先用前512对。每张图从保存的RGB像素重建柱高，再与独立
Decimal答案计算核对；3项编辑/分组/像素测试通过，完整6912张图仍在渲染核验。
没有模型调用，也没有图像必要性、近重复筛查或科学现象通过结论。

基础发现测试最终171项，71通过/100依赖跳过，10.936秒。此前v1因隔离测试缺少原生
validator的Python路径而1项错误，修正测试导入环境；生产隔离Worker自带可信加载路径，
源码未据此改动。失败与v2/v3日志均保留。当前资料不支持VETO的性能主张。

## E77 — 只读存储调查与用户清理候选请求（2026-09-10）

用户表示没有另一大容量目录，要求调查可清理对象；这不是对具体删除的授权。
ours/storage_review_20260910.md和results/veto-storage-inventory-20260910-v1.json
先列出冻结模型引用与保留关系。最终模型、机制分叉与运行空间按保留口径约521GiB，
当前约120GiB可用，正式扩展仍关闭。全目录元数据盘点发现MIMIC-CXR-JPG约588GiB，
原始医学数据与当前VETO无直接关系，是否保留须另行决定。旧32B/7B模型缓存与pip下载
缓存另列候选；不清理当前anaconda3、原生运行环境或Chess活动数值参考。

weight-only两个model_world_size_1_rank_0.pt重新读取完整文件，各20700225459字节，
均为SHA 1e3d65762a6e078375b7a8da5198cd47cb1cb6355995b2adeeeac4fa1ed7d0dd，
当前是不同inode。若获用户批准，可硬链接保留两路径和全部字节，回收约18.309GiB；
独立data.pt与恢复元信息保留。核验记录在veto-checkpoint-deduplication-review-20260910-v1.json。
截至该记录，没有删除、移动或替换既有文件；GitHub未push。

## E76 — 共享原生视觉配对评测入口与成本计数（2026-09-10）

ours/visual_native_evaluation.py提供成对manifest→bytes-only Parquet及
已初始化AgentLoopManager→共享VisualHarnessDataset/原生视觉回路→E1 AuditReceipt。
Method为P_C=mean[v(answer,y)*v(answer_prime,y_prime)]，与E4训练共用提交答案/二元
verifier和掩码。完整原始像素、问题、标签和覆盖在生成前核对；原生过滤少一侧或批次
不能均分给worker时拒绝。按样本身份合并返回，重新检查parser、reward、最终token
位置和掩码；原生批次归档，失败留下failure与已完成批次，不补零生成整集准确率。
角色沿用manifest，engineering记录不能输入现有VETO接受器。

visual_harness_loop新增实际生成次数与返回assistant token计数。原生num_turns含用户/
工具反馈，不能当作模型调用次数；原生turns、返回token和最终保留loss token分别报告，
E3排序未变。两个harness原生测试验证计数随批次合并保留。原五函数行为和原生训练目标
保持；冻结Chess/联合搜索输入没有变更。

CPU测试执行原生Manager分片、Worker.generate_sequences、Hydra视觉agent、crop、
processor和DataProto存取。一对工程图片的完整执行有3次脚本请求：一侧直接回答、
另一侧crop后回答；实际调用次数[1,2]与原生turns[2,4]分开。强制倒序返回仍正确配对。
额外回放篡改reward/重复样本身份均拒绝；缺失配对、不能均分worker及第二批中断也
验证。中断情形只再执行1次脚本请求，总4次；没有完整结果文件或伪造准确率。

v1因冷启动先导入native agent而未恢复verl.tools失败；使用既有prepare_worker后，
v2三项原生检查47.126秒通过。新增归档输出故障测试后的最终评测v3一项23.076秒通过；
基础125通过/99依赖跳过，1.20秒（随后仅修改测试，新增情形由v3执行）。Manager/Worker
构造器被绕过，Ray RPC和模型server是夹具，权重/解码身份在本测试中也是明确的合成值。
0实际模型/API调用、0优化器步数、新GPU作业为0；归档存取发生于临时测试目录，保留
证据是日志和来源哈希。不能称为真实视觉准确率、GPU服务启动或VETO收益。

results/visual-evaluation-integration-check-20260910.json含原始失败/通过日志、源码及
16份冻结记录的重哈希；155个实际代码/配置文件一致，大型权重未重哈希，test未加载。
实际视觉GPU启动/冻结计划、worker权重/解码核验、proposer及VETO接受仍待接通。

## E75 — 联合第二轮h5完整评测（2026-09-10）

223908于10:26:54–10:45:12 COMPLETED/exit0，1098秒/2H800/0.610000GPUh。
固定MH32上6/32 solved，32调用/256806输出token，29次长度截断/无thinking结束。
完整原生/token/棋盘、worker与Slurm证据通过，见
results/controlled-mh-223908-completion.json。h4与h5同为6/32，第二轮仍在评测
h6/223909，尚未选择。这里只报告MH优化集测量，不是留出结果或组件因果收益。

本快照项目已知终态25.849444GPUh，正在运行的h6未计入终态；API保守0.613044CNY、
82已结/0未结，45元内可预留44.386956元，非官方账单或余额。账户GPU总3700h，
已用787.416667h，剩2912.583333h，21.28%；主目录剩120.26GiB。核验没有新增模型/
API调用，没有读取test数据，也没有自动重启控制器或重复提交已有评测。

## E74 — 可执行视觉harness与原生数据/工具派发（2026-09-10）

新增ours/visual_harness.py、visual_harness_dataset.py、visual_harness_loop.py、
独立agent YAML及visual_harnesses/base_harness.py，通过继承接入原生dataset工厂和
ToolAgentLoop。Method是共享E3候选空间h=(观察,工具参数,反馈,parser,追加请求)，
E4仍按提交答案的原二元reward筛选成功轨迹；VETO接受规则未由这些文件接通。
五个函数只收到可见文本、请求参数或初始可见图像尺寸，不收到GT或reward metadata。
候选SHA参与缓存/每行身份，rollout在生成前拒绝身份不符，源文件变化也会拒绝。
prompt-only要求两条字面提示以外AST相同；实际FST编排还须显式调用该验证。

两项原生CPU测试最终25.575秒通过：实际关键词dataset工厂、Hydra实例化、默认h0
请求/token/像素/mRoPE一致、160→128的原生crop、反馈、parser、预算内追加请求、
零loss观察mask及有/无工具DataProto合并。测试共8次脚本请求；模型/优化器/API均
未执行。有效crop参数、返回像素哈希及追加次数进入DataProto。强视觉h0和正式视觉
协议尚未建立，实际视觉proposer/评测编排、actor更新与VETO接受仍未验证。

初次测试因原生AgentLoopMetrics不能按dict读取而失败；其封闭schema也不保留自定义
计数，已改为extra_fields并检查实际DataProto内容。v1失败、v2–v4阶段通过日志均
保留。复查补上可变位置/关键词默认参数的拒绝，避免此类跨轨迹状态；v5最终通过。
AST验证不等于OS隔离，也不保证任意Python程序无副作用。基础125通过/98依赖跳过，
1.12秒；不是98项验证通过，也不是视觉准确率或因果消融。

最终results/visual-harness-integration-check-20260910-v2.json记录全部来源、测试日志
哈希和16份冻结执行记录。旧“153个唯一输入”实际为source_sha256里的153个路径
字符串，解析后对应149个文件；本次同时核验joint_source_sha256，共155个实际
代码/配置文件且内容均相同。此为计数口径澄清，旧报告保留；大型权重未重新哈希。
未读取test题目/答案，未提交新GPU作业，未修改原生WHALE源文件。

## E73 — 联合第二轮h4完整评测（2026-09-10）

223905于10:07:54–10:26:39 COMPLETED/exit0，1125秒/2H800/0.625000GPUh。
固定MH32上6/32 solved，32次调用、258316输出token，30次长度截断/无thinking结束。
完整原生/token/棋盘、worker和Slurm证据重哈希通过；Method为原WHALE的
MH(theta1,h4)。第二轮仍在评测h5/223908，未选择候选，也没有留出结果。

证据见results/controlled-mh-223905-completion.json。已知项目终态累计25.239444GPUh；
该快照API保守记账0.613044CNY、82已结/0未结，45元内可预留44.386956元，不是官方
余额或账单。账户GPU总3700h，用786.950000h，余2913.050000h，21.27%；与项目成本
不同。主目录余120.27GiB。核验没有新增模型/API调用，运行中的h5成本尚未计入终态。

## E72 — 联合h2固定回复解析诊断与请求匹配修正（2026-09-10）

复用ours/recorded_parser_comparison.py，在同一h2记录回复上执行h0与h2 parser。
提示词和声明预算相同，parser/辅助合法性代码不同；31/32次动作不同。5个成功轨迹
中4个与h0 parser动作不同，3个成功仍含长度截断。辅助is_legal_action结果不同
不等于独立棋盘合法性差异。此检查没有新模型生成，不是重生成的组件消融。

附加跨运行请求比较最初错误地按generation账本行序配对任务；并发完成顺序不代表
任务顺序，所得17个相同请求的数字无效。已依据每题原生harness_trace的唯一raw回复
匹配实际generation记录，逐项断言唯一匹配：32/32请求体相同，32/32输入token相同，
仅3/32输出token序列相同。因此不能把总分1→5全部归因于parser，也不能仅从请求seed
推断后端严格确定性。原生得分和固定回复parser统计不受此附加配对错误影响。

旧results/controlled-joint-h2-parser-comparison-20260910.json保留；权威修订为
results/controlled-joint-h2-parser-comparison-20260910-v2.json，含匹配方法、源哈希及
supersedes说明。Method对应E3组件诊断，不提供预设贡献比例或留出收益证据。

## E71 — 联合第一轮h2/h3完整评测与原生选择（2026-09-10）

h2/223896完成1099秒/2H800/0.610556GPUh：5/32 solved，32调用/253487输出token，
30次长度截断。h3/223898于09:46:40–10:05:29完成1129秒/2H800/0.627222GPUh：
3/32 solved，35调用/258373输出token，31次长度截断。两者COMPLETED/exit0，完整
原生/token/棋盘及worker证据通过。第一轮h0–h3为[1,1,5,3]/32，原生选择器保留h2。
四次评测共131调用/1032116输出token/2.513889GPUh，重复使用相同MH32，不能当作
128道独立题。完整五轮搜索、第二批GPU恢复和留出评价仍未完成。

results/controlled-mh-223896-completion.json与223898-completion.json保留终态；后者
含四次评测的再次核验及原comparison.json选择快照。本阶段是MH(theta1,h0)的原生
首轮，不是VETO收益。至223898项目已知终态24.614444GPUh；API保守0.613044CNY。

## E70 — 共享原生视觉工具回路与最终答案验证（2026-09-10）

ours/visual_evidence_tool_loop.py继承固定上游ToolAgentLoop及Math的ImageZoomInTool。
已有training_bootstrap已恢复同仓库工具包，本次实际导入通过，没有新增依赖或修改
原Chess运行环境。原生工具parser、状态机、bbox处理、crop、反馈和release保持；
工具直接接收模型实际看到的初始PIL像素，避免再次fetch/resize改变坐标，且metadata
不能提供额外图片/路径。原crop执行不使用其构造器创建的Ray池，本适配不创建该池。

Method对应共享观察I_next=native_crop(I_initial_visible,b)，E4为最后连续assistant
mask段的r=v(answer,y)。原生工具反馈和图像mask=0，生成token mask=1；模型只调用
工具而没有最终A/B时得0。原生无效bbox的-0.05工具反馈保留为metadata，不加入二元
任务奖励。不是VETO E1–E3，也不是新的loss；所有视觉条件必须共用。

独立agent/tool YAML通过实际Hydra实例化；最后1项CPU原生测试含6情形：crop正确/
错误、无效框、无图、无最终答案、直接回答。10次脚本请求，裁剪像素与当前初始图像
的实际crop相同；正常工具交换含44个assistant loss token、107个零mask观察token。
原生DataProto的像素/grid、图像token、mRoPE和奖励位置通过；无图请求不能借metadata
恢复图片，工具实例状态释放。最终20.384秒通过；无图情形现在提供真实可用的隐藏PIL图片，不能只用无法加载的
路径排除泄漏。之前4情形19.483秒、6情形17.972秒、YAML19.806秒通过的阶段日志保留；
较弱的路径探针不是充分的隐藏图像证明。基础套件125通过/96依赖跳过，1.24秒。

模型server是脚本夹具；0实际模型/API调用、0优化器步数/新GPU作业。没有真实actor
前反向、分布式Ray服务、完整trainer或视觉准确率证据。完整可执行视觉候选派发、
评测/搜索入口及VETO接受位置仍待接通。16冻结记录/153唯一代码配置输入保持一致，
未重新哈希大型权重；test题目/答案未加载。最终来源/日志哈希与边界见
results/visual-tool-integration-check-20260910-v3.json；之前报告保留为中间快照。

## E69 — 联合第一轮h1完整评测（2026-09-10）

223895完成COMPLETED/exit0，1134秒/2H800/0.630000GPUh；32题原生/token/棋盘、实际
worker及Slurm证据通过，1/32 solved，32调用/260128输出token。全部达到8129-token
上限且缺少thinking结束。该h1没有提高当前MH总分；它与独立harness-only选中的h1
是不同代码，名字仅在各自搜索内有意义，不能混用。第一轮尚未完成选择，控制器1413742
已继续h2作业223896。Method仍为原WHALE的MH(theta1;H)，不构成VETO或test收益。

终态与重哈希核验见results/controlled-mh-223895-completion.json。项目已知终态累计
23.376667GPUh；09:29账户GPU总3700h，用785.116667h，余2914.883333h（21.22%）；
主目录余120.29GiB。API保守记账0.528809CNY/70已结/0未结，预算内可预留44.471191元；
不是官方账单或余额。本次证据核验没有新增模型/API调用，test未加载。

## E68 — 共享视觉奖励进入原生成功筛选的CPU检查（2026-09-10）

新增ours/visual_evidence_reward.py复用已有严格A/B verifier。Method对应E4的
r_i=v(answer_i,y_i)及原WHALE B+={tau_i:r_i>0.5}，所有视觉条件共用；不新增损失，
不计为VETO E1–E3。数据域/标签错误明确失败，普通错误或不合规回复返回0，metadata
自报reward不能覆盖标签。正确答案本身不证明使用图像。

原生dataset动态工厂、processor、SingleTurn、reward loader/manager/worker、score
collation和整个原生RSFT筛选循环在CPU执行。现有一对工程图片上，4条脚本回复得分
[1,0,1,0]，只选0/2行且图像张量、grid和回复掩码保持；另4条不合规回复全部为0并
跳过更新。奖励位于最后一个有效回复token，模型请求不含标签或reward metadata。
模型回复、Ray RPC传输及actor/checkpoint操作是夹具：0模型/API调用、0优化器步数，
没有实际视觉前反向、分布式reward服务或完整视觉harness/search。

首个原生probe因裸模块字符串被视为文件路径失败，保留v1日志；改为原生要求的
pkg://ours.visual_evidence_reward，并实际测试dataset的pkg://导入。v2两项通过，
21.047秒；基础套件125通过/95依赖跳过，1.20秒。原dataset文档的同类路径已纠正。
16份既有冻结执行记录及153个唯一代码/配置输入仍一致；此次未重哈希大型权重。
证据见results/visual-reward-integration-check-20260910.json及三个原始测试日志。
上游未修改，test题目/答案未加载，研究阶段门保持未通过。

## E67 — 首个联合h0完整评测与实际worker交接（2026-09-10）

223824于08:48:06–09:07:29 COMPLETED/exit0，1163秒/2H800/0.646111GPUh。
独立223622→223757的theta1在两个实际Qwen3_5ForConditionalGeneration worker中通过
8坐标及seed42检查；其中5坐标区别于共同theta0、8个区别于历史数值参考。抽查不等于
所有服务参数的完整证明。完整32题原生/token/棋盘审计通过，1/32 solved，32次调用，
260128输出token；32次均达到8129-token上限且缺少thinking结束。原生legal_rate不是
独立合法率，不能拿它解释成功率。

共同theta0/h0同为1/32，但实际2题成功标签改变、4题步骤或终止记录改变。总分相同
不能推导为相同预测或权重未改变，也未显示一次更新改善总正确率。这是MH优化集上的
单次评测，不是完整联合条件结果或test结果。Method对应MH(theta1,h0)权重交接与测量。
控制器1413742已按冻结计划完成第一轮3候选生成并提交h1作业223895；09:11实际RUNNING。

完整终态及逐项证据重哈希见results/controlled-mh-223824-completion.json，实际worker
运行快照单独保留。项目已知终态累计22.746667GPUh；09:10账户GPU总3700h，用784.55h，
余2915.45h（21.20%），主目录余120.30GiB。API保守记账0.528809CNY/70已结/0未结，
45元预算内44.471191元可预留；原生日志美元估价不能代替GLM项目CNY记账，后者也不是
官方账单或账户余额。核验没有新增模型/API调用，没有复用其他独立条件的选择结果。

## E66 — 首个独立联合MH计划及控制器启动（2026-09-10）

在真实whale/seed42训练223622和规范导出223757完整闭合后，联合搜索计划准备及原生
check_plan均通过。计划SHAba2f78c3a326470e83b3a035d3f1c49ee3a50de4c1b2241e4573f9010803dfad；
目标权重清单5b07bc6e3ee39a3e96279571f1474b66e43d016cea0c8fef0e9bce649dec14a0。
保持同一MH32、h0、seed42、5轮×3候选、每评测2H800/16CPU/128GiB/40分钟上限。
首次h0将重新运行，不能复用共同theta0或独立weight-only结果。原生选择/满分停止与
预声明的提案元数据兼容不变；VETO仍关闭。8个实际worker拟抽查坐标与额外CPU区分性
报告完全相同，其中5个区别于共同theta0。Method对应原WHALE的MH(theta1,h0)，不是新机制。

一次性导出收尾进程1404374于08:42:27退出，结果为EXPORT_CERTIFIED_JOINT_SEARCH_PLAN_PREPARED。
随后记录启动意图并只启动一次控制器1413742；其主函数正在再次执行启动核验，数据root及
首个GPU评测要在该核验后创建/提交。没有自动重启，不把进程已启动当作评测完成。
现有g4空闲；1GPU串行、4GPU同时候选布局未验证且不属于冻结方案，沿用两分片布局。

启动时API保守记账0.474527CNY/62已结/0未结，45元内44.525473元可预留，非官方余额或账单。
实际账户GPU总3700h、已用783.933333h、剩2916.066667h（21.187387%）。本次启动尚无新模型/API
调用；其后真实调用由原账本与逐评测审计记账。原始quota和启动时刻见search-start记录。

## E65 — 首个独立联合theta1规范导出完整闭合（2026-09-10）

223757于08:29:56–08:34:00 COMPLETED/exit0，244秒/1RTX4090/0.067778GPUh。
全部723活动张量/4,539,265,536元素的导出值等于原生FP32参数的精确BF16转换。
相对共同theta0，原生3,832,832,389元素/426张量变化，L2=0.005843913813354426；
导出9,013,513元素/283张量变化，L2=0.00033853721758805215；maxabs均1.1920928955078125e-7。
这些是数值交接证据，不是准确率提升。原生model/extra/data.pt保持原字节与phase1身份。

导出权重文件SHA12252ff9ecf89c4c32bf6100f1d4f293c263ffe558a31929644bcf737b315cf8，
权重清单摘要5b07bc6e3ee39a3e96279571f1474b66e43d016cea0c8fef0e9bce649dec14a0。
完整计划/原生状态/配置资产/Slurm终态核验通过，result保留whale/seed42/phase1及1次更新；
results/joint-checkpoint-export-223757/result.json与完整transition记录可复核。

另外实测原有8个worker抽查坐标：5个与共同初始权重不同，8个与历史数值参考不同。
无需修改冻结的worker实现，即可在随后实际加载检查中区分这两种错误来源；当前只有
CPU坐标测量，尚无实际联合推理worker。该检查不等于所有服务参数的逐元素证明。
Method对应theta1→下一轮MH的权重交接，不新增优化机制。

项目已知终态累计22.100556GPUh；08:36主目录余120.31GiB。联合搜索计划正在从这次独立
导出准备，尚未开始评测或调用GLM。没有生成新视觉/test结果，原研究阶段门不变。

## E64 — 独立WHALE第一阶段规范导出提交（2026-09-10）

223622完整审计、原生更新、sampler/scheduler/RNG与Slurm通过后，生成导出plan SHA
baa012c103e1b7438231e0769f4a74b28a290341ff5c2deb7b8ef8d487197e4f。
再次核对140来源、共同theta0和原生恢复状态通过。原生model SHA
6231ba08153d700200478244599231979345e872985e1a2af5925f98cb7e72c3。

08:29实时队列为空，GPU总3700h/已用783.90h/剩2916.10h，主目录剩129.96GiB。
Slurm test-only预估立即可启动（假想223756未实际提交）；随后仅提交真实223757，
1RTX4090/4CPU/64GiB/20分钟上限。实际运算是CPU序列化与全参数检查，增加GPU不加速
此路径；最小GPU分配时间仍全部计费。提交意图及返回job id分别保留，未自动重试。
Method对应theta_native→BF16(theta_native)的权重交接，不改变优化目标。

仅计划准备及预检通过，真实导出结果尚待终态。联合theta1上的搜索仍未启动，不复用
独立weight-only的checkpoint或harness-only选择。test未读取，0新增模型/API调用。

## E63 — 首个独立WHALE第一阶段完整审计与更新通过（2026-09-10）

223622于07:46:21–08:23:08 COMPLETED/exit0，2207秒/2H800/1.226111GPUh。
64条新轨迹全部通过原生回复重放、token/mask及独立棋盘检查；实际题序与冻结的真实
DataLoader预检一致。68次请求全部返回、517481输出token、0错误/未结/未知usage。
6条accepted：00BQD和007fJ各3条，45996个loss token；原生1次SFT更新，loss0.6284344159357715，
grad_norm5.947546005249023，actor98.538076秒，保存34.842459秒，末次权重同步0.883265秒。

唯一global_step_1及model/extra/data.pt核验通过；sampler yielded8，scheduler last_epoch1，
CPU/CUDA/NumPy/Python RNG完整，原生Adam不保存行为保留。没有进行GPU恢复或把状态存在
当作恢复成功。初始同步释放6GiB闲置CuPy；当前冻结106来源文件与新增绘图后仍逐字节一致。

Method仍是theta1=RSFT(theta0,h0;B1)，没有新增优化目标，也未发生联合MH选择；不能将
6/64相对独立weight-only第一批3/64的采样差异解释为联合方法收益或6道独立题成功。
规范导出计划在完整终态后准备，尚未有真实联合导出或联合MH评测。

原生完整audit/result/Slurm位于results/controlled-joint-phase1-223622/，旧运行快照保留。
项目已知终态累计22.032778GPUh。08:25账户实时GPU总3700h、用783.90h、余2916.10h（21.19%）；
CPU总44400h、用3927.40h、余40472.60h。账户计量与项目终态成本单独记录，不互相代替。
0新增审计模型/API调用；test题目/答案仍未加载。

## E62 — 完整原生搜索的真实轨迹图（2026-09-10）

plot_search_trajectory在原生环境重新运行完整搜索验证，与已保存seed42 completion证明
逐项相同后提取16个候选的题数、调用数、token和Slurm GPU成本。基础绘图环境生成PNG/PDF、
CSV和图像哈希收据，未修改冻结训练依赖。直接核对全部16个分数及6个真实选择时点通过，
图像已目视检查。Method对应对现有E3选择过程的观测，不引入优化机制或新损失。

h0/初始为1/32；第一轮三个候选全部结束后保留h1/8/32，后四轮仍保留h1。累计仅本次
候选评测9.165556GPUh；训练/导出/提案成本另外记录。全纵轴为0–32，展示全部候选，
不汇总为独立512题，不加单种子统计误差条，不宣称heldout、消融或因果收益。
保留原冻结五轮预算，不能依据单次平台期回改协议。0新增模型/API调用，test仍未加载。
产物：results/controlled-harness-only-seed42-search-trajectory-20260910-v1.*。

## E61 — 原生视觉输入保留与规范导出资产等价（2026-09-10）

原RLHFDataset在构建消息时pop images，导致后续长度过滤无法读取原payload；原harness
提示替换还会把带图像的user content替换成字符串。使用共同4B的真实HF processor及现有
一对工程图片确认：原hook启用后两行均无图像内容。诊断及缺少qwen_vl_utils的首个失败
日志保留。官方固定版本qwen-vl-utils0.0.14和av18.1.0单独安装到显式启用的visual overlay，
wheel SHA与PyPI固定版本记录相同；原冻结训练环境未改变。

ours/VisualEvidenceDataset继承原dataset，复制原行再调用原builder，恢复单图证据，并让
捕获的提示环境进入filter缓存身份；构建后改变提示环境即拒绝。无图控制使用同一类。
Method对应共享观察(I,q)→policy的输入箭头及长度约束，不属于VETO E1–E3，不改RSFT目标。

最终两项原生CPU测试35.515秒通过：每图120个图像token；176token完整提示在175阈值下
正确过滤；提示变更cache身份不同；两图像素不同但问题token相同；无图控制输入相同。
原SingleTurn请求保留image_data，未传入答案；原postprocess得到pixel_values[480,1536]、
position_ids[1,4,187]和2个回复mask token，DataProto保留multi_modal_inputs。回复及reward
明确为合成夹具；模型权重未加载，没有实际视觉生成/actor前反向/任务准确率结论。

共同4B与223620规范导出后的processor在两图×视觉/无图×thinking开关共8个案例中，
所有输入张量及原生mRoPE完全一致；这是资产语义检查，不能代替推理worker或视觉训练。
基础环境124通过/94依赖跳过。14份既有冻结执行记录、131个唯一来源输入及旧导出103个
指定扩展实现文件重哈希一致，上游checkout干净。完整绑定见
results/visual-native-interface-implementation-check-20260910.json。没有加载test任务或调用模型/API。

## E60 — 首个独立WHALE第一批正式启动（2026-09-10）

07:46:21提交并运行223622，冻结plan SHA e828995883605be964319dd7671d69ff3e16ef3cf2f0cf347b290256c3663868。
独立whale/seed42从共同theta0和原h0开始，2H800/16CPU/160GiB/60分钟，64条新轨迹、累计target1。
它不复用weight-only的训练前缀或harness-only选中结果。Method为原WHALE的
 theta1=RSFT(theta0,h0;B1)，完成后才进入自己theta1上的MH搜索和原生第二阶段恢复。

模型加载及初始原生同步已完成；真实释放6GiB闲置CuPy缓存，活跃池为0。08:01运行15:09，
请求日志增长，尚无完整batch审计或成功率。start/effective-config及实际提交/资源快照保留。
此前唯一名额协调器和harness搜索均已退出。2卡使用已验证的actor/rollout布局；1卡未验证，
4卡并行试验的Ray隔离未验证，不能以减少排队为由混合trial。

07:45实际账户配额GPU总3700h、已用782.75h、剩2917.25h（21.16%）。07:57主目录剩148.23GiB，
可完成下一阶段但不足以保留后续完整矩阵约430GiB。存储路径问题继续待用户回复；没有删除checkpoint。

## E59 — seed42完整搜索及独立weight-only规范导出结束（2026-09-10）

h15/223617于07:34:33 COMPLETED/exit0，1051秒/2H800/0.583889GPUh；完整32题4成功，33请求、
255413输出token，原生回复/token/棋盘/实际worker及seed检查通过。原五轮均保留h1，全部16个
评测完成；h0..h15分数依次[1,8,5,0,6,7,8,6,7,7,8,6,6,8,8,4]/32。完整搜索证明SHA
1594ab2b45501db89dd6512526c5e14265534324fa566952a1173c179c8c8745。
累计515次任务模型调用、3,991,282输出token、9.165556GPUh；512次题目评测重复同一MH32，
不是512独立任务或test数据。三次中断及显式恢复保留，没有重提案或重复已审计推理。

一次性协调器1330451验证完整搜索并重查独立weight-only导出计划后，仅提交一次223620；
两个试验仅共享提交名额，h1未进入weight-only训练。导出07:37:55–07:43:24 COMPLETED/exit0，
329秒/1RTX4090/0.091389GPUh。真正执行的是CPU序列化，必要GPU分配如实计账。

全部723活动张量/4,539,265,536元素满足export=BF16(native)。相对共同theta0，原生FP32
3,853,212,360元素/426张量变化，L2=0.00588364107729743；导出BF169,033,512元素/283张量
变化，L2=0.0003396180583945256。两者maxabs均1.1920928955078125e-7。空第二批对应两份
原生model字节相同；model/extra/data.pt保留。导出manifest SHA
7d17434003944c9699dcce2063c884e5b77e477b381ec2d6ed51bd29b95ab8d9。
结果保留weight_only/seed42/step2、真实1次更新及原预检题序偏差。参数变化不是准确率提升。

results/controlled-checkpoint-export-223620/result.json及完整transition记录已封存。
已知项目终态累计20.806667GPUh，运行中的223622另计。07:51保守项目API0.474527CNY、
62已结、0未结，45元预算内44.525473可预留；不是官方账单或账户余额。没有加载test题目/答案。

## E58 — 剩余六份独立计划与一次性作业衔接（2026-09-10）

harness-only seed43/44和WHALE/FST seed43/44第一批共六份计划全部CPU预检PASS，
后四项执行真实DataLoader/原生Hydra解析，均共同theta0/h0且各自独立。未启动相应GPU
训练或搜索。新independent_harness_search继承原循环，预声明既有候选别名/报告路径
兼容；当前seed42及其修订来源不改。2项原生测试/0.960秒覆盖seed43/44五轮和满分停止，
分数、提案服务及Slurm为合成fixture；基础124 PASS/92 SKIP/1.22秒。
results/remaining-trial-plans-implementation-check-20260910.json绑定六份计划、源码与日志，
确认原8个冻结执行记录不变。新调用0，test读取0，没有将预检计作训练结果。

seed42旧规范导出完整check PASS；一次性finish_seed42_search_then_export PID1330451
于07:24:16等待现有控制器，完整搜索证明和旧导出复核通过且名额空闲才单次sbatch。
它不重启搜索或修改条件依赖；出错停止，SUBMITTING状态不能被当成尚未提交而重试。
语法检查通过，真实后续导出仍待搜索结束。Method中这是资源排程；新种子搜索仍是
h_star[s]=MH(theta0,h0;M_H,s)，没有新增目标或模型。

只读存储预估07:13可用158.02GiB，剩余16个native checkpoint和15个规范导出按实际尺寸
预计435.60GiB，未计回复/缓存/临时文件、未扣除相同文件。当前下一阶段可执行；已询问
可用共享存储路径。证据results/controlled-pilot-storage-projection-20260910.json；无删除、
去重或其他项目改动。这不是实验阻塞终态或未来空间承诺。

## E57 — h14完整审计与最后候选启动（2026-09-10）

h14/223615于07:00:01–07:16:56 COMPLETED/exit0，1015秒/2H800/0.563889GPUh；
32题8成功、32次请求、236698输出token，原生完整审计/实际worker来源/终态归档
results/controlled-mh-223615/。已知终态项目累计20.131389GPUh。
最后候选h15/223617运行中，07:19:22快照中两个worker权重/seed PASS。h13/h14未超过h1；
完整五轮尚待h15。API保守项目仍0.474527CNY/62次已结/0未结，不是官方账单或余额。

## E56 — 最终留出封存、四分片执行与三种子汇总（2026-09-10）

heldout_cohort核验12个最终trial身份；controlled_heldout/heldout_evaluation/独立审计器
支持固定四个16题逻辑分片在1/2/4卡分波，每分片fresh worker与实际权重/seed证明。
终态核对全部回复/token/独立棋盘、启动/分波、Slurm状态及GPU/CPU/RAM分配。
heldout_comparison重验保存的finalizer结果，输出12行CSV、JSON和Markdown；只在三种子
齐全时报告均值/样本SD，保留配对差值及缺失原因。成本是所列终态记录的已知小计，
包含失败分配，未列来源保持null；不冒称完整训练/搜索成本或统计显著性。
Method对应A[c,s]=mean_i exact_solved(theta[c,s],h[c,s],x_test[i])及描述性配对差值。

最终6项原生CPU测试PASS/19.174秒，2项汇总测试PASS/0.024秒；基础124 PASS/90 SKIP/1.09秒。
四分片fixture用旧MH32题每题复制两次生成64个合成ID，native回复/token/棋盘重放执行；
子进程/GPU/Slurm及来源矩阵是fixture。父搜索/训练/导出finalizer在路由测试中使用stub，
其真实小模型/历史回复检查另有既有记录。没有正式cohort、heldout计划、test题目读取、
新模型/API调用或test成绩。推理无RSFT loss mask，不声称检查该项。
results/heldout-implementation-check-20260910.json绑定新源码/日志，并确认8个冻结执行
记录和旧seed42导出实现未变；正式多种子训练、联合闭环及实际留出执行仍未完成。

## E55 — 第四轮完成、h13审计与h14启动（2026-09-10）

h12/223507于06:22:30–06:39:35 COMPLETED/exit0，1025秒/2H800/0.569444GPUh；
32题6成功、32次请求、238780输出token，完整审计归档results/controlled-mh-223507/。
第四轮原生选择仍为h1。第五提案GLM11次请求0.117990CNY，项目保守累计0.474527CNY，
62次已结/未结0，45CNY内可预留44.525473；非官方账单/余额。
h13/223573于06:42:31–06:59:37 COMPLETED/exit0，1026秒/2H800/0.570000GPUh；
32题8成功、32次请求、246575输出token，完整审计归档results/controlled-mh-223573/。
已知终态项目累计19.567500GPUh。07:05:54快照中h14/223615运行、两个worker权重/seed
PASS，控制器仍PID1169570；第五轮尚未完成，h13未超过h1，未声称留出提升。

## E54 — h11完整审计及h12启动（2026-09-10）

h11/223466于06:04:58–06:22:02 COMPLETED/exit0，1024秒/2H800/0.568889GPUh；
完整32题6成功、32次请求、248721输出token。原生回复/token/独立棋盘、实际worker
权重/seed、完整档案哈希与Slurm终态通过，归档results/controlled-mh-223466/。
已知终态项目累计18.428056GPUh。h12/223507于06:22:30开始，两worker证明PASS；
06:25:54快照已保存。第四轮仍待h12，当前接受第三轮的h1，五轮搜索尚未结束。
保守API项目累计仍0.356537CNY/51次已结，未结0；审计未新增模型/API调用。

## E53 — 连续weight-only多种子终态及导出闭合（2026-09-10）

新增controlled_continuation_result、controlled_continuation_export、Slurm wrapper以及
controlled_checkpoint_export_result。前者核验seed43/44的完整两批原审计、真实sampler和
scheduler、启动配置及Slurm终态；导出通过进程内接点复用joint_checkpoint_export的原生
serializer/逐参数比较/终态入口，条件、种子、目录及kind独立，退出后恢复默认行为。
原joint导出尚未有正式计划，增加接点不改变其默认结果。seed42旧冻结导出器/计划不改，
新只读finalizer保留预检偏差并核验参数报告、原训练来源、执行回执和Slurm终态。
Method对应固定h0的theta1=RSFT(theta0,h0;B1)、theta2=RSFT(theta1,h0;B2)，原trainer
连续执行，另核验theta_eval=BF16(theta2)。空更新和BF16不变均保留，不作为失败或重采样理由。

2项continuation终态检查通过/11.925秒；4项原生导出检查通过/54.619秒；5项旧格式终态
及joint第一/第二阶段回归通过/27.808秒，共11项。基础最终119通过/87依赖跳过/1.08秒。
六次真实小VLM原生导出（34张量/42576元素）覆盖joint两阶段和continuation两种子身份，
零变化与仅FP32变化正确保留。小fixture无image processor，不验证视觉输入。
continuation终态fixture使用旧两批指标/state与合成seed/模型/Slurm元数据，原档案不改；
旧seed42闭合fixture的训练证明与serializer是stub，实际逐参数比较/回执/终态核验执行。
既有64条第二批原生重放/空批计数/phase2导出计划回归也通过。不能将这些fixture认作新实验。
results/continuation-finalization-implementation-check-20260910.json绑定源码、日志，核验五份
冻结训练/搜索计划、三次恢复修订的原source未变；旧seed42导出计划97个代码源同样未变。
没有新增真实训练/导出GPU作业或任务模型/API调用。最终12-trial来源矩阵与64题执行器仍待接入，
test任务/答案没有加载；完整多种子实跑、联合训练与留出结果仍待完成。

## E52 — 原生frontier及retry语义核验（2026-09-10）

只读调用原compute_pareto和retry_budgets，并使用已有h0–h9完整分数与真实h9代码。
原frontier用turns <= best_turns，保留同调用数但低分的候选；前三轮原标记为
h1/h6/h5/h8/h9/h4/h7/h2，严格非支配仅h1/h6。原_best独立按分数降序、调用升序
排序，实际接受h1一致。复现不修改原行为、档案、提案反馈或搜索；论文不得把原标记
冒称严格非支配证明。证据results/native-frontier-retry-semantics-20260910.json。
原retry环境是默认值，harness声明优先：h9在默认1/1下实际为1/2。留出协议中的
retry值指共同环境默认，选中harness原属性继续生效；详见ours/native_chess_semantics.md。
本检查0新增模型/API调用，未读取test任务；冻结的两个协议文件与记录不改。

## E51 — h10完整审计及h11启动（2026-09-10）

h10/223408于05:47:59–06:04:52 COMPLETED/exit0，1013秒/2H800/0.562778GPUh。
完整32题8成功、32次请求、249684输出token，原生回复/token/棋盘重放、实际worker
权重/seed、档案哈希与终态通过，归档results/controlled-mh-223408/。
已知终态项目分配累计17.859167GPUh。06:04:58自动进入h11/223466，两个实际worker
证明PASS；06:07快照已保存。第四轮仍待h11/h12，当前接受第三轮的h1；尚未完成五轮。
API保守累计仍0.356537CNY/51次已结，无未结请求；h10审计没有新增模型/API调用。
分数仍是同一MH32优化集结果，不是留出收益或VETO消融。

## E50 — 未查看test任务前固定留出评测（2026-09-10）

ours/controlled_heldout_protocol.md及results/controlled-heldout-protocol-20260910-v1.json
固定四条件×seed42/43/44、64题一次原生完整评测、mean/sample SD与逐种子原始成绩。
只读取既有split manifest的ID/哈希并校验test文件SHA，未加载题目、局面、答案或回复。
四个sorted-ID交错逻辑分片均16题；合并覆盖64个唯一ID。沿用8129 assistant tokens、
temperature1/top_p1/top_k20/thinking、单分片并发4、原turn/retry规则，无新增best-of。
所有12个trial的最终身份或失败/未完成状态必须先封存；只有完整来源的模型/harness对可评测。
空训练或BF16零变化不剔除；seed42预检偏差保留；test不得进入GLM或反馈到选择/训练。

06:01:39实时账号检查为4GPU/48CPU/1提交作业；GPU配额3700h、已用779.866667h、
剩余2920.133333h（21.077477%使用）；这是查询时快照。四逻辑分片可用4/2/1 H800
并行或分波，预估17–20/34–40/68–80分钟，各约1.13–1.33GPUh。固定逻辑分片减少
物理并行度变化对任务/seed分配的影响；不声称GPU逐位确定性。4卡可能排队更久。
资源证据保留在results/heldout-resource-plan-20260910-v1.json；当前搜索仍占用唯一
提交名额，没有提交test作业。协议run_ready=false，最终来源矩阵及执行入口尚待准备。
这是最终评价框的预声明，不是新机制收益、完成的对照实验或实际test吞吐测量。

## E49 — 第三轮完整审计及第四轮启动（2026-09-10）

h8/223273于05:28:06 COMPLETED/exit0，1027秒/2H800/0.570556GPUh；完整32题
7成功、32次请求、249983输出token。h9/223332于05:45:57 COMPLETED/exit0，
1047秒/2H800/0.581667GPUh；完整32题7成功、32次请求、243851输出token。
两者完整原生回复/token/棋盘审计、实际worker来源、Slurm终态与哈希均通过，归档
results/controlled-mh-223273/及223332/。第三轮原生选择仍保留h1（8/32）。
第四提案9次GLM请求0.084025CNY；截至05:53:45保守项目累计0.356537CNY/51次已结，
未结0，45CNY预算内可预留44.643463CNY。这不是官方账单或账户余额。
h10/223408运行中，两实际worker权重/seed证明PASS，启动快照保留。已知项目终态
分配合计17.296389GPUh，h10待终态。五轮尚未完成，分数仅为MH32优化集成绩。

## E48 — 第二阶段审计、终态及两阶段规范导出（2026-09-10）

新增joint_resume_receipts、audit_joint_training_phase2、joint_training_phase2_result；
joint_checkpoint_export扩展到phase1/phase2。原冻结训练/搜索实现保持不变。
Method对应theta2=RSFT(restore(C1),h_star;B2)以及theta_eval=BF16(theta_native)。
恢复记录必须绑定同条件/seed/checkpoint，完整actor布局与RNG/scheduler及pending data；
随后完整核验64条实际轨迹、token/mask/reward、Slurm终态与实际B2 sampler。
第二批本地更新数与此前累计更新分开，空更新仍按原生流程保存；导出逐值检查原生FP32
到BF16并测量相对共同theta0的变化，零变化与仅FP32变化都保留。

最终12项原生集成PASS/64.651秒（joint-phase2-audit-export-native-integration-20260910-v2.log）；
基础117通过/84依赖跳过/1.11秒，两个Slurm入口语法及diff检查通过。单项完整B2重放
64条/64请求、0accepted/0loss，与未改动的旧档案一致；临时来源及GPU/Slurm身份是fixture。
两阶段各执行原生小VLM导出，34张量/42576元素，覆盖零变化与1个FP32元素变化但BF16
不变的结果，native文件不变。fixture无image processor，不是视觉输入验证。
最初终态测试遗漏fixture run-name替换，最初集成遗漏tmp/upstream链接；仅修fixture，
失败及修复后的日志均保留。没有放松真实验证条件。
results/joint-phase2-terminal-export-implementation-check-20260910.json绑定源码与日志，
检查五份既有冻结训练/搜索计划及三次恢复修订的执行来源未变；不重算大模型manifest。
这些CPU检查没有产生新任务模型/API调用或joint GPU作业，也不代表实际联合checkpoint恢复。
正式joint第一阶段、MH搜索、第二批执行及留出评测仍未完成。

## E47 — h7完整审计及h8启动（2026-09-10）

h7/223272于05:10:54 COMPLETED/exit0，1046秒/2H800/0.581111GPUh；
完整32题6成功、32次请求、250228输出token，完整原生回复/token/棋盘重放及
来源哈希通过，证据位于results/controlled-mh-223272/。已知终态累计16.144167GPUh。
05:10:59控制器自动提交h8/223273，两worker权重/seed证明PASS。第三轮尚未选择，
h8/h9待完成；当前接受h1，未新增GLM提案调用。所有分数仅属MH32优化集。

## E46 — 联合第二阶段入口与无副作用的原生恢复观察（2026-09-10）

新增ours/joint_training_phase2.py、Slurm wrapper、native_resume_observation.py及
search_completion.py。Method对应h_star=MH(theta1;H)到
 theta2=RSFT(restore(C1),h_star;B2)的衔接，不新增优化目标或收益结论。
C1包含原生FP32 actor、scheduler、RNG及data.pt，按原流程不恢复Adam moments。
完整搜索检查逐轮重算候选/FST拒绝、分数、Pareto、选择与停止，核对全部评测档案；
只有五轮完整或原生满分停止可进入phase2，不在真实档案中重写frontier。
恢复观察在原loader返回前核对全部actor值及RNG/scheduler；driver读取pending loader
状态，不调用会新建迭代器的state_dict。正式phase2必须使用同条件/同seed联合来源，
共同theta0的harness-only和weight-only前缀不能代替。GPU仍按原fit顺序同步后生成。
尚未准备的联合搜索入口预先接受一致candidate别名；当前harness-only冻结源不改。

实际Hydra/原生RLHFDataset、旧data.pt→正确第二批、原生小模型完整加载与后续随机数
不变、组合hook在上游cwd中幂等通过。旧weight-only checkpoint只是明确的CPU loader
fixture，无正式joint checkpoint/phase2计划或新joint GPU作业。全参数观察没有测量
vLLM接收端，真实GPU恢复、第二批完整轨迹/终态审计及留出测试仍待完成。
最终12项原生集成通过（56.843秒，phase2-recovery-native-integration-20260910-v2.log）；
基础116通过/81依赖跳过（1.37秒），Slurm语法及diff检查通过。
首次集成负例只删除第五轮comparison，未删第五提案请求，被额外请求检查先拦截；
负例夹具修正后通过，v1失败日志保留。真实档案和完成标准未变。
results/joint-phase2-implementation-check-20260910.json绑定最终源码、日志和五份既有
冻结计划的全部source文件哈希；这份CPU实现报告不重新计算大模型完整manifest。

## E45 — h6终态、第二轮选择及第三提案显式恢复（2026-09-10）

h6/223204于04:42:38 COMPLETED/exit0，1022秒/0.567778GPUh；完整32题8成功、
32次请求、247598输出token，审计及终态保留results/controlled-mh-223204/。
原第二轮比较保留h1；h4/h5/h6分别6、7、8/32。随后GLM以7次调用生成h7/h8/h9，
04:44:44因candidate字段不在旧别名集合而停止。原候选编号、路径和父候选无歧义；
新增staged_candidate_recovery.py只增加一致candidate映射，保留全部字段及候选字节。
显式修订b3ed1b32e1a9d7b8f2f4dd019029c24b7b29d2d9e06ac7a2eadd8c6593cc8d9e
冻结七个评测、前两轮选择、原第三提案与前两份修订；原文件不改。三项原生测试
比较三次中断与连续五轮/满分停止，顺序与演化记录一致、没有重复生成，评分为合成。
控制器PID1169570从原start_iteration=3、iterations=3继续；重放原第三提案0次API。
h7/223272于04:53:28开始，两实际worker检查PASS，启动/恢复快照保留。
API保守累计0.272512元/42次已结调用；第三轮7次0.071667元，未结0。
45元预算内可预留44.727488元；这是项目请求记账，不是官方账单或账户余额。
Method对应E3候选导入/执行兼容，不修改原评分、选择或研究机制。

## E44 — h5完整审计及h6启动（2026-09-10）

h5/223199于04:25:19 COMPLETED/exit0，1033秒/2H800/0.573889GPUh。
完整32题7成功、32次请求、246051输出token；全部原生回复/token/独立棋盘及
来源文件哈希核验通过，证据保留results/controlled-mh-223199/。
已知终态累计14.995278GPUh。04:25:36自动提交h6/223204，两实际worker
参数/seed证明PASS。第二轮仍待h6，原选择器尚未比较本轮；API仍累计0.200845元。
现有recorded_parser_comparison对h5全部32次回复进行零推理诊断：对原h0 parser
动作和合法性分歧均0，7个成功都能保持原动作，5个正常结束thinking、2个包含长度截断。
终止分布为assistant_token_budget23、wrong_move2、solved7。报告与原回复哈希归档；
这不是新的prompt/parser受控生成消融，也不证明全输入parser等价或任何收益比例。

## E43 — 联合theta1来源核验与原生MH入口（2026-09-10）

新增joint_staged_search及离线wrapper，继承原StagedSearch、原五轮×3候选选择和
完整两分片审计，仅改路由和提案元数据格式兼容。Method为h1=MH(theta1,h0)的交接。
正式prepare要求独立joint phase1训练和导出完成记录，核对condition/seed/两个作业、
完整模型manifest和保留的native checkpoint；target来自实际theta1导出。
原harness-only计划生成器仅提供共同数据、推理和预算设置，临时模板丢弃，联合计划
独立核验。共用数值reference仅作8坐标区分，绝不作为初始化；空更新仍可保持theta0。
FST继续使用原AST提示词限制。没有真实联合checkpoint，故无正式联合搜索计划或作业。

3项原生测试通过（joint-staged-search-native-tests-20260910-v2.log）：WHALE/FST
五轮及满分停止与原调用链一致，FST非提示词改动拒绝；新两worker路由重放旧32题/34次
调用，原始档案不改；来源condition/seed/配置/数据绑定改动拒绝。模型分数是合成的，
来源fixture终态认证被stub，历史回复仅工程证据；不声称真实联合搜索已经通过。
初次fixture错误读取原propose_claude没有的next_names参数，原失败日志保留，仅修正fixture。

joint export完成记录显式提供规范化的Slurm终态路径及绝对artifact路径，避免不同CLI
路径写法导致同一证据不相等。原生小VLM导出2项检查验证实际merger、完整参数转换及
相对/绝对close参数结果相同。基础环境112通过/76依赖跳过，launcher语法检查通过。
活动harness-only冻结源未修改，新的训练/导出/搜索GPU作业均未提交。

## E42 — 第二轮h4完整审计及h5启动（2026-09-10）

h4/223135于04:07:51 COMPLETED/exit0，1036秒/2H800/0.575556GPUh。
完整32题6成功、32次请求、255996输出token；全部原生回复/token/独立棋盘及
来源文件哈希核验通过，证据保留results/controlled-mh-223135/。
已知终态累计14.421389GPUh。04:08:06控制器自动提交h5/223199，两实际worker
参数/seed证明PASS。第二轮尚未选择，h5/h6仍待完整评测；API累计仍0.200845元。

## E41 — 联合第一阶段终态与规范导出交接（2026-09-10）

新增ours/joint_training_result.py与ours/joint_checkpoint_export.py及Slurm wrapper。
Method对应theta1=RSFT(theta0,h0;B1)的完成证据及theta_export=BF16(theta_native)
到E3搜索的权重交接，不新增研究损失或收益结论。共同起点、h0及已有冻结训练计划不改。
终态要求完整64行/请求计数、原生成功过滤与ceil(accepted/8)次更新、Slurm成功退出，
仅global_step_1、tracker=1、完整model/extra/data.pt、实际DataLoader的sampler进度。
调度器按原worker每非空批推进一次，独立于该批的SFT minibatch数量；空更新有效。

导出prepare重新验证完整联合训练结果，拒绝weight-only来源。规范导出使用原FSDP merger，
逐参数检查共同BF16起点、原生FP32和导出BF16，保留原生恢复文件、恢复推理配置资产语义。
close另核对导出Slurm终态和资源、模型manifest及完整参数计数；成功导出不等于GPU已加载。
实际计划要等独立联合第一阶段完成才能准备，当前没有真实联合导出计划或作业。

3项原生完成/保存状态测试通过（最终v4日志）；历史第一批metrics和sampler/extra在
明确synthetic provenance下使用，原档案不改。验证两条件计划绑定、截断audit、失败终态、
重复日志、额外checkpoint及改动模型文件拒绝。扩展fixture初次v2因临时目录下相对路径
缺失失败；仅修正fixture的绝对来源路径，v3/v4均通过，原失败日志保留。
2项导出测试通过，实际子进程执行小Qwen3.5原生merger：34活动张量、42576元素。
零变化及1个FP32元素变化而BF16全不变均PASS；精确转换、tokenizer/推理配置一致，
保留全部原生文件。fixture缺少image processor的原生警告保留，不宣称视觉输入测试。
基础环境112通过/73依赖跳过，shell语法检查通过；新增验证没有模型生成或GPU分配。
真实联合两阶段、搜索及留出集比较仍待执行。

## E40 — 第一轮选择及第二轮元数据恢复（2026-09-10）

h3/222969于03:38:37 COMPLETED/exit0，1022秒/2H800/0.567778GPUh。
完整32题审计0成功、33次请求、260128输出token；两worker参数/seed检查及全部
原生回复/token/棋盘验证通过，终态归档results/controlled-mh-222969/。
已知终态累计13.845833GPUh。03:39:05原生第一轮比较选中h1：h0/h1/h2/h3
分别1/8/5/0，分母32。成绩属于优化集；不能用它代替独立留出集或因果消融。

第二轮GLM生成h4/h5/h6，8次relay请求，99.094秒。03:40:45控制器因
`Expected allocated candidate metadata`退出；原始pending_eval.json是裸列表、
用harness标识候选。候选字节已经生成且未评测，原始提案及失败记录保持不变。
API保守累计0.200845元/35次已结请求，其中第二轮0.062230元，未结0；非官方账单。

新增ours/staged_metadata_recovery.py和独立冻结修订，仅对无歧义元数据格式作
兼容处理，保留所有原字段；原作者循环以start_iteration=2、iterations=4继续。
已完成首轮历史、四个审计和第二提案均复用；候选代码、预算、数据与原选择规则不变。
原候选报告不作真实因果分析或收益证据。再次异常不会隐式重试。
三项原生CPU测试通过：完整五轮和第二轮满分停止的选择历史、分配、评测与逐行
演化记录均与无中断运行一致；旧证据、候选、首轮比较字节不变，冲突/越界仍拒绝。
基础回归111通过/69跳过；原生恢复测试使用合成分数，不含真实模型/API调用。
实际冻结预检通过，修订SHA707a71fa1eca99e4f4bf63f7f92e151c342037e69e523ff2891fbd5f4db92c70。
03:50:35控制器PID1101994完成复用、零新增API调用，h4作业223135已在2H800运行；
两worker实际参数检查通过，启动快照保留。完整搜索及联合试验仍未完成。

## E39 — h2完整评测及h3启动（2026-09-10）

222958于03:04:04–03:21:06 COMPLETED/exit0，1022秒/2H800/0.567778GPUh。
完整MH32结果为5/32，32次模型请求、256392输出token；原生回复/token重放和独立棋盘
核验通过，证据归档results/controlled-mh-222958/。已知终态累计13.278056GPUh。
03:21:35原控制器自动提交h3作业222969；两个worker初始参数检查通过、正常推理，
启动快照results/controlled-mh-222969-start-snapshot.json。当前首轮h0=1、h1=8、h2=5，
分母均32；h3未完成，原选择器尚未做首轮选择。API估算仍0.138615元，27次已结调用。

## E38 — 联合条件第一批入口与完整轨迹审计准备（2026-09-10）

joint_training_phase1复用原共享训练配置和原生phase_overrides，以共同theta0/原h0
独立启动WHALE、WHALE-FST的第一批8题×8轨迹。累计target1、online iterations0，
保存model/extra/data.pt，Adam不保存；不池化已完成weight-only前缀。所有题目仍来自
固定train128，预写两批真实DataLoader顺序，当前只采样第一批，最多520256输出token。
每个第一阶段2H800/16CPU/160GiB/60分钟，最多2GPUh；未提交实际GPU训练。

两份seed42计划prepare及独立CLI check均通过完整Hydra、共同权重、源码、真实数据集/
DataLoader核验。WHALE计划SHAe828995883605be964319dd7671d69ff3e16ef3cf2f0cf347b290256c3663868；
FST计划SHAdf3ec6831c02e0e1d16541a87e42420015ef6f5032fc8fa8caa2001315aae8ed。
两配置经原RayPPOTrainer构造后的训练步数/optimizer配置与新审计器逐字段一致，
记录results/joint-phase1-runtime-audit-20260910.json。

2项原生测试10.840s通过：范围/重复计划拒绝，以及原_fit_online_rsft在0→1和1→2时
都只生成64条合成轨迹。accepted为3和0的情况均保持保存和权重同步，空集不调用SFT。
这里的1→2从设定global_steps=1开始，未调用原checkpoint loader，不能算真实GPU恢复。

audit_joint_training_phase1核对唯一批次、原生顺序/64行、全部请求、token/mask及独立棋盘。
其兼容测试复制222801历史第一批并仅重映射fixture来源元数据，通过完整64条/75次请求
重放，得到原3条accepted、22989训练token；额外batch2会被拒绝。原归档未修改。
首次fixture因临时目录下相对dataset路径失败，日志保留；只修正fixture为原数据绝对路径，
第二次20.654s通过。不存在重采样或把旧数据冒充新联合条件结果。
基础环境109 passed/68 skipped，shell语法通过。相关日志均在results/joint-phase1-*。

Method对应E4的theta1=RSFT(theta0,h0;B1)及到E3的保存箭头，不新增优化目标。
本项只准备第一阶段；联合MH checkpoint来源核验、第二阶段启动/真实恢复、终态封存与
导出仍需完成，F0–F5状态不变。

## E37 — 受控h1完成及固定回复parser诊断（2026-09-10）

222891于02:46:33–03:03:49 COMPLETED/exit0，1036秒/2H800/0.575556GPUh。
共同未训练theta0、seed42、同一MH32下h1=8/32，h0=1/32；32次调用、245056输出token，
完整请求/token/原生回复重放与独立棋盘审计通过。终态和审计在results/controlled-mh-222891/。
已知项目终态累计12.710278GPUh。03:04:04控制器自动提交h2/222958；h2/h3尚未完成，
原选择器尚未执行本轮选择。API仍27次已结请求、项目保守估算0.138615元，不是账单。

新recorded_parser_comparison只使用已审计回复，不调用模型、API或打开测试数据。
全部32次回复的新旧parser动作与合法性判断完全一致；8个成功都保留旧parser动作，
其中6个每次调用均有thinking结束标记，2个含length截断。23个失败因预算耗尽，
另1个是合法但错误的走法。候选的system/user prompt及parser/legality函数AST均有改写，
预算仍format1/illegal1/maxturn9；不能仅凭GLM报告中的机制描述归因，也不能把固定回复
分析当作旧parser+新prompt重新生成的因果消融。源码/输入哈希与逐调用记录在
results/controlled-h1-222891-parser-comparison.json。此结果不是VETO收益或留出集性能。

## E36 — 后续weight-only种子的真实DataLoader预检（2026-09-10）

新的controlled_data_order调用原create_rl_dataset、collate_fn、RayPPOTrainer._create_dataloader
及StatefulDataLoader，实际读取前两批训练输入；题目ID及保存sampler状态再与独立索引
探针核对。父进程Python/NumPy/Torch CPU随机状态完整恢复。测试三种子42/43/44全部
通过，seed42两批还与222801实际128条轨迹一致；独立探针矛盾会拒绝。
3项原生CPU测试44.583s通过；基础环境108 passed/66 skipped，未新增模型或API调用。

controlled_pilot_continuation复用原weight-only两批配置/启动，仅为尚未训练的43/44写新
真实题序合同及TorchData版本/源码绑定，种子42原结果和偏差不改、不重采样。
seed43计划SHA53c06268f64545b2562f17a8ef3504990a1b1ed3b4fd50cb62d5c192cd4d0e61；
seed44计划SHAec45befd91fc3c456c32cf12a15f9bfa389a3fd61ed4c1967a2b35c6143025fd。
两计划prepare后均再次通过独立CLI --phase check，完整Hydra配置和实际DataLoader输出相同。
结果位于results/controlled-weight-only-seed43/44-{prepare,preflight}-20260910.log及对应plan/YAML。
它们各保持128新训练轨迹、1040512输出token上限、共同theta0/原h0/连续两批，无proposer。
未提交训练；历史2H800训练4098秒仅作资源校准，不当作当前排队预测。
原完整轨迹审计器可读取新合同的planned_batch_ids；联合WHALE/FST阶段仍需单独接通。
另将两个新冻结配置交给真实RayPPOTrainer DataLoader构造过程，原生训练步数/optimizer
schedule修改后的完整配置与现有runtime_configuration审计器逐字段一致，结果保存在
results/controlled-continuation-runtime-audit-20260910.json；没有进行新训练轨迹重放。
Method对应E4数据采样和随机种子复现设置，不改变原损失或新增方法贡献。

## E35 — 受控checkpoint规范导出准备（2026-09-10）

controlled_checkpoint_export绑定222801终态、完整批次证据及两个checkpoint哈希，选定
global_step_2/actor，输入参照共同未训练theta0。新计划
results/controlled-weight-only-seed42-export-plan-20260910-v1.json已冻结，SHA
fa4a3dd05ab8152051fc15a13087b2190f6c60d0f1d8715e0311c02f75925174。
目标data/native-rsft/controlled-weight-only-seed42-222801/hf-step-2-canonical；
原生model/extra/data.pt保留，训练题序预检偏差在导出计划/未来报告中继续披露。

原合并器仍执行参数加载、合并及BF16转换；规范参数名和共同起点配置复用此前验证实现。
全部723独立活动张量逐元素比较，要求每个导出元素精确等于原生BF16 cast，同时报告
FP32更新与BF16保留变化。UNCHANGED也是可通过的参数审计结果，不预设学习收益。
GenerationConfig自动保存的源缺省差异仅允许output_hidden_states/output_attentions
由False变None，或完全无差异；归档生成资产并恢复共同源缺省后，核验完整推理合同。
其他解码差异直接拒绝。新入口不证明joint-arm前缀共享或GPU恢复已完成。

5项相关原生CPU测试17.910s通过：新资产恢复/错误解码拒绝、真实小张量的零变化及
FP32变化被BF16舍入抹去的完成路径，以及已有原合并器/规范key/完整cast/图漂移检查。
新完成路径的serializer和tokenizer合同使用显式fixture替身；原合并器另由真实小型
Qwen3.5模型测试验证，实际共同起点的配置/tokenizer资产则由单独新测试核验。
这是软件测试，不是4B参数转换结果。基础环境107 passed/64 skipped；shell语法检查通过。

prepare已核对真实训练输入及checkpoint哈希，未提交GPU作业、调用模型或API。
CPU导出分配计划1RTX4090/4CPU/64GiB/20分钟，必要GPU分配完整计费，上限1/3GPUh。
完整harness-only搜索占用唯一提交名额，导出须在无竞争的调度时机执行。
Method对应E4保存/交接的逐元素BF16等式，F0–F5状态不变。

## E34 — 共同初始h0完成、首轮报告路径中断与恢复验证（2026-09-10）

222889于02:11:21–02:28:54 COMPLETED/exit0，1053秒/2H800/0.585GPUh。
共同未训练theta0、seed42、MH32完整审计：1/32，33次请求、260128输出token。
结果及全部私有证据哈希在results/controlled-mh-222889/result.json，完整审计及Slurm
终态并列归档；此前16次完成的运行快照不能替代本结果。已知项目终态累计12.134722GPUh。

02:31:10原控制器在首轮提案后停止。GLM用6次relay请求生成h1/h2/h3及
logs/iteration_1/report.md；原作者系统说明允许logs/iteration_*/report.md，本地
scoped_proposer只接受iteration_001。这是兼容性中断，没有候选被评测或选择。
三个原始候选通过原native.validate_candidate；项目保守API累计0.138615元、27次已结
请求，当前首轮增量0.051539元。不是官方账单或账户余额；报告里的机制/因果描述不作证据。

新staged_search_recovery通过独立修订冻结失败现场、原始请求/候选/报告/SSE/公开证据
哈希及新代码哈希，原冻结计划和已完成评测不修改。只接受当前未补零报告，将原字节归档，
再调用原完整性检查；额外槽位、修改已有证据、跨轮报告、双报告及软链接仍拒绝。
首轮从原始隔离视图复制校验，保留原视图；在原run_evolve回调中导入同一候选及规范化
slot/name元数据，避免提前占用槽位。h0走已审计缓存，第一轮不再调用proposer。
后续四轮仍使用原多候选适配器和原5×3搜索/满分停止。不会泛化为自动故障重试。

三项native CPU测试通过：与不中断运行比较五轮候选/评测顺序及选择历史完全一致，
独立满分fixture也保持提前停止；首轮只生成一次、h0只评测一次，篡改候选导致恢复拒绝。
所有测试零API/模型调用、评分合成，详见results/staged-search-recovery-native-tests-20260910.log。
基础环境107 passed/62 skipped；原始三个真实候选仅做原生校验，尚无候选成功率。
恢复修订results/controlled-harness-only-seed42-recovery-amendment-20260910-v1.json已预检冻结。
02:46:33恢复控制器PID1019931从原run_evolve重用h0及原首轮三个候选，实际请求参数与
已冻结首轮请求完全一致，没有额外GLM调用。h1作业222891已运行，两worker参数检查
通过并开始推理；启动记录results/controlled-harness-only-seed42-recovery-start-20260910.json。
恢复代码提交5ce25ac，原计划源码不变。h1运行中的成本和分数尚未结算。
本项属于E3执行设置，VETO/F0–F5无新科学结论。受控checkpoint导出及联合GPU恢复仍待接通。

## E33 — 两批训练完成、原生恢复核验及预检顺序偏差（2026-09-10）

222801 COMPLETED/exit0，01:02:44–02:11:02 HKT，4098秒/2H800/2.276667GPUh。
139次请求全部完成，1039118输出token，无错误或未结请求。128条轨迹完整重放、token/
mask核验及独立棋盘检查通过；第1批3/64 accepted，00BQD两条、007fJ一条，22989 loss
token，1次SFT更新；第2批0/64，原生跳过更新。两个checkpoint均保存并记录模型及恢复
状态文件哈希。两份原生模型文件字节完全一致，SHA均为
`1e3d65762a6e078375b7a8da5198cd47cb1cb6355995b2adeeeac4fa1ed7d0dd`；
数据加载器进度仍分别保留，完整参数相对共同theta0的比较/规范导出尚待执行。
终态报告results/controlled-pilot-222801/result.json，实际训练更新数不是批次数2。

CPU恢复探针最初两次失败：下一批与旧预检ID矩阵不符；随后确认第1批实际ID也与旧
矩阵不符。原因是旧native_order直接遍历RandomSampler，而Torch2.11/TorchData0.11的
多进程DataLoader初始化及reset会构造两次sampler迭代器，推进了随机流。新
native_loader_schedule通过原RayPPOTrainer._create_dataloader产生索引，记录两次
迭代器构造的RNG哈希。其第一批及保存generator状态完全匹配实际记录；恢复真实
RLHFDataset/data.pt后，下一批也完全匹配。三种子只用索引的CPU探查均完成，没有模型调用，
记录于results/native-dataloader-schedules-20260910.json；seed43/44的完整训练配置仍须另行冻结。

真实seed42两批分别为
`[0095W,003o0,00BQD,007fJ,008qL,004Ao,003md,004b0]`、
`[004u0,00Ar2,005gP,003r5,005nD,00AB1,003UW,00B7G]`。
旧计划及其ID列表保持原样；偏差审计只替换内部诊断期望矩阵，重用完整原审计过程。
顶层状态显式为PASS_NATIVE_EXECUTION_WITH_PREFLIGHT_DEVIATION，训练终态同样带偏差标记。
未按成功率选题、丢弃轨迹或重新采样；后续计划必须修正DataLoader预检，不能用旧helper
声称具体题序已验证。偏差详情及全部原始失败日志均保留。

native_phase_resume对应E3→E4的原生状态恢复，设置累计target1/2并要求data.pt存在。
原FSDPCheckpointManager的CPU小模型测试恢复了模型、scheduler、Python/NumPy/Torch RNG；
即使放置损坏的optimizer文件也不读取，符合原model+extra、不恢复Adam的契约。
真实data.pt探针只记录actor RPC，未执行GPU参数加载；不据此许可共享weight-only前缀。
4项原生检查通过，11.434秒；基础套件105通过/61依赖跳过。完整轨迹审计及封存不新增推理。

控制器PID930536在名额释放后自动提交222889，02:11:21启动harness-only h0/32题评测。
两张H800的实际worker权重8坐标和进程种子42证据均已落盘。此时无新GLM提案，累计
保守API估算仍0.087076CNY；项目已知终态GPUh合计11.549722，222889成本待终态。
受控checkpoint的完整值比较/导出、联合条件原生GPU恢复、四条件×三种子及视觉闭环
仍未完成。两批训练题不同，3/64与0/64不能作为参数更新后退化的直接证据。

## E32 — 完整原生搜索的跨节点调度（2026-09-10）

新增staged_search/staged_evaluation/audit_staged_evaluation，Method对应共享E3执行边界
`h_next=MH(theta_fixed,h_incoming)`，保持原run_evolve的5轮、每轮3槽、frontier及early-stop。
登录节点常驻原循环；每个候选由独立Slurm作业的两个H800分片完成32题；完整记录原生
输出、请求起止、实际token、worker权重坐标，独立重放及棋盘验证后才允许分数回流。
分片失败、缺失证据、改变文件或非成功Slurm终态会阻止后续付费提案，不记成0分。
中断控制器不自动新开试验，原目录保留；实际提交响应含糊时也不重试sbatch。

6项原生CPU检查通过，21.020秒。覆盖完整5轮选择历史、FST越界候选拒绝、原满分提前
停止、失败评测不触发proposer、重复启动拒绝、提交名额等待。旧h0/h1共69次真实回复/
511216 tokens逐条重放；新合并器读取旧h1两分片得到原6/32并通过完整审计。所有这些
检查均0新增生成、0付费调用、0GPU分配；合成搜索分数与旧真实模型记录分开标记。
最新基础测试101通过/59依赖跳过，日志为results/staged-search-*-20260910-v2.log。

首个harness-only seed42计划SHA为
`ade2e2a9ed315ab5abddb0b52e4679cfb5f3d9641b6f2703c89a4abf6a52d59d`。
目标为共同未训练theta0，工程theta2仅用于选择8个数值不同的worker检查坐标，不用于
初始化/训练。每次评测2H800、16CPU、128GiB、40分钟，最多16次/512轨迹/4162048
assistant输出token/60次GLM请求/21.333333GPUh；这是上限，不是已花成本。
实际API仍受全项目45CNY journal约束，当前累计保守估算0.087076CNY，held0。

01:40:52调度预测被QOSMaxSubmitJobPerUserLimit拒绝，未提交作业；随后确认222801
仍是唯一活动作业。因此在sbatch前等待账号队列清空；不对实际提交进行自动重试。
单卡预计约35–45分钟，双卡历史实际同规模评测约17–18分钟、0.57–0.59GPUh；保留
已验证的双卡确定性ID分片。当前队列等待时间尚不可预测，不把旧forecast当作现值。
01:41附近配额为222000GPU分钟/已用46393；这是查询时快照，非当前账单。

222801于01:47:33仍RUNNING。第1批日志3/64 accepted、1次SFT更新、loss token22989，
actor50.597秒、保存35.644秒、同步0.896秒；两批独立审计待完成。当前91个请求完成，
返回648926输出token、0请求错误，48请求未结束。日志中的response-region tokens含
环境文本和重编码，不能冒充服务端completion_tokens。第1批指标和时点快照已保存。
F0–F5仍未通过，未读取留出测试内容，无视觉/VETO收益结论。

源码提交a18422f后，登录节点控制器于01:50:50进入WAITING_SUBMISSION_SLOT，实际PID930536。
会话44254、计划及h0请求的不可变启动快照保存在
results/controlled-harness-only-seed42-start-20260910.json；当时0新增GPU作业/0新增API。
活动日志不加入Git，现存目录禁止重复启动。两份已开始计划的源码哈希均复核一致。

## E31 — 多候选原生循环及两批次审计接口（2026-09-10）

scoped_proposer保留真实iteration和候选槽位。原生WHALE/FST循环各执行2轮、每轮3槽，
h0和h1–h6全部进入原生评估路径；得分来自显式CPU fixture，不是模型表现。
三项范围检查拒绝历史证据修改、越界候选、重复/冲突/陈旧metadata；缺失metadata时调用
原生recover_pending_candidates，不让上一轮pending文件冒充本轮提案。没有付费API调用。
4项原生测试通过。原生完整5轮GPU staging仍未接通。

两批次auditor复用原生请求重放、token/mask核查及独立棋盘验证，增加每批原生采样顺序、
完整128轨迹和两批统一请求池覆盖。实际原生dataloader将总步数及actor/critic调度器步数
设为2，与审计配置一致，CPU fixture通过。完整真实批次审计尚待222801完成。
最新基础套件99通过/55依赖跳过。

## E30 — 首个共同起点weight-only对照启动（2026-09-10，运行中）

222801，seed42，2H800、16CPU、160GiB、90分钟上限；最多3GPUh，无API调用。
原生过滤保留全部128训练题，seed42预先固定两个批次、共16题×8轨迹。h0固定，两个
批次连续执行；不重启Adam或采样engine。在线iterations显式设0，总native batch steps为2。
首次准备误读HARNESS_PATH字段，CPU预检失败；改用原生CHESS_PUZZLE_HARNESS_PATH后通过，
未为该错误分配GPU。实际GPU启动再次解析完整配置并核对数据顺序后才进入原生训练。

2H800可立即启动，4H800预测到9月11日18:57；单卡不支持当前已验证的分离角色配置。
01:12附近：64个请求开始、8个完成、63743输出token、0错误，当前仍运行。真实Ray hook
和初始CuPy清理成功。空accepted批次若出现须如实记录，不能重抽样选好结果。
这仍是128/32/64小规模协议，非原论文16384训练题/256MH题规模；F0–F5均未完成。

## E29 — 共同未训练初始权重完整审计通过（2026-09-10）

222794 FAILED/exit1，77秒/1RTX4090/0.021389GPUh。所有条件将使用
同一theta0=Q_BF16(theta_pretrained)，没有优化器更新或生成请求。序列化破坏检测
通过：拒绝改值、alias不一致、缺少/新增活动参数。原任务在cast audit之后因推理配置
合同不一致退出，未写入完整结构化参数报告，故恢复必须重审。原模型没有generation_config，
save_pretrained新增的文件让output_hidden_states/output_attentions从False变为None。
失败源代码、配置和终态均归档。222797只移走该新增文件、保留权重字节并重新完整审计。
恢复预检同时核对原始两分片index与新单文件的区别；首版未提交计划由v2取代。
完整worker CPU fixture最初误用recorder属性名，修正测试断言后通过；没有修改训练hook。
最新基础套件96通过/53依赖跳过。恢复前已知项目累计9.246667GPUh。

222797 COMPLETED/exit0，00:44:40–00:46:15 HKT，95秒/1RTX4090/0.026389GPUh；
项目已知累计9.273056GPUh。723个独立活动张量/4,539,265,536个元素完整，均精确等于
原始权重的BF16 cast。原BF16元素无变化；原48个FP32张量/3840元素中有2791个元素/
24张量经舍入变化。0次优化器更新、0次生成/API请求。完整审计报告已写入并归档。
只把新增generation_config移入归档，权重文件SHA前后相同；模型/生成/词表/控制ID/
prompt token语义检查通过。共同模型manifest SHA为
260337087e4d739db98c2744cf429678cd2f87674e55cad9f22d3cfbe60c3d1a。
这不是一次独立F0试验或任务提升；原始未转换模型与共同BF16起点必须分别标注。


## E28 — 实际三种子与四条件搜索边界（2026-09-10）

222792 COMPLETED/exit0，114秒/2H800/0.063333GPUh；已知项目累计9.225278GPUh。
真实三个worker的model/torchCPU/torchCUDA初始种子分别42/43/44，初始RNG哈希各异，
重复读取不会推进RNG。每个engine的8个实际embedding坐标均等于已审计theta2。
每种子8个相同短prompt且request seed=None，分别返回23/20/21 tokens，总24请求/64 tokens。
CUDA RNG随生成推进；CPU RNG未推进。正常关闭日志及NCCL进程组退出警告一并保留。
这是独立vLLM工程探针，不是三次训练、完整Ray worker验证或任务准确率。

6项原生CPU集成通过：Python/NumPy/Torch种子和启动时hash seed检查、真实数据sampler、
原生vLLM CLI解析、传入harness作为h0、FST只允许prompt变化、weight-only禁止搜索。
CLI fixture只把无CUDA登录节点的设备发现设为CPU，不启动engine。三种子原生Hydra配置
另行实际解析通过。最初CLI fixture因无CUDA设备识别失败；诊断保留，修正fixture后通过。
基础套件96通过/52依赖跳过。无新增GLM请求，保守API累计估算仍0.087076元。

## E27 — 新vLLM worker数值与单请求核验完成（2026-09-09）

222779 COMPLETED/exit0，23:55:24–23:56:44 HKT，80秒/1H800/0.022222GPUh；项目已知
累计9.161944GPUh。两个模型的推理配置/生成配置/词表/控制token及工程prompt token
一致。官方worker extension读取8个实际embedding值，全部等于新导出且区别于本轮
输入；坐标从两个真实checkpoint重新选取，不复用上一轮的区别值。

随后1次工程请求返回[9175,248046]共2 tokens、文本“Yes”、finish_reason=stop。
prompt要求ready，因此不计为指令遵循成功；也不是任何训练/MH/test题结果。
数值证据在生成之前写入request receipt，完整measurement与终态均保存。engine日志
确认Shutdown complete，随后有destroy_process_group未显式调用的警告，Slurm退出0；
不隐去该警告，也不将其等同于已经观测到持续的资源泄漏。没有遗留Slurm运行任务。

当前完成的是共享E4训练→canonical序列化→新进程实际加载；没有核验原生NCCL
接收端的全部数值，没有F0四条件×三种子结果，尚无VETO/heldout增益。无新增GLM调用。

## E26 — 本轮实际输入到原生/导出的完整参数比较（2026-09-09）

222703 COMPLETED/exit0，23:44:30–23:49:02 HKT，272秒。CPU merger/audit，按QOS要求
分配1RTX4090并计0.075556GPUh；已知项目累计9.139722GPUh。原生输入checkpoint
哈希在合并及审计后不变。723个独立活动张量/4,539,265,536个元素完全保留，alias值
一致；每个导出元素精确等于原生FP32的BF16转换。相对本轮全BF16输入，原生FP32
有3,845,402,884个元素/426个张量变化，L2=0.00933611255；BF16导出后保留
14,479,791个元素/286个张量变化，L2=0.000778378142。两者最大绝对差均2.38418579e-7。
这是参数更新证据，不是任务准确率或科学效果结论。

导出入口最初传入str给既有Path接口，CPU预检拒绝；修正类型后重新完整预检，故未
为该错误分配GPU。新worker预检初次在基础解释器读取torch版本失败，改用实际原生
runtime后通过；同样未提交GPU。诊断保留，不混入模型失败或消耗。

222779现已提交：前后模型配置/生成配置/词表/控制token一致，8个新的区分坐标由本轮
两个真实checkpoint计算。单H800、最多1次16-token工程请求，不用train/MH/test题。
实际worker数值、请求结果及最终退出状态仍待核验；未新增GLM调用。

## E25 — 实测CuPy缓存释放后完成两次原生更新（2026-09-09）

222640 COMPLETED/exit0，368秒/2H800/0.204444GPUh；项目已知累计9.064167GPUh。
真实fixture释放64MiB而保留活跃sentinel；原始初始/更新后NCCL同步各释放6GiB+512
闲置字节，池total归零，used保持零。CPU记录最初把额外512字节也要求为纯6GiB，
归档断言拒绝；检查实际事件后保留完整字节数，没有修改GPU输出或训练结果。

完整64条缓存/11条accepted/30392 loss token，原生8+3 mini-batch完成2个step。
loss1.223704905831255是两个mini-batch均值之和；grad_norm8.163288593292236是两步
均值。更新141.718s、保存35.324s、同步0.875s；Torch分配/保留峰值75.803/77.352GiB。
真实20.70GB native checkpoint已保存并哈希，导出和实际推理参数验证仍待完成。
关闭时有DataLoader worker Killed traceback，出现在完整metrics/save/sync之后；
Slurm终态COMPLETED/exit0。归档保留该限制，不删除失败信息。

此修复属于E4资源生命周期，未改变损失/token/AdamW/NCCL传输。源69次模型调用仍
计入222292，无新增GLM/采样。基础95PASS/47SKIP，原生CPU3集成+2清理边界测试PASS。

## E24 — fused128反向OOM与NCCL缓存生命周期修复（2026-09-09）

222615终态FAILED/exit1，23:12:16–23:17:24 HKT，308秒/2H800/0.171111GPUh；
前次222512为305秒/0.169444GPUh。原生fused128推进到固定词表梯度矩阵乘法后
OOM：申请1.19GiB，余527.06MiB，进程78.66GiB、PyTorch已分配70.54GiB。
64行缓存、11行accepted、30392 loss token保持。无最终更新计数/模型，较早内部
更新数未知。完整终态/日志/冻结来源已归档；无新增推理/GLM调用。

检查原始NCCL finalize发现CuPy双3072MiB缓冲置空后，仅调用torch.cuda.empty_cache。
新增ours/native_transport_memory.py通过注册子类，在原始finalize后同步设备并释放
闲置CuPy块，记录池used/total/free与设备空闲显存。活跃块不释放；不改公式、张量、
优化器或实际NCCL传输。2项原生CPU边界检查PASS9.743s。GPU端须先以原生prepare/
finalize与活跃sentinel验证分配，再在真实同步中测量释放，未预先宣称解决OOM。

## E23 — 同一审计批次的原生Torch后端恢复（历史记录，2026-09-09）

**后续终态：222512 FAILED/exit1，原生fused512反向OOM；以下保留提交时记录。**

完整恢复计划SHA`c4af93ddc688713ef2a38444c6590f908603e9f7c84b3a48eb6d292398c4ec4c`，
绑定114项来源；源码提交`4e7589a`。model和actor同时启用原生fused开关，backend
显式保持Torch。整个配置与222292只允许这组后端/恢复hook和运行目录变化。首次
Hydra误用新增已有backend键被拒绝，改为覆盖现有键后完整预检PASS，未启动GPU。

`recovered_alternation_bootstrap.py`继承原trainer，以完整审计批次替换一次生成，
逐项检查实际入队顺序/任务内容，沿用新的UUID分组。原success-filter保留11条轨迹，
30392训练token，裁剪无效回复尾部至6321，prompt4096保持。mini8/micro1产生8+3
两个mini-batch；仍用原actor/优化器/checkpoint/NCCL。两个服务生成入口显式拒绝调用。

3项CPU测试通过真实dataset/8-worker loader和原native trainer循环、异地目录初始化、
配置漂移及新推理拒绝；GPU调用/save边界使用fixture，不计作真实训练。基础95通过/
44依赖跳过。Torch fused在之前FP32/BF16 CPU fixture通过，但BF16不是逐位等价；
正式对照必须共用后端。缓存返回的原生日志吞吐量不能用作推理速度。

H800立即/2H100预测9月10日20:42，选择保持原actor+rollout两角色的2H800/160GiB。
估计6–12分钟、0.2–0.4GPUh；硬上限20分钟/0.666667GPUh，不是实测。作业222512
于22:57:32 HKT提交，后续查询g4 RUNNING/1:49，actor日志已确认fused=True，尚在
原生初始化。还没有优化器/checkpoint/实际新权重交接结果，也无新GLM或采样调用。

## E22 — 选中h1后的在线训练审计准备（2026-09-09）

**完成采样与批次审计，训练前向OOM。** 222292终态FAILED/exit1，22:00:33–22:33:45
HKT，1992秒/2H800/1.106667GPUh，项目已知累计8.519167GPUh。64条轨迹/69次调用/
452201输出token完整；11条accepted/30392训练token，覆盖2个训练题。原生请求和
配置/token/mask重放、独立参考续着全部PASS。42条预算停止、7条wrong_move、4条
malformed。成功轨迹数不能当作独立题目数，亦不是与上一批不同题目的受控对比。

原actor前仅裁无效回复尾部至6321，提示区域仍4096，所有有效token/样本未删除。
随后lm_head F.linear申请4.82GiB，GPU仅剩3.68GiB而失败。没有最终actor指标或
checkpoint，内部是否已执行较早mini-batch更新未知；没有可核验完整更新阶段。
终态文本、日志、88项冻结来源与完整batch审计均已归档。下一修复复用已审计批次，
不重新生成。原仓库已有torch分块输出实现通过小型混合Qwen的FP32与BF16原actor
损失/梯度CPU测试；各自损失相同，聚合梯度相对L2差异分别1.16865e-7和0.00409952，
55个梯度张量。FP32逐张量容差2e-5/2e-6；BF16预设聚合容差为4倍该类型epsilon。
不宣称BF16逐位一致，正式条件必须使用同一后端。尚无真实4B/FSDP GPU恢复结果。
基础环境最终95通过/41依赖跳过，另有4项原生batch审计、1项参数序列化及2项
原生分块后端CPU测试通过；未重复采样，未新增GLM调用。

证据：[失败结果](results/alternation-rsft-222292/result.json)、
[完整批次审计](results/alternation-batch-audit-222292.json)。以下为运行中的历史记录。

222292保持同一在线运行。22:27:57 HKT查询RUNNING/27:24，69次调用开始、56次返回，
已知376769输出token、无错误，13次消耗未完全返回。没有完整batch、新优化器更新
或新checkpoint结果。88项冻结来源哈希无变化；未提交重复采样或额外GLM调用。

`audit_alternation_training.py`使用已保存回复重放原ChessPuzzleAgentLoop，核验实际
请求/采样、选中的harness、完整配置哈希、训练token/mask和元数据，再独立走棋检查
合法参考续着及重试边界。实际在线请求配置哈希为
`31ce33d0a64287c9f8ad693618410ebd6c044b1659c0edd67f85955cc1a843b7`，与冻结配置经
原生预初始化迁移后的结果相同。响应区不足以放入反馈时，原native即使解完棋仍给
零训练奖励，审计明确保留这一区别。若回复在写事件前被丢弃，则拒绝通过并保留证据。

4项原生CPU检查通过：多轮/重试，篡改请求、奖励和mask拒绝，独立错误走法拒绝，
预算停止、解题后反馈截断，以及未记入事件的回复不能被忽略。都是明确的回复fixture，
不是模型分数。首轮沙箱异步测试停滞被定点停止，日志保留；关闭tokenizer并行并使用
宿主执行后测试完成，未声称确定根因。另修正审计器成功标签win→原solved后全部通过。

`verify_alternation_transition.py`比较本轮实际canonical输入与新native FP32/BF16输出，
禁止新增/缺失活动参数，并检查共享head值。真实小型序列化文件测试通过：单纯类型
转换计零变化、FP32微更新与BF16落盘后保留变化分开计量，错alias/graph/导出值拒绝。
上一输入已全BF16，因此本次不存在第一次原始FP32小张量转换的额外混合项。
实际4B输出尚未产生。这两个模块对应E4测量与交接，未改变原成功筛选目标。

## E21 — 明确授权后的GLM提案与下一步连接（2026-09-09）

**最终MH结果：h0=0/32，h1=6/32（18.75%），原规则选择h1，独立审计PASS。**
222212正常COMPLETED，1023秒/2H800/0.568333GPUh；h1共34次调用、251088输出token，
平均7846.5，28次调用length停止且无thinking结束。6条成功中4条完整结束thinking，
2条从截断文本被原parser提取，全部参考续着经独立棋盘核验。失败为22条预算终止、
3条wrong_move、1条malformed。项目已结算累计7.412500GPUh，旧亚秒成本另列。
这是单种子MH优化集结果，未修改候选、未计为VETO收益、未替代留出集或F0矩阵。

下一训练最终计划绑定h1和新8题、双方审计及88项源码/证据哈希；完整实际Hydra配置
CPU预检PASS。一actor/一rollout两H800，160GiB RAM，90分钟/3GPUh硬上限。
H100排队预测到9月11日18:57，H800即时；当前GPU配额剩175765分钟。最终计划不
声称已训练。222292在22:00:33 HKT于g4启动，完整GPU前置检查和原生初始化后进入
在线采样。22:07:48快照64次调用开始、8次完成/63083输出token、无错误；剩余消耗
未知，没有完整batch或优化器更新结果。仍是同一个运行作业，未重启或重复采样。

对已有69次MH回复应用两个实际parser：h0的35次全部一致，h1的34次有33次一致，
另一次h0读出b5c6而h1判歧义。6条h1成功回复在旧parser下全部读出同样动作。因此
新parser不能被解释为这6条成功的来源。这是固定回复诊断，不是重采样消融，也不
分离提示词与观察前缀的因果作用。下一步正式FST控制及组件消融仍有必要。
下方为提案和评测进行中的历史观察。

用户明确批准上一轮列明的材料发送后，GLM提案成功，新增7次请求/77.3145秒，
保守新增0.039819元，累计0.087076元、held0。先前审批拒绝未执行的记录仍保留。
h1源码未人工修补；review发现说明中的last-tag-wins并未实现，多个不同标签仍被
拒绝，未完成thinking中的标签仍可能被解析。proposer实际没有读取逐条轨迹。
该候选属于full harness，与FST无关；不依据修改说明预设收益。

222212使用冻结v2 MH计划，在g4上2H800执行完整32题h1评测，h0缓存不重复采样。
两实际worker权重检查通过；21:48:44 HKT共有25次完整返回、193148输出token，
无错误记录，未完成调用消耗未知。完整评分与独立审计尚未完成。

为下一次E4准备train顺序的第8至15行，共8题64轨迹，h0/h1初始提示最长808/930。
训练入口先核验完整搜索选择、双方审计、数据角色、checkpoint及整个有效配置。
原frontier/early-stop目标一致性及真实Hydra配置拒绝模型/harness/损失漂移共2测试通过。
实际选择、最终计划和GPU训练仍待当前搜索结束。HF初始化与显式工程数据游标不
声称为原全量trainer/optimizer/RNG/sampler状态恢复；原公开Chess脚本本身仅保存/
加载model和extra，不保存optimizer。所有差异在下一冻结计划中列明。

另补WHALE-FST的prompt-only组件：上游未发布对应skill，因此从公开full contract
加明确文本搜索限制重建。原AST规则掩盖整个prompt RHS，CPU实证允许random.seed
副作用；增加执行前纯字符串/惰性str注解检查，原AST校验保留。4测试覆盖原wrapper
装配、原搜索循环、代码变化拒绝及副作用拒绝；测试分数是fixture，非FST实测。
此组件须在独立FST计划中显式接入，不修改当前full-harness阶段。


## E20 — 更新checkpoint上的完整MH划分与分阶段执行（2026-09-09）

完成记录：222196为COMPLETED/exit0，1069秒、2H800，0.593889GPUh；h0为0/32，
35次调用共260128输出token，32次length停止且无thinking结束标记。轨迹终态30次
assistant_token_budget、2次malformed。全部原runner回复重放、独立棋盘/token/模板
和请求计量检查通过；无新审计调用。项目累计已计量6.844167GPUh，旧亚秒成本另列。
终态来自本轮成功的scontrol查询；晚些时候保存全文时控制器记录已过期，sacct禁用。

登录节点GLM提案被自动审批拒绝，理由为这批具体内部轨迹的对外发送需明确授权。
命令未执行，API预算仍0.047257元、held0；本轮无h1，没有搜索收益或下一次更新。
已生成[7文件发送范围与脱敏说明](notes/mh_proposer_transfer_review.md)及SHA清单。
[完整结果与审计](results/updated-mh-baseline-222196/result.json)保留0/32，不能把CPU
fixture的分数或作者论文数字替入。下方为启动和运行时的历史快照。

使用完整32题MH/seed42，2个确定ID分片、每片16题；原始h0和GLM生成的一个h1使用
同一更新checkpoint/预算/运行时/分片方案。`mh_phase.py`围绕原`run_evolve`建立
GPU→登录节点proposer→GPU的阶段边界，保留原runner、验证器、汇总与frontier目标。
原结果ID均匿名为redacted，合并使用proposer不可见的内部顺序边车记录。

3项CPU原生集成测试通过：全32题的异质fixture输出与两分片逐值相同，完整原搜索循环
暂停/恢复只评测h0一次，并验证普通零分平局与候选满分early-stop；实际GPU worker子类
在原load入口后读取CPU fixture的模型值，加载边界使用测试桩，不认领GPU启动成功。
proposer入口拒绝缺失baseline独立审计。94项基础测试通过/27跳过。

完整服务参数解析第一次因登录节点无自动设备类型而失败；限定CPU设备类型占位后，
原参数解析器验证worker类及服务参数通过，此检查未初始化GPU。初步完整预检通过，
随后增加审计前置检查并冻结v2，再做最终启动预检。此时没有新GPU或API模型调用。
H100排队至次日，H800当时可立即；选择两H800以缩短预计完成时间，每GPU阶段40分钟/
1.333333GPUh硬上限。预计时间不是测量。正式F0仍保留weight-only/harness-only/FST/WHALE
各3独立种子；此一次搜索阶段不替代该矩阵，也没有VETO科学收益。

E20运行更新：v2完整CPU预检通过后，222196于21:00:31 HKT在g4分配2张H800。
两个真实服务worker的8个参数值均通过，21:15:38实录25次调用返回195096输出token、
无记录错误，未完成调用消耗未知；尚无完整32题MH结果。运行源码哈希保持冻结。

新增独立MH审计器，使用实际保存回复重放原runner/输入/解析/请求，再用python-chess
单独核验参考续着与合法性；不调用模型。CPU多轮、错误动作/奖励及格式异常检查通过。
另备好下一步在线训练记录+尾部裁剪子类，原actor分发CPU检查通过；`RSFT_HARNESS_PATH`
已接入原launcher，完整Hydra CPU配置确认可用新checkpoint与指定harness。这只是h0
占位验证，不是候选选择或下一轮训练。当前222196不依赖这些新增/修改文件，未被改动。

## E19 — 真实原生更新、规范导出与精度归因（2026-09-09）

**NATIVE_STEP_AND_CANONICAL_EXPORT_PASS; FRESH_VLLM_HANDOFF222175_RUNNING.**
恢复222156完成1次原优化器更新，原生checkpoint保存及NCCL调用返回；复用221967审计
批次、2条accepted/16258 loss token，新模型调用0。SFT loss0.627908975，梯度范数
9.324229240；actor更新33.321s、save44.206s、同步0.990s，Slurm整体274秒/2H100/
0.152222GPUh。Torch峰值allocated69.294504GiB、reserved70.337891GiB。
结束清理时DataLoader worker被终止的Ray日志保留；最终Slurm为COMPLETED、exit0。
缓存恢复的原生吞吐统计不能用作生成速度；Qwen3.5的MFU0为不支持该估算，不是实测。

原生checkpoint SHA256为`b1345308ed7e257e6ad85da8643cf9f6fa755812f4a479c3c13a6969dc42c06b`。
零GPU CPU导出被当前QOSMinGRES拒绝，没有作业或分配成本；按该要求分配1张4090，
CPU执行且隐藏CUDA。222164原merger写出文件后，严格参数名检查拒绝297个视觉名字被
HF旧格式回写为语言前缀的结果，150秒/0.041667GPUh。旧结果与来源保留。

`ours/canonical_native_export.py`通过原merger子类指定`save_original_format=False`，
保留原加载/合并/BF16转换；使用经严格差异审查的base推理配置/分词器，防止原生训练
增加的EOS/pad字段改变推理停止条件。原配置派生EOS为248044，tokenizer EOS为248046；
不将两者误认为相同。真实配置检查与小型模型全部原始存储名/数值/重载共2项测试通过。

222165在148秒/0.041111GPUh完成规范导出及全参数审计，原生文件前后哈希不变。
723个独立活动张量含4,539,265,536元素；native FP32有3,783,464,114元素变化，
L2=0.005828377；每个导出值都等于native BF16 cast。按base存储精度分组后：

| 原始存储精度 | 张量数 | 元素数 | 导出变化元素 | 变化L2 | 归因 |
|---|---:|---:|---:|---:|---|
| BF16 | 675 | 4,539,261,696 | 8,991,538 | 0.000338018 | 训练变化在导出后仍存在 |
| FP32 | 48 | 3,840 | 2,791 | 0.101289711 | 混合训练与BF16转换，不能全归因于学习 |

正式比较须统一序列化精度或加入仅转换的control，不能把整个导出L2当作学习强度。
这也不提供准确率收益证据。项目已计量累计6.131389GPUh，旧亚秒占用未知另列。

单H100作业222175在20:21:53 HKT启动，15分钟硬上限0.25GPUh。CPU预检通过后，
核对新vLLM worker内8个已知改变的BF16 embedding位置，再执行一条最多16token的
工程请求。不访问test，不调用付费API，也不代替原in-place NCCL接收端的指纹检查。
首次CPU预检暴露Transformers返回BatchEncoding；显式`return_dict=False`修复并验证
非空int列表，失败日志保留。最新基础测试94通过/23跳过。

证据：[训练归档](results/native-rsft-recovery-222156/result.json)、
[失败导出](results/native-checkpoint-export-222164/result.json)、
[规范导出归档](results/native-checkpoint-export-222165/result.json)、
[精度归因](results/native-export-precision-attribution-222156.json)、
[推理检查冻结计划](results/updated-vllm-handoff-plan-20260909.json)。
本阶段对应E4共享训练/序列化/执行测量。E1–E3未变，F0–F5未通过，无VETO收益声明。

E19后续：222175加载规范checkpoint后，`apply_model(partial(...))`被原生msgpack
拒绝，生成入口未执行。主进程异常后分配仍在，主动释放；Slurm最终CANCELLED，
实际206秒/0.057222GPUh，新模型调用0，累计6.188611GPUh。源码、原计划、完整日志
及实际控制器记录归档于`results/updated-vllm-handoff-222175/`。改用官方
`worker_extension_cls`和具名`collective_rpc`，仅发送坐标；不启用pickle回退。
CPU验证原编码器/解码器往返、扩展解析与真实BF16模型读取通过。v2计划再次冻结，
完整预检待完成后才提交，不将本次加载记录冒充完整handoff成功。

E19最终：222176的8个实际worker参数值全部匹配新导出且区别于base，1次生成完成，
返回token IDs `[9175,248046]`，独立分词器解码为`Yes`，与原始日志及结果文件一致。
提示要求`ready`而未遵从，因此只记PASS_ENGINEERING_HANDOFF_ONLY，不记准确率成功。
结果写完后Python/vLLM子进程仍存活，主动取消释放；Slurm最终CANCELLED，实际222秒/
0.061667GPUh。此后补显式10秒engine shutdown，未重复GPU运行，清理修订尚待实际验证。
累计已计量6.250278GPUh，另有旧亚秒占用未知；本阶段新增付费API调用0。
最终基础测试94通过/24跳过；原生数值RPC测试1项通过。完整成功测量、退出状态、来源
及费用：[222176归档](results/updated-vllm-handoff-222176/result.json)。
权重更新/规范导出/新进程推理链路已验证，完整MH→后续RSFT交替与全部科学门仍未通过。

## E18 — 原生恢复入口与参数变化审计（2026-09-09）

恢复222151在74秒后因Ray工作目录的相对路径问题失败；222154在202秒后因
原RLHFDataset全零占位字段被严格schema拒绝而失败。两次均0新增模型调用/0更新，
分别计0.041111与0.112222 GPUh。完整结果、源码和日志分别归档。
修复异目录路径及zero uint8[N,1]占位字段后，完整原生CPU循环通过：真实数据集、
原8-worker DataLoader、64条缓存、原reward与accepted筛选、原actor分发入口，
实际2条accepted/16258loss token/mini8/micro1一致。GPU RPC、save和NCCL为明确测试桩，
不能据此声称真实更新或保存。尾部裁剪的混合Qwen原actor损失/梯度3项再次通过。
基础环境94通过/21跳过。v3冻结计划启动检查通过；222156于19:50:12 HKT启动，
2H100/20分钟上限。此前项目已计量5.896389 GPUh，此作业待结算。

新增reviewed transition比较器，绑定已审查15个inactive MTP名字及输出共享别名，
逐张量分别比较原始BF16→原生FP32及原始BF16→导出BF16，并要求导出恰为native BF16 cast。
原生3项针对性测试通过；真实checkpoint仍待产生。CPU导出脚本使用原生FSDP merger，
预设4CPU/64GiB、0GPU/20分钟，尚未提交。该部分仅对应E4测量，不改E1–E3。


## E17 — 完整原生批次审计与中断步骤恢复（2026-09-09）

221967失败成本1.280556 GPUh完整保留，完整64条记录独立审计PASS：69次调用、
520256输出token、2条成功，56预算停止/4格式错误/2错误落子。失败在原生log_softmax
临时分配15.16 GiB；没有已完成更新。源码与请求/张量/日志归档于
`results/native-rsft-pilot-221967/`。项目已计量GPU成本5.743056 GPUh。

E4恢复保留原始64条批次和原生成功筛选，仅在实际2条accepted batch上裁全无效的
响应尾部，宽度16384→8144；提示4096、loss token16258、mini8/micro1均保持。
原训练目标仍为 `-sum(mask * log_probability) / sum(mask)`。小型4层混合Qwen3.5
通过原actor损失与梯度对照（FP32），数据布局/失败关闭测试通过。左侧补齐裁剪曾导致
损失差0.000465393而未通过容差，不采用该路径，失败日志可查。

原生8-worker StatefulDataLoader而非独立sampler才能还原此批次顺序，已实测一致。
完整配置解析包含`model_engine: dp`；运行时迁移经原批次配置哈希核验。
恢复使用同一未更新base；只返回缓存一次，禁止新采样，不把69次原调用再次计为新调用。
保留原trainer的save与更新后NCCL同步。2H100/20分钟上限0.666667 GPUh。首个恢复222151因worker相对路径问题在74秒后退出，
尚未初始化GPU模型，0新增采样/0更新，成本0.041111GPUh归档。路径修复已通过异目录
实际恢复子类测试；v2作业222154于19:42:08 HKT启动，实际更新待验证。
基础测试93通过/16跳过，新原生裁剪3项、批次恢复1项、配置迁移1项通过。
此阶段验证工程恢复，不构成新数据、独立种子、留出集提升或VETO机制证据。

## E-20260909-16: original merger tensor API compatibility

**State: SYNTHETIC_EXPORT_PASS; ACTUAL4B_EXPORT_PENDING.** A tiny, randomly
initialized Qwen3.5 conditional-generation fixture was saved in the native
single-rank FSDP layout and passed through the released `FSDPModelMerger`.
Reloading all35 active state-dict names produced exactly the expected BF16
values. The output head alias was serialized separately and re-tied on reload.
Tokenizer/processor export functions were replaced with None in this test;
no actual4B checkpoint, GPU training or model inference was performed.

This one native-runtime test passed in10.330s. It verifies the tensor API path
with the installed Transformers version; it does not replace full export,
configuration/tokenizer checks, parameter-delta measurement or fresh inference.
Evidence: [result and source identities](results/native-checkpoint-export-fixture-20260909.json),
[original output](results/native-checkpoint-export-fixture-20260909.log).
The unchanged job221967 remained live around27minutes with40 complete returned
calls and0 journal errors. No complete tensor batch/checkpoint was available.
This is E4 shared execution preparation, not a scientific result or new mechanism.

## E-20260909-15: recorded training-batch audit preparation

**State: UNIT_CHECKS_PASS; FULL_LIVE_BATCH_PENDING.** The same native job221967
remained RUNNING after16m16s. The live prefix contained69 request starts,24
complete returns,0 recorded errors and189838 returned usage tokens; unfinished
requests have unknown consumption. This is not a completed-batch result.

`ours/audit_native_training_batch.py` checks a closed batch receipt and exact
schema, consumes matching recorded messages/replies with their actual sampling
parameters, and replays the original AgentLoop on CPU. It compares padded inputs,
responses, loss/attention masks, reward locations and attended Qwen position IDs.
Separate board/action checks do not invoke the native verifier. Native fallback
parsing from unfinished reasoning remains explicitly acknowledged. No optimizer,
parameter transition or performance claim follows from these checks.

Eight targeted tests passed in the native runtime; base runtime passed7 and
skipped the chess-dependent test. They cover loss-mask corruption, reward/position
changes, receipt/shape integrity, duplicate reply matching under distinct budgets,
incomplete journals and independent board rejection. Full live64-row CLI replay
and its configuration-bound schema check remain pending until the receipt exists.
[Validation and request snapshot](results/native-batch-audit-validation-20260909.json).
No new model calls or GPU jobs were made by these checks. Running-plan sources
remain unchanged. This is E4 evidence support; E1–E3 and scientific gates are unchanged.

## E-20260909-14: native accepted batch, optimizer OOM and bounded remediation

**State: NATIVE_ACCEPTED1_OF64; NO_COMPLETED_UPDATE; SCALAR_OPTIMIZER_PROBE_PASS;
V5_JOB221967_STARTED.** Job221921 completed64 new trajectories, with67 native
calls and520256 returned usage tokens. The native verifier accepted one row.
An independent limited dump review verified eight occurrences of each declared
training board and checked that the accepted002vV move a1h1 is the reference
move and legal checkmate. Its last decoded response has no thinking terminator:
the original fallback parser accepts a move from unfinished reasoning. This is
native verifier success, not a finalized-answer claim. The default decoded dump
lacks exact request boundaries, token IDs and masks; full replay is unavailable.

The first AdamW step raised CUDA OOM in `_multi_tensor_adam` at foreach sqrt,
after backward. It requested432MiB with19.06MiB free. No completed optimizer step,
checkpoint or trained-weight handoff was produced; in-memory optimizer state may
have changed before the exception. End18:21:03HKT, runtime2290s on2H100,
1.272222GPUh. [Complete archive](results/native-rsft-pilot-221921/result.json).

`probe_native_optimizer.py` invokes the original ZeRO-1 optimizer factory with
explicit foreach modes. CPU fixtures matched after three steps per mode. The
bounded single-H100 probe221964 used synthetic tensors with all723 independent
4B shapes, text-only gradients and6GiB CuPy transport reservation. Scalar AdamW
completed in the16-second job; peak Torch allocation73719085056bytes. This checks
optimizer memory only, without native FSDP activations or model forward/backward.
Its0.004444GPUh brings known project cost to4.462500GPUh, excluding the historical
unmeasured subsecond allocation and new job. No model/API calls came from probes.

The pilot launcher now supplies only two new resolved overrides: foreach=false
through the existing native optimizer config, and the previously CPU-tested
recording hook. [Complete config comparison](results/native-rsft-pilot-config-diff-20260909-v5.json)
and the original startup gate passed. The [v5 plan](results/native-rsft-pilot-plan-20260909-v5.json)
binds source/config/probe identities. Job221967 started18:32:36HKT with the same
two-GPU layout,90-minute cap and64 fresh trajectories. Live complete training,
recording, export and updated-checkpoint inference are still pending. The change
belongs to E4 shared execution; it must be common to future comparison conditions.
No scientific gate passes and no VETO performance improvement is claimed.

## E-20260909-13: checkpoint graph review during the native training pilot

**State: PRE_EXPORT_RULES_REVIEWED; ACTUAL_EXPORT_PENDING.** The original4B
safetensors contain738 names, while the installed HF meta model has724 state-dict
names. All723 common names have matching shapes. The15 stored-only names are
the exact `mtp.*` parameters ignored by the installed HF and vLLM loaders;
the sole active-only name, `lm_head.weight`, is tied to the language embedding.
The frozen pilot disables MTP. The reviewed comparison must cover all723
independent active names and verify any serialized alias equality; arbitrary
missing keys cannot be discarded. The original FSDP merger also converts tensors
to BF16, so FP32 optimizer deltas and exported deltas must be distinguished.

This CPU meta-graph/source inspection loaded no model weights and made no model
or API calls. The strict transition checker is unchanged and still rejects
different name sets. Adaptation must wait for the actual export and keep the
reviewed exclusions explicit. All running-plan source hashes remain unchanged.
At18:20HKT job221921 was still RUNNING,37m26s, with no complete batch/checkpoint.
No optimizer update, export success or scientific gate is claimed. This is E4
measurement preparation; E1–E3 are unchanged. No new tests are needed for this
documentation-only change; the previous test results are not relabelled as reruns.

Evidence: [meta graph](results/native-checkpoint-graph-preflight-20260909.json),
[source-bound review](results/native-checkpoint-graph-review-20260909.json),
[Slurm snapshot](results/native-rsft-pilot-221921-slurm-1820.txt).

## E-20260909-12: native recording support and successful-retry replay

**State: RECORDER_CPU_PASS; LIVE_RECORDING_PENDING; JOB221921_STILL_RUNNING.**
The same job remained RUNNING at18:10HKT (about27 minutes), with generation-GPU
activity measured at18:03HKT and no complete rollout batch yet. Its frozen sources are unchanged; it is not restarted
or retroactively instrumented. The previous goal turn made concrete progress:
complete bootstrap audit, native startup repairs and an initialized training job.

Original `_postprocess` exports only numeric reward fields to the default rollout
dump, so per-turn events, termination strings and response masks are lost there.
`ours/native_training_trace.py` supplies subclasses around the original Chess
chat manager and disaggregated trainer. Requests are journaled before invocation;
responses/errors are recorded without changing returned objects. Exact padded
tensors, masks, events and selected provenance are compressed, with a hash receipt
written only after the batch closes. An unfinished/error/partial journal keeps
unknown usage explicit. Accounting summarizes records read, not job completion.
These private training traces contain reference metadata and cannot be sent to
the proposer as-is. Token-ID `generate` calls are outside this chat recorder.

Seven tests passed, including delegation through the actual native class hierarchy
using fixtures, unchanged padded tensors/masks, exception/cancellation propagation
and incomplete accounting. CPU hook import/idempotence and Hydra override parsing
passed. Actual distributed Ray recording is not exercised. The first test used
the wrong upstream class name; the first CLI check misplaced positional overrides
after Hydra flags. Both failed logs are retained. Base suite:86 pass,10 skips.
[Validation](results/native-training-trace-validation-20260909.json).

The archived seed46 group also passed original AgentLoop replay:8 examples,
10 calls replayed,1 accepted (with a format retry),0 new model calls or updates.
Original generated IDs equal native reencoding for4/8 cases; no exact-token claim
is made for the other four. This extends the earlier success-without-retry check.
[Replay](results/native-agent-loop-replay-221898-seed46/result.json).
Everything here supports E4 measurement; E1–E3, upstream and running job sources
are unchanged. No extra GPU allocation, paid proposer request or scientific gain.

## E-20260909-11: complete bootstrap audit; native one-step training started

**State: BOOTSTRAP_AUDIT_PASS; RSFT_JOBS221906_AND221907_FAILED_STARTUP;
JOB221912_PROACTIVELY_CANCELLED; JOB221921_RUNNING;
NO_VERIFIED_WEIGHT_UPDATE.**
Job221898 completed normally at17:19:45HKT after2204 allocated seconds on two
H100s (1.224444GPUh). The complete independent audit verifies all64 trajectories,
eight prompts, seeds42–49, requests, tokens, board transitions and artifact hashes.
Two trajectories solve two distinct training puzzles. There were70 model calls,
520237 output tokens and63 call-level length stops; terminal trajectory reasons
were53 assistant-token-budget stops,2 solved,4 malformed and5 wrong moves.
These are training-only diagnostics, not scores on the earlier eight inputs or
held-out results. The full byte-identical archive is under
`results/chess-bootstrap-221898/`; no Stage A extension to256 is launched.

The [frozen native RSFT plan](results/native-rsft-pilot-plan-20260909.json), SHA256
`6c37edfae17b5e6ee68946709ac0fe01552792d5022ab44f49dcafab73758be4`,
requires the complete nonempty bootstrap audit and source/runtime/checkpoint
identities. The actual resolved Hydra configuration is compared with the reviewed
configuration at job startup; only the job-dependent output name may differ.
CPU gate, config validation and native data loading pass. Job221906 started
17:23:31HKT on gpucluster-g18 with two H100s,16CPU/160GiB and a90-minute/3GPUh cap.
One GPU is the native actor and the other the rollout worker. A fresh eligibility
check showed this layout could start immediately; an earlier three-GPU test was
denied by QOS. ALL email notification is configured, delivery is not independently
verified. No paid proposer is involved.

Job221906 passed its real startup gate and constructed the4B actor/optimizer,
then failed because `Checkpoint engine nccl not registered`. An explicit CPU
import exposed `ModuleNotFoundError: No module named cupy`; the native package
had swallowed this optional ImportError. No training trajectory or optimizer
update occurred. Its87 allocated seconds cost0.048333GPUh; the complete source
snapshot, run logs and failure summary are under `results/native-rsft-pilot-221906/`.

Following the [official CUDA13 CuPy installation instructions](https://docs.cupy.dev/en/stable/install.html),
the single CuPy14.2.0 wheel was hash-pinned in an additive checkpoint dependency
lock and installed without modifying existing packages. Native NCCL engine import,
registry identity, NCCL library version22809 and pip dependency consistency pass.
The CPU startup gate now imports the selected native transport explicitly.
The [v2 plan](results/native-rsft-pilot-plan-20260909-v2.json), SHA256
`9210225f1f1e98f7adc45756154b2ecef4468689004bbc8b495abbaaa7a3bd1e`,
preserves the original resolved training configuration and records the failed
attempt. CPU preflight passes; job221907 started17:28:15HKT with the same resource
cap. All training and actual weight-propagation claims remain pending.

Job221907 then initialized the vLLM service and AgentLoop workers but failed at
the initial actor-to-rollout transfer. The native NCCL engine transports complete
tensors; the engineering128MiB bucket was too small for the2542796800-byte FP32
embedding tensor. This was our earlier engineering override; the original launcher
defaults to3072MiB.211 allocated seconds cost0.117222GPUh, with0 generated training
trajectories and0 updates. Complete sources/logs are archived under
`results/native-rsft-pilot-221907/`. Its failure note initially proposed2560MiB;
the final reviewed correction instead restores the original3072MiB default.

The CPU gate now reads safetensors headers and computes the largest tensor at
the actor's configured precision. It rejects undersized buffers before submission.
Two tests verify that stored BF16 size cannot substitute for FP32 transport size
and that unknown precision is rejected. Full resolved-config comparison confirms
that128→3072MiB is the only Hydra change. The [v3 plan](results/native-rsft-pilot-plan-20260909-v3.json),
SHA256`16570532220b702f6fda43eb1fc9e070ab451afee86efffb0e54bc77814233e2`,
passed CPU preflight. Job221912 started17:35:08HKT with the same sampling, loss,
data, model and90-minute/two-H100 resource bound.

Job221912 completed the original initial weight-sync call and entered Online RSFT
generation. Further source review found that native asyncio.gather submits64
tasks to eight generation slots, while the420s HTTP timeout includes queue waiting.
The run was proactively cancelled at17:40:04HKT to avoid the predicted queue-timeout
failure; **no timeout was observed**.296 allocated seconds cost0.164444GPUh.
No complete native rollout batch was saved, and actual call/token counts remain
unknown. The first metrics probe used the wrong loopback address; the correct
node-IP endpoint responded but exposed no generation counters with native
disable_log_stats. These monitoring limits cannot be replaced with zero costs.
No optimizer update occurred. Its exact source is local commit`dfb23e7` and its
run/plan/stop record is under `results/native-rsft-pilot-221912/`.

The [v4 plan](results/native-rsft-pilot-plan-20260909-v4.json), SHA256
`f6d17d845a667c6d70d6c39f9c3714348273bcb1b8523a068b8f28ed15ea41df`,
changes only the communication deadline to4800s (original launcher600s, prior
engineering420s). Full resolved-config comparison and CPU gate pass. This is
an explicit queue allowance for the bounded eight-slot pilot, not an increased
token budget. The90-minute job cap remains. Job221921 started17:42:53HKT and
collects64 fresh trajectories; earlier interrupted generation is additional work,
not erased from cost accounting. Initialization weight sync does not establish
post-optimization propagation or tensor-value equality.

The original trainer collects64 fresh on-policy trajectories from the same eight
training prompts and performs at most one RSFT iteration. This is additional
collection cost, not cached-bootstrap replay. A nonempty accepted set, finite
nonzero optimization, actual tensor changes, HF export and live weight propagation
remain to be verified. Validation is disabled for this engineering step.
`ours/check_checkpoint_transition.py` compares tensor values in bounded CPU chunks;
serialization, metadata and exact dtype conversion alone are not weight updates.
These checks correspond to E4 execution and measurement, not a new VETO formula.

Base tests:81 pass and8 native-dependency skips. All eight native tests separately
pass in the training environment, including malformed/illegal UCI distinctions,
real tensor changes, serialization-only changes and nonfinite rejection.
Known project GPU use including both failed starts and the intentional stop totals
3.185833GPUh, plus unmeasured old job221576 subsecond use and pending221921. No API spending changed,
upstream remains clean, no scientific gate passed and nothing was pushed.

## E-20260909-10: native success sampling running; training interface replayed

**State: JOB221898_RUNNING; NONEMPTY_PARTIAL_SUCCESS_SET; NO_WEIGHT_UPDATE.**
The [frozen plan](results/chess-bootstrap-plan-20260909.json), SHA256
`25ce73d4a111a3880c3c18108738d1cd3615ae6f1e60f958227badb6decdb861`,
selects the first32 lexicographically sorted IDs from the existing128-case pilot
training split. Chunk0 takes its first8 prompts, each with seeds42–49. Two H100
replicas get four seeds each and use unchanged native h0/runner/verifier and the
same frozen4B checkpoint. No proposer API is called. Job221898 started16:43HKT
with16CPUs/96GiB, ALL email notification and a one-hour/2GPUh allocation cap.

At17:03HKT four seed groups have completed32 trajectories, with2 native complete
successes. These solve two distinct training puzzles and passed a
[partial independent board replay](results/chess-bootstrap-221898-partial-success-audit.json).
This is not a completed64-trajectory audit, an improvement over the old inputs,
or a held-out result. A first manual partial replay incorrectly assumed that a
successful trajectory cannot contain retries; the corrected replay checks the
retry budget before continuing. The general chunk auditor handles retries.

[Offline native training AgentLoop replay](ours/replay_chess_agent_loop.py)
uses the archived server responses, checks every requested message/sampling cap,
and verifies reward agreement and nonassistant token masking. It passed the
historical8-case h0 set (0 accepted) and the completed new seed42 set (1 accepted).
This does not run a model or optimizer. Native reencoding preserves original
generated token IDs on7/8 and3/8 trajectories respectively; the differing cases
are recorded, not relabelled as exact token-sequence equality.

The initial sandboxed CPU replay was interrupted; a second90-second attempt
stalled with an idle executor and was killed after its timeout grace period.
The host-IPC attempt advanced immediately and revealed a diagnostic schema
error: AgentLoopMetrics is not a dictionary. Accessing the native reward-info
field fixed that error. All attempt logs are retained; these failures are not
low model scores. No claim isolates the exact sandbox syscall responsible.

The [native4B pilot config](results/native-rsft-pilot-resolved-20260909.yaml) and
[dataset preflight](results/native-rsft-pilot-dataset-preflight-20260909.json) pass.
The original trainer will collect64 fresh on-policy trajectories and run one
RSFT iteration, with validation disabled for this engineering step. This extra
collection is separate from the bootstrap feasibility sample. It cannot count
as an update if the accepted batch is empty. GPU training, optimizer memory,
checkpoint export and service weight propagation are not yet exercised.

81 existing CPU tests and3 independent malformed/illegal/legal UCI checks passed.
All implementation changes are in `ours/`; E1–E3 and upstream remain unchanged.
The running job's final GPU usage is pending. The next action is to finish and
audit the same job221898, then freeze and submit the native training step;
do not restart a live job or automatically extend sampling to256 after success.

## E-20260909-09: native GLM search completed; no accuracy improvement

**Conclusion: ENGINEERING_SEARCH_AND_REPLAY_PASS; NO_SCORE_GAIN; NO_RSFT_UPDATE.**
The unchanged native `run_evolve` completed one logical harness-only iteration
using GLM-5.3-Flash and the frozen local Qwen3.5-4B target, with VETO off.
Three jobs contributed distinct stages; their failures and resource use remain
part of the record. This is eight previously accessed engineering cases, not
an original-paper result, a three-seed comparison or a VLM experiment.

| Harness | Solved | Policy calls | Output tokens | Length stops |
|---|---:|---:|---:|---:|
| h0, unchanged native baseline | 0/8 | 8 | 65032 | 8 |
| h1, actual GLM candidate | 0/8 | 8 | 65032 | 8 |

Native selection accepted h1 because accuracy and mean calls were tied and its
results appeared first in filesystem traversal. This is **not an improvement**.
Independent replay found eight format retries for h0 and seven format retries
plus one legal wrong move for h1. All input token IDs, output decoding, usage,
board transitions and native verdicts matched. The first audit incorrectly
assumed h0 wins every tie; its failed log and source are retained. The corrected
audit verifies objective optimality and tied-set membership, without claiming
that an unrecorded filesystem enumeration was made deterministic. Formal
comparisons require a shared reproducible tie policy before VETO attribution.

- **221733:** h0 completed, then the CLI could not reach the local relay because
  Slurm inherited login-node HTTP proxies at127.0.0.1:27182. An unauthenticated
  direct HEAD from gpucluster-g18 reached the official GLM endpoint with HTTP401.
  The disconnected proposer was stopped; no paid request reached the relay.
  A secondary import guard encountered the CLI's own temporary `latest` symlink.
  FAILED,749 allocated seconds on1H100,0.208056GPUh.
- **221743:** proxy handling and temporary-state isolation were corrected.
  GLM completed its proposal in102.31seconds with CLI exit0. Seven real requests
  used5978 input,24896 cache-read input and2029 output tokens:0.038990CNY under
  the conservative project rates. Import rejected `slot:h1` because native full
  mode expects `name:h1`. Candidate bytes and original metadata were retained.
  FAILED,186 allocated seconds,0.051667GPUh.
- **221754:** imported the exact h1 after strict, unambiguous slot-to-name
  normalization and native source validation, then evaluated and selected it.
  It reused complete hash-bound h0 through native caching and made no new API
  calls. COMPLETED,600 allocated seconds,0.166667GPUh.

The total is0.426389GPUh for this milestone, including both failures, and
1.631389known project GPUh overall; old job221576 has unmeasured subsecond usage.
The persistent journal now totals15 real requests including the original direct
probe,0.047257CNY conservatively accounted, no outstanding reservation, and a
45CNY cap. Neither this estimate nor the CLI's USD estimate is a provider invoice
or account balance. Reprinting a saved CLI session during import incurs no new
model request. The successful authenticated calls establish this node's GLM
reachability, not unrestricted connectivity on every compute node.

The candidate changes only SYSTEM_PROMPT, USER_PROMPT and whitespace handling
in format_observation. Parsing, legality checks and retry/turn limits are
unchanged. GLM read the baseline and aggregate frontier, but did not inspect the
rollout traces; its failure-cause hypothesis is not validated. h0 and h1 use the
same data and sampling settings, but the target service restarted between them.
The default proposer, effort, search size and runtime compatibility differences
are declared in the [final frozen plan](results/native-search-plan-20260909-v3.json).

Evidence: [independent search audit](results/native-search-221754-summary.json),
[candidate source review](results/native-search-candidate-review-20260909.json),
[network diagnosis](results/native-search-network-diagnosis-20260909.json),
[budget summary](results/glm-budget-summary-20260909-search.json), and the complete
archives under `results/native-search-221733/`, `221743/` and `221754/`.
The first two driver/launcher versions are archived with their failed jobs;
their frozen plans must not be checked against the newer driver.

In parallel, the original converter/splitter prepared128 train/32MH-validation/
64test pilot cases from pinned Lichess, excluding all64 previously inspected IDs
and position groups. No model-based filtering occurred. An independent pass
verified disjointness, all trigger positions and790 legal solution plies.
[Manifest](results/chess-pilot-manifest-20260909.json),
[data audit](results/chess-pilot-audit-20260909.json). These inputs were not used
in this search.81CPU tests passed in the documented base test environment;
native imports and prepared-candidate validation passed in the isolated training
environment. Initial diagnostic commands using the wrong test/runtime interpreter
failed before being corrected; they were not model runs.

Stop repeating the same eight-case decode check. Next use the isolated pilot for
nonempty native RSFT, checkpoint export and actual weight propagation, followed
by the four original-domain conditions with three seeds. E1–E3 are unchanged;
all new code is shared execution/data measurement. F0–F5 remain unpassed, no
weights were trained, the original and old project were not modified, and no
GitHub push occurred.

## E-20260909-08: released default4B on the same engineering inputs

**Conclusion: RUNTIME_AND_INDEPENDENT_REPLAY_PASS; ZERO_SUCCESSFUL_TRAJECTORIES.**
Job221725 completed on one H100 in670 allocated seconds (0.186111GPUh), exit0.
The [frozen plan](results/chess-vllm-4b-plan-20260909.json) replaces only the model
size/checkpoint from the2B vLLM diagnostic; dataset, harness, seeds42/43, thinking,
temperature1/top_p1/top_k20 and8129 budgets remain fixed. Four requests run per
shard. All22 source/input dependencies are hash-bound; this is an engineering
comparison with eight previously accessed puzzles, not original paper accuracy.

The4B model completed8 calls and65032 output tokens. All8 hit the token limit;
all8 raw outputs were ambiguous to the unchanged parser, yielding one format
retry followed by exhaustion of the assistant budget.0 solved,0 correct moves,
0 accepted RSFT trajectories. Native legal_rate=1 again does not measure actual
move legality. The [independent audit](results/vllm-runtime-221725-summary.json)
replayed all actions and matched prompt token IDs, decoded output, usage and
native verdicts. All logs and controller records are archived in
`results/vllm-runtime-221725/`.

The original AgentLoop formatting helpers were also executed on all8 prompts
and matched the4B processor/tokenizer inputs exactly. During CPU plan preparation,
passing plain string messages to the newer processor's tokenize=True interface
raised TypeError. Structured text messages fixed that diagnostic, and their
tokens matched plain-text tokenizer input. The native AgentLoop uses
tokenize=False followed by processor(text=...), which passed unchanged. These
are API-format checks, not new model mechanisms.

No training or Meta-Harness optimization occurred. No new paid API calls were
made: the persistent conservative journal remains0.008267CNY, held0, limit45CNY.
Known project GPU use including failed jobs is1.205GPUh;221576subsecond time is
still unmeasured.4B did not resolve this fixed-input/budget failure, but eight
cases cannot establish population performance or rule out sampling/data effects.
Next is the original bounded Meta-Harness search and evaluation integration;
nonempty RSFT and actual checkpoint propagation remain prerequisites for claims.

The2B source snapshot is local commit`476baed`. Its frozen launcher hash differs
from the current optional-plan launcher; replay its source checks at that
snapshot. The4B plan binds the current launcher. No upstream or old-project files
were changed, and no push occurred.

## E-20260909-07: vLLM CUDA execution and native RSFT preflight

**Conclusion: GPU_RUNTIME_AND_TRACE_AUDIT_PASS; NO_SUCCESSFUL_RSFT_TRAJECTORY.**
This is E4 shared execution infrastructure; E1–E3 remain unchanged. No training
step, proposer search, scientific gate or method improvement is claimed.

Job221663 completed on1H100 in472 allocated seconds (0.131111GPUh). The fixed2B
checkpoint and same eight previously accessed Chess inputs were evaluated by the
unchanged native runner, harness and verifier. Two consecutive four-request shards
used request-local seeds42/43, temperature1/top_p1/top_k20, thinking enabled, and
8129 total/per-call assistant tokens. vLLM0.20.0/torch2.11.0/Transformers5.5.2
completed8 calls,65032 output tokens,8 length stops,0 puzzles solved. Independent
Python-chess replay found5 format retries,2 illegal retries and1 legal wrong move.
Every input token ID matched local chat-template tokenization; every returned
output token sequence decoded to the recorded response. All sources, dataset
bytes, plan and result artifacts are hash-bound. See the
[audit](results/vllm-runtime-221663-summary.json) and
[frozen plan](results/chess-vllm-runtime-plan-20260909-v3.json).

The earlier HF run and this vLLM run both exhausted all eight budgets. Changed
backend versions, batching and RNG streams preclude a controlled backend accuracy
claim. These are eight engineering cases, not an original paper split or baseline
score. Empty acceptance cannot demonstrate an optimizer update or weight handoff.
The next model to check is the released default4B, rather than changing puzzle
answers or selecting successful outputs after seeing them.

The [official Qwen3.5-4B model](https://huggingface.co/Qwen/Qwen3.5-4B) is pinned to
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. All11 runtime files were downloaded;
both safetensors shards (9,319,828,096bytes total) match official LFS SHA-256 values.
The combined asset/weight identity is
`bba57bcc77d3f063465bc0b62560247253898c099f5587ec815de3e95c65bbdc`.
[Checkpoint manifest](results/qwen35-4b-checkpoint-20260909.json). No4B inference
occurred at this milestone; downloading an additional model is not an experiment.

Failures are retained with their original plans and driver/launcher snapshots:

- 221641:45 allocated seconds, configuration validation failure before generation.
  With chunked prefill disabled, batched tokens16384 did not cover context32768.
  vLLM also warned that Qwen3.5 does not support disabling chunked prefill. The
  corrected plan enables it; this is a declared common runtime deviation from the
  release launcher default. Scheduler validation now runs before submission.
- 221643:330 allocated seconds, one request began but timed out at180s. No complete
  response was returned, and actual generated-token count is unknown. Observed
  single-request throughput was approximately43 tokens/s, insufficient for8129
  tokens within180s. No numerical score is recorded for this incomplete request.
- 221663:420s request timeout, four concurrent requests per shard, one shared GPU.
  Observed total generation throughput was approximately164 tokens/s while all
  four requests were active. This is a runtime log observation, not a controlled
  throughput benchmark. No other task or scientific setting was tuned from scores.

All three jobs had20-minute caps, ALL notifications and no requeue. Controller
records were captured before purge. Their total allocated cost is847/3600=
0.235278GPUh. Known project GPU time is1.018889GPUh including these failed attempts;
the earlier221576 subsecond failure remains unmeasured. No new paid API calls;
the GLM conservative journal remains0.008267CNY against45CNY authorization.

The native RSFT launcher is prepared but has not been submitted. Hydra originally
failed because `data/legacy_data.yaml` is absent in all three released domains.
The exact `data` mapping from the pinned Chess generated config is now in
`ours/config/data/legacy_data.yaml`; Chess/Math/SearchQA generated mappings are
structurally identical. See [restoration evidence](results/legacy-data-restoration-20260909.json).
The worker setup hook applies the same declared `verl.tools` package restoration
in fresh Ray workers. No upstream source has been changed.

Complete Hydra resolution and native `validate_config` passed. Native dataset
loading also passed for8 rows using Qwen3VLProcessor, with independent prompt
lengths308–747 tokens. It returns raw prompts and a dummy tensor; the AgentLoop
constructs training tensors later. The first CPU attempt hit sandbox IPC denial;
the second diagnostic wrongly expected immediate `input_ids`; the third counted
BatchEncoding fields as tokens. All attempts are preserved, and explicit
`return_dict=False` corrected the final token count. These were preflight defects,
not measured model failures. The final
[dataset report](results/native-rsft-dataset-preflight-20260909.json) records the
actual contract. Actual Ray GPU training and checkpoint-to-server transfer remain
unverified.76 offline tests passed in0.99s, including overlapping local requests
and serialized, consistent usage records.

## E-20260909-06: official GLM proposer transport

**Conclusion: PASS_TOOL_TRANSPORT_ONLY.** The user authorized the domestic Zhipu
API and45 CNY first-round budget. The direct256-token-cap request returned
`glm-5.3-flash`,17 input/26 output tokens and the requested OK text in1.08s.
The documented [Anthropic-compatible endpoint](https://docs.bigmodel.cn/cn/guide/develop/claude/introduction)
was used; the [official model page](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash)
confirms the model code and thinking/tool support. No credential is in this repo.

The signed official Claude Code2.1.236 binary was installed only under `data/`.
Manifest signature and binary hash passed against Anthropic's published signing
fingerprint; [installation record](results/claude-runtime-install-20260909.json).
The original WHALE wrapper builds/executes the CLI and parses its events. The
probe only supplies process-local environment, a budget relay, bounded turns/time
and an extra filesystem restriction. It does not invoke a formal search loop.

Automatic approval initially rejected potential private-source export. A fresh
fetch of the pinned public GitHub base harness proved exact byte equality. The
Linux Landlock launcher passed an allowed-read/forbidden-read canary check, and
the bounded public-file probe was then approved. Only the disposable workspace
is writable and other user files are unreadable. ABI1 filesystem controls do not
claim network or metadata-operation confinement. The real API key stays in the
parent relay; the native child receives only an ephemeral loopback token.

Attempts1–4 each completed one real GLM request and one successful file Read,
then timed out (180/180/90/90 seconds). No file was changed. Diagnostic attempts
3/4 captured complete SSE, including message_stop. The official Python SDK parsed
the saved response successfully with0 external requests. A separate loopback-only
replay with Content-Length completed the native client's two turns immediately.
Adding an explicit Connection: close header alone had not fixed the live stall.
The final relay buffers the provider's exact response body and supplies its
Content-Length; it does not alter model text or token counts. This changes time
to stream delivery, not the proposed content. After the fix, attempt5 made3 API
requests and completed Read→Edit→DONE. AST checks proved that only the specified
USER_PROMPT sentence changed; no improvement was requested or measured.

Across the direct probe and all paid attempts: **8 real requests,4831 input plus2176 cache-read input and
315 output tokens**. The persistent journal's conservative1/4 CNY per million
input/output rates give **0.008267 CNY**, no unresolved reservations. This is not
an account balance query or provider invoice. Before each request, the relay
reserves the full published context and requested output, enforces the project's
45 CNY limit, and leaves uncertain calls reserved. It rejects fallback models
and paid server tools. CLI dollar estimates are not used; the offline replay
itself displayed a nonzero estimated dollar cost despite making0 external calls.

Evidence: [passing tool probe](results/glm-proposer-tool-probe-20260909-attempt5.json),
[budget summary](results/glm-budget-summary-20260909.json), archived attempts/logs
under `results/glm-proposer-probes-20260909/`, and the offline SDK/framing records.
The specified-edit candidate is an engineering artifact, not an optimized harness.
Formal proposer-search iterations and training updates remain0.

Final CPU suite:72 passed in0.96s. New document links and a scan of156 publishable
files for the actual API credential passed. Raw logs retain their recorded bytes;
code/document whitespace checks are reported separately from raw artifacts.

## E-20260909-05: isolated training dependency candidate

The existing qwen-vl environment could not import the trainer because Ray was
absent. A separate historical environment imported the trainer but used
Transformers4.57.6/vLLM0.12.0 and was not modified. A new project-local Python3.12
environment was resolved and installed with237 packages, pinned download URLs and
SHA256 hashes; pip check passed. The high-level requirements explicitly declare
torch2.11.0, vLLM0.20.0, transformers5.5.2 and numpy2.2.6.

This is a compatibility variant: WHALE's numpy<2 conflicts with OpenCV>=4.13,
required by the evaluated Qwen3.5-capable vLLM versions. The minimal conflicting
pip dry-run exited1 with ResolutionImpossible; its log is retained. Earlier
vLLM0.17.1–0.19.0 metadata also caps Transformers below5, while0.20.0 permits the
chosen5.5.2. The candidate's successful resolution is not a claim that the
released requirements work unchanged. Primary constraints:
[vLLM0.20.0 metadata](https://pypi.org/pypi/vllm/0.20.0/json),
[OpenCV4.13 metadata](https://pypi.org/pypi/opencv-python-headless/4.13.0.92/json).

Actual imports passed for torch, Transformers, vLLM, the original disaggregated
trainer and actor classes; Qwen3.5's local config/model class was recognized.
The Chess agent-loop import initially failed: its release directory omits
`verl.tools`. `ours/training_bootstrap.py` explicitly imports that namespace from
the same pinned WHALE Math directory; schemas.py is byte-identical to SearchQA's
copy. The Chess trainer, workers, actor and agent-loop imports then all passed.
All fallback source hashes are recorded. No replacement PyPI verl is installed.
GPU kernels, actual RSFT, and live checkpoint propagation remain unverified.

Evidence: `results/training-{resolution,imports,imports-restored}-20260909.json`,
resolution/install logs, `ours/training_requirements.txt` and
`ours/training_resolved.lock`. This work belongs to common implementation details,
not a new Method term. Existing environments and upstream source were unchanged.

## E-20260909-04: full-budget local Chess decoding diagnosis

**Conclusion: COMPLETE_DIAGNOSTIC; NO_SUCCESSFUL_RSFT_TRAJECTORIES.** The same8
preselected engineering examples were retained as two disjoint4-example shards.
Two seeds42/43 initialize the shard streams and row order; these are not two
independent research seeds. Temperature1, top_p1, top_k20, thinking enabled and
8129 assistant/per-call budget restore the released JSON decoding values. The
local HF2B client still differs from the released4B/served-client setting, and
handling of separate reasoning in the missing original client is unknown.

Fresh queue tests predicted H800 starts tomorrow or later;4 H100s were rejected
by QOSMaxGRESPerUser, while1/2 H100s could start immediately. Two fixed shards were
chosen, with a30-minute parent limit and1 GPUh maximum. Both source/data/processor
prechecks passed before sbatch. Job221597 completed on2 H100s,13:44:30–14:07:26 HKT,
exit0:22min56sec, **0.76444 allocated GPUh** including idle time. ALL email settings
were configured; delivery was not independently verified. Driver580.178.04 and
81559MiB per GPU were read inside the allocation. Slurm later purged the job before
the raw controller text was saved, and sacct reports accounting storage disabled;
the resource summary transcribes the earlier successful live controller lookup.

All8 generations reached8129 tokens: **65032 output tokens,8 length stops,0 solved**.
Independent parsing found6 ambiguous-move format retries,1 illegal-move retry and
1 legal wrong move. Seven cases then exhausted their total assistant budget;
none executed a second call. The native legal_rate reports1.0 because it only
checks terminal illegal events, so it must not be presented as1.0 step legality.
The independent board legality count is1/8.

The audit verifies all planned IDs exactly once, source/data/checkpoint identities,
first-input hashes, every decoded raw token sequence, native event outcomes and
all saved artifact hashes. It explicitly rejects multi-call inputs and does not
claim a general multi-turn audit. Both full shards and input plans are archived.
Evidence: [audit](results/chess-decode-221597-summary.json),
[resources](results/chess-decode-221597-resources.json),
[pre-run study plan](results/chess-decode-study-plan-20260909.json).
The result is not a formal WHALE baseline, a controlled thinking comparison,
evidence of alternation failure, or VETO effectiveness. Do not run an empty RSFT
filter and count it as a training update. Historical221577 plans bind earlier
driver bytes and require the60b836c/a542169 snapshot rather than this shard driver.

## E-20260909-03: original Chess evaluator with a local completion client

**Conclusion: PASS_RUNTIME_CONNECTION_ONLY; F0_NOT_PASSED.** The released Chess
runner, harness and verifier executed without source changes. The missing client
was replaced by a declared local HF implementation in `ours/compat`; it was not
recovered from the original authors. No training or proposer search occurred.

- Model: [official Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`. Downloaded4,548,221,488 weight bytes
  plus processor/config/license files. The downloaded weights and tokenizer
  matched the official LFS SHA256 values. Content-bound identity and all asset
  hashes: [checkpoint record](results/qwen35-2b-checkpoint-20260909.json).
- Runtime: existing qwen-vl Python3.10.20, torch2.4.0+cu121, transformers5.4.0;
  isolated chess1.11.2 installed under `data/runtime-deps/` using its pinned PyPI
  source hash. Existing environments were not modified. Installation provenance:
  [pip report](results/chess-dependency-install-20260909.json).
- Data: Lichess/chess-puzzles revision
  `479ea9bc9f681385f5adb23fa27a96c2dc8ae599`; first64 raw records, then first8
  unique valid positions with <=9 solution plies, using the released converter.
  No selection used model output. The engineering manifest lists all64 accessed
  IDs for exclusion when future scientific splits are built; those splits do not
  yet exist. Final data: `data/chess-engineering-v3/engineering.parquet`.
- Data preparation attempts1/2 wrote valid files but exited134 during native
  thread cleanup. Explicit iterator.close alone did not fix it. This matches the
  failure described in [Arrow issue45214](https://github.com/apache/arrow/issues/45214).
  Synchronous `ParquetFile.read_row_group(use_threads=False)` with pre-buffering
  disabled completed with exit0. All64 source rows and the final8-row Parquet
  remained byte-identical across attempts. Failure evidence is preserved in
  `results/chess-input-preparation-20260909-attempt{1,2}.json`.
- Processor preflight initially had an overly strict field assertion: Qwen3.5
  returns `mm_token_type_ids` even for text. The corrected check verified all
  modality IDs are zero, all eight prompts contain the proper board, and saved
  actual token hashes. Input lengths310–749 are recorded individually in
  [the processor report](results/chess-processor-preflight-20260909.json).
- Failed job221576: orchestration continued despite the failed CPU preflight and
  submitted before the plan existed. It exited1 before model loading,0 model
  calls,0 whole seconds reported by Slurm; subsecond allocation is unmeasured.
  [Failure report](results/chess-smoke-221576-failure.json) and
  [controller record](results/chess-smoke-221576-controller.txt) are retained.
  The driver now has a CPU `--check-only` path. Subsequent submission explicitly
  required its successful return code; the earlier failure was not hidden.
- Successful job221577 used the frozen
  [run plan](results/chess-smoke-plan-20260909.json). One H800,4CPU/24GiB RAM,
  10-minute limit,ALL notifications,no requeue. Controller:13:21:26–13:21:51 HKT,
  exit0, **25 allocated seconds (0.00694 GPUh)**. Python elapsed21.62s; summed
  generation5.83s. Peak allocated4.258GiB, reserved4.377GiB. The optional fast
  linear-attention dependencies were absent; Transformers used its torch fallback.

| Fixed engineering configuration | Actual result |
|---|---:|
| Greedy, thinking disabled, total budget256, per-call128, seed42 | 8 examples |
| Model generations / output tokens | 8 / 98 |
| Legal first moves | 8/8 |
| Correct first moves / solved puzzles | 0/8 / 0/8 |
| Length-truncated generations | 0 |

Independent audit matched input token hashes to the CPU preflight, identified
eight distinct prepared FENs in actual messages, parsed raw moves independently,
checked legality with python-chess and compared them to the stored first solution
move. Every raw output, token count and wrong-move stop agreed with the original
runner's traces. Weight, plan, source and result hashes also matched.
Evidence: [raw generations](results/chess-smoke-221577-generations.jsonl),
[trajectories](results/chess-smoke-221577-trajectories.jsonl),
[full result](results/chess-smoke-221577-result.json),
[independent audit](results/chess-smoke-221577-summary.json),
[controller completion](results/chess-smoke-221577-controller.txt).

This result is not a WHALE baseline score: the client/backend, data subset,
temperature, thinking mode and token budget differ from the paper setting. All
cases ended on the first wrong move, so real multi-turn generation was not tested.
There are no successful trajectories here for RSFT; do not train on failed rows
or report an empty filtered update as training. Restore original decoding and
validate success/retry/multi-turn behavior before the next baseline stage.
The full test suite passed **66 tests in1.17s** and shell syntax passed. The earlier
Git whitespace check covered tracked changes only. Checking the full local snapshot
also flagged trailing whitespace in raw Slurm controller outputs and a final blank
line in the frozen compatibility namespace. Those recorded bytes are preserved
for hash verification; these formatting notices are not runtime failures. Proposer
calls, paid API calls and weight updates remain0.

## E-20260909-02: real local visual input check

**Conclusion: PASS_INPUT_CONNECTION_ONLY; JOINT_LOOP_NOT_READY.** This run used
real cached Qwen2.5-VL-3B-Instruct inference. It did not train weights, search a
harness, call a proposer, or pass the F0–F5 scientific gates.

- Pre-run plan: [visual-smoke-plan-20260909.json](results/visual-smoke-plan-20260909.json),
  frozen before submission. Fixed 8 engineering pairs, two image sides, visual and
  no-image conditions, seed17, greedy, at most8 new tokens: exactly32 generations.
  E1 input connection was the question being tested; E2–E3 were not exercised.
- Model: existing Qwen2.5-VL-3B-Instruct revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`; existing qwen-vl environment,
  torch2.4.0+cu121 / transformers5.4.0 / Pillow12.1.1. Offline loading only, no
  dependencies installed or weights downloaded. This differs from WHALE's model
  family and cannot be described as the original baseline.
- CPU processor preflight passed32 inputs. Each visual input had154 image tokens
  agreeing with its image grid and pixel tensor; no-image inputs had0. Questions
  and image pixels alone entered model messages; labels, IDs and paths stayed
  outside. Both CPU preflights are preserved under `data/visual-processor-preflight-v*`.
- Queue comparison before submission: 1/2/4 RTX4090 test-only jobs all predicted
  2026-09-10 15:07 HKT; one H800 and one H100 predicted immediate start. One H800
  was chosen for the small32-request workload,4CPU/24GiB RAM/10-minute limit.
  Multi-GPU sharding would add startup/merge work to a seconds-scale generation
  workload. These were point-in-time queue predictions.
- Job221552 completed with exit0, 2026-09-09 12:50:27–12:51:11 HKT: **44 allocated
  seconds, 0.01222 GPUh**, no requeue. Python elapsed31.99s; summed generation5.41s;
  peak allocated7.217GiB, reserved7.379GiB. This is not sustained serving throughput.
  ALL email notifications were configured for the latest project address; actual
  delivery is unverified. Slurm accounting storage is disabled, so completion and
  allocation use the saved [controller record](results/visual-smoke-221552-controller.txt).

| Engineering condition | Correct images | Correct pairs (E1) |
|---|---:|---:|
| Actual image input | 16/16 (100%) | 8/8 (100%) |
| No-image input | 8/16 (50%) | 0/8 (0%) |

All32 outputs were single A/B labels ending on EOS. No-image token inputs were
identical and the model always answered A. Independently recomputing direct label
equality and pair products from the raw ledger agreed with the saved E1 receipts.
The rescore also verified exact coverage, image hashes, frozen source files,
all model weight/processor asset hashes, and decoding identities.

Evidence: [raw generations](results/visual-smoke-221552-generations.jsonl),
[full result](results/visual-smoke-221552-result.json),
[independent rescore and artifact hashes](results/visual-smoke-221552-summary.json).
Local original outputs remain in `data/visual-smoke-221552/`; exported raw and
result files are byte-identical. Final CPU tests: **62 passed in1.09s**; shell
syntax and Git whitespace checks passed. No external API calls or cost occurred.

The easy engineering images have ceiling performance. They establish readable
input for this model and expose the loss of evidence when images are removed;
they do not show alternation-induced shortcuts, generalization, or VETO benefits.
Do not include this table in a method-performance comparison. A local visual
target run also does not measure a local proposer's coding or tool-use ability.
The baseline dependency and checkpoint-handoff gaps from E-01 remain unresolved.

## E-20260909-01: offline implementation and release preflight

**Conclusion: OFFLINE_MODULE_VERIFIED; END_TO_END_NOT_READY.** No research model
calls, training, GPU submissions or paid proposer calls occurred. No benchmark
performance or scientific gain is available.

- Hypothesis: Method E1–E3 can be expressed as one switchable acceptance module
  with an exact delegation path when disabled; incomplete/stale evidence must not
  influence selection. This entry checks software behavior, not the VETO thesis.
- Source parent: local `1bf0b6e`; pinned WHALE
  `fbe125eb7abea7f760c99ab9acc1a6261e708fc6`, unchanged. Source content hashes are
  recorded in `results/offline-verification-20260909.json` for this working snapshot.
- Environment: existing Anaconda base Python 3.12.7, pytest/Pillow; no installations.
  Read-only capability checks also used the existing qwen-vl interpreter. The
  snapshot records package versions. Base tests ran on CPU in approximately 1–2 s.
- Validation: 54 tests passed after final numerical-boundary review. Tests include E1
  correctness and manifest binding, stale checkpoint/data/decode/verifier rejection,
  same-data controls, Pareto-before-gate counterexample, disabled upstream selectors,
  resume JSON roundtrip/tampering, rendered image oracle, provider isolation and
  explicit absence handling, and billing arithmetic. These fixtures are not results.
- Data: 8 generated engineering source pairs, 16 PNGs; seed17; same question with
  different correct answers after swapping bar heights. Independent pixel oracle
  checked every saved PNG, and one example was visually inspected. Manifest identity
  `dcdef9a751c7f4be49281eede40421fb0f3db54852e0209cf7f009c39bf63aa7`.
  Local path `data/engineering-chart-pairs-v1/manifest.json`; role=engineering.
  No benchmark/train/validation/test dataset was opened or downloaded.
- Preflight: base and qwen-vl reports in `results/preflight-20260909-*.json`.
  They identify missing packaging metadata, math environment, SearchQA MH config,
  FST skills, Chess client, Claude CLI, and unverified live weight handoff. A later
  v2 report uses the direct SearchQA MH source reference instead of its launcher.
  qwen-vl has torch/transformers but lacks vllm/ray/chess/client.
- Actual documented-install check:
  `python -m pip install --dry-run --no-deps --no-index -e upstream/WHALE/domains/chess_puzzles/verl`
  rejected the directory because both setup.py and pyproject.toml are absent.
  Log: `results/upstream-editable-install-20260909.log`. No package was installed.
  This is a real packaging failure, not a numerical baseline failure.
- Account snapshot: 2026-09-09 around 12:18 HKT, queue empty; quota 222000 GPU-min,
  used45862, remaining176138 (2935.633 GPUh); concurrency4GPU/48CPU. No resources
  reserved. Queue/resource data are point-in-time, not a future availability promise.
- Costs: Claude/API spend0, allocatedGPUh0. Hypothetical Opus and GLM budgets are
  separate JSON artifacts with `kind=budget_scenarios_not_measurements`.

Reproduction commands are in `ours/README.md`. Next: select/configure proposer
billing/endpoint, restore traceable dependencies and verify checkpoint-to-server
handoff, then connect the shared visual environment. No scientific gate is passed
by these offline checks. Preserve all F0–F5 tests and all planned strong controls.

The user asked about cost and then local/GLM alternatives. Those questions did not
authorize a purchase or specify an API spending ceiling. GLM/local profiles were
prepared without calling a service; no credential value was read into any report.
# E100 — Corrected visual RSFT and fixed-h0 followup completed

Job224347: 64 fresh W trajectories, 46 accepted, 92 assistant loss tokens,
6 native optimizer updates. All46 real backbone inputs match the native successful
batch, with46 finite nonzero visual feature gradients. Job224358 verifies changes
in all723 native FP32 parameter tensors, including all297 vision tensors. Exported
BF16 changes remain in455 tensors (vision163/297), with exact native BF16 casts.

Same128 V images and h0: before and after both117/128 marginal and53/64 paired
correct. Four answers change. Native processor pixels/positions/prompts match
exactly. One fixed-harness engineering batch, no independent test or corrected
model repeatability result; no alternation effect or VETO advantage established.
Bootstrap95 intervals for differences: marginal[-3.125,3.125]pp, paired[-6.25,6.25]pp.
These condition on one model comparison, not independent seeds.

Corrected-weight candidate selection and selected-harness continuation remain
pending. Old theta1 acceptance must not be reused for a new weight identity.
Current pilot allocation cost6.898056GPUh, no pending reservations. Index:
`results/visual-corrected-engineering-milestone-20260910-v1.json`.

# E98 — Pixel-forwarding defect and corrected first-batch restart

The pinned WHALE `dense_common` fused backend silently omits Qwen3.5 image kwargs
at actor loss computation. Real tiny-model reproduction: changing pixels leaves
original likelihood unchanged, visual backbone never executes, vision gradients
are absent. A shared E4 adapter restores native multimodal forwarding and keeps
the native fused likelihood; four CPU tests include native-HF likelihood/gradient
equivalence and actual input/gradient observations. No new method component.

Previous job224041 and its checkpoints remain archived as text-loss-updated
weights from image-bearing rollouts. Their inference scores are real, but they
cannot establish the intended visual training-feedback mechanism. Restart from
the common initial model with identical h0/W sampling/loss/filter. GPU gradients
and full parameter changes remain pending. Protocol amendment:
`results/visual-pixel-forward-correction-20260910-v1.json`.

E99 — Actual h1 selection on the archived theta1 lineage: job224335 completed
512 image evaluations, 1.0375GPUh. First H128/128, C64/64 pairs; repeat H128/128,
C63/64. E2–E3 early-stop and resume both choose h1; the original selector also
chooses h1. Acceptance is bound to old theta1 and cannot certify h1 under corrected
weights. No independent-test result or VETO efficacy claim. Formal51 stays paused.

## Complete throughput and first updated-model H/C measurement (2026-09-15 22:00 HKT)

233485 completed for0.508056GPUh. Complete batch8/16/32 inference times on512 V
images were939.78/729.22/528.15s. Batch32 was selected by speed, with all31
changed answers and11 changed correctness outcomes retained versus batch8.
All1536 raw answers pass the fixed host scorer; this is E1 execution calibration,
not a VETO effect. Controller2431747 then submitted233540 on1H800 for the
authorized setup H128+C64 measurement of the selected updated seed42 model.

The complete measured forecast is80.66GPUh total,23.52GPUh for final evaluation.
A proposed5h transfer from core to evaluation preserves100h total; current caps
remain unchanged pending consent. Retention would bring the projected peak from
700.51 to373.63GiB, but no deletion is authorized or executed by these reports.
