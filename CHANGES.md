# 来源与修改记录

## VETO-v3完整候选审计与预注册停止（2026-09-17）

- 完成seed42/theta1上h0--h3的同一H128与fresh C256档案。H正确数依次为
  108、106、63、107；C两图同时正确数为203、186、93、205；C单图正确数为
  439、414、240、442。
- 不确定性感知审计将h1、h2判为FAIL，将h3判为UNCERTAIN；WHALE、VETO和匹配点估计
  门最终均选择h0。按预注册去留规则停止V3续训、其他种子和正式矩阵。
- 新增版本化`fast_chart_search_report_v3`，动态核验C规模并保留三种V3选择档案；旧冻结
  `fast_chart_search_report.py`未修改。论文构建器按报告协议验证相应源码哈希。
- 新增完整诊断，保留候选失败、全部GPU/API成本、证据身份和停止原因。该档案对应
  E1-v3测量、E2-v3审计和E3-v3最终选择；因选择没有分叉，不执行E4续训。
- 结果作为负诊断进入正文和补充材料，不解释成独立方法收益；V/T/R与ChartQA仍未读取。

## VETO-v3候选恢复与h1真实审计（2026-09-16 22:32 HKT）

- 正式认证h0基线完成：H108/128，C配对203/256，C单图439/512。
- GLM付费提案因外层字段和回调参数类型不匹配无法直接载入；新增确定性恢复入口，保留
  原始工作区，只转换`slots`元数据、私有局部标识符及`len(assistant_turns)`接口表达。
- 恢复后的全部回调用真实宿主参数类型做隔离预检；5项单元测试覆盖字符串不变、精确
  turn替换、私有属性拒绝、特殊标识符拒绝和名称冲突拒绝。该代码属于proposer到候选档案的E3传输边界，
  不新增方法公式。
- h1首次执行错误及成本完整登记；修复后作业234749完成H/C256，H106/128、C配对
  186/256、C单图414/512。V3部分档案将h1判为FAIL，但最终选择等待h2/h3闭合。
- h2、h3计划均已通过预检；共享名额释放后，h2作业234797已启动，h3将顺序执行。
- 论文补充材料加入h0/h1的部分V3优化档案，明确h2/h3和续训结论待定；`paper-build-v26`
  成功编译并逐页检查。历史V2候选表因报告源码身份不同而未被重新认证或迁移。

## VETO-v3真实基线执行与可变审计规模认证（2026-09-16 18:32 HKT）

- 完成H-pair128反馈评测和fresh C256的首次真实联合评测；后者640条回答齐全，但因旧认证器
  固定要求C128而不能登记为正式基线，保留为诊断运行。
- `visual_search_evaluation`现在从冻结manifest读取各partition图像数，H128和C512分别进入
  原生dataset；重复性记录也按冻结规模核验。
- `visual_search_bridge`的认证分母及记录数改由冻结partition推导，并强制C恰有每对两张图；
  覆盖旧C64与新C256的CPU回归通过。
- 共享评测wrapper按计划类型分派校准套件与单个配对评测；提交器接受经过注册的C256
  90分钟上限。两次失败作业及实际成本均保留，没有静默重试或修改结果。
- 修复只对应“真实模型评测到E1/E2-E3档案”的证据管线，不新增或改变Method公式。
  可认证v5计划已通过预检，因共享Slurm单作业名额被其他项目占用而等待提交。

## VETO-v3安全审计方法修订（2026-09-16）

- `evidence_v2`在预登记R512继续门失败后停止；负结果原样保留，已打开子集只作为方法
  开发信息。
- 新增`CounterfactualSafetyAcceptance`：配对能力/脆弱性分解、经多重比较调整的来源配对
  bootstrap判定、PASS/FAIL/UNCERTAIN处理，以及只在有支持的候选中沿用普通H排序。
- 新增H配对反馈摘要和三个固定、机器可读的机制槽位；C/V/T/R结果仍不提供给proposer。
- 原生RSFT保持不变。重复查看的顺序审计与crop/zoom搜索暂缓，等待各自的统计和实验证据。
- 已完成无模型调用的数据冻结：H-pair128和历史全部划分之外的fresh C256均通过像素/答案
  检查，C来源重叠为0；H-pair有8题按最小变更规则重新分配任务以满足数值答案区间分离。
- 搜索评测已支持显式fresh-C物化输入，V3基线强制核验256对；联网proposer与本地contract
  已统一为V3，旧V1/V2入口保持原语义。
- H-pair与fresh-C256的seed42/theta1/h0正式计划均已通过只读预检，尚未提交GPU；计划显式
  绑定weight phase、模型、harness、数据、解码器和源码身份。

## VETO v2独立R512诊断完成并停止扩展（2026-09-16 16:04 HKT）

作业234367在固定theta1上完成h1/h3的同源R512比较。h1为405/512配对、881/1024
单图，h3为416/512配对、900/1024单图；h1−h3分别为−2.15和−1.86个百分点，
来源成组bootstrap 95%区间为[−5.08,+0.78]个百分点。预登记继续条件失败，
`evidence_v2`不再扩大。该结果只诊断选择规则，不含新训练，也不是三种子最终结果。
作业耗时2191秒、0.608611 GPUh，逐样本结果及失败判定完整保留。

## VETO v2配对优先选择与H反馈输入（2026-09-16）

在普通H准确率不低于incoming的候选中，VETO现在按配对正确率、普通准确率、turn数依次
排序；新增完全匹配的单图排序对照。初始、普通、提前停止和恢复路径均由同一版本化入口
处理。旧接受门及其结果保持`gate_v1`语义。

新增固定、可核验的H失败摘要并直接写入实际proposer请求，只传H最终行为与评分，不传
私有审计集或臆测错误原因。新增seed42旧候选档案回放和同一theta1上的h1/h3 R512诊断
入口；它不训练新模型，T仍封存。Method对应E2-v2/E3-v2；E1评分和E4原生RSFT未改变。
实现及边界见[协议修订](ours/chart_selection_v2_20260916.md)。

## 实验完整性与训练接口修复（2026-09-16）

已按用户授权落实代码审阅第1–7项：100处关键运行assert改为显式校验，失败作业与候选
完整绑定，先闭合槽位再发布选择并区分额度与实际尝试；新增labels接口保护、共享分片
张量读取和统一终点评测身份。新计划冻结当前源码，旧记录保持原哈希。ChartQA评分未改。

历史244个文件、2816条回答及三种规则的选择/恢复决策复核一致。旧源码由本地标签
`before-integrity-hardening-20260916`及run内快照保留。实施、测试和E1–E4对应详见
[交付记录](ours/integrity_hardening_20260916.md)。本轮未增加科学结果或方法主张。

## 首个完整视觉闭环已完成，资源决定待确认（2026-09-16 10:08 HKT）

seed42第二阶段训练233990和后续完整核验234097均已完成；当前没有GPU作业运行，
控制器在完成有界任务后正常退出。第二阶段使用216/256条成功轨迹、28次原生更新；
全部成功图像消费、非零梯度、相对第一阶段的完整参数改变和精确BF16导出已核验。

| 开发集状态 | 单图正确/512 | 两图同时正确/256 |
|---|---:|---:|
| 初始模型，h0 | 452 | 214 |
| 第一阶段后，h0 | 448 | 207 |
| 第二阶段后，h3 | 447 | 212 |

相对第一阶段配对+5对、单图−1题；相对初始模型仍配对−2对、单图−5题。三种选择规则
全部选择h3，而且最终评测同时改变权重与策略，因此不能将这次变化归因于VETO接受门。
三个阶段共1536条开发回答及256条续训回答均已重新核对解析和评分。

[完整诊断与证据](ours/first_cycle_diagnostic_20260916.md)；
`data/fast-chart-20260915-v1/first-cycle-report-v2/result.json`。
新增报告对应E4完整训练/交接及E1描述性开发测量，未新增方法模块；8项报告与证据绑定
检查通过。正文/补充`paper-build-v19`已加入真实闭环，正式比较仍保留明确占位。

已结15.938333/100GPUh，无GPU预留；现有新目录约123.31/400GiB。扣除已完成工作后，
全范围预测82.40GPUh；最终评测23.52小时仍超过原20小时阶段额度。全部保留约700.78GiB，
按之前具体回收清单执行时暂估峰值374.05GiB，仍须明确删除授权。
[两项待确认资源安排](ours/fast_chart_resource_decisions_20260916.md)已更新为现有路径和实测结果；
未删除任何模型，未改阶段额度，未新提交正式条件。种子43/44和独立终点仍未完成，
科学去留判据尚不能在三个种子上评定；goal未完成。18条件逐项状态已记录于
`results/fast-chart-matrix-status-20260916-v1.json`，不将未执行对照代填为h3的结果。
本轮新增结果和稿件保留本地，未自动推送GitHub。

## 续训恢复与首批评分已核验（2026-09-15 23:44 HKT）

233990仍在执行真实第二阶段训练。已核验恢复后的724个张量条目（含别名）与原生FP32
检查点逐值一致，学习率调度和随机状态一致；Adam moments按原生约定未保存或加载。
真实loader恢复的data.pt状态与修正计划相符，首轮权重同步已返回，服务端8个区分初始
模型的坐标核验通过。服务端完整图比较仍须等待训练后导出核验，不能由8个坐标代替。
续训第一个批次（global_step5）有64条轨迹，共享评分重算48条正确，64条reward均一致。
尚不表示四批训练结束或最终模型性能提升。58个冻结运行源文件均未变化。
证据：`results/fast-chart-stage2-restore-and-first-rollout-review-20260915-v1.json`。
23:45追加：首批48条成功轨迹已完成6次原生SFT更新调用，5348个assistant loss token；
更新后原生权重同步成功返回。64条轨迹的题目身份及每题8次采样与修正预检一致，成功
过滤的子集也逐项一致。最终参数变化仍须等完整导出审计，未由更新调用代替证明。
证据：`results/fast-chart-stage2-first-update-review-20260915-v1.json`。

预算更新按任务身份扣除已完成的4次H/C测量，本次续训及后续核验按setup最多3GPUh
计算一次，没有再把同一工作计入待运行core。包含其最坏耗时与剩余重跑额度，当前完整
范围预测约81.79/100GPUh；最终评测仍约23.52小时，超过原20小时阶段额度。全部保留
约700.61GiB；需另行授权回收的滚动方案约430.48GiB，均非已获资源放行。完整矩阵继续
等待既有资源修订，当前闭环不受影响。该估计须在实际续训、导出和结算后再更新。
证据：`results/fast-chart-partial-cycle-remaining-cost-review-20260915-v1.json`。

## Real selection reconstruction and illustrated limitation (2026-09-15 23:17 HKT)

Completed the real E1–E3 report from all four seed42 H/C allocations. Final and
resume boundaries reconstruct WHALE=VETO=marginal=h3. All1024 actual answers
were revalidated; equivalence is reported without assigning a VETO contribution.
The planned E4 continuation is running its actual-parent CPU sampler preflight.

The paper builder accepts an explicit post-hoc case, validates its raw answers,
fixed scorer and image identities, and copies unchanged images into the PDF.
This is E1 interpretation evidence: the observed h2 response reads changed
values but reverses the subtraction order. The method text now states that
paired failure alone cannot identify visual non-use or a latent mechanism.
The v15 main/supplement compile, and all six rendered pages were inspected;
final method tables remain unmeasured and the case float layout is provisional.
Rendering uses isolated pypdfium2 5.13.0; the training environment is untouched.

## Complete-archive reporting for E1–E3 (2026-09-15 22:50 HKT)

22:58 update: the same report now diagnoses answer-type parseability using the
fixed host parser/verifier and complete sample identities. Five targeted tests
pass (`fast-chart-search-report-tests-20260915-v4.log`). The actual h0/h1
descriptive diagnosis was computed from512 H/C records; it is not a causal
claim or a score/protocol modification. h2 has completed, h3 runs as233759.

`ours/fast_chart_search_report.py` certifies every actual H/C measurement and
reconstructs native WHALE, paired and marginal selections in both final and
resume paths. All three allocated slots must be accounted for; failed slots
receive no invented score. The generated table labels H/C as optimization
data and explicitly reports equivalent decisions without attributing benefit.
Its cost field covers only candidate/incoming H/C allocations, including failures.

`ours/fast_chart_paper.py --search-report` imports the table only with complete
slot coverage, matching source/record hashes and a reproducible table body.
The supplement retains a pending placeholder until an actual complete archive
is available. Four targeted tests pass; real reconstruction and PDF integration
await h2/h3 completion. These changes do not alter frozen experiment code.

## Bounded first-cycle handoff (2026-09-15 22:39 HKT)

`ours/fast_chart_first_cycle_controller.py` connects the existing E1 candidate
measurements to native E2–E3 selection and one prospectively fixed VETO E4
continuation. It reuses active job233589, permits only the three allocated
seed42 candidates, four continuation batches and one full followup, all in the
existing setup budget. No paid proposal, cleanup, other condition or sealed
endpoint is launched by this controller. Four targeted tests cover duplicate
and ambiguous submission, delayed artifacts and explicit failed-slot evidence.

The actual GLM proposal completed once, with3 candidates and6 settled requests.
It did not read available H feedback; that limitation is retained and candidates
are not replaced. Its CLI latest-log alias was converted reversibly to a hard
link of identical content; no checkpoint was removed and no frozen source or
storage check was relaxed. See the first-proposal-tool-use-review,
cli-log-alias-normalization and first-cycle-controller-launch receipts in results.

## Completed shared calibration and bounded throughput measurement (2026-09-15 21:27 HKT)

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

## 2026-09-15 — 三项学习率的训练均完成

233361完成1e-5四批训练：256次轨迹尝试、144条成功轨迹（52/35/42/15）、20次原生
更新，四次权重同步均返回。运行30分36秒，计1.02GPUh；累计视觉支出12.781944GPUh。
控制器自动衔接233399，全部144条图像输入和视觉梯度核对通过，完整参数审计及V512
仍待完成。成功轨迹数和训练更新数不用于挑选学习率；选择继续依据三项完整开发准确率。
证据：results/fast-chart-third-lr-training-completion-20260915-v1.json。

## 2026-09-15 — 防止回收后的模型缺失导致空间低估

完整成本预测现在要求记录中的原生模型及服务权重文件实际存在且非空；缺失目录或仅剩
元数据的检查点不再被算成零/小模型。5项相关检查通过，包括同inode去重计量。两个已
完成LR模型的真实目录均通过清单存在性核查，活动233361的训练源码身份不变。
此项为E1/E4资源测量，未进行模型删除，也不替代原完整参数/哈希核验。若以后获准清理，
继续使用保留的清理前测量和明确保留方案，不能从缺失模型重新外推完整矩阵空间。
证据：results/fast-chart-storage-forecast-handoff-20260915-v1.json。

## 2026-09-15 — 当前研究入口与审稿证据清单

README由9月10日Chess快照更新为已授权的精简图表协议，原文完整保留。明确18个逻辑
条件、共享前缀、当前训练验收与尚未完成的VETO效果比较；历史F0/大矩阵不再误作当前指令。
新增论文审稿清单，逐项列出方法边界、必要对照、真实视觉更新、统计与资源证据；未知项
保留pending。本地链接及上游固定版本核对通过，上游源码未修改，没有新增实验分数。

## 2026-09-15 — GLM正式搜索前费率复核

官方详细价格表核实GLM-5.3-Flash输入0.8元、输出2.8元、缓存命中0.23元/百万token；
模型页核实1M上下文及128K最大输出。现有1/4保守费率覆盖上述价格，45元总上限不变。
保存公开来源及哈希，未改写历史账本或未结算预留，没有新增付费请求；不称为服务商账单。

## 2026-09-15 — 第二项校准完成与共享磁盘交接恢复

233258正常完成V512：448题单图正确、207对同时正确，比共同初始分别下降0.781和
2.734个百分点。逐题重算一致，13对改善、20对变差；保持校准结果与VETO效果的边界。
评测耗费0.331944GPUh，累计视觉支出11.761944GPUh，尚未选择学习率。

Slurm完成记录先于共享磁盘上的最终followup结果可见，控制器因此退出。原始结果随后
完整出现，计划、参数、服务评测与终态身份核验通过，没有补写或重建结果。控制器增加
300秒可见性等待，以及保持初始计划和失败记录的显式恢复；5项检查通过。恢复进程
2231441继续第三项1e-5预检，前两项GPU实验不重跑。此项仅为E4结果交接，无算法改变。
原源码与故障记录保存在ours/history/fast-chart-controller-visibility-20260915-v1/；
完整恢复依据见results/fast-chart-controller-visibility-recovery-20260915-v1.json。

## 2026-09-15 — 第二项真实训练完成与论文来源补齐

233172完成1e-6四批训练：192条成功轨迹、25次更新，1.257778GPUh。233258接续核验，
完整图像消费、视觉梯度、FP32参数变化和精确BF16导出均通过；V512结果仍待完成。
根据CVF官方记录及Qwen官方model card补齐PlotQA和模型引用，明确固定模型revision
和所有方法共同的BF16初始化/FP32训练参数/BF16服务权重流程。新增引用不改变实验。
v11遇到离线字体缺少Courier的URL排版错误，保留失败日志，改用现有正文字体的链接；
v12正文与补充均编译通过，无字体替换、溢出或未定义引用警告。没有新增性能主张。

## 2026-09-15 — 完整成本预测计入更新后模型的加载时间

已完成的1e-7模型加载耗时280.66秒，高于初始模型校准的最大251.69秒。forecast现从
完整、哈希匹配的LR开发记录读取加载耗时，采用已观测最大值计入每次未来评测，保留
实测批量、每图耗时和调度开销。3项相关检查通过；仅这一项对42次最终评测的外推增加
约0.338GPUh，不会按较快初始模型低估成本。实际全部LR/吞吐测量仍未完成，未准入扩展。
这是E1/E4共享服务的成本记录，不改变方法、正在运行的训练或阶段额度。

## 2026-09-15 — E1与单图准确率门的解释边界

正文与补充加入E1的恒等式P=2M−1+q0及解析示例，说明相同单图准确率可以有不同配对
正确率；固定权重候选档案中q0不变时，两种零容差门等价。也明确在固定M时提高P会
同时提高两边均错比例，不能据此宣称所有错误都减少。没有新增方法模块或训练改动。
复用已有逐题验证与误差分解，对完整V记录重算：共同初始为18/24/214，1e-7训练后为
25/19/212（两边都错/仅对一边/两边都对）。这些是不同权重上的校准描述，不是C上的真实门选择差异。
paper-build-v10正文与补充编译通过，无字体、溢出或未定义引用警告；此前源码保存在v9。

## 2026-09-15 — 学习率校准到论文证据表的连接

fast_chart_paper增加显式--configuration入口，只有三个固定学习率全部完成后，
才根据原始结果、训练计划和参数交接记录生成补充表；复核普通准确率与较小LR平局规则。
选中的seed42前缀明确标注为后续共享计算，不额外计作独立实验。E1/E4共同配置说明，
不是VETO性能主张。当前真实未完成的校准状态被入口拒绝；paper-build-v9编译通过，
LR结果继续显示待完成，无虚构数字。实际三行结果表的填充与编译仍待校准结束。

首个1e-7试验随后完成V512：443题正确、212对同时正确，相比共同初始452题/214对
分别下降1.758和0.781个百分点。按原始答案与固定验证器重算一致，6对改善、8对变差。
233025正常结束，耗费0.418056GPUh；累计视觉支出10.172222GPUh。这是校准观察，
不能解释为VETO收益或因果失效机制。控制器继续1e-6和1e-5，不根据首项结果改范围。

## 2026-09-15 — 完整四批视觉训练与逐图核验

232741正常完成四批256条轨迹，191条成功轨迹产生25次原生更新，耗费1.27GPUh。
新增fast_chart_backbone_audit，按接受档案逐项核对实际模型消费的图像张量、网格、
原生精度转换和对应视觉梯度。3项检查通过；真实191条输入和梯度全部匹配。
followup在导出前执行该审计；原版源码按既有交付清单SHA复原留档。
233025完整参数比较通过：原生FP32与服务BF16均有真实变化，视觉部分亦通过。
此项对应E4图像进入原生RSFT及实际参数更新的验收，不增加VETO机制，不宣称性能收益。
V512评测正在进行，尚未完成学习率校准或正式方法比较。

实际HF导出使用新版合并配置布局，不能要求其文件哈希与初始模型逐字相同。
endpoint保留两套原始身份，并严格核对加载后的模型/生成配置、全部tokenizer后端
（含BPE合并）、图像/视频处理器和模板；真实导出的三种CPU图像尺寸的全部像素、网格、
token结果亦完全一致。9项相关检查通过，行为变化或原文件漂移仍拒绝执行。
此项为E1/E4统一输入处理契约，未更改正在运行的作业或已准备吞吐计划的冻结源码。
证据：results/fast-chart-real-processor-contract-20260915-v1.json。

## 2026-09-15 — 恢复goal并修复首批训练后的显存交接

232265首批完成53条成功轨迹、7次原生优化更新后，在NCCL准备CuPy缓冲时OOM。
新增fast_chart_transport，在同步边界仅释放闲置PyTorch/CuPy缓存，核对活跃分配不变，
再调用原生prepare；保留原finalize、FP32权重、Adam状态和3GiB桶。Method对应E4
权重交接基础设施，不是VETO的新机制。真实参数与接收权重核验仍保持。

9项相关检查通过。恢复控制器对旧/新配置、模型、采样顺序和轨迹预算逐项比对；
只允许计划及输出路径变化。首个同配置重跑计入retry阶段，后续新LR仍计setup。
修复计划、源码旧版本和失败终态保留；新v3完整CPU预检通过，232741已RUNNING，
232741首批56条成功轨迹完成7次SFT更新；同步前释放24.395GiB闲置显存，活跃分配
不变，原生权重同步返回，已通过旧OOM位置。完整四批和最终参数核验仍待完成。
修正了控制器退出时留下WAITING状态及跨目录计划名碰撞的问题。活动goal已恢复。
v2预检误继承未限制的CPU线程环境，实际72线程；未提交GPU作业时停止并保留记录，
v3按与训练脚本一致的单线程设置重做完整预检。训练源、实际sampler检查均保持。

新增独立的共享评测吞吐测量入口：同一完整V512、同一h0/模型/评分，比较batch16/32
与已完成batch8；只按速度选择，并报告全部答案变化。对应E1调度基础设施，非新增方法。
2项检查通过：拒绝提示词/token预算/样本范围变化，不按高分或不完整测量选择批量。
计划已生成，尚未提交GPU，等待当前校准结束；不与活动控制器争用单作业额度。

接通统一endpoint执行设置：完整吞吐测量及GPU终态通过后，固定一次共享批量设置，
所有方法/种子的T、R与ChartQA使用同一配置；独立评测提交后禁止再选择设置。
训练、搜索、V校准和token预算保持。7项检查通过，包括真实DataProto接口的CPU
模拟批量32完整512题覆盖、逐题评分、16份批次证据和两个256题分项；不是VLM结果。
forecast只有显式提供已完成测量才使用提速估算，未测量时仍保留33.84GPUh评测缺口。
上述改动对应共享E1评测调度，不属于VETO的新模块。

新增fast_chart_costs，逐项核对GPU终态与项目API账本；共享资源采用显式等额分摊，
每项必须覆盖一次，失败费用和未结算上界不可遗漏。2项检查通过，已生成真实总账快照；
正式方法分摊等待实际分支完成，不把预留费用标成实际账单，也不把共享计算标成免费。
这是论文成本表的报告基础设施，不增加E1–E4方法机制。

## 2026-09-15 — 精简VETO实施与真实共同起点校准

按新授权将执行范围收缩为18个逻辑条件、100GPUh、400GiB和项目API累计45元。
旧协议和结果保留；不回溯宣布旧科学门槛通过，不启动51组/Chess/CLEVR/2B。

- E1：chart_answer_protocol固定host解析器，候选parse_answer不参与得分；
  plotqa_multitask在原源分组/图像上建立比较、单值、差值三类题及非重叠数值容差。
- E2–E3：fast_chart_search复用原生run_evolve及现有接受模块，单轮3槽、固定incoming、
  完整有效档案、失败留痕；正常、提前停止和分段恢复走同一入口。真实新搜索待执行。
- E4：visual_sampler_reference修复固定旧参考；fast_chart_training及hooks使用实际
  保存的sampler，保留原生成功过滤、损失和Adam-reset约定；观察像素/梯度/接收权重。
  fast_chart_followup核对全参数、规范导出及续训前后原生FP32变化。
- 对照：fast_chart_augmentation保持每阶段256条轨迹，替换16题为8个完整C配对。
- 共同评测：ChartQA固定原始开放答案512题，485张图；保留来源与原始评分解释。
  endpoint注册限制T/R/外部评测只能使用预算确定的权重；statistics/figures只读完整记录。
- 论文：可编译正文、补充及方法图；三张结果表保持待实验，未生成虚构收益图。

22项新集成、投入门和终点完整性检查通过。真实校准231666三种均完成，按单图准确率
选structured（452/512；配对214/256）；耗费0.996111GPUh，累计7.894167GPUh。
后续LR控制器仅执行校准与开发复评，受20/100GPUh约束。暂未产生VETO科学收益。
第一项LR1e-7训练232265已启动；冻结57份源依赖复核一致。paper-build-v8正文、补充均
编译成功，补充含真实共同提示词校准；其余结果表和收益图继续等待实验。
修复了字体编码和参考文献回链，当前构建无字体替换或栏宽溢出警告；仍采用明确标注
为2026版本的官方作者模板，2027最终投稿格式待核验。CPU跟进进程在校准结束后自动
形成实测成本/存储测算，失败则保存诊断，不自动启动正式矩阵。
fast_chart_admission禁止把纯精度转换或无图像梯度当作有效共同起点；完成但无更新的
正式分支仍可记录负面结果，不按更新大小筛选终点。forecast依据实际终态/吞吐测算，
当前串行最终评测外推33.84GPUh，超过阶段20GPUh，正式扩展尚未准入。
新增实现均在ours；上游子模块未修改。未删除文件，未进行新的GLM付费调用或推送。


## 2026-09-10 — E1误差分布与简单门的可区分性

新增pair_error_decomposition，核对P=2M−1+q0，并按实际配对数据分组报告误差分布。
它解释E1与同数据marginal_gate的区别，不增加方法机制。3份完整C与3份V记录的q0
均为0；当前C档案上两个零容差门等价，V仅作描述。两项CPU检查包含相同M不同P的
反例和非法reward拒绝，避免把当前等价夸大为普遍结论。0新增GPU/API/训练或清理。

## 2026-09-10 — 保存回复的E1解析诊断

新增visual_answer_parser_replay：完整H/C分区的回复、真值、候选与批次哈希匹配后，
分别运行候选和参考解析器。真实h1的H回放完成，128与120的差异反映8条格式依赖。
0新模型调用/训练/费用；属于事后评分诊断，不能充当prompt消融或VETO贡献比例。
固定主评分、当前候选与原始结果保持，后续训练解释增加明确的格式漂移排查要求。

## 2026-09-10 — 接受后续训模型的固定h0复测入口

新增visual_resumed_followup及Slurm包装器，复用现有规范导出、真实视觉评测与逐题输入
匹配。对应E4实际参数更新→固定h0下E1复测，不增加训练损失或接受模块。prepare先要求
真实续训完成、成功轨迹审计、FP32 theta1→theta2完整比较和Slurm终态；尚不能生成真实
可执行计划。E1继续用既有64对V开发子集；既有推理波动、受控分叉与独立测试仍需验证。

新组合逻辑把真实FP32训练变化与incoming BF16参照的舍入残差分别保存，后者不能当作
新更新。2项CPU数值测试通过：没有训练变化但有舍入残差；有真实微小更新但BF16导出
尚不改变。真实导出/评测未执行，0新增GPU/API、0检查点或清理。所有既有冻结源码保持。

## 2026-09-10 — 真实视觉提案与续训证据核验

按本次明确授权执行已有GLM入口，生成一个真实h1并启动同theta1的H/C两遍测量。
E1仍由共享逐样本评测给出，E2–E3依旧使用固定incoming参照；未手工改写生成候选。

新增audit_native_visual_resume复用首批核验规则，逐项重放下一批图像/提示/答复、原生
成功子集，绑定恢复和前后两次权重传递回执。新增verify_visual_resumed_parameters
比较原生FP32 theta1与theta2的全部独立张量及共享别名，不产生额外模型副本。
Method对应E4轨迹/参数更新的证据检查，变化量为||theta2-theta1||_2；不是新损失或方法
模块，也不能把参数变化写成性能收益。原审计器与上游代码保留，新增文件均在ours/。
参数比较的2项小张量CPU测试通过，涵盖真实变化、不变、别名/形状/精度/非有限值拒绝；
真实GPU续训与新审计器仍待真实接受后运行。本轮没有额外清理或GitHub推送。

## 2026-09-10 — 视觉接受回执到原生续训入口

新增visual_training_resume、visual_resume_bootstrap、visual_resume_model_worker和Slurm入口，
对应E2–E3接受回执→E4原生恢复/权重传递/成功轨迹训练。prepare拒绝未完成搜索或仅有
accepted_harness.txt的状态；固定原训练/搜索/导出血缘、W输入与接受代码。保存model/extra
及sampler，不新增Adam状态或损失。继承原checkpoint manager和trainer，复用完整actor
恢复核对；原NCCL同步后从receiver读8个坐标，拒绝仍为旧权重。原多模态轨迹记录保留。

6项CPU测试通过：配置变化限定、未完成选择拒绝、原生小张量恢复且后续RNG抽样不变、
BF16接收值核对、同步/检查顺序、重复worker初始化保持。真实W/sampler的新配置预览
通过但使用已有h0，actor RPC仍是记录；无真实h1、GPU恢复、第二批更新或方法收益。
新代码不属于之前待授权的GLM发送清单，未执行任何外发或额外数据清理。

## 2026-09-10 — 真实H/C记录接入原搜索与训练接受边界

visual_search_evaluation已完成h0真实H/C两遍评测，串行方式仍有2个H和1个C答案变化；
保留首遍选择和所有重复记录。visual_search_bridge复用原run_evolve，通过离线评测、
联网提案两个边界接入共享视觉任务，E1记录供给E2–E3最终训练策略接受。完整候选档案
与固定h0参照保留，C接受回执位于proposer可读H目录外；原搜索排序/停止记录单独保留。
4项CPU检查覆盖原生普通/提前停止循环、off无审计、恢复篡改与H公开字段；真实h0初始
接受完成，真实h1、最终选择和续训未完成。GLM调用被自动审批拒绝，待43,161字节的
明确外发范围授权，无新API调用。visual_search_contract.md是proposer输入约定。

visual_resume_data_probe对应E4恢复的数据通道，使用真实W数据和原生saved sampler；
CPU检查通过，下一批8题与前批不重叠并含图像。actor RPC仅记录，不能当GPU恢复证明。
原sandbox本地进程socket失败保留；允许本地socket后的相同检查通过，无模型调用。
新增实现均在ours/，原生训练损失及上游代码保持，未额外清理或发布。

## 2026-09-10 — 首次视觉训练后复评完成

native_visual_followup复用原生merger、完整活动参数比较及共享视觉评测，对应E4→E1。
224121完成：参数改变保留到导出并进入实际服务；同题单图117/128→113/128、配对
53/64→49/64，不作为VETO效果。audit_visual_followup_inputs核对128道题处理后像素、
提示和位置编码一致；原始请求输入/解码也一致。新增visual_followup_repeatability
复用相同评测器，对两个固定模型各重复一次全套题，检查推理波动，不择优挑选结果。
无新损失或接受机制，上游未改，原生恢复产物保留，正式矩阵和Chess续训继续暂缓。

## 2026-09-10 — 从视觉作答进入原生RSFT训练

新增native_visual_training.py及运行入口，从既有原生actor配置接入共享视觉数据、
奖励和采样，首批限定32道W子集中的8题×8次生成；未将开发题用于训练，原生损失和
成功过滤保持。visual_training_bootstrap记录完整多模态batch及成功子集，不使用
旧的仅文本裁剪。对应E4执行与来源证据，不是VETO E2–E3贡献。

224008生成64次后因reward-worker分支丢失sample ID而在更新前失败；原始记录和费用
保留。visual_training_identity继承原视觉loop，仅将ID放进原生extra_fields，独立
agent注册避免父类覆盖。2项原生CPU分支/归档检查通过。224041已完成同一数据/配置
的新轨迹：47条成功轨迹、原生报告6次优化器更新，检查点已保存；参数变化尚未核验。
audit_native_visual_training完成W输入像素/token/回复及原生成功子集的独立重放。
核验器v1的TensorDict迭代失败保留，v2显式遍历keys后通过，不能代替实际权重变化或科学效果。

工具初始提示校准223996已提前停止；保存部分结果，不声称完整V分数或选定正式h0。
当前先验证direct条件的真实更新，正式方法比较及51组矩阵仍未启动。

## 2026-09-10 — 直接视觉验证与64对开发图表对照

真实4B简单图表直接回答16/16。原工具试验0/16的裁图坐标和答案格式原因单独归档，
不删除零分、不作为VETO收益。新增NormalizedEvidenceZoomTool对应共享视觉工具输入；
canonical_answer_harness统一明确末行答案解析，BlindVisualHarnessDataset用于真正
无图对照；这些都不是E2–E3接受模块的新贡献。实际crop的2项CPU检查及无图/解析2项
原生检查通过，早期测试失败保留。所有新增实现仍在ours/，上游未改。

visual_coordinate_diagnostic已完成两项简单图表对照；visual_development_screen现对
预先固定的64对PlotQA派生开发图表执行看图/无图/归一化裁图三条件，属于E1测量。
223990已完成，direct 117/128、53/64对；blind 64/128、0/64对；normalized_zoom
6/128、0/64对。新增summarize_visual_development_screen对应E1独立重算与源图表
匹配区间，真实全部结果与批次归档核验通过；不代表跨种子或VETO效果。实际0.274167
GPUh、不训练、不新增模型检查点，新目录实际分配约0.372GiB。
维持现有磁盘空间限制，51组历史矩阵暂停，Chess第二阶段续训暂缓。

## 2026-09-10 — 按用户要求优先视觉验证，完整矩阵不作为当前队列

停止仅在等待的旧顺序控制器，暂缓其自动Chess续训/导出；保留最后一项正在运行的
Chess评测与全部证据。新增ours/visual_priority_controller.py，仅复用既有预算/提交
与真实16图评测入口；本次不生成新模型检查点，不删除其他文件，MIMIC保留。
对应E1配对答题测量的优先执行，不增加Method机制。2项新路径/失败检查和3项已有
计费/提交检查通过。原51组JSON保留为历史研究设想，当前机器可读范围以
results/veto-visual-priority-amendment-20260910-v1.json为准；无正式矩阵自动队列。
新增控制器启动不等于视觉模型已经生成答案，真实结果以其完成回执为准。

## 2026-09-09 — 授权后候选评测、下一E4连接与FST搜索边界

GLM生成h1并保留原字节，记录说明/行为差异与7次实际API费用；提交同冻结计划的
222212候选评测。新增`ours/prompt_subspace.py`及重建prompt-only说明，保留原AST
校验并在执行前约束纯文本赋值；4项原生CPU检查通过，未认领作者原skill或实测FST。
新增`alternation_training.py`/`run_alternation_rsft.sh`，从完整搜索/审计绑定下一批
新在线RSFT；2项CPU目标/配置检查通过。下一8题来自固定训练顺序，未按结果挑选。
这些文件不进入当前MH阶段依赖。Method对应原E3搜索空间对照和E3→E4连接，非VETO贡献。


## 2026-09-09 — MH完整审计与后续在线RSFT接线

新增`ours/audit_mh_phase.py`及棋盘续着/异常测试；记录精确回复的原runner重放与独立棋盘
检查，真实32题批次222196已正常完成且独立审计PASS（0/32）。新增`compact_recorded_training_bootstrap.py`
及原actor分发测试，保留新在线采样/记录，仅复用已验证尾部裁剪；训练launcher增加
`RSFT_HARNESS_PATH`输入。完整Hydra CPU占位配置通过，不代表最终选择或GPU训练。
这些文件不在该MH作业冻结依赖中，运行来源核验一致；均为E4↔E3共享连接与测量。
归档基线完整计量和失败类型；GLM提案被自动审批拒绝，未产生新调用，已准备具体
发送范围清单供授权审核。未启动h1评测或下一轮训练。

## 2026-09-09 — 更新权重上的分阶段原生MH

新增`ours/mh_phase.py`、`phase_vllm_worker.py`、`run_mh_phase.sh`及原生集成测试。
通过原搜索循环的阶段边界将GLM调用移到登录节点；两GPU对完整32题MH做固定分片，
保留匿名输出，以私有顺序映射复原原汇总。每worker在原load后验证实际参数。
普通目标平局使用固定读入顺序，原评分和early-stop保留。3项CPU检查通过，真实执行
待最终预检与提交；代码属于E4→E3连接，未改上游或引入VETO收益声明。

## 2026-09-09 — 原生单步成功与规范权重交接

- 归档222156真实优化器更新及222165规范导出的全活动参数数值检查，保留222164的
  HF视觉参数名回写失败；区分训练差异和3840个原始FP32元素的BF16转换影响。
- 新增`ours/canonical_native_export.py`及2项原生检查，通过继承原merger保留规范
  参数名和原base推理配置，不更改上游的训练算法或参数合并规则。
- 新增`ours/probe_updated_vllm.py`与限时入口，CPU已验证权重/分词/生成合同；
  222175回调序列化失败归档后，官方数值RPC修订222176通过8参数值检查和1次生成。
  完整结果保存后退出清理未释放进程，已主动取消；随后补显式engine shutdown，
  此后置清理修改未额外GPU复跑，不改变原完成测量。
- Method对应E4中的参数更新与传递测量，不新增科学机制。正式对比要求匹配精度，
  全部失败及成本保留；94项基础测试通过/24原生依赖跳过，上游未改、未push。

## 2026-09-09 — 完整恢复循环与活动参数核验

修复恢复worker的仓库相对路径解析及原数据集占位字段兼容，增加真实数据集到原生
trainer/actor入口的CPU集成测试；成功样本、loss mask和梯度累积配置保持。
归档222151/222154失败及真实资源成本，v3作业222156运行中。新增独立
`ours/verify_native_transition.py`、比较器测试与CPU导出launcher，检查已审查活动图、
真实参数差异及BF16舍入。后者不在运行任务冻结源码中，不修改原WHALE文件。


## 2026-09-09 — 审计批次的原生训练恢复

新增`ours/compact_native_batch.py`、`recovered_training_bootstrap.py`、`recovery_gate.py`
及显式launcher：一次性复用221967完整审计批次，禁止新采样；只裁无效回复尾部。
原生loss、optimizer、成功筛选、checkpoint与权重传递继续由上游执行，对应Method E4。
新增原生混合Qwen损失/梯度、真实64行恢复及配置迁移检查；保留未通过的左补齐实验日志。
归档221967全部来源/请求/批次/失败记录，更新累计成本与研究状态。上游未改，未push。

## 2026-09-09 — 原生checkpoint导出接口检查

- 新增`ours/tests/test_native_checkpoint_export.py`，用小型合成Qwen3.5参数执行原
  FSDP merger和HF重载，35个张量名对应的BF16值一致、共享输出头正确恢复，测试通过。
- processor/tokenizer导出在fixture中替换为None，真实4B导出仍待训练结果；对应E4
  共享执行检查。未改运行中的221967源码，未新增GPU作业/模型调用，未push。

## 2026-09-09 — 原始请求与训练张量审计

- 新增`ours/audit_native_training_batch.py`，将完整请求记录、原AgentLoop CPU重放、
  精确张量/mask及独立棋盘核验连接起来；对应E4测量层，未更改生成或训练逻辑。
- 新增8项有针对性的异常/完整性测试，训练环境全部通过；完整64条真实批次CLI
  仍待221967落盘验证。当前同一作业持续运行，新增调用/作业0；冻结源码不变，未push。

## 2026-09-09 — 原生Adam显存诊断与v5训练计划

- 归档221921的64条原生采样和首次Adam OOM；原验证器接受1条，但未完成优化器更新。
  保存有限独立棋盘/覆盖核验，并明确未结束思考的fallback解析及原dump无法完整重放。
- 新增`ours/probe_native_optimizer.py`和限时单GPU入口，调用原ZeRO-1构造器；CPU
  数值对照及真实单H100合成显存检查通过。只改变pilot中的原生foreach配置并启用
  既有记录hook，完整Hydra比较确认两项差异。所有改动对应E4共享执行，不增加公式。
- v5启动预检查通过，221967于18:32:36 HKT启动，64条新的在线轨迹；真实更新和导出
  待验证。此次未重跑无变化的基础测试，CPU/GPU probe与shell语法检查提供新增验证。
  上游未改、未push；旧失败成本全部保留。

## 2026-09-09 — checkpoint模型图审查

- 记录官方4B权重与原生HF活动图的15个MTP遗漏项及共享输出头，冻结精确核验规则；
  723个共同张量形状一致。此为E4测量准备，实际导出、参数变化及权重推理尚待验证。
- 未改运行中221921的源码或配置；完整计划哈希检查通过。更新18:20 HKT运行快照，
  未新增模型/API调用或GPU作业，未重跑既有测试，未push。

## 2026-09-09 — 原生训练记录子类与重试轨迹核验

- 新增`ours/native_training_trace.py`和显式Ray hook，通过继承保留原生请求与返回、
  完整训练张量/mask和逐轮事件；中断请求保留未知计量，完成快照不等于作业完成。
  属于E4共享测量层，不增加优化目标或VETO机制。
- 7项记录器测试、实际类继承/hook导入和CPU配置解析通过；修正一次上游类名错误和
  一次CLI参数顺序错误，日志保留。当前221921未启用该hook，分布式实录仍待验证。
- 原seed46的8条轨迹在原AgentLoop重放通过，包括1条格式重试后的成功轨迹，0新增
  模型调用。最新基础测试86项通过、10项原生依赖跳过。无新增GPU作业/API调用；未push。

## 2026-09-09 — 训练启动成功集与原生单步接线

- `ours/`新增冻结采样、独立棋盘/token审计及原AgentLoop离线重放：原h0与4B权重不变，
  完成64条训练轨迹，其中2条成功；记录全部失败、调用和资源成本，不认领性能提升。
- 原生RSFT launcher增加显式输入参数，新增冻结单步入口、完整配置启动检查与逐张量
  参数变化审计。代码对应E4训练及checkpoint传递测量，E1–E3和上游均不变。
- 首次GPU启动221906因缺CuPy失败，87秒占用及完整源码/日志保留；新增独立哈希锁文件
  安装官方CuPy14.2.0，显式原NCCL导入检查通过。221907随后暴露传输缓冲太小的问题，
  恢复上游3072MiB默认值并增加FP32张量容量预检查。两次失败共
  0.165556GPUh，0训练轨迹。实际优化器更新、导出和新权重推理仍未验证；基础81项
  及原生8项测试通过，未push。
- 221912首次原生权重同步通过，进入生成后因源码中可确认的队列/超时配置风险主动
  取消，没有观测到超时；0.164444GPUh计入成本，实际调用/token数未知。保持模型、
  采样和job上限，将通信等待改为4800秒；完整配置比较通过，重试221921运行中。

## 2026-09-09 — 原生GLM搜索、独立重放与无污染pilot数据

- 新增`ours/native_search.py`与Slurm入口，调用原`run_evolve`、验证器、frontier和选择器，
  VETO关闭。proposer使用受约束的副本；评测失败向上抛出，不能冒充零分成绩。
- 增加完整h0缓存和已生成候选的哈希核验/导入，保留所有失败作业，避免重跑基线和API。
  仅将无歧义slot=h1归一化为name=h1；不改候选源码、评分或搜索范围。计算节点启动时
  清除登录节点代理，CLI的本地请求显式绕过代理；临时CLI状态不进入候选导入。
- `audit_native_search.py`独立重放输入/输出token与棋盘判定。原实现平分时依赖文件
  遍历顺序；第一版审计错误假定h0优先，已保留并修正为核验最优平分集合。
- 三个作业完成一个真实工程搜索迭代；h0/h1均0/8，h1平分被接受，无性能提升。
  新增7次GLM请求，累计保守0.047257元；本阶段0.426389 GPUh，包含失败占用。
- `prepare_chess_splits.py`委托原converter/splitter，固定来源并排除全部64工程ID/局面组，
  产出128/32/64题。独立续着与交叉重叠检查通过，未调用模型。
- 上述代码对应共享候选生成/评测/E3选择及数据协议；不增加VETO公式或贡献点。
  81项CPU测试通过。原域四条件/三seeds、非空RSFT和实际权重切换仍未完成；未push。

## 2026-09-09 — 固定4B运行与原AgentLoop输入核验

- `run_vllm_runtime.sh`增加可选计划参数；同一驱动执行固定官方4B、同8题、同预算。
  2B已提交源码快照`476baed`保留旧launcher哈希，旧计划须在该快照回放。
- 221725完成8次生成，11分10秒；全部截断且无成功，独立token/落子重放通过。
  CPU检查确认原AgentLoop格式函数与4B服务token一致，保留一次准备接口错误。
- 新增4B运行计划、输入核验、完整输出和控制器资源证据。累计已计量1.205GPUh，
  无新增付费API调用；无梯度更新、正式搜索、科学增益或push。

## 2026-09-09 — 原生vLLM执行、独立重放及训练配置恢复

- 新增`ours/vllm_runtime.py`、`run_vllm_runtime.sh`及独立`audit_vllm_runtime.py`，
  在原Chess评测器入口接入本地HTTP客户端，保存完整token证据。新代码对应E4共享
  执行/测量层；不修改E1–E3，不认领WHALE的成功过滤或SFT目标为新贡献。
- 新增`run_native_rsft.sh`并为`training_bootstrap.py`增加显式Ray worker初始化hook；
  `ours/config/data/legacy_data.yaml`来自固定上游生成配置，三领域data映射相同。
  原训练解析、配置验证和8题数据契约通过，GPU训练和权重切换仍未执行。
- 作业221663的8次真实生成和逐步重放通过，但8次均截断、0成功。两次先前启动/请求
  失败与CPU诊断错误保留。新增4项接口测试，总76项通过。GPU计量包含失败占用。
- 固定官方Qwen3.5-4B版本，为下一步原默认模型检查准备。无新增智谱调用，无push。

## 2026-09-09 — 智谱proposer实接、完整解码诊断与隔离训练依赖

- 用户提供智谱API授权及45元预算。新增`ours/probe_glm.py`、`glm_gateway.py`及国内
  provider配置：密钥只存仓库外；持久化预留/核销、指定模型与端点、完整响应计量。
  不将CLI估算美元金额视为真实账单；API连接与文件工具排错共8次请求。
- `run_proposer_probe.py`调用未改源码的WHALE wrapper；`confined_exec.py`限制子进程
  文件访问。公开上游文件字节核验及越界读取拒绝检查通过，自动审批复审放行。
  四次Read后超时记录保留，完整SSE离线解析/重放定位到响应定界；原样传输响应加上
  Content-Length后第五次Read→Edit检查成功，仅指定提示语改变，无正式搜索。
- `run_chess_smoke.py`支持1–8题计划，`run_chess_decode.sh`固定两分片，
  `audit_chess_decode.py`逐题独立核验本次单调用诊断。221597完成8题但全部截断，
  原生legal_rate定义与逐步合法率的差异已记录，不把0/8报告为正式baseline成绩。
- `training_requirements.txt`及完整哈希锁文件定义隔离兼容环境，明确NumPy约束改动。
  `training_bootstrap.py`只补接同固定WHALE版本中的verl.tools；训练入口/actor/rollout
  CPU导入通过，GPU训练和真实权重传播未验证。
- 上述文件对应共享proposer/target/训练执行层及实验核验，不新增E1–E4公式或科学贡献。
  上游源码、旧项目和已有环境未改；本地保存源码与真实证据，未push。

## 2026-09-09 — 原Chess执行器的本地兼容接线

- 本地实现与证据快照为 `60b836c`；完整快照检查补充记录了原始Slurm输出末尾空白及
  已冻结compat模块末尾空行，保留其字节以便核验。此前工作树检查未覆盖未跟踪文件。
- 新增 `ours/local_completion.py`、`ours/compat/autoharness_textarena/`，按公开runner
  消费的接口提供本地Qwen3.5推理、实际token日志和权重身份核验；这是本项目兼容实现，
  不冒充恢复的原client。runner/harness/verifier和全部上游文件未改。
- 新增 `prepare_chess_runtime.py`、`run_chess_smoke.py` / `.sh` 及4项权重/配置完整性测试。
  它们属于共享实验设施，不增加Method公式；总计66项测试通过。
- 隔离安装chess1.11.2，下载并核对官方Qwen3.5-2B固定版本；固定Lichess来源取64条、
  按原转换器事先选8题。两次Arrow退出异常通过同步读取修复，样本字节未变化。
- 保留错误提交221576的失败证据，增加CPU check-only和成功返回码后的提交步骤。
  成功作业221577完成8次生成、25秒H800占用；8次合法落子、0题解出，无截断。
- 所有数值是未训练、小样本、修改解码配置的工程检查。未运行proposer、训练、
  原配置多步复现或科学消融；没有成功轨迹可用于当前RSFT样本。未push或修改旧项目。

## 2026-09-09 — 本地视觉输入与E1真实接线检查

- 新增 `ours/visual_task.py`：共享的图像/问题输入与固定答案校验；对应E1输入端，
  不增加优化公式。新增 `run_visual_smoke.py` / `.sh`，保留每次真实生成、来源、
  权重、processor与输入哈希。原接受模块和上游文件未改。
- 新增8项固定答案格式测试，总计62项CPU测试通过。固定8对工程图像的32次GPU生成
  在作业221552完成；1张H800，44秒，约0.01222 GPUh，外部API费用0。
- 新增运行前计划、原始生成、控制器完成记录和独立重评分。更新当前状态及公式映射。
  工程图表有图正确率达到天花板，不能支撑VETO增益；未训练、未搜索、未通过F0–F5。
- GLM/本地proposer仍为未连接配置。未安装新依赖、购买额度、push或修改旧项目。

## 2026-09-09 — 新Goal的独立实现与proposer成本/替代方案

- 新增 `ours/evidence.py`（E1）、`acceptance.py`（E2–E3）、`adapter.py`（E3边界），
  `select.py`用于保存评测的选择重放；未修改上游RSFT/E4或任何verl文件。
- 新增 `chart_pairs.py` 的工程图像生成/独立像素oracle，`preflight.py` 的只读复现
  检查，`budget.py` 的假设费用计算，`proposer_profiles.py` 的隔离后端配置及54项测试。
  这些基础设施和测试不是额外的科学创新；真实VLM adapter/训练/搜索仍未接通。
- 新增EXPERIMENTS.md、ours/README.md、notes/proposer_options.md和真实离线报告。
  根据新Goal更新README/PROJECT_STATUS/thesis的实施状态；F2–F5阈值未改，原要求
  仍可用initial-review标签回溯；本轮无性能数据可被阈值变动影响。
- 补查CHILL-Harness、Don't Blink和GLM官方模型/价格；完整论文新颖性审核仍未完成。
- 未安装依赖、购买额度、调用付费API、提交GPU或push；旧项目文件未动。

## 2026-09-09 — 阶段 0/1 研究笔记

- 来源：`https://github.com/krafton-ai/WHALE.git`，commit `fbe125eb7abea7f760c99ab9acc1a6261e708fc6`，该提交日期 2026-09-04。
- 论文版本：[arXiv:2609.00196v1](https://arxiv.org/abs/2609.00196v1)，2026-08-31 提交；代码与论文作为两个独立版本核查。
- 只新增本项目 Markdown 研究文档，完整复制 LICENSE、NOTICE；未修改上游 checkout，未写研究实现或启动实验。
- `LICENSE` SHA-256：`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`。
- `NOTICE` SHA-256：`e1dbcce250ef1fc2e176522987cdb47bbc714eb0c95f167525a6f827e808fbdf`。
- 本项目文稿及图示为重新撰写；没有搬用原论文段落、公式排版或图表。上游原仓库资产仅作为完整来源 checkout 保留，不是我们的论文资产。
- 以后每项源文件修改记日期、路径、目的，并留可对照此 commit 的 patch；新增实现逐文件注明本项目新增。任何基础设施修复统一应用于所有条件，不能计为方法贡献。

## 同日 — 用户目标更新与文档核验

- 按更新的轻量/视觉方向重写候选选择：主线 attack #10，辅助 #5；真实分叉 lookahead 不再作为正式算法。新增 VETO thesis、状态文档，并纳入 CAT、AutoDesign、CPL、PAPO、CFPO、DeFacto 等碰撞结果。
- 没有把论文/方法假设写成已观察结果，没有创建 ours/、results/ 或 paper/，没有实现、训练或推送。
- 实际文档核验通过：7份 Markdown、18个本地链接、44个固定 commit 源码链接的文件/行号存在性；attack surface 恰好10项；无尾部空白。链接存在性不等于运行时正确性验证。
- 实际 git status 显示上游 checkout clean，HEAD 与固定来源一致；顶层 LICENSE/NOTICE 与上游逐字节相同，SHA-256 与上列值一致。
- 本轮的验证是文档与来源完整性检查，不是 baseline 数值复现。研究执行链之外的 vendored 模块未声称逐行审计。

## 2026-09-09 — 启用本地版本管理

- 根据用户请求初始化本项目 Git（main），提交当前文档并以 initial-review 标记初始快照。
- 通过 .gitmodules 将已存在的 upstream/WHALE 登记为 submodule，保留原仓库历史并固定原 commit；没有更改上游源文件。
- 新增 .gitignore，排除本地环境、凭据、缓存、下载数据和 checkpoint；README 与状态文档增加回溯说明。
- 提交身份仅配置在本仓库，沿用现有项目的用户身份；未修改全局 Git 设置，父仓库未配置远端，未 push。
# 2026-09-10: Qwen3.5 visual training interpretation corrected

Real corrected run224347 and followup224358 completed. All46 accepted image
trajectories have matching actual backbone inputs and nonzero feature gradients;
all723 native parameter tensors changed, including all297 vision tensors. Fixed
h0 development scores remain117/128 images and53/64 pairs, with4 changed answers.
Corrected-weight search/continuation remain pending; no VETO efficacy claim.
Five fused-forward CPU tests, seven resume-path tests, and real batch/input/full
parameter audits pass. Old text-loss lineage is blocked from visual continuation.

Confirmed the pinned generic fused forward drops pixel/grid kwargs for Qwen3.5.
Old image-bearing rollouts and changed language weights do not certify multimodal
RSFT. Added an isolated native multimodal-backbone/fused-likelihood adapter and
real backbone input/feature-gradient observations. Four real tiny-model CPU tests
pass. Restart the same first W batch from common initial weights; do not resume
the old text-loss checkpoint as a corrected method comparison. Method: shared E4.

GLM h1's full evaluation and early-stop/resume acceptance completed on old theta1:
H128/128, C64/64, repeat C63/64. Both original and paired selection choose h1;
no independent VETO benefit or formal expansion follows from this result.

Complete h0/LR tables now compile in `paper-build-v14` with source/result hashes.
The v13 missing-public-font failure is retained; v14 downloaded the font into
the TeX cache. Formal method tables and empirical figures remain pending.

The bounded setup H/C controller connects completed throughput and the next
authorized E1 measurement, with no automatic retry or broad expansion. Four
tests pass; PID2431747 is live. This is the E1 measurement handoff, not an
additional VETO mechanism. Batch16 completed512 images at1.424s/image; batch32
is still required before selecting shared endpoint concurrency.

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
