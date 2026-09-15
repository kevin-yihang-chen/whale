# Project status

更新于2026-09-10，香港时间。**共同theta0首轮完整审计为h0=1/32、h1=8/32、h2=5/32、h3=0/32；原选择器选中h1。第二轮h4=6/32、h5=7/32、h6=8/32均已审计，原选择器仍保留h1。第三轮h7=6/32、h8=7/32、h9=7/32已完整审计，仍保留h1；第四轮h10=8/32、h11=6/32、h12=6/32已完整审计，仍选h1；第五轮h13/h14均为8/32且已审计，最后h15作业223617运行中。WHALE/FST三种子第一阶段计划均已通过预检，尚未训练；weight-only seed43/44的终态与规范导出入口也已通过CPU检查。**
本页只记录最新状态。[完整历史快照](ours/history/PROJECT_STATUS_20260909_before_export.md)和
[实验日志](EXPERIMENTS.md)保留之前的0/8、失败及运行中记录。

## 正在执行的对照试验

222801于01:02:44–02:11:02在两张H800运行4098秒，计2.276667GPUh。固定h0、共同未训练
BF16起点、seed42、128条新轨迹；139次请求全部返回，1039118输出token、0错误，未超预算。
完整原生回复/token/mask重放与独立棋盘检查通过：第1批3/64 accepted，来自00BQD两条及
007fJ一条；22989个loss token、1次SFT更新。第2批0/64，原生跳过更新，两个checkpoint均保存。
[终态与checkpoint哈希](results/controlled-pilot-222801/result.json)、
[完整轨迹审计](results/controlled-pilot-schedule-deviation-audit-222801.json)保留以下预检偏差。

**原预检直接遍历RandomSampler，遗漏多进程StatefulDataLoader初始化的额外随机流推进，
因此预写的具体题序不正确。** 从冻结的原生代码/配置/seed推导真实DataLoader，得到的
两批题序与128条实际轨迹完全一致；第一批保存的sampler随机状态也逐字节一致。
原计划和原审计器保持不变，新报告显式标记预检偏差，不宣称与原预写ID列表完全一致。
没有丢弃轨迹、改题或重新采样。后续训练计划必须采用真实DataLoader预检。
[计划](results/controlled-weight-only-seed42-plan-20260910-v2.json)、
[运行快照](results/controlled-pilot-222801/snapshot-20260910-014733.json)、
[四条件试验协议](ours/controlled_pilot_protocol.md)。

原生Ray worker已执行共享种子hook，初始NCCL同步后实际释放6GiB+512字节闲置CuPy缓存。
WHALE/FST的多候选适配器通过2轮×3候选原生CPU循环检查；测试评分是合成数据，未调用API。
新审计入口只修正诊断用的预期ID矩阵，保留完整配置核验、全部轨迹重放及偏差声明。
新的[分阶段搜索器](ours/staged_search.py)已通过6项原生CPU检查：5轮选择历史、满分提前
停止及FST拒绝行为与原循环一致；旧h0/h1的69次真实回复全部重放通过，h1原始两分片
也通过新合并/审计入口。测试未产生新模型或API调用，不等于真实完整搜索已完成。
[harness-only seed42计划](results/controlled-harness-only-seed42-plan-20260910-v1.json)预检通过，
共同theta0、5轮×3槽、MH32。账号当前限制同时只有1个提交作业；调度器等待名额释放后
提交离线2H800评测，仅在完整审计与Slurm COMPLETED/exit0后交还原选择器并调用GLM。
01:50:50控制器等待提交名额；训练结束后自动提交222889，02:11:21开始两张H800的h0评测。
h0于02:28:54 COMPLETED/exit0，1053秒/2H800/0.585GPUh。两worker的共同初始权重
8坐标抽查及进程seed42记录通过；完整32题审计为1/32，33次请求、260128输出token。
[完整审计与终态](results/controlled-mh-222889/result.json)。
[02:22快照](results/controlled-mh-222889-start-snapshot.json)：两分片共16次调用完成、
130064输出token、0错误；这是运行进度，不是完整32题成功率。
[启动快照](results/controlled-harness-only-seed42-start-20260910.json)记录计划、h0请求及进程。
02:31:10原控制器因GLM生成`logs/iteration_1/report.md`而停止；原作者允许这类报告，
本地白名单仅接受`iteration_001`，属于报告路径兼容问题。没有候选GPU评测或选择发生。
第一轮6次GLM调用已生成h1/h2/h3；三者原生校验通过，原始候选、报告和失败日志保留。
[显式恢复修订](results/controlled-harness-only-seed42-recovery-amendment-20260910-v1.json)
冻结中断现场，仅规范当前报告的文件名；重用已审计h0和已付费首轮提案，从原循环恢复。
三项原生测试验证五轮选择历史、满分停止、候选顺序一致，且不重复h0或第一轮生成；
保护证据改写、额外候选、跨轮报告及重复报告仍拒绝。恢复不是自动重启；再次中断须另行核验。
02:46:33恢复控制器PID1019931从原循环重用h0及首轮提案，零新增API调用，提交h1作业
222891；两侧实际worker参数检查通过。h1于03:03:49 COMPLETED/exit0，1036秒、
0.575556GPUh；完整32题审计为8/32，32次请求、245056输出token。
[h1审计与终态](results/controlled-mh-222891/verified-summary.json)。03:04:04自动进入h2，
新作业222958运行中；原选择器仍等待h2/h3完整结果，没有提前采纳h1。
[h2运行快照](results/controlled-mh-222958-start-snapshot.json)保留当时进度；完整结果为5/32、
32次请求、256392输出token，03:21:06 COMPLETED/exit0，1022秒/0.567778GPUh。
[h2终态及审计](results/controlled-mh-222958/verified-summary.json)。h3作业222969于03:21:35
开始，两worker参数检查通过，[h3快照](results/controlled-mh-222969-start-snapshot.json)记录运行状态。
[恢复启动及原候选哈希](results/controlled-harness-only-seed42-recovery-start-20260910.json)。
h3于03:38:37 COMPLETED/exit0，1022秒/0.567778GPUh；完整32题0成功，33次请求、
260128输出token，[终态审计](results/controlled-mh-222969/verified-summary.json)通过。
03:39:05原选择器在[第一轮完整比较](results/controlled-harness-only-seed42-round1-comparison-20260910.json)
中选中h1。随后GLM用8次调用生成h4/h5/h6；03:40:45因候选元数据为裸列表、使用
`harness`字段而触发本地严格解析错误。候选未进入GPU评测，原始提案与失败现场保留。
[第二次显式修订](results/controlled-harness-only-seed42-metadata-amendment-20260910-v1.json)
仅包裹裸列表、映射无歧义name/slot/harness/id字段，并保留原字段；候选代码不改。
在原生`run_evolve(start_iteration=2, iterations=4)`内重放第二提案，已审计h0–h3
不重新推理，首轮历史不重复追加。三项新原生测试验证两次中断后的五轮完整历史及
满分停止与连续运行一致；额外文件、保护证据改写及冲突元数据仍拒绝。原冻结计划不改，
新的修订绑定源文件、两次恢复证据、原始提案、四个评测和规范化前后元数据。
03:50:35第二恢复控制器PID1101994重放原提案、零新增API调用，提交h4作业223135。
两worker参数检查通过，[启动快照](results/controlled-harness-only-seed42-metadata-start-20260910.json)
记录原候选哈希、元数据映射、两份实际worker证明和Slurm状态；完整五轮搜索尚未完成。
h4于04:07:51 COMPLETED/exit0，1036秒/0.575556GPUh；完整32题6成功、32次请求、
255996输出token，[终态审计](results/controlled-mh-223135/verified-summary.json)通过。
04:08:06自动提交h5/223199，两worker参数和种子检查通过；[启动快照](results/controlled-mh-223199-start-snapshot.json)
记录实时状态。第二轮尚待h5/h6，仍保留第一轮接受的h1；未产生新的GLM提案调用。
h5于04:25:19 COMPLETED/exit0，1033秒/0.573889GPUh；完整32题7成功、32次请求、
246051输出token，[终态审计](results/controlled-mh-223199/verified-summary.json)通过。
04:25:36自动进入h6/223204；两实际worker检查通过，[启动快照](results/controlled-mh-223204-start-snapshot.json)
保留当时状态。第二轮尚未比较/选择，当前仍接受h1；完整五轮搜索未完成。
[h5固定回复诊断](results/controlled-h5-223199-parser-comparison.json)显示32次回复在候选
与原h0 parser下动作/合法性均无分歧，7个成功都保留原parser动作；5个成功结束thinking、
2个含长度截断。它没有显示parser扩展改变本次动作，也不能据此估计提示词或parser的因果收益。
h6于04:42:38 COMPLETED/exit0，1022秒/0.567778GPUh；完整32题8成功、32次请求、
247598输出token，[终态审计](results/controlled-mh-223204/verified-summary.json)通过。
[第二轮原生比较](results/controlled-harness-only-seed42-round2-comparison-20260910.json)仍保留h1。
随后GLM用7次请求生成h7/h8/h9；04:44:44因裸列表使用`candidate`字段而停止。
候选编号/路径/父候选一致，代码尚未进入评测。新的[第三次显式修订](results/controlled-harness-only-seed42-candidate-amendment-20260910-v1.json)
仅增加一致的candidate别名映射，保留全部字段、提案和失败证据；原冻结文件不改。
控制器PID1169570从原`start_iteration=3, iterations=3`接续，重用七个完整评测和原第三提案，
没有重复API/模型调用。三次中断与连续五轮及满分停止的原生测试通过，演化历史一致。
h7/223272于04:53:28开始，两实际worker权重/seed证明PASS；[恢复及运行快照](results/controlled-harness-only-seed42-candidate-start-20260910.json)
记录原候选哈希和0次重提案调用。第三轮与完整五轮尚未结束。
h7于05:10:54 COMPLETED/exit0，1046秒/0.581111GPUh；完整32题6成功、32次请求、
250228输出token，[终态审计](results/controlled-mh-223272/verified-summary.json)通过。
h8/223273于05:10:59开始，两实际worker检查PASS，[启动快照](results/controlled-mh-223273-start-snapshot.json)
记录当时来源与预算。h8于05:28:06 COMPLETED/exit0，1027秒/0.570556GPUh；完整32题
7成功、32次请求、249983输出token，[完整审计](results/controlled-mh-223273/verified-summary.json)通过。
h9/223332于05:28:30–05:45:57 COMPLETED/exit0，1047秒/0.581667GPUh；完整32题
7成功、32次请求、243851输出token，[完整审计](results/controlled-mh-223332/verified-summary.json)通过。
[第三轮原生比较](results/controlled-harness-only-seed42-round3-comparison-20260910.json)仍保留h1。
第四轮GLM完成9次请求，生成h10/h11/h12；[05:53快照](results/controlled-mh-223408-start-snapshot.json)
保留h10运行状态。h10于05:47:59–06:04:52 COMPLETED/exit0，1013秒/0.562778GPUh；
完整32题8成功、32次请求、249684输出token，[完整审计](results/controlled-mh-223408/verified-summary.json)通过。
h11/223466于06:04:58开始，两worker权重/seed证明PASS，[06:07快照](results/controlled-mh-223466-start-snapshot.json)
保留当时来源与预算。h11于06:22:02 COMPLETED/exit0，1024秒/0.568889GPUh；完整32题
6成功、32次请求、248721输出token，[完整审计](results/controlled-mh-223466/verified-summary.json)通过。
h12/223507于06:22:30开始，两worker权重/seed检查通过，[06:25快照](results/controlled-mh-223507-start-snapshot.json)
保留实际状态。第四轮尚待h12，仍接受第三轮的h1；完整五轮尚未结束。
[原生选择语义检查](results/native-frontier-retry-semantics-20260910.json)确认原frontier会保留
同调用数的低分候选，不能解释为严格非支配集；原最高分优先接受规则不变。环境retry值
是默认值，选中harness的声明按原逻辑优先；留出时保留其原行为，不强制覆盖。
联合WHALE/FST第二阶段启动、恢复证据核验、完整轨迹审计、终态与导出均通过CPU检查，
实际联合GPU恢复仍未执行。

weight-only最终checkpoint的[规范导出计划](results/controlled-weight-only-seed42-export-plan-20260910-v1.json)
已冻结：保留原生model/extra/data.pt，逐元素检验导出值等于原生BF16转换，并分别计量相对
共同theta0的FP32/BF16变化。保持原推理配置资产语义，记录原题序预检偏差，不把空更新
或舍入后的不变结果判为失败。5项相关原生CPU测试通过；真实全参数比较尚未执行。
导出仅用CPU，Slurm计划需最小1RTX4090/4CPU/64GiB，20分钟上限、最多1/3GPUh。
当前唯一提交名额由搜索占用，导出没有提交，不与搜索控制器争用名额。

[seed43计划](results/controlled-weight-only-seed43-plan-20260910-v1.json)及
[seed44计划](results/controlled-weight-only-seed44-plan-20260910-v1.json)已完成Hydra解析和
实际RLHFDataset/StatefulDataLoader的两批预检，并与独立索引探针的ID/保存sampler状态
一致。各128条训练轨迹预算、固定h0和连续原生两批行为不变；没有启动训练或增加API调用。
新的continuation入口只处理尚未开始的两种子，保留seed42原文件。三种子真实数据加载器
回归及父进程Python/NumPy/Torch随机状态恢复测试通过。
新计划还通过[原生运行配置与现有审计器的逐字段一致性检查](results/controlled-continuation-runtime-audit-20260910.json)；这是CPU配置检查，不是新训练轨迹审计。

[controlled_continuation_result](ours/controlled_continuation_result.py)现接收seed43/44的原生完整
128条审计，核验两批指标、实际sampler/scheduler、唯一step1/2、启动配置及Slurm终态。
[导出适配器](ours/controlled_continuation_export.py)复用同一原生规范导出器，独立保留
weight_only条件、种子、累计更新和终态，不把它们重标为联合条件。共享导出只增加进程内
配置接点，原联合行为回归通过；尚无正式联合/continuation导出计划，原冻结训练源未变。
[seed42旧格式导出终态](ours/controlled_checkpoint_export_result.py)另外接通，保留原预检偏差，
不改原已冻结导出器与计划。11项原生检查通过，基础119通过/87依赖跳过；
[实现记录](results/continuation-finalization-implementation-check-20260910.json)绑定源码与日志。
两种子终态测试使用明确标注的历史state/指标及合成身份，未生成真实seed43/44结果。
六次原生小VLM CPU导出覆盖原joint两阶段及continuation两种子身份；零变化与仅FP32变化
均保留。旧seed42闭合测试使用合成serializer，实际逐参数比较及终态拒绝路径执行。
没有新增任务模型/API调用、真实训练或导出作业，也没有加载test任务。最终来源矩阵和
64题留出执行器仍待接入；这些准备不能代替三种子的实际训练。

[WHALE seed42第一阶段](results/controlled-whale-seed42-phase1-plan-20260910-v1.json)和
[WHALE-FST对应计划](results/controlled-whale-fst-seed42-phase1-plan-20260910-v1.json)已通过
共同theta0、原h0、实际DataLoader及完整Hydra预检。它们各运行1批8题×8轨迹、累计
target=1，保存原生model/extra/data.pt后停止；不复用已完成weight-only前缀。
新审计器完整重放历史第一批64条/75次请求的fixture，3条accepted、22989 loss tokens
与旧审计一致；只重映射fixture来源元数据，原档案不改，不算联合条件的新实验。
两项原生循环测试覆盖first阶段与已处于step1的循环边界、成功/空更新均保存和同步；
没有执行GPU checkpoint恢复。两个新第一阶段计划尚未提交；第二阶段审计/终态入口已通过CPU fixture检查，真实GPU恢复与第二批执行仍待完成。第一阶段每条件2H800/16CPU/160GiB/60分钟，最多2GPUh。

第一阶段的[终态封存](ours/joint_training_result.py)和[规范导出入口](ours/joint_checkpoint_export.py)
现已接通并通过CPU检查。终态必须包含完整64行审计、全部请求记账、真实SFT更新数、
Slurm退出证据、唯一global_step_1及与预检相同的sampler状态；模型/extra/data.pt保留。
导出逐参数检查原生FP32和共同theta0的变化及精确BF16转换，另核对推理资产与导出作业终态。
两项真实小VLM导出测试覆盖34个活动张量/42576元素，零变化和仅FP32有变化均通过；
三项完成记录测试覆盖原生历史metrics/保存状态、完整报告、导出计划绑定及篡改拒绝。
历史数据仅用于明确标记的fixture，没有被重标为联合结果；临时fixture相对路径失败日志
与修复后的v3/v4通过记录均保留。没有提交新的训练/导出GPU作业；实际联合checkpoint、
联合MH实跑、真实第二阶段GPU恢复及留出集结果仍未完成。

新的[联合搜索入口](ours/joint_staged_search.py)已实现，继承原五轮搜索与两分片完整审计。
它要求同条件/同seed的独立phase1训练、规范导出及两作业终态，绑定可恢复的原生
checkpoint；actual target来自该导出。只复用原harness-only生成器的数据/推理/预算设置，
不复用其评测或把它的共同theta0资格当作联合权重证明。WHALE-FST仍受提示词AST限制。
三项原生CPU测试通过：五轮、满分停止和FST拒绝与原流程一致；两历史分片32题/34次
回复由新worker路由合并并完整重放；混合condition/seed、配置或来源改动被拒绝。
来源fixture的训练/终态证据是合成stub，历史分片仅属工程检查；不能视为真实联合运行。
新增export终态路径采用绝对表示，真实小VLM导出检查覆盖相对/绝对CLI路径闭合一致。
尚无独立联合第一阶段checkpoint，故没有生成正式联合搜索计划或新增GPU作业。

第二阶段的[joint_training_phase2](ours/joint_training_phase2.py)要求同条件/同seed的完整联合
搜索证明，选中harness及phase1原生checkpoint一致。[search_completion](ours/search_completion.py)
逐轮重算原生候选合法性、FST拒绝、分数、Pareto、选择与停止，核对全部私有审计及公开
档案，不能将少于五轮且没有满分停止的搜索交给训练；不改写原档案。
[native_resume_observation](ours/native_resume_observation.py)在原loader返回前比较全部actor
FP32值、scheduler及CPU/CUDA/NumPy/Python RNG，读取尚未消费的DataLoader保存状态，
不调用会创建迭代器的state_dict()。Adam保持原生不恢复约定。原生循环仍负责权重同步。
CPU检查包括完整原生小模型加载、随后随机数一致、实际旧data.pt→正确第二批题序，以及
上游工作目录中的组合hook幂等性。历史weight-only数据仅作明确标记的配置/loader fixture，
没有生成正式phase2计划，也没有把它认作联合checkpoint。真实GPU恢复、第二批实际轨迹、
联合训练与留出集比较仍待完成；第二批审计/终态实现的后续检查见下。
[本轮实现检查](results/joint-phase2-implementation-check-20260910.json)记录12项原生集成通过、
基础116通过/81项缺少原生依赖跳过，以及五份既有冻结计划的全部source文件哈希未变。
首次集成的负例夹具遗漏删除第五提案请求，先被额外请求检查拦截；修正夹具后v2通过，
失败日志保留，没有因此修改真实档案或放松完成条件。

第二阶段的[完整轨迹审计](ours/audit_joint_training_phase2.py)、[恢复证据核验](ours/joint_resume_receipts.py)
及[终态封存](ours/joint_training_phase2_result.py)现已接通。检查64条实际回复/token/mask、
独立棋盘reward、完整GPU恢复记录、B2 sampler状态与Slurm终态；分别记录本批及累计更新数。
空第二批仍保存checkpoint，不能把第一批的一次更新记成第二批更新。
[joint_checkpoint_export](ours/joint_checkpoint_export.py)现支持phase1/phase2，同一原生导出器
逐值验证BF16转换并保留完整native恢复文件；FP32/BF16均不变也属有效结果。
最终12项原生集成通过（64.651秒）；基础117通过/84依赖跳过。64条旧第二批仅作明确的
重放fixture；两个阶段各自的零变化/仅FP32变化由真实小VLM CPU导出验证（34张量、42576元素）。
合成Slurm/GPU元数据没有被记录为真实联合实验；小模型无image processor，不验证视觉输入。
首次终态及集成测试分别遗漏fixture run-name替换、upstream链接，修复fixture后通过，失败日志保留。
[实现检查](results/joint-phase2-terminal-export-implementation-check-20260910.json)绑定源码、日志、
五份既有训练/搜索计划和三次恢复修订的未变源文件。Method对应
`theta2 = RSFT(restore(C1), h_star; B2)`与`theta_eval = BF16(theta_native)`两条交接箭头。
真实联合第一批checkpoint、联合搜索、第二批GPU执行和最终留出评测仍未完成。

h12/223507于06:39:35完成，1025秒/0.569444GPUh；32题6成功、32次请求、238780 token，
[完整审计](results/controlled-mh-223507/verified-summary.json)通过；[第四轮原生比较](results/controlled-harness-only-seed42-round4-comparison-20260910.json)仍保留h1。
h13/223573于06:59:37完成，1026秒/0.570000GPUh；32题8成功、32次请求、246575 token，
[完整审计](results/controlled-mh-223573/verified-summary.json)通过。h14/223615随后启动，
07:05:54的[快照](results/controlled-mh-223615-start-snapshot.json)记录两个worker权重/seed PASS。
第五轮尚待h14/h15；h13未超过h1，尚无完整五轮搜索结果。

h14/223615于07:16:56完成，1015秒/0.563889GPUh；32题8成功、32次请求、236698 token，
[完整审计](results/controlled-mh-223615/verified-summary.json)通过。h15/223617为最后候选，
07:19:22的[快照](results/controlled-mh-223617-start-snapshot.json)中两个实际worker证明PASS。
当前五轮仍未结束；h13/h14均未超过h1。

[六份剩余种子预检](results/remaining-trial-plans-implementation-check-20260910.json)全部PASS：
harness-only seed43/44固定共同theta0、原h0、各自五轮；WHALE/FST seed43/44独立第一批
均通过真实DataLoader及原生Hydra配置核验，尚未训练。新[independent_harness_search](ours/independent_harness_search.py)
继承原循环，预声明已经验证的无歧义候选别名/报告路径兼容；原seed42冻结源不改。
2项原生CPU测试覆盖seed43/44完整五轮及满分早停，分数/Slurm为合成fixture，0实际调用。

一次性[排程入口](ours/finish_seed42_search_then_export.py)PID1330451已启动，状态位于
`data/seed42-search-export-handoff-20260910/state.json`。它等待原控制器完整结束，生成
原生五轮完成证明，再复核独立weight-only seed42导出并在空闲作业名额上只提交一次。
07:24:16状态为WAITING_FOR_COMPLETED_SEARCH，尚未提交导出；失败会停止并保留现场。
这是两个独立条件之间的资源排程，harness-only候选不会成为weight-only训练输入。
导出[完整预检](results/controlled-weight-only-seed42-export-recheck-20260910.log)已PASS；
当前已有两个活跃协调进程，不应另行提交竞争作业或仅凭旧状态文件重启它们。

[存储预估](results/controlled-pilot-storage-projection-20260910.json)：07:13时可用约158.02GiB，
按真实文件大小估算剩余16个native checkpoint及15次规范导出需435.60GiB，未计日志/临时文件、
未扣除可能的相同文件。当前可继续下一次导出和训练阶段，完整保留矩阵需要更多容量或核实后的
存储复用。已询问可用共享路径；未删除产物、做物理去重或改动其他项目。

## 留出执行与最终汇总接线

[heldout_cohort](ours/heldout_cohort.py)封存全部12个trial身份或缺失原因；完整条目重新核验
原生搜索结束、同条件/种子两批训练及规范导出。零更新有效，seed42原题序预检偏差继续保留。
[controlled_heldout](ours/controlled_heldout.py)及[四分片执行器](ours/heldout_evaluation.py)
固定64题、四个16题逻辑分片，1/2/4卡按预声明方式分波；每分片新进程与worker权重/seed证据。
[完整审计](ours/audit_heldout_evaluation.py)重放实际回复、token和独立棋盘；终态还核对
Slurm、启动身份、分波顺序、实际GPU/CPU/RAM。推理没有RSFT loss mask，未声称检查该项。
[汇总器](ours/heldout_comparison.py)输出JSON/CSV/Markdown，三种子齐全才给mean/sample SD，
保留逐种子配对差值与缺失原因。成本是所列终态（含失败）的已知小计，未列来源保持空值。

6项原生CPU测试通过/19.174秒，2项汇总测试通过/0.024秒，基础124通过/90依赖跳过。
分片fixture将旧MH32题各复制两次为64个明确标记的ID，只验证调度/覆盖/原生重放；
训练/导出父终态及Slurm在封存测试中使用stub，不计作新试验。没有正式cohort、留出执行
计划、test题目访问或新模型调用。[实现记录](results/heldout-implementation-check-20260910.json)
绑定源码/日志，并核验8份冻结执行记录及旧seed42导出源不变。Method对应
`A[c,s] = (1/64) sum_i exact_solved(theta[c,s],h[c,s],x_test[i])`的最终测量框，未改变优化目标。

## 当前可核验结果

| 环节 | 结果 | 证据 |
|---|---|---|
| 共同theta0受控MH | seed42前三轮h0–h9依次1、8、5、0、6、7、8、6、7、7/32；仍选h1；第四轮h10/h11/h12为8、6、6/32，仍选h1；第五轮h13/h14均8/32已审计，h15运行中 | [h0终态](results/controlled-mh-222889/result.json)、[h1终态](results/controlled-mh-222891/verified-summary.json)、[h2终态](results/controlled-mh-222958/verified-summary.json) |
| 原生MH搜索 | 同一32题优化集，h0=0/32，GLM生成的h1=6/32；原选择器选中h1 | [h0审计](data/chess-alternation-step1/baseline-audit.json)、[h1审计](data/chess-alternation-step1/candidate-audit.json) |
| h1在线轨迹 | 64条/69次调用完整审计；11条accepted，来自2个训练题，30392训练token | [批次审计](results/alternation-batch-audit-222292.json) |
| 原生更新222640 | COMPLETED/exit0；原8+3两个mini-batch完成2次优化器更新、保存、NCCL同步 | [完成结果](results/alternation-recovery-222640/result.json) |
| 导出222703 | COMPLETED/exit0；723个活动参数完整，全部导出值精确等于原生BF16转换 | [参数审计](results/alternation-checkpoint-transition-222640.json) |
| 新worker加载222779 | COMPLETED/exit0；8个实际数值与新权重一致且区别于旧权重；1次请求返回2 tokens | [实际worker证据](results/alternation-vllm-handoff-222779/result.json) |
| 三种子222792 | COMPLETED/exit0；实际worker种子42/43/44、RNG及参数抽查通过，24个短请求/64 tokens | [种子证据](results/trial-randomization-222792/result.json) |
| 共享对照接线 | 6项原生CPU集成通过，三种子原生Hydra解析通过；FST拒绝非prompt变更 | [集成日志](results/controlled-native-fixtures-20260910.log)、[Hydra配置](results/controlled-native-hydra-preflight-20260910.json) |
| 共同初始权重222797 | COMPLETED/exit0；723个活动张量全部精确等于原始权重的BF16转换，0次训练更新 | [完整审计](results/canonical-initialization-20260910.json)、[终态与成本](results/canonical-initialization-finalization-222797/result.json) |

6/32是MH优化集分数，不是留出测试结果。6个成功中，4个正常结束thinking，2个由原生
解析规则从截断回复取出走法；离线旧parser也能解析这6条成功回复，尚不能把收益归因于
parser变化。11条训练成功轨迹不是11个独立题目，也不能与上一批不同题目直接比较。

新受控h1的32次固定回复在新旧parser下动作及合法性判断全部一致；8个成功中6个
正常结束thinking、2个含length截断。失败为23个预算耗尽、1个合法错误走法。
[固定回复诊断](results/controlled-h1-222891-parser-comparison.json)因此不能把8/32归因于
parser修改，也不等于“旧parser配新prompt”的重新生成消融；真实受控消融仍需另行运行。

## 本轮修复与训练证据

222292完成采样后前向OOM；恢复222512/fused512、222615/fused128均反向OOM。
三次失败及完整轨迹都保留。最终222640沿用同一完整缓存、原生fused128、success-filter、
学习率1e-7及8+3更新；新增NCCL子类在原生finalize后释放闲置CuPy内存。真实初始和
更新后同步各释放6GiB+512字节，活跃池计数不变。未删除有效token、修改优化目标或重复采样。

222640运行368秒/2H800，实际actor约141.72秒，保存35.32秒，更新后同步0.875秒。
Torch已分配/保留显存峰值75.803/77.352GiB。梯度范数8.16329是两步的原生均值；
loss1.2237049是两个mini-batch token平均损失之和，不能直接与上一轮单mini-batch比较。
关闭时有DataLoader worker Killed日志，发生在完整指标、保存及同步之后；Slurm为exit0。

相对本轮实际BF16输入，原生FP32中3,845,402,884个元素变化（426个张量），导出BF16后
14,479,791个元素保留变化（286个张量）。参数变化证明更新已写入文件，尚不证明任务性能提升。

新模型目录：`data/native-rsft/alternation-replay-222640/hf-checkpoint-canonical`。
原生checkpoint SHA：`f42e2219040b036e0b926fdd5c2ad0d0d0c7082d715c0177cc069c3433b68ada`。
导出权重manifest SHA：`0ee36bbe88866adb5f62ced2bcbe3db3740a6fba87c56f63bd486bf9e29e8518`。

新推理进程确实读到8个改变的embedding值；一次请求返回“Yes”/2 tokens，未按要求
回复“ready”，故不计作指令遵循成功。engine记录Shutdown complete，Slurm为exit0，
另有未显式destroy_process_group的退出警告，原样保留。这是新进程加载证据，不是
原训练进程内NCCL接收端的全参数指纹。

截至07:19:22，项目记录的保守API累计估算为0.474527元，62次已结relay调用；
首轮6次约0.051539元、第二轮8次约0.062230元、第三轮7次约0.071667元、第四轮9次0.084025元、第五轮11次0.117990元。
45元项目预算内可预留44.525473元；无未结请求，不是官方账单或账户余额。
成功恢复0.204444GPUh，CPU导出的必要GPU分配0.075556GPUh，新worker检查0.022222GPUh；
截至222779，已知项目分配合计9.161944GPUh。历史缓存返回耗时不当作推理吞吐量。
222792新增114秒/2H800/0.063333GPUh，已知项目累计9.225278GPUh。其生成前后CUDA随机状态变化，CPU随机状态不变；没有对GPU逐位确定性作保证。222794因自动生成的配置资产不一致而FAILED/exit1，77秒/0.021389GPUh已计入，累计9.246667GPUh。恢复222797仅移走新增generation_config并重做完整审计，COMPLETED/exit0；95秒/0.026389GPUh。项目当时累计9.273056GPUh。222801新增2.276667GPUh，222889新增0.585GPUh，222891新增0.575556GPUh，222958新增0.567778GPUh，222969新增0.567778GPUh，223135新增0.575556GPUh，223199新增0.573889GPUh，223204新增0.567778GPUh，223272新增0.581111GPUh，223273新增0.570556GPUh，223332新增0.581667GPUh，223408新增0.562778GPUh，223466新增0.568889GPUh，223507新增0.569444GPUh，223573新增0.570000GPUh，223615新增0.563889GPUh，已知终态累计20.131389GPUh。223617运行成本待终态。通知配置保留，未验证邮件实际送达。

## 四组对照的共同起点

共同模型目录：`data/models/qwen3.5-4b-common-bf16-v1`；manifest SHA
`260337087e4d739db98c2744cf429678cd2f87674e55cad9f22d3cfbe60c3d1a`。
723个独立活动张量/4,539,265,536个元素完整；原BF16参数完全不变，原48个FP32张量/
3840元素中有2791个元素（24张量）经BF16舍入改变。这些是精度转换，不是学习收益。
恢复前后权重文件哈希相同，推理配置/生成配置/词表/控制token/prompt token合同通过。
222801已从该共同初始化开始真实训练；harness-only另行从同一未训练manifest做MH评测。

## 研究状态与下一步

**VETO（Visual Evidence Tests for Optimization）仍是待验证假设，F0–F5均未通过。**
当前没有运行完整VLM训练/搜索闭环，也没有VETO或留出集收益数据。上述修复属于共享E4
成功筛选训练及权重交接；新机制候选仍是E1配对视觉验证、E2固定入阶段参考的非退化约束、
E3按任务分数再按调用数选择。源码与数学对应见[ours/README](ours/README.md)。

1. 分块后端、CuPy清理、三种子传播及共同初始权重已完成相应检查；接下来跑F0的
   weight-only、harness-only、WHALE-FST、WHALE四条件×3独立种子。FST提示搜索契约有
   明确标注的公开重建部分，不冒称恢复了未发布作者文件。
2. [留出评测协议](ours/controlled_heldout_protocol.md)及[冻结记录](results/controlled-heldout-protocol-20260910-v1.json)
   已在未加载test题目/答案前固定：每条件64题×3种子、原解码/评分、4个固定16题逻辑分片。
   首次打开test前封存全部12个trial的终态或未完成原因；仅完整模型/harness对进入评测。
   封存、四分片执行、完整审计与三种子汇总入口已通过CPU检查；当前尚未封存正式trial矩阵，没有正式最终评测计划或test分数。优先4H800并行，可按同一分片用2卡/1卡
   分波；[资源比较](results/heldout-resource-plan-20260910-v1.json)给出预估17–20/34–40/68–80分钟，
   各约1.13–1.33GPUh，不是test实测时间。当前搜索仍占用唯一提交名额。
3. 接通视觉任务闭环，验证视觉依赖与配对证据是否有效，再决定扩大数据和模型规模。
   最后再做强对照、多种子消融和第二CV任务；不预设90%贡献或中稿结论。

测试：最新基础环境124通过/92依赖跳过；未来独立种子搜索2项原生测试通过；留出执行/封存6项原生CPU测试、汇总2项测试通过；多种子终态/导出及联合回归11项原生检查通过；第二阶段轨迹/终态/两阶段导出12项原生集成通过；联合搜索3项原生检查通过；新增3项联合终态/计划绑定、2项真实小VLM规范导出测试通过；联合第一阶段2项边界测试、1项64轨迹归档重放fixture通过，两个实际第一阶段预检通过。3项真实数据加载器/continuation测试通过，seed43/44实际预检通过。另有3项搜索中断恢复、5项规范导出相关原生测试通过。4项状态恢复/真实sampler回归检查通过，真实RLHFDataset恢复到正确下一批。原checkpoint loader的小模型CPU测试验证模型、调度器、Python/NumPy/Torch RNG恢复及Adam不加载；GPU恢复仍待联合阶段验证。另有6项分阶段、4项多候选、1项两批次配置检查，以及3项原生恢复集成、2项CuPy清理边界、真实GPU
清理fixture及4B原生更新/全参数导出检查通过。所有实现留在ours/；上游固定提交
`fbe125eb7abea7f760c99ab9acc1a6261e708fc6`，未修改。更改仅本地提交，未push；
`beyond-entropy`未改。用户现有GLM调用授权仍有效，密钥不写入仓库。
