# VETO 存储与保留清单（2026-09-10）

## A组执行完成；B项明确保留（14:26，香港时间）

用户明确回复：“我授权A组，B组不授权。不清理就运行不下去了吗？”
本次仅执行A1–A4，B项MIMIC原始图像不获授权，保持原位。此授权不覆盖其他清理。

[完整执行回执](../results/veto-authorized-a-cleanup-20260910-v1/result.json)
及同目录authorization.json、preflight.json、operations.jsonl记录授权、逐路径身份和操作。
原提案及其元数据副本保留，不回写历史提案中的approved字段。

| 已执行对象 | 实际释放的文件分配空间 GiB | 保留核验 |
|---|---:|---|
| A1：32B模型14个权重blob及对应symlink | 62.10 | 配置、分词器、索引和revision仍保留；再次使用需下载权重 |
| A2：7B模型5个权重blob及对应symlink | 15.45 | 同上 |
| A3：pip http-v2的1480个缓存文件及空目录 | 3.80 | 已安装环境保持原位 |
| A4：step2重复权重替换为step1硬链接 | 18.31 | 两个路径、全部权重字节、独立恢复状态保留 |
| **合计** | **99.65** | **B项未清理** |

执行前119.46 GiB可用，完成时219.14 GiB可用；文件系统实际增量约99.69 GiB。
与逐文件分配空间的差异含目录块和并发作业写入，不能把两者混作同一计量。
两份checkpoint先完整重算SHA256，替换后再次完整核验保留inode，SHA均为
`1e3d65762a6e078375b7a8da5198cd47cb1cb6355995b2adeeeac4fa1ed7d0dd`。
两个路径现共享inode2365560、nlink2；将来如需原位修改，必须先复制为独立文件。
40项保留元数据/恢复文件哈希和33项活动冻结源码哈希复核通过。

**不清理也能继续近期工程验收。** 清理前119.46 GiB已高于顺序控制器的续训门槛
70 GiB（30 GiB工作空间+40 GiB余量）；导出和视觉服务门槛分别50/42 GiB。
原约520.72 GiB估算是正式矩阵保留模型、机制分叉及工作空间的整体规划，并非一次
训练要求。A组释放后，近期工程验收的余量增加，但完整矩阵的存储门槛仍未通过。
下一轮扩展应实测顺序执行、相同权重复用及可精确重建的存储方案；未经核验不能宣称
219 GiB足以容纳全部51个条件，也不能自动删除后续检查点或把B项当作清理后备。

本次脚本authorized_storage_cleanup_20260910.py属于复现资产管理；保存E4权重与恢复
状态的身份，不改变E1–E4计算、损失或实验预算。

## 用户要求扩大盘点后的具体候选（执行前历史清单）

以下保留最初提案及估算；授权与执行状态以本页上方完成记录为准。

| 分组 | 具体对象 | 可释放 GiB | 保留与影响 |
|---|---|---:|---|
| A1 | `/userhome/cs3/yihangc/Data/hf_cache/hub/models--Qwen--Qwen3-VL-32B-Instruct` 中清单列出的14个权重blob及对应symlink | 62.10 | 保留config、tokenizer、index、revision和独立元数据副本；当前VETO不用，旧项目再用需重下载 |
| A2 | `/userhome/cs3/yihangc/Data/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct` 中清单列出的5个权重blob及对应symlink | 15.45 | 同上；旧医学/7B模型试验可能引用，不称为无用文件 |
| A3 | `/userhome/cs3/yihangc/.cache/pip/http-v2` | 约3.83 | pip下载缓存；安装好的Python环境不在此目录 |
| A4 | `data/native-rsft/controlled-weight-only-seed42-222801/global_step_2/actor/model_world_size_1_rank_0.pt` | 18.31 | 全文件重新核验后与step1完全相同；拟用step1的硬链接保留step2路径和全部字节，两个名字共享只读权重；独立data.pt及恢复状态保留 |
| **A组合计** | 缓存清理＋完全相同权重去重 | **约99.69** | 当前可用空间可增至约219 GiB，仍不够完整正式矩阵 |
| B（另行决定） | `/userhome/cs3/yihangc/Data/MIMIC-CXR-JPG/files` | **588.28** | 原始医学图像；保留父目录的CSV、许可证、README、IMAGE_FILENAMES和SHA256SUMS。VCD、VCD_clean、LOCE-CCD-MIMIC-clean共123个代码文件提到MIMIC；重跑需恢复数据，重新下载需原访问权限和大量时间 |

A1/A2的具体blob/symlink、revision和已保存元数据见
[缓存清理提案](../results/veto-cache-cleanup-proposal-20260910-v1/proposal.json)。
A4的两个文件各20,700,225,459字节、不同inode，全文件SHA相同；见
[无损去重核验](../results/veto-checkpoint-deduplication-review-20260910-v1.json)。
B项的精确占用、保留元数据SHA和各项目引用见
[原始数据集独立提案](../results/veto-mimic-storage-proposal-20260910-v1.json)。

没有为MIMIC图像确认第二份完整本地备份。它不是软件缓存，当前VETO不使用它也不意味着
其他项目可以放弃。提案时A、B均未授权；随后用户只授权A组，明确拒绝B项。

## 初次冻结依赖与容量调查（执行前历史快照）

本清单为只读调查；没有删除、移动或修改既有模型。具体绝对路径、占用和匹配到的报告见
[机器可读清单](../results/veto-storage-inventory-20260910-v1.json)。引用扫描为 results/ 中
不超过 20 MiB 的 JSON 文本匹配；包含路径前缀匹配，数量是待核查引用的上界，不能据此
证明某个文件无引用。正式删除需要逐项核验依赖并再次取得用户明确同意。

| 路径（相对仓库） | 占用 GiB | 已发现关系与保留方案 |
|---|---:|---|
| `data/native-rsft/replay-222156/hf-checkpoint` | 9.65 | 历史导出与恢复证据引用；先保留，需逐项区分原导出和 canonical 路径 |
| `data/native-rsft/replay-222156/hf-checkpoint-canonical` | 9.65 | alternation 训练、导出和 worker 比较的基准；保留 |
| `data/native-rsft/replay-222156/global_step_1` | 18.12 | 历史原生模型、恢复及全参数核验输入；保留 |
| `data/native-rsft/alternation-replay-222640/hf-checkpoint-canonical` | 9.65 | **当前 joint 搜索冻结计划中的数值参考**，不能删除 |
| `data/native-rsft/alternation-replay-222640/global_step_1` | 18.32 | 原生恢复状态和 canonical 转换核验输入；保留 |
| `data/native-rsft/controlled-weight-only-seed42-222801` | 46.29 | 两批训练、最终模型及受控比较证据；保留 |
| `data/native-rsft/controlled-whale-seed42-phase1-223622` | 27.91 | **正在进行的搜索与下一阶段恢复输入**；保留 |
| `data/models/qwen3.5-4b-common-bf16-v1` | 8.47 | 受控实验共同初始化和视觉服务输入；保留 |
| `data/models/qwen3.5-2b-15852e8` | 4.25 | 规模扩展模型来源；保留 |

现有可用空间为 **119.80 GiB**（查询时间见 JSON）。按当前模型目录占用估计，正式矩阵
45 个 4B 条件和 6 个 2B 条件若各保存一份最终模型，需要 **406.72 GiB**。六个
harness-only 条件只引用共同权重、避免重复模型副本后，仍约需 **355.89 GiB**。
这是储存上界规划，不是假设每次训练都必然产生不同权重。

此外，三个机制检查点暂估 54 GiB，六个分叉最终 BF16 模型约 50.83 GiB，活动训练、
导出与临时空间至少预留 60 GiB。按去重口径，模型与分叉及活动空间已约 **520.72 GiB**，
尚未包含全部数据、轨迹、失败尝试和可视化。后续根据实际检查点与轨迹吞吐更新。

目前没有核准为可安全删除的模型目录。即使把上面两个旧工程运行的全部约 65.4 GiB
释放，也不足以支撑完整矩阵，而且会破坏仍使用它们的核验链。应优先落实可供计算节点
读取的大容量共享路径，再制定内容寻址、相同权重复用和逐阶段保留方案。

现有空间支持接下来的 Chess 收尾和单个视觉服务工程验收。顺序控制器在训练前保留
额外 30 GiB 工作量、导出前 10 GiB、视觉服务前 2 GiB，并始终保留 40 GiB 余量。
空间不足时停止提交，不自动清理。**完整正式矩阵的存储门槛尚未通过。**
