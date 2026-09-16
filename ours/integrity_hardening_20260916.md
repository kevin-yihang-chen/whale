# 代码审阅修复交付（2026-09-16）

依据用户“那就改啊”的授权，落实[外部建议复核](code_review_20260916.md)中的第1–7项。
本轮修复输入校验、证据绑定和兼容性，不改变VETO选择公式、RSFT损失、成功过滤、
ChartQA评分口径或既有科学结论。全部实现位于`ours/`，未修改上游WHALE。

| 修复 | 行为与论文对应 |
|---|---|
| 运行校验 | 四个实际运行模块的100处`assert`改为明确的`ValueError`校验，普通及优化模式均检查边界、样本、reward、源码、检查点和作业身份。E1/E2–E3证据完整性及E4交接。 |
| 参数与训练接口 | 比较前校验参数键、FP32类型、有效张量数量；保留别名及有限值检查。非空`labels`或缺失`input_ids`在进入骨干网络前报错，继续使用原生外部`response_mask`。E4。 |
| 分片权重 | `checkpoint_tensor`从单文件或多个分片读取唯一指定张量，检查索引、缺失及重复。初始坐标与实际receiver检查共用此入口。E4模型交接。 |
| 终点评测身份 | `endpoint_identity`由prepare/check/run共同使用，绑定权重、数据、解码、评分器和阶段；ChartQA使用独立评分器身份，不继承配对审计字段。E1及共享评测。 |
| 失败候选 | 登记和报告重建共用验证：提交计划路径/哈希、作业ID、真实终态、终态文件哈希、时长、GPU数、成本、候选源码及failure凭证均需一致。运行中、排队中、成功或未知状态不能登记为失败。E2–E3接受档案。 |
| 候选完成顺序 | 先形成全部槽位的成功/失败凭证及`candidate-slots.json`，再发布各规则的selection，最后发布完成结果；selection绑定槽位清单哈希。中断后复核清单并恢复发布，不重新请求候选。E2–E3到E4的交接。 |
| 成本计数 | 分开记录候选额度、实际生成数、评测尝试数、成功评测数和失败槽位数。未生成/未评测/失败均不赋零分、不自动补抽。属于实验协议记录。 |

第8项未按建议去除逗号：`chartqa_open_answer.py`保持原有版本和评分行为，避免改变
比较口径。当前真实RSFT不传`labels`；此次接口修复不是既有训练错误的证据。

## 校验与历史证据

CPU回归覆盖普通结束、提前停止、恢复、`off`原选择行为，及部分候选、选择发布中断、
错误计划/作业/终态/费用/源码、分片重复与缺失、实际receiver的新旧权重判别、端点
prepare/check/run身份一致性。tiny Qwen3.5仍通过原生HF与修复后forward的数值及梯度
等价检查。测试使用构造输入，不作为模型成绩。

最终通过51项unittest与4项检查点pytest；其中38项另在`python -O`下通过。
测试记录见`results/integrity-hardening-validation-20260916-v1.json`及其日志目录。
一次原生dataset检查需要本地多进程通信，在沙箱外离线运行；未提交GPU作业或调用API。

历史复核检查244个文件，并重新核对1024条H/C回答、1536条开发集回答和256条训练回答。
四个真实候选的E1计数、三个选择规则和恢复决策均与原记录一致；WHALE、VETO、marginal
仍全部选择h3。复核记录：
`results/integrity-hardening-historical-audit-20260916-v1.json`。
这次复核没有重新执行模型或扫描完整训练张量，不替代原参数变化审计。

## 版本边界与复现

修复前源码保存在本地提交`c131b8511ed9328aaf2f49955025d4570f7b4a6e`及标签
`before-integrity-hardening-20260916`；逐文件哈希及原报告哈希保存在
`ours/history/integrity-hardening-20260916-v1/before.json`。运行目录内原`source/`快照保留。

旧计划、源码哈希和报告未改写。新计划生成时冻结当前运行源码，引用的旧完成记录仍以
原哈希验证。新搜索报告要求schema v2；旧搜索使用保存的源码或本次绑定原运行快照的
专门审计。不能把旧计划哈希替换成新源码哈希来恢复执行。

本次历史复核脚本位于上述history目录的`recheck_historical.py`：先核对原快照与冻结
哈希，再以当前加固验证器复核输出，并另行记录验证器版本。其输出采用独占创建，
不会覆盖已有审计报告。

复现CPU检查时，在仓库根目录设置：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=data/visual-runtime-overlay-v1:ours/compat:upstream/WHALE/domains/chess_puzzles:.
data/training-runtime-v1/bin/python -B -m unittest \
  ours.tests.test_research_integrity_hardening \
  ours.tests.test_visual_resumed_parameters ours.tests.test_qwen35_visual_fused \
  ours.tests.test_fast_chart_search ours.tests.test_visual_search_bridge \
  ours.tests.test_fast_chart_endpoint_integrity ours.tests.test_fast_chart_search_report \
  ours.tests.test_research_execution_budget
```

同一命令增加`-O`复核优化模式；完整兼容性模块及使用现有系统pytest的检查见验证记录。
本轮仅本地归档，未推送GitHub，未改变研究资源额度或删除模型。
