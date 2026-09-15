# 本轮GLM拟发送材料与研究判断

222196的32题基线已正常完成并通过独立审计。0/32的主要可观测失败是思考耗尽预算及
输出解析失败：32次调用length停止且未结束thinking，轨迹终态30次预算耗尽、2次格式错误。
这并不能证明棋力足够，但也不能用作GLM能力不足的证据；当前解题者为本地Qwen3.5-4B
更新checkpoint，GLM负责下一步修改harness，本轮尚未调用。

原作者报告初始2048条Chess轨迹中2025条达到8129 token上限；搜索后的harness会整理
棋子与合法走法、保留关键历史，并区分思考草稿与正式落子，再与成功轨迹RSFT交替。
这是原WHALE的机制，不是本项目新增贡献。[原论文附录D.3](https://arxiv.org/html/2609.00196v1#A4.SS3)

下一步在同一更新checkpoint、同32题MH、同8129预算上评测一个GLM候选h1，与h0按
原成功率及调用数规则选择。选择和独立审计完成后，绑定选中的源码启动新在线训练。
这一批是单种子的优化数据，不能替代留出集或三种子的四条件正式比较。

## 可审核的发送范围

目的地：智谱官方`https://open.bigmodel.cn/api/anthropic/v1/messages`，模型
`glm-5.3-flash`。最多12次API请求、12个CLI turn、每请求4096输出token、300秒；
继续使用现有45元账本。当前保守已记0.047257元、held0，未获得官方账单。

研究输入目录为`data/chess-alternation-step1/search/`，7文件共1,961,513字节：

| 文件 | 字节 | 用途 |
|---|---:|---|
| harnesses/h0/harness.py | 1954 | 公开上游原始harness |
| logs/accepted_harness.txt | 3 | 当前选择h0 |
| logs/frontier_val.json | 370 | 原评分与调用数 |
| logs/iteration_001/proposer_prompt.txt | 628 | 单候选任务指令 |
| logs/mh_val/h0/hf-checkpoint-canonical/ChessPuzzle-v0_policy_trace.jsonl | 642291 | 模型回复、解析动作和token计量 |
| logs/mh_val/h0/hf-checkpoint-canonical/trajectories.jsonl | 1315123 | 棋局输入、回复及验证反馈 |
| logs/mh_val/h0/hf-checkpoint-canonical/val.json | 1144 | 完整32题汇总与运行元数据 |

另外使用原公开Chess harness搜索说明，以及CLI系统指令、工具协议、工作目录元数据。
模型按需读取上述文件，后续工具结果和生成的候选也会进入对话；1.96MB为可读取研究
文件总量，不是精确的最终网络载荷或计费token数。

本地检查确认32条标准答案字段均为空、example_id均为redacted，嵌套字段没有参考续着
或私有题目ID，文件中没有已保存的凭据值。棋局、模型推理和实验分数会对智谱可见，
部分元数据包含本地路径及账户名。train128/test64、私有顺序映射、权重文件、其他项目
不在该研究目录中。API key由relay作官方鉴权，不放入模型上下文。

文件读取用既有Landlock隔离；它不限制网络或仅元数据操作，不能宣称为完整网络沙箱。
[机器可读清单与全部SHA](../results/updated-mh-proposer-transfer-review-20260909.json)
已保存；原搜索材料仍未更改，后续启动还会核验冻结哈希。

## 首次审批记录（现已解决）

自动审批首次拒绝了GLM启动命令，理由是一般GLM使用授权尚未覆盖这批具体研究轨迹的
对外发送。那次被拒绝命令未执行，未创建proposer工作区，也未新增调用或预留。
随后用户明确回复“我授权”，批准此范围；已保存[授权记录](../results/updated-mh-proposer-authorization-20260909.json)。
重试启动获准且完成，新增7次API/0.039819元保守费用；旧JSON清单保留原审核时状态，
不会将此前未发送的快照改写为已发送。实际h1代码/工具读取情况见[审查](../results/updated-mh-h1-semantic-review-20260909.json)。

代码对应Method中原E4更新权重到E3搜索、再到E4训练的连接。审计检查原
`A(h;theta)=sum_i success_i/32`，新训练hook只裁无效尾部并保持成功筛选的SFT目标；
这些共享执行与测量改动不构成VETO收益声明。
