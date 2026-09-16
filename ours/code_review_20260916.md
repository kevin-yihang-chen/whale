# 外部AI建议复核（2026-09-16）

范围：用户提供的八条建议；审阅当前源码、实际计划和调用链，使用小型CPU输入复现边界问题。本轮没有修改运行源码、评分规则或历史实验记录，没有提交GPU/API任务。

结论：优先修1、2、6；3需要接口保护但不能归因为既有训练故障；7值得完善；5是已复现的分片兼容问题；4的“现有reference残留identity”说法不符合当前实际计划，但可加强计划身份的一致性；8属于评分协议选择，不能直接按建议去逗号。

## 1、2：关键认证不能使用可被优化移除的assert

位置：`visual_search_bridge.py:44–103`、`verify_visual_resumed_parameters.py:15–23`。fast-chart认证确实调用前者，stage2确实使用后者。

在正常Python与`python -O`下调用实际函数，小型复现结果：

| 输入 | 普通Python | python -O |
|---|---|---|
| once遇到已有JSON与新值不一致 | AssertionError | 接受，已有边界文件不变但冲突未报错 |
| after多一个参数键 | AssertionError | 返回CHANGED，额外参数未纳入比较 |
| after是float64而非float32 | AssertionError | 返回CHANGED |
| 声明的tensor_count不正确 | AssertionError | 返回CHANGED |

应将运行时输入、证据、状态和身份校验改为明确的异常；可复用已有require辅助函数。测试代码中的assert不需要机械替换。优化模式测试应使用子进程并检查退出状态或unittest断言，不能让测试自身也被-O去掉。增加针对关键认证模块的AST检查即可，不必为了修补而大规模改写上游。

官方语言定义明确说明-O不生成assert代码：https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement 。

这证明认证存在条件性绕过，不证明历史作业以-O运行或现有分数已被污染。历史启动脚本没有显式-O，但不能据此排除所有环境变量影响；应先用保留源码、普通解释器和已有原始结果重核，无需直接重训。

## 6：失败候选绑定不足，且未限定真正的失败终态

位置：`fast_chart_search.py:82–91`、`fast_chart_search_report.py:91–101`；提交凭证见`fast_chart_submit.py:71–93`。

隔离输入仅替换search.load为一个固定合法阶段，真实调用failed_evaluation：submission的plan_sha256和job_id均不匹配时仍被接受；将allocation.state从FAILED改为RUNNING，保留failure.json，仍被接受。

因此建议比原建议更完整：同时校验submission的计划哈希/路径、作业ID、terminal文件哈希、解析出的终态、时长和GPU数，以及候选与本阶段harness身份。运行中、排队中或终态未知都不能记作已失败并消费槽位。把共同逻辑用于失败登记和结果重建，不只加一条plan_sha256检查。

现有seed42三个候选均完成，未发现这轮使用错配失败凭证的证据。

## 3：labels隐患真实，但当前RSFT未走该路径

位置：`qwen35_visual_fused.py:52–57`。用已有tiny Qwen3.5 fixture给labels前两位填-100，实际抛出`RuntimeError: index -100 is out of bounds for dimension 1 with size 64`。原生fused实现使用gather，-100不属于有效token编号。

调用链核对：上游`verl/workers/actor/dp_actor.py:244–251`及`:352–359`给模型传input_ids和视觉参数，不传labels。RSFT在同文件`:735–739`用response_mask计算外部损失。因而当前证据支持“未覆盖的接口危险”，不支持“现有训练错了token”或“导致当前退化”。

不要悄悄丢弃调用者传入的labels。最小修补可明确限定当前input_ids+外部mask契约并对不支持的labels报错；如果需要支持HF式labels，则明确ignore-index、移位和loss-mask语义并补测试。现有labels=None的数值与梯度等价测试应继续通过。

## 4：原建议部分误判，保留一致性改进

位置：`fast_chart_endpoint_evaluation.py:80–121,234–236`；reference生成位置为`fast_chart_evaluation.py:73–86`。

实际核查了选中LR的V计划和已完成seed42第二阶段的V计划：kind均为compact_chart_pair_evaluation，二者都没有identity字段。该字段是在评测执行时组装到receipt中的。因此当前不能确认“deepcopy必然带入旧identity”的故障。

确实会继承phase等描述性字段；ChartQA路径还保留配对任务的部分元信息。建议建立单一endpoint_identity构造函数，prepare/check/run一致使用；明确新phase，ChartQA使用自己的scorer身份，清理不适用的配对字段。应作为防错完善，不宣称已发现独立评测错配；目前也尚无T/R/ChartQA模型终点结果。

## 5：分片兼容问题成立，当前单文件配置可运行

位置：`native_visual_service.py:92–98`、`fast_chart_training_worker.py:28–29`。

CPU构造合法的两分片safetensors及index：checkpoint_manifest接受两个权重文件，embedding_coordinates因找不到model.safetensors抛出FileNotFoundError。

建议提取只查找指定tensor的共享读取器，支持单文件/分片、缺失和重复tensor检测，复用现有probe_updated_vllm的处理原则。不是直接调用changed_embedding_coordinates：后者需要两个模型且必须发现参数变化，语义不同。当前固定单文件模型的实验不需要因这一兼容问题重跑。

## 7：应在最终选择前闭合所有候选槽位

位置：`fast_chart_search.py:178–196`；部分候选可被接收的行为见`scoped_proposer.py:27–66`。

当前先写selection，再补缺失slot的failure记录，正常执行到底时report会要求各slot恰好有一份成功或失败凭证；问题主要在中途退出后留下不完整的“已完成选择”，以及candidate_attempts=3混淆预算槽位与实际产出。

建议先给所有槽位确定终态，再选择、写带失败清单身份的receipt，最后发布完成标记。区分candidate_budget、实际生成数、实际评测数及失败数。缺失或失败候选应不评分且消费预算，不按0分填充，也不自动追加直到产生有利候选。

## 8：不能无条件去逗号并沿用原评分名称

位置：`chartqa_open_answer.py:15–29`。当前两个方向的1000/1,000比较均返回False。

这与所参考的Pix2Struct relaxed_correctness直接float(text)、不移除逗号的行为一致，不能仅凭与自建图表解析器不同就认定不符合参考指标。参考源码：https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py 。当前零值规则已在模块文档显式说明与参考实现不同。

若需要改善用户输出的格式容错，应将严格验证的千分位规范化作为单独、版本化的辅助评分，并在所有方法上同时重算；保留原评分。不能全局replace(',', '')，否则1,2等不规范字符串也会被改成12。外部真实评测尚未启动，适合在首次评测前把两种口径和名称确定清楚。

## 修改顺序与证据保护

1. 第一批：1+2+6。通过普通/优化模式、篡改输入、错误job及未终止allocation的负例检查。
2. 第二批：3+7。明确forward契约、候选失败终态和可恢复的完成顺序；保持实际正常RSFT和选择规则不变。
3. 第三批：5+4。共享分片读取和统一endpoint身份；先补小型一致性检查。
4. 第8条先确定评分协议，不作为默认修复直接落地。

修改前保留源码版本和已有run内source快照。源码哈希变化会使旧计划按当前源码验证失败，不能把旧计划哈希直接改成新值来绕过。新审计应绑定原运行源码和新验证器版本，输出新的报告，旧记录不覆盖。已完成实验先重核证据，只有确认执行或评分受影响才讨论重跑。

对应论文：1/4/6/7支撑E1与E2–E3证据完整性；2/3/5支撑E4的更新与模型交接；8属于共享外部评分协议。均是可靠性和复现修补，不是VETO的新方法贡献。
