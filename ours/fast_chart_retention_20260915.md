# 精简计划的存储保留方案（待授权）

这是清理候选清单，尚未删除任何文件。完整吞吐成本核算仍在进行。

当前实测模型大小外推：全部保留约699.23 GiB；只顺序回收最终原生权重仍约429.10 GiB；再移除两组未选中校准权重，暂估峰值约372.36 GiB。后续吞吐输出和真实运行可能提高这个数，400 GiB上限保持不变。

## 当前四个可清理目录

下列路径均相对于 `/userhome/cs3/yihangc/Documents/whale-delta/data/fast-chart-20260915-v1`。精确文件身份与已知引用保存在[机器可读清单](../results/fast-chart-retention-proposal-20260915-v1.json)。

| 目录 | 实际占用GiB |
|---|---:|
| `lr-calibration-v3/lr-1e-07-seed42/training/checkpoints/global_step_4` | 18.145 |
| `lr-calibration-v3/lr-1e-07-seed42/followup/hf-canonical` | 9.651 |
| `lr-calibration-v3/lr-1e-05-seed42/training/checkpoints/global_step_4` | 19.295 |
| `lr-calibration-v3/lr-1e-05-seed42/followup/hf-canonical` | 9.651 |

合计56.741 GiB。这两组学习率没有被预定规则选中。删除后不能直接加载其模型；原始逐题回答、所有指标、训练配置、图像/梯度与参数核验、身份哈希、费用和失败记录全部保留。被选中的1e-6原生检查点和服务权重保留。

## 将来逐个回收的15个最终原生检查点

这些路径目前不存在，只固定拟议的正式输出布局。只有对应真实续训、完整参数核验、BF16导出及服务交接都完成，且没有活动作业或后续续训引用时，才能按授权范围逐个回收。正式条件计划必须与清单匹配。

| 种子 | 条件 | 拟回收目录 |
|---|---|---|
| 42 | weight_only | `formal-v1/seed42/weight_only/training/checkpoints/global_step_8` |
| 42 | whale | `formal-v1/seed42/whale/training/checkpoints/global_step_8` |
| 42 | veto | `formal-v1/seed42/veto/training/checkpoints/global_step_8` |
| 42 | marginal_gate | `formal-v1/seed42/marginal_gate/training/checkpoints/global_step_8` |
| 42 | counterfactual_augmentation | `formal-v1/seed42/counterfactual_augmentation/training/checkpoints/global_step_8` |
| 43 | weight_only | `formal-v1/seed43/weight_only/training/checkpoints/global_step_8` |
| 43 | whale | `formal-v1/seed43/whale/training/checkpoints/global_step_8` |
| 43 | veto | `formal-v1/seed43/veto/training/checkpoints/global_step_8` |
| 43 | marginal_gate | `formal-v1/seed43/marginal_gate/training/checkpoints/global_step_8` |
| 43 | counterfactual_augmentation | `formal-v1/seed43/counterfactual_augmentation/training/checkpoints/global_step_8` |
| 44 | weight_only | `formal-v1/seed44/weight_only/training/checkpoints/global_step_8` |
| 44 | whale | `formal-v1/seed44/whale/training/checkpoints/global_step_8` |
| 44 | veto | `formal-v1/seed44/veto/training/checkpoints/global_step_8` |
| 44 | marginal_gate | `formal-v1/seed44/marginal_gate/training/checkpoints/global_step_8` |
| 44 | counterfactual_augmentation | `formal-v1/seed44/counterfactual_augmentation/training/checkpoints/global_step_8` |

这是永久删除最终FP32训练状态；BF16服务权重不能逐位还原它。每个方法的最终服务模型继续保留，用于完整评测和复查；三个共同分叉检查点也保留。harness-only没有新的训练权重，使用其共享模型身份。

## 授权边界

- 不涉及旧项目、B组、选中校准模型、共享初始模型、数据或逐题证据。
- 这份清单本身不构成删除授权；需用户明确同意。
- 删除前重查实际引用和文件清单，保留清理前的完整大小测量。
- 未完成的训练、未核验的导出、活跃作业使用的模型不得删除。
- 清理与正式实验仍受100 GPUh、阶段额度、45元API和400 GiB限制；预测超限时暂停扩展。
