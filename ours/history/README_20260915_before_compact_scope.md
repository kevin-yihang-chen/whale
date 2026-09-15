# WHALE-based research: VETO prototype and visual input check

本项目以 [krafton-ai/WHALE](https://github.com/krafton-ai/WHALE) 的公开代码为 baseline 和 infrastructure。WHALE 的交替优化框架、RSFT、Meta-Harness、环境与训练器均不是本项目贡献。

必须引用：Haechan Kim, Yoonho Lee, Gisang Lee, Chelsea Finn, Kangwook Lee. **WHALE: A Simple Recipe for Joint Harness-Weight Optimization.** [arXiv:2609.00196v1](https://arxiv.org/abs/2609.00196v1), 2026.

截至2026-09-10 10:58香港时间，受控seed42的harness-only五轮搜索已完整结束：原选择器最终保留h1，固定MH32上为8/32，h0为1/32；这些是优化集结果。独立weight-only两批128条轨迹产生1次SFT更新，原生checkpoint及最终规范BF16导出均已核验，原预检题序偏差明确保留。WHALE seed42独立第一阶段及导出完成，联合搜索第一轮h0–h3分别为1/32、1/32、5/32、3/32，原选择器保留h2。第二轮h4/h5均已完整核验6/32，h6作业223909正在评测，尚未完成第二轮选择。其余条件/种子已准备，尚未提交。

共享视觉dataset、二元奖励及原生crop工具回路已通过CPU检查。新增可执行harness的五个函数可改变观察、crop参数、反馈、答案解析与预算内追加请求；原生dataset工厂、Hydra加载、实际像素裁剪、回复掩码和有/无工具批次合并通过，默认h0在工程样例上与现有回路相同。模型回复及actor/Ray传输检查使用明确标注的夹具；配对评测入口现已通过原生Manager/Worker批次执行、样本身份合并与中断保留的CPU检查，能输出保留数据角色的E1记录；实际服务启动、视觉模型前反向、视觉proposer和完整训练/搜索仍未验证。最终12-trial封存、64题留出及汇总代码已准备；test题目/答案未加载，没有正式test分数。

**VETO — Visual Evidence Tests for Optimization 仍是待验证假设；F0–F5均未通过。** 最新完整结果、成本与下一步见[PROJECT_STATUS.md](PROJECT_STATUS.md)，源码与Method公式对应见[ours/README.md](ours/README.md)。所有对照保留原WHALE主体逻辑，不预设提升比例。

- [代码解剖](notes/whale_anatomy.md)
- [10 个研究切入点](notes/attack_surface.md)
- [待用户 review 的 thesis 与证伪实验](notes/thesis.md)
- [相关工作碰撞检查](notes/novelty_screen.md)
- [状态与阶段门](PROJECT_STATUS.md)
- [来源及修改记录](CHANGES.md)
- [实现、Method公式对应与验证命令](ours/README.md)
- [工程记录与真实复现前置失败](EXPERIMENTS.md)
- [Claude、GLM与本地proposer的接线和预算](notes/proposer_options.md)

上游完整 checkout 在 `upstream/WHALE/`，固定于 `fbe125eb7abea7f760c99ab9acc1a6261e708fc6`。顶层 [LICENSE](LICENSE) 与 [NOTICE](NOTICE) 是上游原文件的完整副本，上游副本也保留。没有修改上游文件；未来新增实现只放 `ours/`，不改任何 `verl/`，对上游修改必须提供明确 patch 和修改说明。

本次新 Goal 明确要求代码产出，因此推进独立模块和离线验证，覆盖初始快照的仅文档暂停状态。F0仍是实际联合实验的前置条件：共享依赖和权重交接已完成工程验证，但四条件、多种子和完整实验协议尚未完成，不宣称原baseline复现成功。所有训练从三个独立种子起步，报告 mean ± sample std；禁止虚构数字、只报最佳 run，或将小样本工程跑通包装成论文数值复现。所有工作保留本地，未获明确授权不 push。

研究代码保持一个可关闭的接受模块；视觉数据、环境和训练接线是共享基础设施，须单独记账。已有 CAT / CPL / PAPO / CFPO 等工作覆盖了视觉捷径诊断和反事实训练的大量思想。只做领域迁移或拼接既有损失，增量不够；本项目不预设提升比例，也不以故意弱化 baseline 或增加无用途脚本制造贡献。

## 本地版本回溯

本项目使用本地 Git，主分支为 `main`；初始文档快照标记为 `initial-review`。`upstream/WHALE` 是 submodule，父仓库记录确切上游 commit，保留其独立历史。父仓库未配置远端。

在本项目目录执行 `git log --oneline --decorate` 查看历史，执行 `git show initial-review:notes/thesis.md` 查看初始 thesis。后续阶段形成可审核状态时提交本地快照；push 仍需明确授权。环境、凭据、缓存、下载数据与 checkpoint 不纳入版本库，研究文档和结果摘要正常跟踪。
