# Semantic Evidence KT

知识追踪实验：在固定题目语义先验与历史作答之间，研究冷暖题统一表示和可以逐条核算的预测校正。

本项目以 XES3G5M 与 BAIM/sparseKT 为基础，检验两个候选机制：

1. **冷暖题统一的语义残差学习。** 共享语义适配器负责跨题迁移，题目 ID 残差按训练支持度缩放，并按题目整组屏蔽以模拟零作答新题。
2. **历史预测误差的证据校正。** 只检索当前预测之前的作答，将相对基础模型预期的偏差转为可相加的 logit 校正，保留每条历史的正负贡献。

它们是待验证的研究假设。普通语义融合、收缩或注意力本身不是本项目宣称的新贡献；在实验完成之前，不预写性能提升或 SOTA 结论。

**2026-10-14 状态：已按用户要求停止训练和自动接续。** 首折四组测试完成：对照82.2314%、机制一82.2552%、机制二82.3079%、完整模型82.2445%。五折和冷题实验未完成；现有微小收益不足以证明两个机制稳定有效，未达到超过SOTA的目标。详见 [当前实验结论](reports/RESULTS.md)；所有结果直接引用保存记录，没有复测。

## 数据与来源

- [XES3G5M 官方数据](https://github.com/ai4ed/XES3G5M)：使用 question-level 官方文件与五折训练／验证分组、固定测试集。
- [BAIM 官方代码](https://github.com/sjin4861/BAIM)：固定 commit `a9501b5e44dafc0e81d2f17e717766a4106d0122`，复用公开四阶段 768 维语义缓存。
- [BAIM 论文](https://aclanthology.org/2026.findings-acl.1125/)：作者报告的 83.21%／83.22% 是参考值，尚未由本项目复现。
- [SINKT](https://arxiv.org/abs/2407.01245) 与 [SLC](https://arxiv.org/abs/2606.14123) 为语义归纳与题目校正的相关工作，具体区别见 `RELATED_WORK.md`。

源码审计发现上述 BAIM 公开 commit 的稀疏注意力二次 softmax 会重新激活未来位置。本项目统一修正这一因果掩码问题后比较四组模型；这是对公开代码的核查，不能据此推断论文当时使用的实现。数值证据与修正说明在 `UPSTREAM_AUDIT.json`，上游许可证保留在 `vendor/LICENSE.BAIM`。

## 运行

依赖 Python 3.12、PyTorch 和 NumPy；测试文件预处理另需 pandas，可使用独立 Python 环境。GPU 环境需要与本机驱动匹配；此项目不自动替换现有 GPU PyTorch。

```bash
python data.py download --root data
python prepare_test_sequences.py --data-root data
python -u train.py --data-root data --output runs/first-pass \
  --budget-hours 18 --warm-hours 14.5 --cold-hours 3
```

官方压缩包未包含主测试的 `test_sequences_quelevel.csv`。预处理脚本从原始 `test_quelevel.csv` 调用固定 BAIM 版本的原始分段函数生成该文件：每段200步，尾段不足3步丢弃，其余右填充；学生划分、题目编号及多知识点保持原样。脚本校验上游源码哈希并保存输入／输出哈希与汇总数量，不执行模型推理。来源记录见 `reports/test_sequences_manifest.json`。

队列先完成 fold0 的 `baseline/residual/evidence/full`，验证集选择冻结后才请求测试文件。然后按预算补齐 baseline/full 的其余四折，以及从头训练的冷题对照。使用 `latest.pt`、优化器、随机状态及批次游标续训；已有完成结果与已完成测试块直接复用。恢复时保持原命令预算和配置不变。

```bash
python make_report.py --run runs/first-pass
```

该命令只读取保存的结果，不加载模型、不重复推理。数据、向量、检查点和逐条预测保留本地，不纳入 Git。带历史校正的模型在首次测试前向中保存首批最多128个目标的逐条贡献，供局部解释核算。

## 比较口径与解释范围

主指标为官方 question-level `testauc`，与额外窗口指标区分。只有全部五折完成后才给出同协议五折汇总；单折高分不能替代五折结果。源代码因果修正与作者报告的差异会随成绩公开。

冷题划分为全题库 70%／15%／15%、seed42。验证及测试冷题的作答均从训练和评估历史排除，每个冷题查询只读取严格过去的暖题历史。缓存 PCA 在全题库文本上拟合，因此本实验属于“预知题库、零训练作答冷启动”，不是完全未见题文的归纳实验。

历史贡献解释的是附加校正项，不代表整个骨干都可解释，也不代表真实认知状态或教育干预的因果效果。

完整预定参数、方法边界与最少程序检查见 `PROTOCOL.md`、`CPU_CHECKS.json`。实际训练结果在 `reports/RESULTS.md`（训练完成后生成）。
