"""Summarize existing result JSON; never load a model or repeat inference."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/first-pass")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run = (root / args.run).resolve()
    rows = []
    for path in sorted(run.glob("*/test.json")):
        obj = read(path)
        config = read(path.parent / "config.json")
        completed = read(path.parent / "complete.json")
        rows.append({"protocol": config["protocol"], "fold": config["fold"],
                     "variant": config["variant"], "seed": config["seed"],
                     "validation": completed["best"]["validation"], "test": obj["metrics"],
                     "stop_reason": completed["stop_reason"],
                     "checkpoint_sha256": obj["checkpoint_sha256"],
                     "test_binding_sha256": obj["test_binding_sha256"]})
    report = root / "reports"
    report.mkdir(exist_ok=True)
    lines = ["# Semantic Evidence KT 实验结果", "",
             "本报告仅汇总已保存的结果，没有重新训练或重复推理。", "",
             "## 官方常规协议：fold0 消融", "",
             "| 模型 | 验证 AUC | 测试 AUC | ACC | NLL | Brier | 结束原因 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    lookup = {}
    for row in rows:
        lookup[(row["protocol"], row["fold"], row["variant"])] = row
        if row["protocol"] == "standard" and row["fold"] == 0:
            m = row["test"]
            lines.append(f"| {row['variant']} | {100*row['validation']['auc']:.4f}% | {100*m['auc']:.4f}% | {100*m['accuracy']:.4f}% | {m['nll']:.6f} | {m['brier']:.6f} | {row['stop_reason']} |")
    base = lookup.get(("standard", 0, "baseline"))
    if base:
        lines.append("")
        for variant in ("residual", "evidence", "full"):
            result = lookup.get(("standard", 0, variant))
            if result:
                gain = 100*(result["test"]["auc"] - base["test"]["auc"])
                lines.append(f"{variant} 相对同协议修正基线的 fold0 AUC 差为 **{gain:+.4f} 个百分点**。")
    lines += ["", "## 常规五折覆盖", "",
              "| 模型 | 已完成折 | 平均 AUC | 折间样本标准差 |",
              "|---|---|---:|---:|"]
    fold_summary = {}
    for variant in ("baseline", "full"):
        selected = [row for row in rows if row["protocol"] == "standard" and row["variant"] == variant]
        values = [row["test"]["auc"] for row in selected]
        if values:
            folds = sorted(row["fold"] for row in selected)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else None
            fold_summary[variant] = {"folds": folds, "mean_auc": mean, "std_auc": std,
                                     "all_five_complete": folds == list(range(5))}
            std_text = f"{100*std:.4f}" if std is not None else "不可估计"
            lines.append(f"| {variant} | {folds} | {100*mean:.4f}% | {std_text} |")
    lines += ["", "只有全部官方五折完成，均值才具有五折比较口径；折间波动不是多个独立训练种子的显著性检验。",
              "BAIM 作者报告参考为 83.21%／83.22%。公开固定代码存在本项目已修正的因果掩码问题，不能据此认证超过干净 SOTA，也不能据此断言论文当时的训练代码存在同一问题。"]
    if fold_summary.get("full", {}).get("all_five_complete"):
        d = 100*(fold_summary["full"]["mean_auc"] - .8322)
        lines.append(f"完整模型五折均值与作者报告 83.22% 的数值差为 **{d:+.4f} 个百分点**，保留上述比较限制。")
    lines += ["", "## 预知题库的零作答冷题协议", "",
              "| 模型 | 测试 AUC | ACC | NLL | Brier | 评分事件数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["protocol"] == "cold":
            m = row["test"]
            lines.append(f"| {row['variant']} | {100*m['auc']:.4f}% | {100*m['accuracy']:.4f}% | {m['nll']:.6f} | {m['brier']:.6f} | {m['targets']} |")
    cold_base, cold_full = lookup.get(("cold", 0, "baseline")), lookup.get(("cold", 0, "full"))
    if cold_base and cold_full:
        gain = 100*(cold_full["test"]["auc"] - cold_base["test"]["auc"])
        lines += ["", f"完整模型相对冷题基线的 AUC 差：**{gain:+.4f} 个百分点**。"]
    lines += ["", "全题库按70%/15%/15%、seed42固定划分；冷题作答不进入训练或预测历史。全题库PCA缓存使该协议属于预知题库的零训练响应冷启动。", "",
              "## 证据和限制", "",
              "四组在同一因果修正、同一数据与训练规则下比较。预算中断的任务单独标注结束原因，不将未充分收敛等同于该方法的最终能力。",
              "两个机制是否提高AUC由消融结果判断；历史贡献的可加性只解释校正分支。程序检查不等于认知参数有效性或教学干预效果。"]
    audits = [row for row in rows if row["test"].get("contribution_audit")]
    if audits:
        lines += ["", "## 历史贡献核算", "",
                  "明细在首次测试的同一前向中保存，没有再次推理；每组最多128个评分目标。逐条贡献保存在本地测试首批NPZ的audit字段中。", "",
                  "| 协议 / 折 / 模型 | 保存目标数 | 最大概率重建误差 | 最大logit重建误差 |",
                  "|---|---:|---:|---:|"]
        for row in audits:
            audit = row["test"]["contribution_audit"]
            lines.append(f"| {row['protocol']} / {row['fold']} / {row['variant']} | {audit['scored_targets_saved']} | {audit['max_final_probability_reconstruction_error']:.3g} | {audit['max_logit_reconstruction_error']:.3g} |")
        lines.append("基准概率的逆sigmoid存在舍入影响；报告同时给出概率误差。局部贡献核算成立不等于整个基础模型可解释。")
    if (run / "queue-summary.json").exists():
        queue = read(run / "queue-summary.json")
        lines += ["", f"队列状态：`{queue['status']}`；累计预算用时：{queue.get('budget_seconds_used', 0)/3600:.3f}小时。"]
    if not rows:
        lines += ["", "尚无完成的一次性测试结果。"]
    (report / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (report / "results.json").write_text(json.dumps({"runs": rows, "fold_summary": fold_summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"completed_tests": len(rows), "fold_summary": fold_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
