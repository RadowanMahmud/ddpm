"""Collect results into Markdown tables (results/report_tables.md) and plot training loss curves.

Reads whatever of these exist:
  results/metrics_test.csv, results/metrics_all.csv        (evaluate.py)
  results/classifier_summary.json, classifier_*.csv         (classify.py label)
  runs/classifier.test.json                                 (classify.py train)
  results/nearest_*_stats.csv                               (nearest.py)
  runs/*.loss.csv                                           (run.py train)
"""
import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(value):
    return f"{100 * float(value):.1f}%"


def metrics_section(path, title):
    rows = read_csv(path)
    body = [(r["mode"], r["category"], r["real_count"], r["generated_count"], r["fid"],
             f'{float(r["kid"]):.4f} ± {float(r["kid_std"]):.4f}',
             f'{float(r["inception_score"]):.2f} ± {float(r["inception_score_std"]):.2f}') for r in rows]
    return f"## {title}\n\n" + table(
        ("Model", "Category", "Real n", "Generated n", "FID ↓", "KID ↓", "IS ↑"), body)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    sections = ["# Results tables"]

    for name, title in (("metrics_test.csv", "FID / KID / IS vs held-out test photos"),
                        ("metrics_all.csv", "FID / KID / IS vs all real photos (train + test)")):
        if (args.results / name).exists():
            sections.append(metrics_section(args.results / name, title))

    summary_path = args.results / "classifier_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        real = summary.get("classifier_on_real_test")
        if real:
            sections.append("## Classifier sanity check (real held-out photos)\n\n" + table(
                ("Category", "Accuracy"),
                [*((c, pct(a)) for c, a in real["per_class_accuracy"].items()),
                 ("**overall**", pct(real["test_accuracy"]))]))
        unconditional = summary.get("unconditional")
        if unconditional and (args.results / "classifier_unconditional.csv").exists():
            rows = read_csv(args.results / "classifier_unconditional.csv")
            sections.append(
                "## Unconditional model: which categories does it generate?\n\n" + table(
                    ("Category", "Generated count", "Generated share", "Training share", "Mean confidence"),
                    [(r["category"], r["generated_count"], pct(r["generated_share"]), pct(r["training_share"]),
                      f'{float(r["mean_confidence"]):.2f}') for r in rows])
                + f'\n\nNormalised entropy of category histogram: {unconditional["normalised_entropy"]:.3f} '
                  f'(1 = all five equally often). Total-variation distance to training shares: '
                  f'{unconditional["total_variation_to_training"]:.3f}. Low-confidence (< 0.5) samples: '
                  f'{pct(unconditional["low_confidence_share"])}.')
        conditional = summary.get("conditional")
        if conditional and (args.results / "classifier_conditional.csv").exists():
            rows = read_csv(args.results / "classifier_conditional.csv")
            headers = list(rows[0])
            sections.append(
                "## Conditional model: is the requested category produced? (confusion matrix)\n\n"
                + table(headers, [[r[h] if h != "accuracy" else pct(r[h]) for h in headers] for r in rows])
                + f'\n\nOverall controllability (classifier agrees with requested label): '
                  f'{pct(conditional["accuracy"])}. Mean probability of the requested category: '
                  f'{conditional["mean_intended_probability"]:.3f}.')

    for kind in ("unconditional", "conditional"):
        path = args.results / f"nearest_{kind}_stats.csv"
        if path.exists():
            rows = read_csv(path)
            sections.append(
                f"## Memorisation and diversity ({kind}, RMS pixel distance)\n\n" + table(
                    ("Group", "Gen → train NN (median)", "Test → train NN (median)", "Gen closer than any test",
                     "Pairwise: generated", "Pairwise: training"),
                    [(r["group"], r["gen_nn_median"], r["test_nn_median"], pct(r["share_closer_than_any_test"]),
                      r["gen_pairwise"], r["train_pairwise"]) for r in rows]))

    losses = sorted(args.runs.glob("*.loss.csv"))
    if losses:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(7, 4))
            for path in losses:
                rows = read_csv(path)
                axis.plot([int(r["iteration"]) for r in rows], [float(r["loss"]) for r in rows],
                          label=path.name.removesuffix(".loss.csv"), linewidth=1)
            axis.set(xlabel="iteration", ylabel="noise-prediction MSE", yscale="log", title="DDPM training loss")
            axis.legend()
            figure.tight_layout()
            args.results.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.results / "loss_curves.png", dpi=150)
            sections.append("## Training loss\n\n![loss curves](loss_curves.png)")
        except ImportError:
            print("matplotlib not installed; skipping loss plot.")

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "report_tables.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {args.results / 'report_tables.md'}")


if __name__ == "__main__":
    main()
