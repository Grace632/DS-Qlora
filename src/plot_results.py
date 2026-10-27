"""Plot lexical evaluation proxies without implying human quality scores."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/evaluation/lexical_summary.json")
    parser.add_argument("--output", default="outputs/figures/lexical_comparison.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.summary).read_text("utf-8"))
    metrics = ("citation_marker_rate", "exact_source_match_rate", "unknown_abstention_marker_rate")
    labels = ("Citation marker rate", "Exact source match per span", "Unknown abstention marker rate")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, metric, label in zip(axes, metrics, labels):
        values = [data["models"][mode]["overall"][metric] for mode in ("base", "qlora")]
        for x, (mode, value) in enumerate(zip(("Base", "QLoRA"), values)):
            if value is not None:
                ax.bar(x, value * 100, color=("#64748b", "#0f766e")[x], width=0.6)
                ax.text(x, value * 100 + 2, f"{value * 100:.1f}%", ha="center")
            else:
                ax.text(x, 5, "N/A", ha="center")
        ax.set_xticks([0, 1], ["Base", "QLoRA"])
        ax.set_ylim(0, 110)
        ax.set_ylabel("Percent")
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Automatic lexical proxies on {data['questions']} questions; not doctrinal quality scores", fontsize=12)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
