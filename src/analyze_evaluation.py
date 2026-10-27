"""Summarize reproducible lexical proxies from the paired evaluation output."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


ABSTENTION_MARKERS = (
    "无法确认", "无法核实", "未能核实", "没有找到", "未找到依据", "不能确认",
    "不能确定", "资料中未见", "给定经文中未见", "不宜断言", "不应断言",
)


def summarize_model(rows, mode):
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)

    output = {}
    all_rows = [(category, row) for category, items in groups.items() for row in items]
    for category, items in sorted(groups.items()):
        output[category] = summarize_rows(items, mode)
    output["overall"] = summarize_rows([row for _, row in all_rows], mode)
    return output


def summarize_rows(rows, mode):
    answers = [row["models"][mode] for row in rows]
    quotes = [quote for answer in answers for quote in answer["quoted_spans"]]
    unknown = [row for row in rows if not row["expected_answerable"]]
    abstentions = sum(any(marker in row["models"][mode]["answer"] for marker in ABSTENTION_MARKERS)
                      for row in unknown)
    return {
        "questions": len(rows),
        "mean_answer_characters": round(sum(len(answer["answer"]) for answer in answers) / len(rows), 1),
        "citation_marker_rate": round(sum(answer["citation_mentioned"] for answer in answers) / len(rows), 4),
        "quoted_spans": len(quotes),
        "exact_source_match_spans": sum(quote["exact_source_match"] for quote in quotes),
        "exact_source_match_rate": round(sum(quote["exact_source_match"] for quote in quotes) / len(quotes), 4)
        if quotes else None,
        "unknown_questions": len(unknown),
        "unknown_abstention_marker_rate": round(abstentions / len(unknown), 4) if unknown else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/evaluation/base_vs_formal.json")
    parser.add_argument("--output", default="outputs/evaluation/lexical_summary.json")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text("utf-8"))
    rows = data["results"]
    assert len(rows) == data["independent_questions"]
    for mode in ("base", "qlora"):
        assert all(mode in row["models"] for row in rows)
    summary = {
        "evaluation_input": str(args.input),
        "questions": len(rows),
        "metrics_scope": "automatic lexical proxies only; no doctrinal correctness or human quality score",
        "models": {mode: summarize_model(rows, mode) for mode in ("base", "qlora")},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
