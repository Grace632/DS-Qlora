"""Run lightweight source, split and artifact checks without loading the model."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/metrics/reproducibility.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/raw/cbeta"
    sources = json.loads((raw / "sources.json").read_text("utf-8"))
    checked = 0
    for work in sources.values():
        for file, expected in work["files"].items():
            assert digest(raw / file) == expected, file
            checked += 1
    splits, hashes = {}, {}
    previous_questions, previous_works = set(), set()
    for split in ("train", "validation", "test"):
        path = root / f"data/sft/sft_{split}.json"
        rows = json.loads(path.read_text("utf-8"))
        questions = {row["messages"][0]["content"].strip() for row in rows}
        works = {row["work"] for row in rows}
        assert len(questions) == len(rows)
        assert not questions & previous_questions
        assert not works & previous_works
        for row in rows:
            assert row["quote"] in (raw / row["source_file"]).read_text("utf-8-sig")
            assert row["quote"] in row["source_passage"]
            assert digest(raw / row["source_file"]) == row["source_sha256"]
            assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
        previous_questions |= questions
        previous_works |= works
        splits[split] = {"rows": len(rows), "works": sorted(works), "topics": dict(Counter(r["topic"] for r in rows))}
        hashes[str(path.relative_to(root))] = digest(path)
    questions_path = root / "data/evaluation/test_questions.json"
    questions = json.loads(questions_path.read_text("utf-8"))
    assert len({q["question"] for q in questions}) == len(questions)
    assert not {q["question"].strip() for q in questions} & previous_questions
    for relative in ("data/evaluation/test_questions.json", "configs/qlora_formal.yaml",
                     "checkpoints/adapter_formal/adapter_config.json", "checkpoints/adapter_formal/adapter_model.safetensors"):
        hashes[relative] = digest(root / relative)
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets", "llamafactory", "matplotlib"):
        packages[name] = importlib.metadata.version(name)
    result = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages,
              "original_files_hash_verified": checked, "splits": splits,
              "split_works_disjoint": True, "literal_quotes_verified": True,
              "evaluation_questions": len(questions), "evaluation_categories": dict(Counter(q["category"] for q in questions)),
              "evaluation_question_exact_overlap_with_sft": 0, "sha256": hashes,
              "limitations": ["Exact question non-overlap does not establish semantic independence.",
                              "Quotes are literal source matches, not expert doctrinal review."]}
    import torch
    result["hardware"] = {"gpu": torch.cuda.get_device_name(0),
                          "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
                          "pytorch_cuda": torch.version.cuda}
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
