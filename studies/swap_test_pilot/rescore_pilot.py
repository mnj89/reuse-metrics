"""Recompute the swap-test pilot (Table VIII) from the stored model answers and compare with the stored summaries.

    python studies/swap_test_pilot/rescore_pilot.py

Makes no API calls: every answer is read from results/swap_test_pilot/{pr,doc}/cache.jsonl.
Exits with an error if any recomputed count differs from the stored one.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "results" / "swap_test_pilot"
MODELS = ["anthropic:claude-haiku-5-5", "anthropic:claude-sonnet-5-5", "ollama:gpt-oss:120b"]
CONFIGS = [("pr", "pr", "token"), ("doc", "doc", "token"), ("pr_alpha", "pr", "alpha"), ("doc_alpha", "doc", "alpha")]

ok = True
print(f"{'configuration':<12}{'theta':>6}{'under-spec.':>13}{'coupled':>9}{'passed':>8}   check")
for name, level, dist in CONFIGS:
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache.jsonl"
        cache.write_bytes((PILOT / level / "cache.jsonl").read_bytes())   # work on a copy
        subprocess.run([sys.executable, str(Path(__file__).with_name("gci_run.py")),
                        "--blocks", str(PILOT / "blocks" / "blocks.jsonl"), "--models", *MODELS,
                        "--origin", "Claude_Code=anthropic:claude-sonnet-5-5", "--theta", "0.3", "0.4", "0.5",
                        "--R", "3", "--q", "0.34", "--prompt-level", level, "--distance", dist,
                        "--per-repo", str(ROOT / "results" / "field_study" / "pooled_per_repo.csv"),
                        "--cache", str(cache), "--out", tmp], check=True, capture_output=True, text=True)
        new = json.loads((Path(tmp) / "summary.json").read_text())
    old = json.loads((PILOT / name / "summary.json").read_text())
    if new["api_calls_this_run"] != 0:
        ok = False; print(f"{name}: unexpected API calls ({new['api_calls_this_run']})")
    for th in ("0.3", "0.4", "0.5"):
        v, w = new["by_theta"][th]["verdicts"], old["by_theta"][th]["verdicts"]
        same = v == w and new["by_theta"][th]["theorem3_monotone"] == old["by_theta"][th]["theorem3_monotone"]
        ok &= same
        u, c = v.get("under_specified", 0), v.get("coupled", 0)
        print(f"{name:<12}{th:>6}{u:>13}{c:>9}{30 - u:>8}   {'matches stored summary' if same else 'DIFFERS'}")
print("All pilot numbers reproduced from the stored answers." if ok else "Mismatch: see lines marked DIFFERS.")
sys.exit(0 if ok else 1)
