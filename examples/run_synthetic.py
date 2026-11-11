"""End-to-end demo: generate a synthetic system with a planted true partition,
run the operational protocol, and compare estimate with truth.

    python examples/run_synthetic.py --seed 3 --alt-models model-A model-B
"""
import argparse, json
from reuse_metrics import ProtocolParams, Prov, counts, generate_system, partition, report

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--units", type=int, default=400)
ap.add_argument("--alt-models", nargs="+", default=["model-A", "model-B"])
ap.add_argument("--theta", type=float, default=0.4)
a = ap.parse_args()

system, truth, oracle, N = generate_system(n_units=a.units, seed=a.seed)
params = ProtocolParams(pattern_set=N, alt_models=set(a.alt_models), theta=a.theta)
est = partition(system, params, oracle)
recon = {e.event_id for e in system.events}

print("parameters:", json.dumps(params.as_dict()))
print("true counts:     ", {k.value: v for k, v in counts(truth).items()})
print("estimated counts:", {k.value: v for k, v in counts(est).items()})
agree = sum(1 for k in truth if truth[k] == est[k]) / len(truth)
print(f"per-unit agreement with truth: {agree:.3f}")
print("metrics on estimate:", {k: round(v, 3) for k, v in report(est, system.events, N, recon).items() if not k.startswith('n_')})
