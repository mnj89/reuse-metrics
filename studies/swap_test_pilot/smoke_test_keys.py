"""Call each model once to confirm keys and model IDs before the real run.

    export ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... OPENAI_API_KEY=...
    python smoke_test_keys.py anthropic:<claude-model-id> google:<gemini-model-id> openai:<gpt-model-id>
"""
import sys
from llm_oracle import make_provider

for spec in sys.argv[1:]:
    kind, model = spec.split(":", 1)
    try:
        text = make_provider(spec).generate(model, "Write a Python function `def add(a, b):` that returns a + b.", 0.2, 0)
        if not text.strip():
            print(f"FAIL {spec}: empty answer (the model used up its output limit or refused)")
        else:
            print(f"OK   {spec}\n     {text.strip().splitlines()[0][:80]} ...")
    except Exception as e:
        print(f"FAIL {spec}: {type(e).__name__}: {e}")
