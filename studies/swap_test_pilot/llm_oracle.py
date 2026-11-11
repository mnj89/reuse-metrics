"""Regeneration oracle over real model APIs, with an on-disk cache.

Implements d^{(q)}(b; model) from Section VIII-C: draw R samples from `model` on
the block's prompt, compute the distance of each sample to the block, and return
the ceil(q*R)-th smallest distance.

Providers
    anthropic:<model-id>   needs ANTHROPIC_API_KEY   (pip install anthropic)
    google:<model-id>      needs GOOGLE_API_KEY      (pip install google-genai)
    openai:<model-id>      needs OPENAI_API_KEY      (pip install openai)
    glm:<model-id>         needs GLM_API_KEY, optional GLM_BASE_URL (pip install openai)
    ollama:<model-id>      open-weights model on Ollama Cloud (OLLAMA_API_KEY) or a local server (pip install requests)
    claudecli:<model-id>   Claude through the Claude Code CLI on a plan login (claude -p); no API key
    fake:<name>            deterministic stub for offline tests (no network)

Distance: 1 - similarity of the token sequences of the two functions after
stripping comments and docstrings and normalising formatting with ast.unparse.
A sample that does not parse, or does not define a function, has distance 1.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
import random
import re
import time
import tokenize
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

MAX_OUTPUT_TOKENS = 8192   # thinking models count their reasoning tokens against this limit

SYSTEM = ("You are a code generation assistant. Reply with Python code only: one function definition, "
          "no explanation, no markdown fences.")


def build_prompt(block: Dict, level: str = "pr") -> str:
    """level 'pr': the pull-request text only (the closest proxy for the developer's prompt).
    level 'doc': the PR text plus the docstring the agent wrote for this function, as a
    behavioural specification. The docstring is removed from the distance, so it cannot
    match by itself, but it does tell the model what the function must do."""
    spec = ""
    if level == "doc" and block.get("docstring"):
        spec = f"Specification (docstring of the function):\n{block['docstring']}\n\n"
    return (f"Pull request description:\n{block['prompt']}\n\n"
            f"File: {block['filename']}\n\n{spec}"
            f"Write the Python function `{block['signature']}` that this pull request adds. "
            f"Output only the complete function definition.")


# ----------------------------------------------------------------------------- distance

def normalise(code: str, alpha: bool = False) -> Optional[str]:
    code = re.sub(r"^```[a-zA-Z]*\n|```$", "", code.strip(), flags=re.M).strip()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fns = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not fns:
        return None
    fn = fns[0]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        fn.body = fn.body[1:] or [ast.Pass()]
    if alpha:
        fn = _AlphaRename().rename(fn)
    return ast.unparse(fn)


class _AlphaRename(ast.NodeTransformer):
    """Rename the function and every local name (parameters and names the function binds)
    to positional ids in order of first appearance, so that renaming a variable does not
    count as a difference. Attribute names, called global names and constants are kept."""

    def rename(self, fn):
        bound = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
        for extra in (fn.args.vararg, fn.args.kwarg):
            if extra:
                bound.add(extra.arg)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                bound.add(n.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                bound.add(n.name)
        self.bound, self.map = bound, {}
        fn.name = "f"
        return self.visit(fn)

    def _new(self, name):
        if name not in self.map:
            self.map[name] = f"v{len(self.map)}"
        return self.map[name]

    def visit_arg(self, node):
        node.arg = self._new(node.arg); return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.bound:
            node.id = self._new(node.id)
        return node

    def visit_ExceptHandler(self, node):
        if node.name:
            node.name = self._new(node.name)
        return self.generic_visit(node)


def tokens(code: str) -> List[str]:
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
                continue
            out.append(t.string)
    except (tokenize.TokenError, IndentationError):
        pass
    return out


def distance(original: str, sample: str, alpha: bool = False) -> float:
    """alpha=False: strict token distance. alpha=True: the same after renaming local names."""
    a, b = normalise(original, alpha), normalise(sample, alpha)
    if a is None or b is None:
        return 1.0
    return 1.0 - SequenceMatcher(None, tokens(a), tokens(b), autojunk=False).ratio()


def short(msg: str, n: int = 160) -> str:
    msg = " ".join(msg.split())
    return msg if len(msg) <= n else msg[:n] + "..."


def classify_error(msg: str) -> Optional[str]:
    """Return a plain-language reason when retrying cannot help, else None."""
    m = msg.lower()
    if "free_tier" in m and "limit: 0" in m:
        return ("this model is not available on your free tier (quota limit 0). Enable billing for this "
                "Google project in AI Studio, or choose a model that the free tier includes.")
    if "credit_balance_exhausted" in m or "insufficient_quota" in m or "credit balance is too low" in m \
            or "insufficient balance" in m or "\u4f59\u989d\u4e0d\u8db3" in m or "'code': '1113'" in m:
        return "the account has no credit left. Add credit on the vendor's billing page, then re-run the same cell."
    if "invalid x-api-key" in m or "api key not valid" in m or "invalid api key" in m or "incorrect api key" in m \
            or "authentication_error" in m or "permission_denied" in m or "token expired or incorrect" in m \
            or "'code': '1000'" in m or "'code': '1001'" in m or "'code': '1002'" in m:
        return "the API key was rejected. Re-enter it in the key cell."
    if "not_found" in m or "no longer available" in m or "model_not_found" in m or "does not exist" in m \
            or "unknown model" in m or "'code': '1211'" in m:
        return "the model ID is not available to this key. Check the exact ID in the vendor's console."
    if "usage limit" in m or "rate limit reached" in m or "limit reached" in m or "5-hour limit" in m or "weekly limit" in m:
        return ("your Claude plan usage limit was reached. Wait for the reset time shown in the message, then "
                "re-run the same command: it resumes from the cache.")
    if "not logged in" in m or "please run /login" in m or "invalid api key" in m and "claudecli" in m:
        return "the Claude CLI is not signed in. Run `claude` once, sign in with your plan, then re-run."
    if "the claude command was not found" in m:
        return "the claude command was not found. Install Claude Code and sign in, or set CLAUDE_CLI to its path."
    if "connection error" in m or "connection refused" in m or "all connection attempts failed" in m:
        return ("cannot reach the local model server. Start it (ollama serve) in the background cell and re-run.")
    if "try pulling it first" in m:
        return "the model is not downloaded in Ollama. Run: ollama pull <model>, then re-run."
    if "empty: run the key cell again" in m:
        return msg
    return None


# ----------------------------------------------------------------------------- providers

class Provider:
    def generate(self, model: str, prompt: str, temperature: float, seed: int) -> str:
        raise NotImplementedError


class AnthropicProvider(Provider):
    """Messages API. Anthropic SDK 2.x no longer has a `temperature` argument, and
    newer models may refuse custom sampling settings. Temperature is therefore sent
    as an extra body field; if the API refuses it, the call is retried without it
    and the model is remembered (reported in summary.json)."""

    def __init__(self) -> None:
        import anthropic
        self.anthropic = anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is empty: run the key cell again")
        self.client = anthropic.Anthropic(api_key=key)
        self.no_temperature: set = set()

    def _text(self, r) -> str:
        return "".join(getattr(c, "text", "") or "" for c in r.content)

    def generate(self, model, prompt, temperature, seed):
        kwargs = dict(model=model, max_tokens=MAX_OUTPUT_TOKENS, system=SYSTEM,
                      messages=[{"role": "user", "content": prompt}])
        if model not in self.no_temperature:
            try:
                return self._text(self.client.messages.create(extra_body={"temperature": temperature}, **kwargs))
            except self.anthropic.BadRequestError as e:
                if "temperature" not in str(e).lower():
                    raise
                self.no_temperature.add(model)
                print(f"  [anthropic:{model}] does not accept temperature={temperature}; using its default", flush=True)
        return self._text(self.client.messages.create(**kwargs))


class GoogleProvider(Provider):
    def __init__(self) -> None:
        from google import genai
        self.genai = genai
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is empty: run the key cell again")
        self.client = genai.Client(api_key=key)

    def generate(self, model, prompt, temperature, seed):
        from google.genai import types
        r = self.client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature, system_instruction=SYSTEM, max_output_tokens=MAX_OUTPUT_TOKENS))
        return r.text or ""


class OpenAIProvider(Provider):
    """Chat Completions API. Some OpenAI models (reasoning models) reject a custom
    temperature; for those the call is retried once without it and the model is
    remembered, so the run records which models used the default temperature."""

    def __init__(self) -> None:
        import openai
        self.openai = openai
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is empty: run the key cell again")
        self.client = openai.OpenAI(api_key=key)
        self.no_temperature: set = set()

    def generate(self, model, prompt, temperature, seed):
        kwargs = dict(model=model, max_completion_tokens=MAX_OUTPUT_TOKENS,
                      messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        if model not in self.no_temperature:
            try:
                r = self.client.chat.completions.create(temperature=temperature, **kwargs)
                return r.choices[0].message.content or ""
            except self.openai.BadRequestError as e:
                if "temperature" not in str(e).lower():
                    raise
                self.no_temperature.add(model)
                print(f"  [openai:{model}] does not accept temperature={temperature}; using its default", flush=True)
        r = self.client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""


class GLMProvider(Provider):
    """Z.ai GLM models through their OpenAI-compatible Chat Completions endpoint.

    GLM_API_KEY   your Z.ai key
    GLM_BASE_URL  optional; default is the pay-as-you-go endpoint
                  https://api.z.ai/api/paas/v4/ . GLM Coding Plan keys must use
                  https://api.z.ai/api/coding/paas/v4 instead.

    GLM thinking mode is switched off so that output tokens go to the answer, not
    to hidden reasoning; if the endpoint rejects that field, it is dropped. GLM
    accepts temperatures strictly between 0 and 1."""

    DEFAULT_BASE = "https://api.z.ai/api/paas/v4/"

    def __init__(self) -> None:
        import openai
        self.openai = openai
        key = os.environ.get("GLM_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GLM_API_KEY is empty: run the key cell again")
        base = os.environ.get("GLM_BASE_URL", "").strip() or self.DEFAULT_BASE
        self.client = openai.OpenAI(api_key=key, base_url=base)
        self.no_temperature: set = set()
        self.no_thinking_field: set = set()

    def generate(self, model, prompt, temperature, seed):
        t = min(max(temperature, 0.01), 0.99)
        kwargs = dict(model=model, max_tokens=MAX_OUTPUT_TOKENS,
                      messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        for _ in range(3):
            extra = {} if model in self.no_thinking_field else {"thinking": {"type": "disabled"}}
            try:
                if model in self.no_temperature:
                    r = self.client.chat.completions.create(extra_body=extra or None, **kwargs)
                else:
                    r = self.client.chat.completions.create(temperature=t, extra_body=extra or None, **kwargs)
                return r.choices[0].message.content or ""
            except self.openai.BadRequestError as e:
                msg = str(e).lower()
                if "thinking" in msg and model not in self.no_thinking_field:
                    self.no_thinking_field.add(model); continue
                if "temperature" in msg and model not in self.no_temperature:
                    self.no_temperature.add(model)
                    print(f"  [glm:{model}] does not accept temperature={t}; using its default", flush=True)
                    continue
                raise
        raise RuntimeError(f"[glm:{model}] request rejected")


class OllamaProvider(Provider):
    """Open-weights model through Ollama's native /api/chat, local or Ollama Cloud.

    OLLAMA_API_KEY   set for Ollama Cloud (then the base URL defaults to https://ollama.com);
                     leave unset for a local server (default http://localhost:11434).
    OLLAMA_BASE_URL  optional override, without a /v1 suffix.

    Temperature and seed are sent as given, so unlike the hosted Claude models this
    one runs at the requested temperature. Thinking is switched off where supported."""

    def __init__(self) -> None:
        import requests
        self.requests = requests
        key = os.environ.get("OLLAMA_API_KEY", "").strip()
        base = os.environ.get("OLLAMA_BASE_URL", "").strip() or ("https://ollama.com" if key else "http://localhost:11434")
        self.url = base.rstrip("/") + "/api/chat"
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.no_think_field: set = set()

    def generate(self, model, prompt, temperature, seed):
        for _ in range(2):
            body = {"model": model, "stream": False,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    "options": {"temperature": temperature, "seed": seed, "num_predict": MAX_OUTPUT_TOKENS}}
            if model not in self.no_think_field:
                body["think"] = False
            r = self.requests.post(self.url, json=body, headers=self.headers, timeout=900)
            if r.status_code == 400 and "think" in r.text.lower() and model not in self.no_think_field:
                self.no_think_field.add(model)
                continue
            if r.status_code == 401:
                raise RuntimeError("[ollama] 401 invalid api key")
            if r.status_code == 404:
                raise RuntimeError(f"[ollama:{model}] 404 model_not_found, try pulling it first: {short(r.text)}")
            if r.status_code >= 400:
                raise RuntimeError(f"[ollama:{model}] HTTP {r.status_code}: {short(r.text)}")
            return (r.json().get("message") or {}).get("content") or ""
        raise RuntimeError(f"[ollama:{model}] request rejected")


class ClaudeCLIProvider(Provider):
    """Claude through the Claude Code command line (`claude -p`), using your plan login.

    Run it on a machine where `claude` is installed and signed in with your plan. The child
    process runs in an empty temporary folder with no tools, no slash commands and no session
    saved, with a bare system prompt, so it behaves as a plain text generator. ANTHROPIC_API_KEY
    is removed from the child's environment, otherwise the CLI would bill that key instead of
    your plan. The temperature cannot be set through the CLI (models run at their default).
    Model IDs: a full ID (claude-sonnet-5-5) or an alias; record the resolved version."""

    def __init__(self) -> None:
        import shutil, tempfile
        self.exe = os.environ.get("CLAUDE_CLI", "").strip() or shutil.which("claude")
        if not self.exe:
            raise RuntimeError("the claude command was not found: install Claude Code and sign in, or set CLAUDE_CLI")
        self.cwd = tempfile.mkdtemp(prefix="gci_cli_")
        self.no_temperature: set = set()   # models seen; reported in summary.json as running at the default temperature
        self.timeout = int(os.environ.get("CLAUDE_CLI_TIMEOUT", "300"))

    def generate(self, model, prompt, temperature, seed):
        import subprocess
        self.no_temperature.add(model)
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        cmd = [self.exe, "-p", "--model", model, "--output-format", "text", "--system-prompt", SYSTEM,
               "--tools", "", "--disable-slash-commands", "--strict-mcp-config", "--no-session-persistence"]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=self.cwd, env=env, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"[claudecli:{model}] exit {r.returncode}: {short((r.stderr or r.stdout or '').strip(), 300)}")
        return r.stdout.strip()


class FakeProvider(Provider):
    """Offline stub. 'fake:origin' reproduces the block with small edits; 'fake:other'
    returns a different function; 'fake:random' flips between the two. The block's
    code is smuggled in via the prompt text, so this is for plumbing tests only."""

    def generate(self, model, prompt, temperature, seed):
        rng = random.Random(f"{model}|{prompt}|{seed}")
        m = re.search(r"Write the Python function `([^`]+)`", prompt)
        sig = m.group(1) if m else "def f():"
        good = (f"{sig}\n    total = 0\n    for it in items:\n        if it is None:\n            continue\n"
                f"        total += int(it)\n    return total{' + 0' if rng.random() < 0.3 else ''}\n")
        bad = f"{sig}\n    raise NotImplementedError('different')\n"
        name = model.split(":", 1)[-1]
        if name == "origin":
            return good
        if name == "other":
            return bad
        return good if rng.random() < 0.5 else bad


def make_provider(spec: str) -> Provider:
    kind = spec.split(":", 1)[0]
    return {"anthropic": AnthropicProvider, "google": GoogleProvider, "openai": OpenAIProvider,
            "glm": GLMProvider, "ollama": OllamaProvider, "claudecli": ClaudeCLIProvider, "fake": FakeProvider}[kind]()


# ----------------------------------------------------------------------------- oracle

class LLMOracle:
    """oracle(block, model_spec, prompt) -> d^{(q)}; model_spec = 'provider:model-id'."""

    def __init__(self, cache_path: str, R: int = 3, q: float = 0.34, temperature: float = 0.2,
                 sleep: float = 0.0, max_retries: int = 10, prompt_level: str = "pr", alpha: bool = False) -> None:
        self.cache_path = Path(cache_path); self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.R, self.q, self.temperature, self.sleep, self.max_retries = R, q, temperature, sleep, max_retries
        self.prompt_level, self.alpha = prompt_level, alpha
        self.cache: Dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.open():
                rec = json.loads(line); self.cache[rec["key"]] = rec["text"]
        self._providers: Dict[str, Provider] = {}
        self.calls = 0
        self.empty_answers: Dict[str, int] = {}

    def _provider(self, spec: str) -> Provider:
        kind = spec.split(":", 1)[0]
        if kind not in self._providers:
            self._providers[kind] = make_provider(spec)
        return self._providers[kind]

    def sample(self, spec: str, prompt: str, k: int) -> str:
        key = hashlib.sha256(f"{spec}|{self.temperature}|{k}|{prompt}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        model_id = spec.split(":", 1)[1]
        last = None
        for attempt in range(self.max_retries):
            try:
                text = self._provider(spec).generate(model_id, prompt, self.temperature, k)
                break
            except Exception as e:
                msg = str(e)
                fatal = classify_error(msg)
                if fatal:
                    raise RuntimeError(f"[{spec}] {fatal}") from None
                last = e
                wait = min(2 ** attempt, 60) + random.uniform(0, 1)
                print(f"  [{spec}] {type(e).__name__}: {short(msg)}; retry {attempt + 1}/{self.max_retries} in {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"[{spec}] still failing after {self.max_retries} attempts: {short(str(last))}. "
                               "The cache keeps everything done so far: re-run the same cell later to resume.")
        self.calls += 1
        if not text.strip():
            self.empty_answers[spec] = self.empty_answers.get(spec, 0) + 1
            print(f"  [{spec}] returned an empty answer (counted as distance 1)", flush=True)
        self.cache[key] = text
        with self.cache_path.open("a") as f:
            f.write(json.dumps({"key": key, "spec": spec, "k": k, "text": text}) + "\n")
        if self.sleep:
            time.sleep(self.sleep)
        return text

    def distances(self, block: Dict, spec: str) -> List[float]:
        prompt = build_prompt(block, self.prompt_level)
        return sorted(distance(block["code"], self.sample(spec, prompt, k), self.alpha) for k in range(self.R))

    def __call__(self, block: Dict, spec: str, prompt: str = "") -> float:
        ds = self.distances(block, spec)
        idx = max(1, math.ceil(self.q * self.R)) - 1
        return ds[idx]
