#!/usr/bin/env python3
"""
AutoDev Loop — One-File Autonomous Builder

Usage example:
python autodev.py --task "Implement add(a,b) with tests" \
  --runner "python -m unittest discover -s examples/buggy_add -v" \
  --workspace demo_ws --max-iters 3 --reset
"""

import argparse, os, sys, shutil, subprocess, pathlib, re, zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -------------------- LLM CLIENTS --------------------

def use_ollama() -> bool:
    return os.getenv("USE_OLLAMA", "0") == "1"

class LLM:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError

class OpenAILLM(LLM):
    def __init__(self, model: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set and USE_OLLAMA!=1")
        import openai
        self.client = openai.OpenAI()
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

class OllamaLLM(LLM):
    def __init__(self, model: str):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")

    def complete(self, prompt: str) -> str:
        proc = subprocess.run(["ollama", "run", self.model],
                              input=prompt, text=True,
                              capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ollama error: {proc.stderr}")
        return proc.stdout

# -------------------- PATCHING --------------------

PATCH_RE = re.compile(r"PATCH::([^\n]+)\n(.*?)\nENDPATCH", re.DOTALL)

def extract_patches(blob: str) -> List[Tuple[str, str]]:
    return [(p.strip(), c) for p, c in PATCH_RE.findall(blob)]

# -------------------- FILE IO --------------------

@dataclass
class Workspace:
    root: pathlib.Path

    def abspath(self, rel: str) -> pathlib.Path:
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError("Path escapes workspace")
        return p

    def write_file(self, rel: str, content: str) -> None:
        p = self.abspath(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content.rstrip("\n") + "\n", encoding="utf-8")

    def read_many(self, rels: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for r in rels:
            p = self.abspath(r)
            if p.exists():
                out[r] = p.read_text(encoding="utf-8")
        return out

    def snapshot_zip(self, zip_path: pathlib.Path) -> None:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for path in self.root.rglob("*"):
                if path.is_file():
                    z.write(path, path.relative_to(self.root))

# -------------------- RUN TESTS --------------------

@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    code: int

def run_tests(cmd: str, cwd: pathlib.Path, timeout_s: int = 180) -> RunResult:
    proc = subprocess.Popen(cmd, cwd=str(cwd), shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        return RunResult(False, "", f"Test timed out after {timeout_s}s", 124)
    return RunResult(proc.returncode == 0, out, err, proc.returncode)

# -------------------- PROMPTS --------------------

SYSTEM_RULES = """You are REVIEWER. Return ONLY PATCH blocks:
PATCH::relative/path
<full file>
ENDPATCH
"""

PLAN_RULES = """You are CODER. Propose initial minimal codebase + tests.
Return only PATCH blocks (no explanation).
"""

INITIAL_TASK_TEMPLATE = """TASK:\n{task}\n\nCreate an initial solution and tests that run with:\n{runner}\n"""

REPAIR_TEMPLATE = """TASK:\n{task}\n\nTEST COMMAND:\n{runner}\n\nFAILURE:\n{report}\n\nCURRENT FILES:\n{files}\n"""

# -------------------- LOOP --------------------

def select_llm(model: str) -> LLM:
    return OllamaLLM(model) if use_ollama() else OpenAILLM(model)

def list_project_files(ws: Workspace, max_chars: int = 20000) -> str:
    parts: List[str] = []
    for path in sorted(ws.root.rglob("*")):
        if path.is_dir() or path.stat().st_size > 80_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        snippet = f"=== {path.relative_to(ws.root)} ===\n{text}\n"
        parts.append(snippet)
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n".join(parts)

def apply_patch_blob(ws: Workspace, blob: str) -> List[str]:
    changed: List[str] = []
    for rel, content in extract_patches(blob):
        ws.write_file(rel, content)
        changed.append(rel)
    return changed

def ensure_workspace(path: str, reset: bool = False) -> Workspace:
    root = pathlib.Path(path).resolve()
    if reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return Workspace(root)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--zip", dest="zip_out", default="final_workspace.zip")
    ap.add_argument("--git", action="store_true")
    args = ap.parse_args()

    task_text = args.task
    if task_text.startswith("@"):
        task_text = pathlib.Path(task_text[1:]).read_text(encoding="utf-8")

    ws = ensure_workspace(args.workspace, reset=args.reset)
    llm = select_llm(args.model)

    # Initial plan
    init_prompt = PLAN_RULES + "\n\n" + INITIAL_TASK_TEMPLATE.format(
        task=task_text, runner=args.runner)
    print("[autodev] requesting initial code…")
    init_blob = llm.complete(init_prompt)
    changed = apply_patch_blob(ws, init_blob)
    print(f"[autodev] wrote {len(changed)} files.")

    # Repair loop
    for i in range(1, args.max_iters + 1):
        print(f"\n[autodev] ITER {i}/{args.max_iters} — running tests…")
        rr = run_tests(args.runner, ws.root)
        if rr.ok:
            print("[autodev] ✅ tests passing.")
            break
        report = (rr.stderr or "") + "\n" + (rr.stdout or "")
        files_context = list_project_files(ws)
        repair_prompt = SYSTEM_RULES + "\n\n" + REPAIR_TEMPLATE.format(
            task=task_text, runner=args.runner,
            report=report, files=files_context)
        print("[autodev] requesting repair patches…")
        blob = llm.complete(repair_prompt)
        patches = extract_patches(blob)
        if not patches:
            print("[autodev] model returned no patches. stopping.")
            break
        changed = apply_patch_blob(ws, blob)
        print(f"[autodev] applied patches: {changed}")

    ws.snapshot_zip(pathlib.Path(args.zip_out))
    print(f"[autodev] 📦 exported workspace to {args.zip_out}")
    final = run_tests(args.runner, ws.root)
    sys.exit(0 if final.ok else 1)

if __name__ == "__main__":
    main()
