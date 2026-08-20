#!/usr/bin/env python3
"""Generate labs/labNN-<slug>/README.md from the SAME single source as the deck,
the Lesson Plan and the Learner Guide (course_data.py + data_domainN.py).

Why this exists: build_learner_guide.py deliberately prefers the DETAILED steps
in each lab's README over the terse course_data steps. If the READMEs were
hand-written they would drift from the deck the first time a lab changed.
Generating them from the same dicts makes drift impossible — the deck shows the
step instruction, the guide shows the instruction plus its note, and both come
from one list in one file.

Output format is the one build_learner_guide._readme_steps() parses:
a '## Steps' section, '### N. <instruction>' headers, note prose, then a fenced
block holding the command (```bash) or the AI prompt (```text).
"""
import os, sys, re, glob, importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import course_data as C

def _load_domains():
    acts = []
    for f in sorted(glob.glob(os.path.join(HERE, "data_domain[0-9]*.py")),
                    key=lambda q: int("".join(c for c in os.path.basename(q) if c.isdigit()) or 0)):
        n = "".join(c for c in os.path.basename(f) if c.isdigit())
        acts += getattr(importlib.import_module(os.path.basename(f)[:-3]), f"DOMAIN{n}", [])
    return acts
ACT = _load_domains()

def _find_repo(start):
    env = os.environ.get("COURSE_REPO")
    if env and os.path.isdir(env): return env
    d = start
    for _ in range(8):
        d = os.path.dirname(d)
        if os.path.isdir(os.path.join(d, "courseware")) and os.path.isdir(os.path.join(d, "labs")): return d
    return os.path.dirname(os.path.dirname(HERE))
REPO = _find_repo(HERE)
LABS = os.path.join(REPO, "labs")

# Shell command vs. natural-language prompt — kept identical to the deck engine's
# _is_shell() so a step is labelled the same way on the slide, in the guide and here.
_SHELL_STARTS = ("pip ", "pip3 ", "python ", "python3 ", "py ", "conda ", "git ", "cd ", "mkdir ", "ls", "dir ",
                 "streamlit ", "gradio ", "jupyter ", "curl ", "wget ", "code ", "set ", "export ", "echo ",
                 "pytest ", "uvicorn ", "winget ", "brew ", "apt ", "source ", "./", "$ ")
def is_shell(cmd):
    return str(cmd).strip().lower().startswith(_SHELL_STARTS)

def slug(title):
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-+", "-", s)

TOPICS = {t["num"]: t for t in C.TOPICS}

def render(a, prev, nxt):
    t = TOPICS[a["topic"]]
    L = []
    L.append(f"# Lab {a['num']} — {a['title']}")
    L.append("")
    L.append(f"> **Course:** {C.TITLE} (`{C.COURSE_CODE}`) · **Topic {t['code']}:** {t['title']}  ")
    L.append(f"> **Learning outcome:** {a['objective'][0].upper()}{a['objective'][1:]}.")
    L.append("")
    L.append("## Goal")
    L.append("")
    L.append(a["desc"])
    L.append("")
    if a.get("why"):
        L.append("## Why this lab matters")
        L.append("")
        L.append(a["why"])
        L.append("")
    L.append("## What you'll build")
    L.append("")
    L.append(f"**{a['build']}**")
    L.append("")
    L.append(f"**Tools:** {a['services']}")
    L.append("")
    if a.get("files"):
        L.append("**Files you end up with:**")
        L.append("")
        for f in a["files"]:
            L.append(f"- `{f}`")
        L.append("")
    L.append("## Before you start")
    L.append("")
    if a["num"] == 1:
        L.append("- A web browser and a Google account — Lab 1 runs entirely in Google Colab, with nothing to install.")
        L.append("- An AI coding assistant available: GitHub Copilot, Cursor, or Claude in a browser tab.")
    elif a["num"] == 2:
        L.append("- Lab 1 completed, so you have seen the vibe coding loop and started `prompts.md`.")
        L.append("- A Windows or Mac laptop you can install software on, with about 3 GB free.")
    else:
        L.append(f"- Lab {a['num']-1} completed, with your `torch-vibe/` workspace and virtual environment active.")
        L.append("- Your AI coding assistant open and able to see the files in the workspace.")
    L.append("")
    # ---- Steps: the section build_learner_guide.py parses back out ----
    L.append("## Steps")
    L.append("")
    notes = a.get("notes", [])
    for i, (instr, cmd) in enumerate(a["steps"], 1):
        L.append(f"### {i}. {instr}")
        L.append("")
        if i - 1 < len(notes) and notes[i - 1]:
            L.append(notes[i - 1])
            L.append("")
        if cmd:
            if is_shell(cmd):
                L.append("**COMMAND** — run this in your terminal:")
                L.append("")
                L.append("```bash")
            else:
                L.append("**PROMPT** — paste this into your AI coding assistant:")
                L.append("")
                L.append("```text")
            L.append(cmd)
            L.append("```")
            L.append("")
    L.append("## Verification — Test it")
    L.append("")
    L.append(a["test"])
    L.append("")
    if a.get("troubleshoot"):
        L.append("## Troubleshooting")
        L.append("")
        L.append("| Symptom | Fix |")
        L.append("| --- | --- |")
        for sym, fix in a["troubleshoot"]:
            L.append(f"| {sym} | {fix} |")
        L.append("")
    if a.get("stretch"):
        L.append("## Going further (optional)")
        L.append("")
        for s in a["stretch"]:
            L.append(f"- {s}")
        L.append("")
    L.append("---")
    L.append("")
    nav = []
    if prev: nav.append(f"[← Lab {prev['num']}: {prev['title']}](../{prev['dir']}/README.md)")
    nav.append("[All labs](../README.md)")
    if nxt: nav.append(f"[Lab {nxt['num']}: {nxt['title']} →](../{nxt['dir']}/README.md)")
    L.append(" · ".join(nav))
    L.append("")
    L.append(f"_{C.ORG} · {C.COURSE_CODE} · Version {C.VERSION} · {C.VERSION_DATE}_")
    L.append("")
    return "\n".join(L)

def index(dirs):
    L = [f"# Labs — {C.TITLE} (`{C.COURSE_CODE}`)", ""]
    L.append(f"{len(ACT)} hands-on labs across {len(C.TOPICS)} topics. Work through them in order — each lab "
             "reuses the workspace, the scripts and the prompting habits of the one before it.")
    L.append("")
    L.append("Every lab follows the same shape: a **Goal**, the **Steps** (each with the exact PROMPT to paste "
             "into your AI coding assistant, or the COMMAND to run), a **Test it** check that tells you what a "
             "correct result looks like, and a **Troubleshooting** table for when it does not.")
    L.append("")
    for t in C.TOPICS:
        L.append(f"## Topic {t['code']} — {t['title']}")
        L.append("")
        L.append("| Lab | Title | You'll build |")
        L.append("| --- | --- | --- |")
        for a in ACT:
            if a["topic"] != t["num"]: continue
            d = f"lab{a['num']:02d}-{slug(a['title'])}"
            L.append(f"| {a['num']} | [{a['title']}]({d}/README.md) | {a['build']} |")
        L.append("")
    L.append("## The ForgeSight project")
    L.append("")
    L.append("Every lab advances ONE project — **ForgeSight**, a deep learning suite for a precision "
             "metal-parts factory — from an empty folder in Lab 1 to a packaged, documented project in "
             f"Lab {len(ACT)}. The factory gives all three data modalities a single home:")
    L.append("")
    L.append("| Data | Model | Labs |")
    L.append("| --- | --- | --- |")
    L.append("| Tabular machine telemetry | Regression (tool wear) and 4-class QC classification | 3–11 |")
    L.append("| Surface-inspection images | CNN defect classifier and transfer learning | 12–16 |")
    L.append("| Vibration sensor readings | LSTM time-series forecasting | 17–20 |")
    L.append("")
    L.append("Each lab states the exact files it expects to already exist, so you can rejoin at any lab "
             "boundary if you fall behind.")
    L.append("")
    L.append("## Data and resources")
    L.append("")
    L.append("All datasets, the image generator and the Lab 6 review script live in "
             "[`resources/`](resources/) — see that folder's README for what each file is and the "
             "baseline every model has to beat. Everything is synthetic and deterministic (seed 42), so "
             "your numbers should match the Learner Guide.")
    L.append("")
    L.append(f"_{C.ORG} · Version {C.VERSION} · {C.VERSION_DATE}_")
    L.append("")
    return "\n".join(L)

def main():
    metas = [dict(num=a["num"], title=a["title"], dir=f"lab{a['num']:02d}-{slug(a['title'])}") for a in ACT]
    os.makedirs(LABS, exist_ok=True)
    for i, a in enumerate(ACT):
        d = os.path.join(LABS, metas[i]["dir"])
        os.makedirs(d, exist_ok=True)
        prev = metas[i - 1] if i > 0 else None
        nxt = metas[i + 1] if i + 1 < len(metas) else None
        path = os.path.join(d, "README.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(a, prev, nxt))
        print("Saved", path)
    with open(os.path.join(LABS, "README.md"), "w", encoding="utf-8") as f:
        f.write(index(metas))
    print("Saved", os.path.join(LABS, "README.md"))

if __name__ == "__main__":
    main()
