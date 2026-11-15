<div align="center">

# 🦇 Flandre Scarlet

### *A tiny local AI companion with a vampire's heart and a Linux tutor's brain.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-ff3355?style=for-the-badge&logo=python&logoColor=white&labelColor=2b0a14)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-lightweight-ff4d6d?style=for-the-badge&logo=flask&logoColor=white&labelColor=2b0a14)](https://flask.palletsprojects.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local-ff6b8a?style=for-the-badge&logo=ollama&logoColor=white&labelColor=2b0a14)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-ff8899?style=for-the-badge&labelColor=2b0a14)](LICENSE)
[![4GB RAM](https://img.shields.io/badge/Runs%20on-4GB%20RAM-ffb3c8?style=for-the-badge&labelColor=2b0a14)](#-minimum-requirements)

*"Oh — you're here. What did you want to talk about?"*

</div>

---

<div align="center">

### 🌸 A dreamy, red‑fogged chat window for your own machine.

Single file. No cloud. No API keys. No telemetry.
Just **you**, **Flandre**, and the soft glow of a terminal.

</div>

---

## 📖 Table of Contents

- [✨ What is this?](#-what-is-this)
- [🎨 Features](#-features)
- [💻 Minimum Requirements](#-minimum-requirements)
- [⚡ One‑Command Install](#-one-command-install)
- [🦇 Manual Installation (step by step)](#-manual-installation-step-by-step)
- [📦 Where Does the AI Model Live?](#-where-does-the-ai-model-live)
- [🚀 Running Flandre](#-running-flandre)
- [⚙️ Configuration](#️-configuration)
- [🧠 The Memory System](#-the-memory-system)
- [🎭 Customizing Flandre](#-customizing-flandre)
- [📁 Files Created](#-files-created)
- [🩺 Troubleshooting](#-troubleshooting)
- [⚠️ Security & Safe Use](#️-security--safe-use)
- [🙏 Credits](#-credits)
- [📜 License](#-license)

---

## ✨ What is this?

**Flandre Scarlet** is a self‑hosted, single‑file Python webapp that gives you a
local AI companion themed after the youngest vampire sister of the Scarlet
Devil Mansion.

She runs entirely on your own machine using a **tiny local language model**
(via [Ollama](https://ollama.com)), keeps **persistent memory across sessions**,
and is genuinely useful for:

- 🧛 Chatting about life, moods, and philosophy
- 🐧 **Linux troubleshooting** and command explanations
- 📜 Reading and explaining config files, logs, and error messages
- 🛠️ Small coding questions
- ☕ …just having someone to talk to at 3 AM

No API key. No cloud. No per‑message billing. No data leaving your computer.

> Originally inspired by a simple terminal Mistral chat script, this project
> rebuilds the experience as a beautiful local webapp — with a smaller model,
> a summary‑based long‑term memory, and a personality.

---

## 🎨 Features

| | |
|---|---|
| 🦇 **Flandre persona** | Sweet, playful, slightly unhinged — but genuinely helpful. |
| 🖋️ **Typewriter streaming** | Replies appear character‑by‑character with a red crystal cursor. |
| 🌫️ **Dreamy red fog UI** | Animated gradient fog, glowing crystals, soft crimson glow. |
| 🧠 **Persistent memory** | Conversations survive restarts, stored in `memory.jsonl`. |
| 📚 **Smart summarization** | When memory grows too large, Flandre summarizes it *herself*. |
| 💻 **Runs on weak hardware** | Tested conceptually against ~4 GB RAM laptops (use `qwen2.5:0.5b`). |
| 🔒 **Fully local** | Talks only to `127.0.0.1`. No external network calls. |
| 📄 **Single file** | `main.py` is the whole app — HTML, CSS, JS, and Python. |
| 🔄 **Rolling context** | Keeps only the last N messages to stay light. |
| 🐧 **Linux‑useful** | Clear commands, honest "I don't know"s, and danger warnings. |

---

## 💻 Minimum Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | Any x86_64 | Any modern CPU |
| **RAM** | ~2.5 GB free | 6 GB+ |
| **Disk** | ~2 GB (model + deps) | 5 GB+ |
| **OS** | Linux, macOS, Windows | Linux |
| **Python** | 3.9+ | 3.11+ |
| **Model** | `qwen2.5:0.5b` (~400 MB) | `qwen2.5:1.5b` (~1 GB) |
| **GPU** | Optional | Nice if you have one |

> ⚠️ **A note on 4 GB laptops:** the 1.5B model works but is tight when a browser
> is open. The **0.5B** model is the safe choice. It's still conversational and
> still helpful with Linux commands, just less poetic.

---

## ⚡ One‑Command Install

Copy, paste, wait. That's it.

### On Linux / macOS (bash/zsh):

```bash
curl -fsSL https://ollama.com/install.sh | sh && \
pip install --user flask requests && \
ollama pull qwen2.5:1.5b
```

> 🪶 **Weak laptop?** Swap the model for the smaller one:
> ```bash
> curl -fsSL https://ollama.com/install.sh | sh && \
> pip install --user flask requests && \
> ollama pull qwen2.5:0.5b
> ```

### On Windows (PowerShell):

```powershell
# 1. Install Ollama
winget install Ollama.Ollama

# 2. Install Python deps
pip install flask requests

# 3. Pull the model
ollama pull qwen2.5:1.5b
```

Then jump to [Running Flandre](#-running-flandre).

---

## 🦇 Manual Installation (step by step)

If you prefer to know what's happening, here's every step.

### 1. Install Ollama

Ollama is the engine that runs the model locally. Install it **once**:

- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **macOS / Windows / manual:** download from
  👉 **[https://ollama.com/download](https://ollama.com/download)**

Verify:
```bash
ollama --version
```

### 2. Pull the AI model

Pick the one that matches your hardware:

```bash
# Recommended for ~4 GB RAM laptops:
ollama pull qwen2.5:0.5b

# Or, if you have a bit more headroom:
ollama pull qwen2.5:1.5b
```

You can also try other small models:
```bash
ollama pull phi3:mini
ollama pull llama3.2:1b
ollama pull deepseek-r1:1.5b
```

### 3. Install Python dependencies

```bash
pip install flask requests
```

### 4. Clone this repo & run

```bash
git clone https://github.com/<your-name>/flandre-scarlet.git
cd flandre-scarlet
python main.py
```

Open **http://127.0.0.1:5000** in your browser. 🦇

---

## 📦 Where Does the AI Model Live?

When you run `ollama pull`, the model is downloaded **once** and stored on
disk. It is **not** inside this repo, and it's **not** in your project folder —
it lives in Ollama's own model directory.

### Default model storage locations

| OS | Path |
|---|---|
| 🐧 **Linux (systemd service)** | `/usr/share/ollama/.ollama/models` |
| 🐧 **Linux (user‑run `ollama serve`)** | `~/.ollama/models` |
| 🍎 **macOS** | `~/.ollama/models` |
| 🪟 **Windows** | `C:\Users\<YourName>\.ollama\models` |

### Changing the storage location

Set the `OLLAMA_MODELS` environment variable **before** starting Ollama:

```bash
export OLLAMA_MODELS=/mnt/bigdisk/ollama-models
sudo systemctl restart ollama     # if using the systemd service
```

Then pull models again (they'll land in the new location).

### Checking what's installed

```bash
ollama list
```

You should see something like:

```
NAME              ID            SIZE      MODIFIED
qwen2.5:0.5b      a8b0c5157701  397 MB    2 hours ago
qwen2.5:1.5b      65ec06548149  986 MB    2 hours ago
```

### Where the app's *memory* lives

The webapp's conversation memory is **separate** and lives right next to the
script:

```
flandre-scarlet/
├── main.py
├── memory.jsonl     ← persistent conversation memory
└── persona.txt      ← optional custom persona (see below)
```

---

## 🚀 Running Flandre

```bash
python main.py
```

You'll see:

```
  ✦ Flandre Scarlet — local companion
  ──────────────────────────────────
  model    : qwen2.5:1.5b
  ollama   : http://127.0.0.1:11434
  memory   : /home/you/flandre-scarlet/memory.jsonl
  persona  : /home/you/flandre-scarlet/persona.txt (optional override)
  open     : http://127.0.0.1:5000
```

Open the URL, and start talking.

### Changing the port

Already running something on port 5000? No problem:

```bash
FLANDRE_PORT=5050 python main.py
```

Or change the default inside `main.py`:

```python
PORT = int(os.getenv("FLANDRE_PORT", "5050"))
```

Check what's using a port on Linux:

```bash
ss -tlnp | grep :5000
# or
lsof -i :5000
```

---

## ⚙️ Configuration

All configuration is via environment variables — nothing to edit unless you
want to.

| Variable | Default | What it does |
|---|---|---|
| `FLANDRE_MODEL` | `qwen2.5:1.5b` | The Ollama model name to use. |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Where Ollama is listening. |
| `FLANDRE_PORT` | `5000` | The port the webapp runs on. |

### Example

```bash
FLANDRE_MODEL=qwen2.5:0.5b \
FLANDRE_PORT=5050 \
python main.py
```

### Tuning the memory system

In `main.py`:

```python
MAX_CHARS    = 10_000   # summarize when memory exceeds this many characters
KEEP_RECENT  = 40       # hard cap on messages kept after a summary
NUM_CTX      = 4096     # model context window
```

For a weaker machine, drop `NUM_CTX` to `2048` and `MAX_CHARS` to `6000`.

---

## 🧠 The Memory System

Flandre **remembers you** across restarts. Here's how, in plain English:

1. **Every turn is saved immediately** to `memory.jsonl` (one JSON object per line).
2. **Normal context** re‑loads the last messages on startup.
3. **When memory gets too large** (over `MAX_CHARS`), a **background thread**
   asks the model to compress it into a short third‑person note:

   > *"The user is named Maya and runs Arch Linux on a ThinkPad. They were
   > debugging a broken GRUB config and seemed tired. They mentioned wanting
   > to learn Rust. Flandre promised to remember."*

4. **The summary replaces the old history**, keeping the app fast forever.

You'll see it happen in the console:

```
[memory] compressing…
[memory] compressed to 1 message(s)
```

### Reset her memory

Click **"reset memory"** in the top‑left corner of the app, or delete the file:

```bash
rm memory.jsonl
```

---

## 🎭 Customizing Flandre

Want her to be a different character? Or a different *kind* of Flandre?

1. Create `persona.txt` next to `main.py`.
2. Write her new personality in plain language — likes, quirks, tone, how she
   handles technical questions.
3. Restart the app. Done.

The bundled default persona is in `main.py` under `DEFAULT_PERSONA`, which is
used whenever `persona.txt` doesn't exist.

### Tips for a good persona

- Keep it **under 800 words** — long personas eat your context window.
- **Explicitly say how she should handle technical help** (e.g. "warn before
  dangerous commands"), or she may lean too far into character.
- Say "**never mention being an AI or assistant**" if you want full immersion.
- Mention her **speech style** (short? poetic? chaotic? formal?).

---

## 📁 Files Created

| File | Purpose |
|---|---|
| `main.py` | The entire app — HTML, CSS, JS, Flask, Ollama client, memory. |
| `memory.jsonl` | Rolling conversation memory (auto‑created). |
| `persona.txt` | *(optional)* Custom system prompt override. |
| `memory.tmp` | Temporary file used during atomic memory saves (auto‑deleted). |

You can safely `.gitignore` the first two:

```gitignore
memory.jsonl
memory.tmp
__pycache__/
*.pyc
.env
```

---

## 🩺 Troubleshooting

<details>
<summary><b>🦇 "Ollama not reachable" / the health dot is red</b></summary>

Ollama isn't running. Start it:

```bash
sudo systemctl start ollama
# or, without systemd:
ollama serve
```

Then verify:

```bash
curl http://127.0.0.1:11434/api/tags
```
</details>

<details>
<summary><b>🦇 "model missing" / "model 'qwen2.5:1.5b' not found"</b></summary>

You forgot to pull it:

```bash
ollama pull qwen2.5:1.5b
# or the smaller one:
ollama pull qwen2.5:0.5b
```
</details>

<details>
<summary><b>🦇 "Address already in use"</b></summary>

Something else is on port 5000. Use a different one:

```bash
FLANDRE_PORT=5050 python main.py
```
</details>

<details>
<summary><b>🦇 The laptop gets very slow / swaps hard</b></summary>

Switch to the smallest model and shrink the context:

```bash
FLANDRE_MODEL=qwen2.5:0.5b python main.py
```

Also stop Ollama when you're not using it:

```bash
sudo systemctl stop ollama
```

Or close your browser's other tabs — Chromium alone can eat 1 GB.
</details>

<details>
<summary><b>🦇 Replies take forever</b></summary>

On a CPU‑only 4 GB laptop, expect **a few tokens per second** with the 1.5B
model. The 0.5B model is much faster. Don't try to run a 7B model — it will
crawl or crash.
</details>

<details>
<summary><b>🦇 Memory file grew enormous / conversation feels repetitive</b></summary>

Either lower `MAX_CHARS` in `main.py` so it summarizes sooner, or reset:

```bash
rm memory.jsonl
```
</details>

---

## ⚠️ Security & Safe Use

- The webapp binds to **`127.0.0.1` only** — it's local, not on your network.
  Don't change it to `0.0.0.0` unless you know what you're doing.
- **Flandre can be wrong.** Small models hallucinate Linux commands. Always
  read a command before running it, especially anything with `sudo`, `rm`,
  `dd`, `mkfs`, or `>` redirections.
- **Never paste secrets** (passwords, API keys, private keys) into any local
  LLM chat — anything in `memory.jsonl` stays in plain text on disk.
- The persona *thematically* jokes about destruction — that's flavor text.
  The prompt explicitly tells the model **not** to reduce usefulness with
  roleplay and to warn before dangerous commands.
- The webapp trusts whatever runs on your machine — it's not hardened for
  public exposure.

---

## 🙏 Credits

- 🧛 **Flandre Scarlet** and the *Touhou Project* universe © **ZUN / Team Shanghai Alice**.
  This is a fan project; not affiliated or endorsed.
- 🦙 **[Ollama](https://ollama.com)** — local model serving, done beautifully.
- 🤖 **Qwen2.5** by **Alibaba Cloud** — the tiny models that make this possible.
- 🐍 **[Flask](https://flask.palletsprojects.com/)** — a webapp in 300 lines.
- ✨ Originally sketched as a terminal Mistral chat; rebuilt for local, tiny,
  and beautiful.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

Use it, fork it, remix it, theme it after your favorite character.
Just don't blame Flandre if you `rm -rf` something you shouldn't have. 🦇

---

<div align="center">

*"Heehee — you can stay a little longer, right?"*

🌸 **[⬆ back to top](#-flandre-scarlet)** 🌸

</div>
