# Team setup guide

How to get a working development environment for the benchmark runner: Python
virtual environment, Inspect AI, Ollama, an optional vLLM server (VT ARC for Virginia
Tech students), and Docker. When you finish, `inspect eval --help` runs and the smoke
test passes against a local model.

## Prerequisites

| Tool | Version | Needed for |
| --- | --- | --- |
| Python | 3.10 or newer (developed on 3.14) | Everything |
| Git | any recent | Cloning, branches |
| Ollama | any recent | Running local models (task 1.5 onward) |
| Docker | any recent | Sandboxed tools (tasks 1.8 onward) |

Docker is not needed to complete this guide, but install it now so you are not
blocked when the sandbox work lands.

## 1. Clone the repository

```bash
git clone git@github.com:AlexJFDev/Boeing-Team-2.git
cd Boeing-Team-2
```

The default branch is `development`. Work on an issue branch named
`{type}-{number}/{name}`, for example `feat-2/inspect-ai-setup`. A GitHub Action
rejects PRs from branches that do not match.

## 2. Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins exact versions so everyone runs the same Inspect AI.
Do not upgrade packages on your own; open an issue and bump the pin in a PR.

In VS Code, the workspace settings select `.venv` automatically. Accept the
recommended extensions when prompted.

### Alternative: dev container

If you use GitHub Codespaces or the VS Code Dev Containers extension, the
`.devcontainer/` folder builds the same virtual environment automatically on
first launch. You do not need to follow this section's manual steps.

## 3. Verify the install

```bash
inspect eval --help
```

This should print the usage text with no errors. Then run the smoke test with the
built-in mock model, which needs no GPU or server:

```bash
inspect eval tasks/smoke_test.py --model mockllm/model
```

The run completes and prints a log path. The mock model returns filler text, so
a score of 0 is expected here; this only proves Inspect itself works.

## 4. Install Ollama and run against a real model

1. Install Ollama from <https://ollama.com/download>.
2. Pull a small model:

   ```bash
   ollama pull llama3.1
   ```

3. Make sure the server is running (`ollama list` should print a table, not an
   error), then run:

   ```bash
   inspect eval tasks/smoke_test.py --model ollama/llama3.1
   ```

   You should see `accuracy 1.000`.

## 5. Run against a vLLM server (optional)

Larger open-weight models can run on a vLLM server. vLLM exposes an OpenAI-compatible
API, and Inspect's `vllm` provider talks to it. Skip this step if Ollama is enough for
your work. Virginia Tech students can use VT's ARC service instead of running a
server (see 5.3).

### 5.1 Point Inspect at the server

If you have not already, copy the template: `cp .env.example .env`. Then add these
lines to `.env`:

```dotenv
VLLM_BASE_URL=<server URL, ending in /v1>
VLLM_API_KEY=<key for that server>
```

Inspect reads these two names directly, so no other setup is needed. The template is
prefilled with the VT ARC URL; replace it if you use a different server. Never commit
`.env` (see "API keys and secrets" below).

### 5.2 Run the smoke test

```bash
inspect eval tasks/smoke_test.py --model vllm/<model> --temperature 0 --seed 42
```

`<model>` is the model name the server lists. You should see `accuracy 1.000`.

### 5.3 Use VT ARC (Virginia Tech students)

VT Advanced Research Computing (ARC) serves open-weight models, such as
`gpt-oss-120b`, through a vLLM endpoint at `https://llm-api.arc.vt.edu/api/v1`. You
need a VT account, an ARC API key, and the VT VPN. ARC adds and removes models over
time; the current list is in the
[ARC LLM API guide](https://www.docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html).

### 5.4 Get an ARC API key

1. Go to <https://llm.arc.vt.edu> and sign in with your VT account. No separate ARC
   account is needed.
2. Open **User profile > Settings > Account > API keys** and generate a key.
3. Put it in `.env` as `VLLM_API_KEY`, with `VLLM_BASE_URL` set to the ARC URL above.

Keys are personal. Do not share yours, do not commit it to git, and do not paste it
anywhere except your `.env`.

### 5.5 Connect to the VT VPN

The ARC endpoint is reached over the VT VPN. Install and connect the Cisco VPN client
using the VT 4Help articles:

- [KB0016110](https://4help.vt.edu/sp?id=kb_article&sysparm_article=KB0016110)
- [KB0016111](https://4help.vt.edu/sp?id=kb_article&sysparm_article=KB0016111)
- [KB0016112](https://4help.vt.edu/sp?id=kb_article&sysparm_article=KB0016112)

Stay connected while you run anything against ARC.

### 5.6 Run the smoke test against ARC

With the VPN connected:

```bash
inspect eval tasks/smoke_test.py --model vllm/gpt-oss-120b --temperature 0 --seed 42
```

To also record ARC's details for the model in the log, and to fail before writing a
log if ARC is unreachable, use the runner instead:

```bash
python -m runner.run configs/arc.json tasks/smoke_test.py
```

ARC does not publish a weights revision, so the log records the model name plus the
`created_at` and `updated_at` ARC lists for it. Treat those as the only version signal.

## 6. Install Docker

Follow <https://docs.docker.com/engine/install/> for your OS. On Linux, add
yourself to the `docker` group so you do not need `sudo`:

```bash
sudo usermod -aG docker "$USER"   # log out and back in afterwards
docker run --rm hello-world
```

## API keys and secrets

Hosted-model providers and VT ARC need API keys. Local Ollama runs do not.

1. Copy the template: `cp .env.example .env`.
2. Fill in only the keys you use. Inspect AI reads `.env` automatically.
3. Check `git status`. `.env` must not appear; it is git-ignored.

Never commit, paste, or log a key. A secret scan (gitleaks) runs on every pull
request and fails the check if a commit contains a key-like string. If a real key
is ever committed, rotate it immediately; removing it from the PR is not enough.

## Viewing results

Eval logs are written to `logs/` (git-ignored). Browse them with:

```bash
inspect view
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `inspect: command not found` | Virtual environment is not active | Run `source .venv/bin/activate` |
| `ModuleNotFoundError: No module named 'inspect_ai'` | Dependencies not installed in this environment | Activate `.venv`, then `pip install -r requirements.txt` |
| `No matching distribution found for inspect-ai` (often preceded by "Ignored the following versions that require a different Python version") | Python older than 3.10 | Install a newer Python and recreate `.venv` |
| `ModuleNotFoundError: No module named 'openai'` when using `ollama/...` | Inspect's Ollama provider uses the OpenAI client | `pip install -r requirements.txt` (it is pinned there) |
| `ConnectError` or `Connection refused` for `ollama/...` | Ollama server is not running | Start it with `ollama serve` (or the Ollama app) and retry |
| `model "llama3.1" not found` | Model not pulled | `ollama pull llama3.1` |
| `permission denied ... /var/run/docker.sock` | User not in `docker` group | See step 6, then log out and back in |
| `APIConnectionError: Connection error.` for `vllm/...` | `VLLM_BASE_URL` is wrong, or you are not on the VT VPN | Connect to the VPN (step 5.5), check the URL in `.env`, then retry |
| `401` or `Incorrect API key` for `vllm/...` | `VLLM_API_KEY` is missing or wrong | Generate a key (step 5.4) and set it in `.env` |
| `429` for `vllm/...` | ARC allows 20 requests in flight per user across all models | Lower concurrency (for example `--max-connections 10`) and retry |
| `error: externally-managed-environment` | Installing outside a virtual environment | Create and activate `.venv` first |
