# Team setup guide

How to get a working development environment for the benchmark runner: Python
virtual environment, Inspect AI, Docker, and Ollama. When you finish, `inspect eval --help`
runs and the smoke test passes against a local model.

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

## 5. Install Docker

Follow <https://docs.docker.com/engine/install/> for your OS. On Linux, add
yourself to the `docker` group so you do not need `sudo`:

```bash
sudo usermod -aG docker "$USER"   # log out and back in afterwards
docker run --rm hello-world
```

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
| `permission denied ... /var/run/docker.sock` | User not in `docker` group | See step 5, then log out and back in |
| `error: externally-managed-environment` | Installing outside a virtual environment | Create and activate `.venv` first |
