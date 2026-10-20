# `scripts/` — environment checks

Standalone scripts to confirm your setup works before you run the app. They don't import the project code.

| Script | Checks | Run it with |
|--------|--------|-------------|
| [`check_ollama.py`](check_ollama.py) | Ollama is running and the model answers a question. | `python scripts/check_ollama.py` (add `--model qwen3.5:9b` to test another model) |
| [`check_data_sources.py`](check_data_sources.py) | StatsBomb Open Data is reachable (schedule, events with xG, line-ups). It also tries FBref and WhoScored if the optional `soccerdata` package is installed; those sources were explored early on and are not used by the app. | `python scripts/check_data_sources.py` |
