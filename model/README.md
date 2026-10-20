# `model/` — the custom Ollama model

[`Modelfile`](Modelfile) defines `mundial-analyst`, the model the app uses by default. It is built on `qwen3.5:9b` (about 5–6 GB of VRAM at 4-bit) and sets:

- **`temperature 0.2`**, so predictions for the same inputs stay consistent;
- **`repeat_penalty 1.1`**, to avoid repetitive analysis text;
- **a system prompt** that sets the analyst persona: reason from statistics, prioritise xG and the match stakes, and always end with the `PREDICTION:` block.

Build it once, from the repository root:

```powershell
ollama pull qwen3.5:9b
ollama create mundial-analyst -f model/Modelfile
```

Then check it answers with `python scripts/check_ollama.py`.

The Python code sends its own detailed system prompt and temperature with every request, and those take precedence. The Modelfile sets the defaults, for example when you talk to the model directly with `ollama run mundial-analyst`. To try another model, pass `--model` to any command that runs the model, e.g. `python -m mundial_llm.predict_match "Spain" "Brazil" --model llama3.1:8b`.
