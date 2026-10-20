# `docs/` — design notes

[`design_notes_pt.md`](design_notes_pt.md) is the original design document I wrote before building the project, in Portuguese. In short, it set out:

- **The principle:** an LLM is neither a calculator nor a crystal ball. It shouldn't guess winners from memory; it should act as an analyst that receives cold, structured statistics and turns them into tactical analysis and probabilities.
- **The hardware limit:** an RTX 3060 Ti with 8 GB of VRAM, which means 7B or 8B models quantised to 4-bit, leaving about 3 GB for context.
- **The candidates:** Llama 3.1 8B, Qwen 2.5 7B and Mistral 7B were considered; the project ended up on Qwen 3.5 9B.
- **The data options:** API-Football, FBref via `soccerdata`, and StatsBomb Open Data, with xG as the key metric. FBref was frequently blocked, so StatsBomb became the single historical source, with ESPN added later for live standings.
- **The pipeline:** extract stats, format them as text, inject them into a prompt, run local inference with Ollama.

The [main README](../README.md) describes how the finished system works, including what was added beyond the original plan: the stakes simulator, the lesson memory, and the backtest.
