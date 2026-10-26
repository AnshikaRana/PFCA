# Running the big models on Kaggle (21 runs, 7 sessions)

These models need more GPU memory than a 6 GB RTX 3050: GritLM-7B, Qwen3-Embed-8B, LLAMA-7B,
Instructor-XL, ColPali and ColQwen2. Kaggle gives free **T4 x2** GPUs (2 x 15 GB), about 30 GPU hours a
week, at most 12 hours per session.

## One-time setup (about 10 minutes)
1. Sign in at kaggle.com and verify your phone number (Settings, needed for GPU + internet).
2. **Datasets → New Dataset**: upload `kaggle/pfca-kaggle-upload.zip` (code + the small task files
   the laptop already built for the DORIS-style candidate pools and IRPAPERS, so Kaggle scores on exactly the same pools). Name it `pfca-kit`.
3. **Code → New Notebook → File → Import Notebook**: upload `kaggle/PFCA_kaggle.ipynb`.
4. In the notebook: **Add Input** → your `pfca-kit` dataset. Settings: **Accelerator = GPU T4 x2**,
   **Internet = On**.

## Each session (A to G)
1. In the first code cell set `SESSION = "A"` (then B, C, ... G).
2. **Save Version → Save & Run All (Commit)**. It runs in the background (you can close the tab).
3. When the version finishes: **Output** tab → download `results_session_<X>.zip` and put it in
   `xaudit-kit/kaggle/returned/` on the laptop. Unzip it into `results/`; the website updates.

| Session | Runs | GPU time (rough) |
|---|---|---|
| A | ColPali + ColQwen2 (A3 IRPAPERS); LLAMA-7B (A2 on all 4); Instructor-XL (A1 LitSearch, MAPLE) | ~7 h |
| B | GritLM-7B: A1 IRPAPERS, LitSearch, MAPLE | ~7 h |
| C | Instructor-XL: A1 + A4 DORIS-MAE | ~5.5 h |
| D | Qwen3-Embed-8B: A4 IRPAPERS + LitSearch (512 tokens); Instructor-XL A4 LitSearch | ~7.5 h |
| E | GritLM-7B: A4 IRPAPERS + LitSearch (512 tokens) | ~6.5 h |
| F | Qwen3-Embed-8B A4 MAPLE (512 tokens); Instructor-XL A4 MAPLE | ~8.5 h |
| G | GritLM-7B: A4 MAPLE (512 tokens) | ~7.5 h |

About 50 GPU hours in total, so roughly two weeks of the free weekly quota.

## Notes
- 7-8B models load in float16 split across both T4s; ColPali/ColQwen2 run in float32 (T4 has no
  bfloat16). The settings are recorded in every result file.
- A4 big models use 512 input tokens (the MAPLE paper used each model's full maximum); this is
  recorded in each result and shown on the website.
- Not run (too large for free GPUs, ~17 GPU hours each): GritLM-7B on A1 x DORIS-MAE, and Qwen3-Embed-8B
  and GritLM-7B on A4 x DORIS-MAE (363K abstracts).
- If a model fails, its result file says why; the other models in the session still run.
