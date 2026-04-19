# Veritas Image Pipeline — Master Project Specification

> **IMPORTANT: Read this file at the start of every session before taking any action.**
> This is the single source of truth for all pipeline decisions.
> Last updated: 2026-04-19

---

## 1. Project Context

**Project:** Innuir Nutrition — Southeast Asian food image dataset for LLM fine-tuning
**Target model:** Gemma 4 E4B (Efficient 4-Billion parameter, vision-fine-tuned variant)
**Vision encoder:** SigLIP (supports 896×896 with 2×2 tiling)
**Training hardware:** MacBook Pro M4 Pro, 48GB Unified Memory
**Database:** Project Veritas — ~4,285 food records (FD000031–FD004315), Southeast Asian focus

**Goal:** Produce a high-quality, multilingual, vision-annotated image dataset for fine-tuning Gemma 4 E4B to recognise and describe Southeast Asian food items with nutritional context.

---

## 2. Image Specifications

| Parameter | Value | Reason |
|---|---|---|
| Resolution | **896×896 px** | SigLIP 2×2 tiling; M4 Pro has headroom for high-res |
| Aspect ratio | **Preserve + pad to square** | Center-crop removes peripheral context (e.g. sambal, side dishes) |
| Background padding colour | **White (255, 255, 255)** | Neutral, consistent |
| Format | **JPEG, Quality 95** | Industry standard; negligible artifacts at Q95; manageable file size |
| Slots per food | **3: hero, macro, in_the_wild** | Visual diversity for training |

---

## 3. Dataset Structure

### 3.1 Directory Layout — FLAT
```
output/
  images/
    FD000031_hero.jpg
    FD000031_macro.jpg
    FD000031_in_the_wild.jpg
    FD000045_hero.jpg
    ...
  annotations.jsonl        ← one line per image
  manifest.jsonl           ← one line per food item (summary)
```

### 3.2 Annotation JSONL Schema (per image)
```json
{
  "image": "FD000031_hero.jpg",
  "food_id": "FD000031",
  "food_name": "Soy Milk with Grass Jelly",
  "food_type": "beverage",
  "slot": "hero",
  "captions": {
    "en": "A glass of soy milk with grass jelly (xiancao), a popular Southeast Asian cold beverage made from soy milk and grass jelly cubes. Rich in plant-based protein and low in calories.",
    "zh": "豆浆仙草冻，一杯由豆浆和仙草冻组成的东南亚冷饮，富含植物蛋白，低热量。",
    "ms": "Susu soya dengan cincau, minuman sejuk Asia Tenggara yang popular, diperbuat daripada susu soya dan kiub cincau. Kaya dengan protein berasaskan tumbuhan dan rendah kalori.",
    "ta": "சோயா பால் மற்றும் புல்-ஜெல்லி, சோயா பால் மற்றும் புல்-ஜெல்லி கியூப்களால் ஆன ஒரு பிரபலமான தென்கிழக்கு ஆசிய குளிர் பானம். தாவர புரதம் நிறைந்தது மற்றும் குறைந்த கலோரி கொண்டது."
  },
  "fm_score": 10,
  "sf_score": 9,
  "clip_score": 0.353,
  "publish": true
}
```

### 3.3 Manifest JSONL Schema (per food item)
```json
{
  "food_id": "FD000031",
  "food_name": "Soy Milk with Grass Jelly",
  "food_type": "beverage",
  "publish": 3,
  "avg_fm": 10.0,
  "avg_sf": 9.3,
  "avg_clip": 0.353,
  "slots_complete": ["hero", "macro", "in_the_wild"],
  "slots_missing": [],
  "elapsed_s": 118
}
```

---

## 4. Scoring & Quality Gates

### 4.1 Dual Scoring Pipeline
1. **Stage 1 — CLIP pre-filter** (HuggingFace `openai/clip-vit-base-patch32`)
   - Computes cosine similarity between image and food-specific text queries
   - CLIP scores range ~0.0–0.40 for real-world image-text pairs
   - Pass threshold: **≥ 0.30** (strong match; equivalent intent to 0.80 in normalised space)
   - Images below threshold are still passed to Gemini but ranked lower

2. **Stage 2 — Gemini FM/SF scoring** (`gemini-2.5-flash`)
   - **FM (Food Match):** Does the image show the correct food? Score 1–10
   - **SF (Shot Fit):** Does the image fit the slot requirements? Score 1–10
   - Model: `gemini-2.5-flash` (thinking model — do NOT set max_output_tokens)
   - Pass criteria: FM ≥ 7 AND SF ≥ 7 AND REAL_PHOTO=YES AND FOOD_MATCH=YES AND WATERMARK=NO

### 4.2 Slot Assignment
- Best candidate = highest combined score (passes gate → fm_score → clip_max)
- Fallback: if no candidate passes the gate, slot is left empty (not filled with low-quality image)
- Minimum publish threshold: **pub ≥ 1** (include partial sets in training data)

---

## 5. Caption Generation

- **Model:** `gemini-2.5-flash` (same API key)
- **Languages:** English (en), Simplified Chinese (zh), Malay (ms), Tamil (ta)
- **Style:** Detailed descriptive, 2–3 sentences
- **Content:** Food name, key ingredients, cuisine type, nutritional context
- **Generated at:** Post-processing step after images are selected and saved

---

## 6. API Keys & Services

| Service | Key | Usage |
|---|---|---|
| Google Gemini | `AIzaSyBRAXQvldywp5YeOwRjSCeDUG56WhT7w9o` | Vision scoring (FM/SF) + caption generation |
| SerpAPI | `0e1646bbfca613ede8085e960f8bfaf71519c04f3fb1fc93ff2dcc0d3596469d` | Google Images search |
| HuggingFace CLIP | Local model (no API key) | CLIP pre-filter scoring |

**Gemini model note:** Use `gemini-2.5-flash` (NOT `gemini-2.0-flash` — deprecated for new users). Do NOT set `max_output_tokens` — it is a thinking model and truncates responses if limited.

---

## 7. Pipeline Architecture

```
Input JSON (FD records)
    ↓
[1] Classifier → food_type (beverage / cooked_dish / raw_ingredient / packaged_product)
    ↓
[2] Query Builder → slot-specific search queries (hero / macro / in_the_wild)
    ↓
[3] SerpAPI Image Search → up to 15 URLs per query
    ↓
[4] Image Downloader → up to 10 candidates per attempt
    ↓
[4a] CLIP Pre-filter → rank by cosine similarity, pass ≥ 0.30 to Gemini
    ↓
[4b] Gemini FM/SF Scorer → score top 5 CLIP candidates
    ↓
[5] Slot Assigner → pick best, pad-to-square, resize 896×896, save JPEG Q95
    ↓
[6] Caption Generator → multilingual 2–3 sentence captions via Gemini
    ↓
Output: flat JPEGs + annotations.jsonl + manifest.jsonl
```

---

## 8. Execution Parameters

| Parameter | Value |
|---|---|
| Workers per batch | **5** (parallel) |
| Batch size | **50 records** (small batches — save results before proceeding) |
| Max attempts per slot | 6 |
| Scope — Phase 1 (smoke test) | First 200 records (4 batches of 50) |
| Scope — Phase 2 (full run) | All 3,775 remaining records in batches of 50 |
| Existing 800 old images | **Re-process** through new pipeline (consistency) |

### Operating Principles (in priority order)
1. **Quality first** — Never include a low-quality image. FM/SF ≥ 7, CLIP ≥ 0.30. Better to have an empty slot than a bad image.
2. **Speed** — 5 parallel workers per batch. Each batch of 50 should complete in ~30–45 minutes.
3. **Save before proceeding** — After every batch of 50:
   - Save images to Mac immediately
   - Update `manifest.jsonl` and `annotations.jsonl`
   - Commit and push to GitHub
   - Update `PROJECT_SPEC.md` with progress
   - Only then start the next batch

This ensures no work is ever lost to a sandbox reset.

---

## 9. File Locations

| Item | Path |
|---|---|
| Pipeline code | `/home/ubuntu/veritas_image_pipeline/pipeline/` |
| Input records | `/home/ubuntu/veritas_image_pipeline/input/` |
| Output images | `/home/ubuntu/veritas_image_pipeline/output/images/` |
| Annotations | `/home/ubuntu/veritas_image_pipeline/output/annotations.jsonl` |
| Manifest | `/home/ubuntu/veritas_image_pipeline/output/manifest.jsonl` |
| Checkpoints | `/home/ubuntu/veritas_image_pipeline/output/checkpoints/` |
| Mac source data | `/mnt/desktop/db_veritas1/` |
| Mac output target | `/mnt/desktop/db_veritas1/food_images_gemma4_training/` |
| GitHub repo | `https://github.com/nuvear/veritas_image_pipeline` |

---

## 10. Key Decisions Log

| Date | Decision | Reason |
|---|---|---|
| 2026-04-17 | Use SerpAPI instead of DuckDuckGo | More reliable, structured results |
| 2026-04-18 | Switch vision scorer from OpenAI to Gemini | OpenAI key invalid; Gemini API confirmed working |
| 2026-04-18 | Add CLIP pre-filter before Gemini | Reduces Gemini API calls; faster pipeline; better image-text alignment |
| 2026-04-18 | Remove max_output_tokens from Gemini calls | gemini-2.5-flash is a thinking model; token limit truncates responses |
| 2026-04-19 | Change from center-crop to pad-to-square | Preserve full dish geometry for LLM training |
| 2026-04-19 | Change from nested to flat directory layout | Better for dataloaders; easier to debug |
| 2026-04-19 | Set JPEG quality to 95 (was 92) | LLM training needs high fidelity |
| 2026-04-19 | Add multilingual JSONL captions (EN/ZH/MS/TA) | Singapore food culture is linguistically fluid |
| 2026-04-19 | Set CLIP threshold to 0.30 (not 0.80) | CLIP cosine similarity max ~0.40 in practice; 0.30 = strong match |
| 2026-04-19 | Set FM/SF minimum to 7 | Quality over quantity for LLM training |
| 2026-04-19 | Include pub ≥ 1 items | Partial sets still provide valuable training signal |
| 2026-04-19 | Start with 1,000-record smoke test | Validate training quality before full 3,775-record run |
| 2026-04-19 | Re-process old 800 images | Consistency in metadata and quality for stable fine-tuning |
| 2026-04-19 | Batch size reduced to 50 records | Save results after each batch; no work lost to sandbox resets |
| 2026-04-19 | Save to Mac + push to GitHub after every batch | Prevents data loss; enables recovery from any point |
| 2026-04-19 | Operating principle: Quality > Speed > Save-first | Quality is non-negotiable; speed is secondary; persistence is mandatory |

---

## 11. Session Recovery Checklist

After any sandbox reset, do the following IN ORDER:
1. **Read this file** (`PROJECT_SPEC.md`) — all decisions are here
2. **Check GitHub** (`git log --oneline -10`) — see what was last committed
3. **Check Mac output** (`ls /mnt/desktop/db_veritas1/food_images_gemma4_training/ | wc -l`) — count saved images
4. **Check manifest** (`wc -l output/manifest.jsonl`) — see how many items completed
5. **Resume from last checkpoint** — do NOT restart from scratch
6. **Do NOT ask the user questions already answered in this spec**

## 12. Progress Tracker

| Batch | FD Range | Records | Status | Images | Pushed to GitHub | Saved to Mac |
|---|---|---|---|---|---|---|
| B01 | FD000031–FD000080 | 50 | PENDING | — | — | — |
| B02 | FD000081–FD000130 | 50 | PENDING | — | — | — |
| B03 | FD000131–FD000180 | 50 | PENDING | — | — | — |
| B04 | FD000181–FD000230 | 50 | PENDING | — | — | — |
| ... | ... | ... | ... | ... | ... | ... |

*Update this table after each batch completes.*
