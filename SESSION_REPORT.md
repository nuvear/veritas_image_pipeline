# Veritas Image Pipeline — Session Report
**Date:** April 19, 2026  
**Status:** Pipeline Rebuilt, Tested, and Ready for Full Run  
**Target LLM:** Gemma 4 E4B (Vision Fine-Tuning)

---

## 1. Executive Summary

This session focused on rebuilding the Veritas Food Image Pipeline after a sandbox reset and re-architecting it specifically to generate high-quality image-text pairs for fine-tuning the **Gemma 4 E4B** vision-language model.

The pipeline now successfully executes a dual-scoring filtration system (HuggingFace CLIP + Gemini 2.5 Flash) to guarantee strict semantic alignment between food records and retrieved images. A 5-item test batch was successfully processed, generating 15 production-quality images across three visual slots (hero, macro, in-the-wild) alongside multilingual JSONL annotations.

All codebase changes, architectural decisions, and operating parameters have been permanently committed to the GitHub repository (`nuvear/veritas_image_pipeline`) to ensure immediate recovery from any future sandbox resets.

---

## 2. Architectural Upgrades for Gemma 4 E4B

To meet the strict data requirements of vision LLM fine-tuning, the following architectural changes were implemented:

### Image Specifications
- **Resolution:** 896×896 px (allows 2×2 or 4×4 tiling for Gemma's 224/448px native patches).
- **Aspect Ratio:** Original aspect ratio preserved and padded to square (black padding). Center-cropping was disabled to prevent loss of edge context.
- **Format:** JPEG at Quality 95 to balance artifact-free training with storage efficiency.
- **Directory Layout:** Flat directory structure (`FD000031_hero.jpg`) to simplify dataloader ingestion.

### Dual-Scoring Filtration
1. **CLIP Pre-Filter (Semantic Alignment):** 
   - Uses HuggingFace `openai/clip-vit-base-patch32` running locally.
   - Calculates cosine similarity between the image candidate and the query text.
   - **Threshold:** Set to **0.30** (a strict threshold for real-world image-text pairs, ensuring only highly relevant images pass to the LLM scorer).
2. **Gemini Vision Scorer (Domain Accuracy):**
   - Uses `gemini-2.5-flash` to evaluate the food match (FM) and slot fit (SF) on a 1–10 scale.
   - **Threshold:** Both FM and SF must score **≥ 7** for an image to be accepted.

### Multilingual Annotations
- A new `caption_generator.py` module was added to produce detailed, 2–3 sentence captions describing the food, its ingredients, and its cultural context.
- Captions are generated in English, Simplified Chinese, Malay, and Tamil (matching the Veritas database schema).
- Outputs are saved directly to an `annotations.jsonl` file formatted for standard HuggingFace/PyTorch dataloaders.

---

## 3. Workflow & Operating Principles

To maximize reliability and prevent data loss during long runs, the orchestrator was rewritten to enforce the following principles:

1. **Quality First:** No image is saved unless it passes both the CLIP 0.30 gate and the Gemini FM/SF ≥ 7 gate.
2. **Batch Processing:** Records are processed in strict batches of 50.
3. **Parallel Execution:** 5 parallel worker processes run concurrently to maximize throughput without hitting API rate limits.
4. **Save-First Progression:** After every batch of 50 completes, the orchestrator:
   - Updates the master `manifest.jsonl` and `annotations.jsonl`.
   - Rsyncs the images to the mounted Mac desktop (`/mnt/desktop/db_veritas1/food_images_gemma4_training`).
   - Commits and pushes the progress to GitHub.
   - Only then does it proceed to the next batch.

---

## 4. Test Batch Results

A 5-item test batch (FD000031, FD000045, FD000080, FD000120, FD000200) was executed to validate the new architecture.

| FD ID | Food Name | Slots Filled | Avg FM | Avg SF | Avg CLIP |
|---|---|---|---|---|---|
| FD000031 | Soy Milk with Grass Jelly | 3/3 | 10.0 | 9.0 | 0.351 |
| FD000045 | Char Kway Teow | 3/3 | 9.7 | 9.7 | 0.329 |
| FD000080 | Nasi Lemak | 3/3 | 9.7 | 9.3 | 0.357 |
| FD000120 | Roti Prata | 3/3 | 9.0 | 9.3 | 0.327 |
| FD000200 | Pearl Milk Tea (Less Sugar) | 3/3 | 9.3 | 9.7 | 0.327 |

**Total output:** 15 images generated, 100% success rate. 

*Observation:* The pipeline performed exceptionally well on complex plated dishes (Nasi Lemak, Char Kway Teow). For beverages (Soy Milk), the `hero` and `macro` slots returned visually similar images. Future iterations may require query adjustments to force distinct macro perspectives for drinks.

---

## 5. Current State & Next Steps

### Current State
- The codebase is fully operational and pushed to `https://github.com/nuvear/veritas_image_pipeline`.
- API keys have been secured in a local `.env` file.
- The pipeline is ready to process the full dataset.

### Next Steps
1. **Full 1,000-Record Smoke Test:** Execute the orchestrator for the first 1,000 records to generate ~3,000 images.
2. **Review Annotation Quality:** Inspect the generated `annotations.jsonl` to ensure the multilingual captions meet the required depth for Gemma 4 E4B fine-tuning.
3. **Full Run (3,775 Records):** Complete the remaining records.
4. **Mac Transfer:** Ensure the background rsync successfully pushes all batches to the local Mac drive.

---
*Cross-Reference: See `PROJECT_SPEC.md` in the GitHub repository for the complete technical specification, API configurations, and disaster recovery checklist.*
