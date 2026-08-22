# Dataset spot-check notes

**Date:** 2026-08-22
**Dataset checked:** `data/roboflow_export3/{train,valid,test}` — confirmed as the live path from `phone_data.yaml` (do not assume `roboflow_export2`; that folder exists too but is a different, non-identical export — `phone_data.yaml` points at `export3`).
**Image counts:** train=702, valid=99, test=30 → **831 total images**, each with a corresponding YOLO `.txt` label file (empty file = 0 boxes, a legitimate negative).

## Scope flag — read this first

The task framing for this check assumed a 2-class dataset (phone, paper-chit) with calculator dropped. **That's not what's live.** `phone_data.yaml` currently defines:

```
nc: 3
names: ['paper-chit', 'phone', 'calculator']
```

and **9 of the 25 randomly sampled images (36%) carry a `class=2` (calculator) box.** This isn't a labeling defect — the calculator boxes look visually consistent with a real small handheld object in every sampled case — it's a mismatch between the assumed project scope and the actual dataset config. Flagging it rather than silently working around it. For the rest of this check, calculator-class boxes were left unevaluated (neither pass nor fail) and only phone(1)/paper-chit(0) labels were judged against what's visible, per the task's actual verification rubric.

## Sample

25 images, sampled with `random.seed(42)` over the combined list of `(split, filename)` across all three splits (sorted, then `random.sample`) — reproducible by re-running the same seed against the same file listing. Landed as 22 train + 3 valid + 0 test (unweighted random draw; not stratified by split).

Sampled filenames:

**train (22):**
```
01-Candidate_was_found_using_a_mobile_phone_in_the_examination_hall-_f0000394_t00015-8s_jpg.rf.5435caae8cf911dcd9eac2eb2b96f052.jpg
01-Candidate_was_found_using_a_mobile_phone_in_the_examination_hall-_f0000418_t00016-7s_jpg.rf.bde413e6c2c18b89fbd7f8b16c96e09e.jpg
01-Candidate_was_found_using_a_mobile_phone_in_the_examination_hall-_f0001212_t00048-5s_jpg.rf.41680860b0dee8c1883eac4378fbeeb1.jpg
01-Candidate_was_found_using_a_mobile_phone_in_the_examination_hall-_f0001236_t00049-4s_jpg.rf.40504dfb0a6badf1d1b33a9d5e2f8f0e.jpg
03-CCTV_Mobile_Usage_f0000791_t00065-5s_jpg.rf.37adeaa66fa150762894b76b74a312e6.jpg
03-CCTV_Mobile_Usage_f0000827_t00068-4s_jpg.rf.1e13ee0acc70c1bde94998529b3ea99d.jpg
03-CCTV_Mobile_Usage_f0000990_t00081-9s_jpg.rf.fea01f66f5bece2acf57201d5538b51c.jpg
03-CCTV_Mobile_Usage_f0001050_t00086-9s_jpg.rf.2d3471d0619b11772312a51b34a812de.jpg
03-CCTV_Mobile_Usage_f0001236_t00102-3s_jpg.rf.1c7cc58368ad26559b3d5b21f5d9dd75.jpg
04-CCTV_Candidate_Talking_f0000944_t00118-0s_jpg.rf.a0eab44a220c70db88e96e1c22096ff5.jpg
04-CCTV_Candidate_Talking_f0001024_t00128-0s_jpg.rf.f7ea9d76dded5e4ba534d54031bf0c32.jpg
05-Crowd_observed_near_the_reception_and_verification_desk-_f0000250_t00010-0s_jpg.rf.8a420203e33b1b45d1987ca0ea5578f4.jpg
05-Crowd_observed_near_the_reception_and_verification_desk-_f0001200_t00048-0s_jpg.rf.584e8aac80548f1386daf99ede880399.jpg
05-Crowd_observed_near_the_reception_and_verification_desk-_f0003500_t00140-0s_jpg.rf.fa6e50425374354a7a144ec6b93c9ab3.jpg
Seat_No-_12_was_seen_taking_a_piece_of_paper_from_the_desk_f0001307_t00052-3s_jpg.rf.c2e607d4c6700d312400793e3d0766df.jpg
clip03_FP_0-7s_097_png.rf.f3a4ae204053eb7feb758f71287dec74.jpg
cluster_0480_jpg.rf.f1d990aafad3d6885a0625064bd04072.jpg
cluster_0661_jpg.rf.ddfde66097d0cdce2e9051fab9dfa29b.jpg
ev4_0411_jpg.rf.e3fbb5087ae46a139bfed80cd9d1b786.jpg
ev5_0301_jpg.rf.c8c8cd43c66f4968c0592a9aec0a0176.jpg
ev6_1890_jpg.rf.4db5167dab4ca8b07c8768ecc76a8544.jpg
ev8_0517_jpg.rf.fad695467bbe399ed3c96ad12dd48181.jpg
```

**valid (3):**
```
05-Crowd_observed_near_the_reception_and_verification_desk-_f0003300_t00132-0s_jpg.rf.a6a16e1629973d86a671901a07418186.jpg
05-Crowd_observed_near_the_reception_and_verification_desk-_f0003900_t00156-0s_jpg.rf.b960cd127dfda23a380575c0667004a7.jpg
05-Crowd_observed_near_the_reception_and_verification_desk-_f0004250_t00170-0s_jpg.rf.8c474a7f3867803a9b49bfe6b164b204.jpg
```

## Result: label rule confirmed followed across the sample — no phone/paper-chit discrepancies found

All 25 images were opened and visually cross-referenced against their label file content (not just file-existence):

- 9/25 had a `calculator` box — visually consistent with a small handheld object at the annotated location in every case; not evaluated further per the scope note above.
- 1/25 had a `phone` box (`Seat_No-_12..._f0001307_t00052-3s`, inside the documented 26–82s paper-copying window for that clip) — zoomed into the exact box region and confirmed it bounds a distinct dark handheld object, separate from the paper visible on the desk. Plausible phone label; the paper itself isn't separately boxed as paper-chit, which is a defensible call (ordinary desk paper ≠ a suspicious chit), not a bug.
- 15/25 had empty label files (0 boxes) — in every one of these, no phone or paper-chit was clearly visible in the frame at the time checked (mostly wide-shot crowd/talking/negative scenes, and a couple of "before the object comes out" frames in clips that later show one). This matches the documented "legitimate Null-tagged negative" pattern, not skipped annotation.
- **0 discrepancies** where a visible phone/paper-chit had no box, or a box existed over empty space.

## Domain-matched frame count vs. 150–300 target

The P2 doc's original target (`scripts/extract_training_frames.py`'s own docstring) was **150–300 domain-matched frames**, with `scripts/split_dataset.py` designed to optionally blend in a separate Kaggle/base pool (kept out of the test split by design).

Findings:
- No Kaggle/external dataset directory exists anywhere in the repo (checked for `*kaggle*`, `*external*`, `*public*` under `data/`).
- `data/phone_dataset` — the only place `split_dataset.py` would have materialized a domain+Kaggle blend — **does not exist**. That pipeline was apparently never used to build the live dataset.
- `data/frames_raw/` contains 34 folders, all named after real clips (`clip01_ev1_takeout`, `clip03_ev4_photo_copy_burst`, `clip06_ev9_photo_5022`, etc.) — unambiguously domain-matched.
- Filename buckets across `roboflow_export3` (831 images): `03-CCTV_Mobile_Usage` (138), `ev*` (134), `05-Crowd_observed...` (121), `Seat_No-_12...` (119), `During_the_exam...` (75), `04-CCTV_Candidate_Talking` (72), `cluster_*` (72), `01-Candidate_was_found...` (43), `clip*` (32), `02-Candidate_was_found...` (25). The first group and the numbered-clip groups map directly to known exam-hall clips.
- **Caveat, stated explicitly rather than guessed:** the `ev*` (134) and `cluster_*` (72) groups — 206 images, ~25% of the dataset — don't correspond 1:1 to any named folder in `data/frames_raw`, so I can't point to a documented extraction record for them specifically. I sampled 8 of these 206 across all three splits and every one carries a genuine CCTV timestamp/location watermark (`LUCKNOW1` or `Mumbai04` — real site names, not stock-photo artifacts), matching the same camera systems as the confirmed domain clips. Based on that evidence, my assessment is these are also domain-matched footage, just extracted via a different/later process than `extract_training_frames.py` — but this is an evidence-based inference from a partial sample (8/206), not a confirmed lineage record, and not an exhaustive check of all 206.

**Bottom line:** best assessment is all 831 images are domain-matched (no evidence of any base/public dataset mixed in), which clears the 150–300 target by roughly 2.8–5.5x. If a documented split between domain and base-dataset frames exists somewhere I didn't find, that would change this conclusion — treat the 206-image caveat above as the one open thread here.
