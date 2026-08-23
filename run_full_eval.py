import os, glob
from src.motion.benchmark_stages import full_pipeline
from src.motion.eval_metrics import evaluate_clip, compute_prf1

def run_all():
    clips = sorted(glob.glob('data/drishti/*.m*'))
    total_tp, total_fp, total_fn = 0, 0, 0
    results = []

    print("[+] Starting evaluation on 8 validation clips...", flush=True)

    for vid in clips:
        clip_name = os.path.basename(vid)
        # GT files use the prefix '01', '02', etc.
        gt_clip = clip_name.split('_')[0]
        
        preds = full_pipeline(vid, clip_name=clip_name, step=30)
        m = evaluate_clip(clip_name=gt_clip, pred_csv_or_json_path=preds)
        
        results.append(m)
        total_tp += m['tp']
        total_fp += m['fp']
        total_fn += m['fn']
        
        print(f"Clip {clip_name}: TP={m['tp']} FP={m['fp']} FN={m['fn']} F1={m['f1']:.4f}", flush=True)

    prf = compute_prf1(total_tp, total_fp, total_fn)
    print('\nOVERALL RESULTS:')
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {prf['precision']:.4f}")
    print(f"Recall: {prf['recall']:.4f}")
    print(f"F1 Score: {prf['f1']:.4f}")

if __name__ == '__main__':
    run_all()
