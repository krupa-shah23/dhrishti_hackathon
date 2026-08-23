from src.motion.eval_metrics import match_events

def run_tests():
    # 1. Exact match
    pred = [{'start': 10, 'end': 20, 'seat_ids': ['seat_7']}]
    gt = [{'start': 12, 'end': 18, 'seat': 'seat_7'}]
    assert match_events(pred, gt)['tp'] == 1, 'Failed exact match'

    # 2. No temporal overlap
    pred = [{'start': 10, 'end': 20, 'seat_ids': ['seat_7']}]
    gt = [{'start': 22, 'end': 28, 'seat': 'seat_7'}]
    assert match_events(pred, gt)['tp'] == 0, 'Failed no temporal overlap'

    # 3. Insufficient overlap (threshold 0.5)
    pred = [{'start': 10, 'end': 20, 'seat_ids': ['seat_7']}]
    gt = [{'start': 18, 'end': 22, 'seat': 'seat_7'}]
    assert match_events(pred, gt)['tp'] == 0, 'Failed insufficient overlap'

    # 4. Wrong seat
    pred = [{'start': 12, 'end': 18, 'seat_ids': ['seat_8']}]
    gt = [{'start': 12, 'end': 18, 'seat': 'seat_7'}]
    assert match_events(pred, gt)['tp'] == 0, 'Failed wrong seat'

    # 5. Unknown seat (no seat in GT) -> matches if time overlaps
    pred = [{'start': 12, 'end': 18, 'seat_ids': ['seat_7']}]
    gt = [{'start': 12, 'end': 18}] 
    assert match_events(pred, gt)['tp'] == 1, 'Failed unknown seat'

    # 6. Duplicate prediction -> only 1 TP, 1 FP
    pred = [{'start': 12, 'end': 18, 'seat_ids': ['seat_7']}, {'start': 13, 'end': 19, 'seat_ids': ['seat_7']}]
    gt = [{'start': 12, 'end': 18, 'seat': 'seat_7'}]
    res = match_events(pred, gt)
    assert res['tp'] == 1 and res['fp'] == 1 and res['fn'] == 0, 'Failed duplicate prediction'
    
    # 7. Point-event GT
    pred = [{'start': 5, 'end': 10, 'seat_ids': ['seat_7']}]
    gt = [{'start': 6, 'end': 6, 'seat': 'seat_7'}]
    res = match_events(pred, gt)
    assert res['tp'] == 1, 'Failed point-event GT'

    print('All synthetic fixtures passed!')

if __name__ == '__main__':
    run_tests()
