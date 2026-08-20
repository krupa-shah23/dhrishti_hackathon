import psutil
import os
import time
from train_risk_model import train, risk_score
from extract_features import extract_features
from mock_track_data import generate_mock_tracks

process = psutil.Process(os.getpid())

mem_before = process.memory_info().rss / (1024 ** 2)  # MB
t0 = time.time()

train()

mem_after = process.memory_info().rss / (1024 ** 2)
print(f"\nTraining time: {time.time()-t0:.2f}s")
print(f"RAM used by training: {mem_after - mem_before:.2f} MB")
print(f"Peak process RAM: {mem_after:.2f} MB")