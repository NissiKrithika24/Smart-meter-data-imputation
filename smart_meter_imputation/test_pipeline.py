"""Quick end-to-end test: data loading + 2 epochs of LSTM training."""
import sys
import time
sys.stdout.reconfigure(line_buffering=True)

import config
from data_loader import prepare_data
from train import create_model, train_model
from utils import set_seed

set_seed()

t0 = time.time()
df, scalers, train_w, val_w, test_w = prepare_data()
print(f"Data prepared in {time.time()-t0:.1f}s")
print(f"Train: {len(train_w)}, Val: {len(val_w)}, Test: {len(test_w)}")

# Quick training test with 2 epochs
model = create_model("LSTM")
print(f"LSTM params: {sum(p.numel() for p in model.parameters()):,}")

t1 = time.time()
result = train_model(model, train_w, val_w, "MCAR", 0.2, num_epochs=2)
print(f"2-epoch training time: {time.time()-t1:.1f}s")
print(f"Train losses: {result['train_losses']}")
print(f"Val losses: {result['val_losses']}")
print("Pipeline test PASSED!")
