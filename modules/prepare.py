import pandas as pd
from typing import Tuple, List
import random
import numpy as np
import torch
from tqdm.auto import tqdm


def split_train_test_by_window(df: pd.DataFrame, all_dates: List, i: int, T: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train=[i-T+1, i] and test=[i-T+2, i+1] by month indices."""
    train_start_idx = max(0, i - T + 1)
    train_end_idx = i
    test_start_idx = max(0, i - T + 2)
    test_end_idx = i + 1
    train_mask = (df['mdate'] >= all_dates[train_start_idx]) & (df['mdate'] <= all_dates[train_end_idx])
    test_mask = (df['mdate'] >= all_dates[test_start_idx]) & (df['mdate'] <= all_dates[test_end_idx])
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def set_seed(seed: int = 42) -> None:
    """Deterministic seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tqdm.write(f"\n[PREP] Seed set to {seed}")