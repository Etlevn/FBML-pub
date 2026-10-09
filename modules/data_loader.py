import os
from typing import Dict, List, Tuple
import pandas as pd
from tqdm.auto import tqdm


def load_sources(global_path: str, files: List[Tuple]) -> pd.DataFrame:
    """
    Read and merge multiple sources.
    files: [(filename, ftype, rename_map[, drop_cols])]
      - ftype in {'stata','sas','csv','parquet'}
      - rename_map: e.g., {'public_date':'mdate'}
    Merge keys: ['permno','mdate']
    """
    print(f"[DATA] Loading from {global_path} ...")
    df = None
    pbar = tqdm(files, desc="Files", leave=True)
    for entry in pbar:
        filename, ftype, rename = entry[:3]
        pbar.set_postfix_str(filename)
        path = os.path.join(global_path, filename)
        if ftype == 'stata':
            temp = pd.read_stata(path)
        elif ftype == 'sas':
            temp = pd.read_sas(path, format='sas7bdat', encoding='utf-8')
        elif ftype == 'csv':
            temp = pd.read_csv(path)
        elif ftype == 'parquet':
            temp = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {ftype}")

        if rename:
            temp = temp.rename(columns=rename)
        if 'mdate' not in temp.columns:
            raise ValueError(f"{filename} missing 'mdate' column; map it via rename to 'mdate'")
        temp['mdate'] = pd.to_datetime(temp['mdate']).dt.to_period('M').astype(str)

        # column drop disabled to match v1

        tqdm.write(f"\n[DATA] Loaded: {filename} (type={ftype}), shape={temp.shape}")

        df = temp if df is None else pd.merge(df, temp, on=['permno','mdate'], how='left')
    pbar.close()
    print(f"\n[DATA] Merging ...")
    return df


def get_all_dates(df: pd.DataFrame) -> List[str]:
    return sorted(df['mdate'].unique())


