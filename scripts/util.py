"""scripts.util

Helper utilities for dataset bookkeeping and behavioral measures used in the
project. Centralizes paths, loads commonly used CSVs, and exposes helpers to
read per-subject trial CSVs and compute behavioral metrics (consumption
score, devaluation ratio, block/run rates, contingencies, etc.).

Key helpers
- `read_trial_info(subid, usecols=None)`: read cached per-subject trial CSV
- `get_consumption_score`, `get_devaluation_ratio`, `estimate_devaluation_ratio`
- `get_block`, `get_block_rate`, `get_run_rate`, `get_mean_rate`
- `get_contingencies`, `get_task_ordering`, `get_devalued_coin`,
  `get_devalued_direction`

Usage
-----
Import functions in analysis scripts::

    from scripts.util import get_devaluation_ratio, get_block_rate

Run as a script to recompute slopes or other aggregated outputs.
"""

# Importing libraries
import os
import sys
sys.path.append("..")
import numpy  as np
import pandas as pd
from   typing import Optional, Tuple, List
import statsmodels.api as sm
# Importing custom libraries
from   scripts.subjects import Subject

# Defining directories and loading data
# Use the location of this file to compute the project root so that
# imports or executions from other working directories still resolve
# the data paths correctly.
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIRECTORY = os.path.abspath(os.path.join(THIS_FILE_DIR, ".."))
RESULTS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "results")
FIGURES_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "figures")
DATA_DIRECTORY    = os.path.join(PROJECT_DIRECTORY, "data")
# CSVs and trial-info directories
CSVS_DIRECTORY          = os.path.join(DATA_DIRECTORY, "csvs")        # CSVs directory
TRIAL_INFO_DIRECTORY    = os.path.join(DATA_DIRECTORY, "trial_info")  # Trial info directory
TRIAL_INFO_CACHE        = {}
# Loading behavioral data
BEHAVIOR                = pd.read_csv(os.path.join(CSVS_DIRECTORY, "behavior.csv"))               # Behavioral data
RATE                    = pd.read_csv(os.path.join(CSVS_DIRECTORY, "rate.csv"))                   # Response Rate data
SUBJECTS                = sorted(BEHAVIOR["subID"].unique().tolist())                             # All subject IDs (n = 199)
HEALTHY_SUBJECTS        = sorted(BEHAVIOR[BEHAVIOR["patient"] == 0]["subID"].unique().tolist())   # Healthy controls (n = 144)
PATIENTS                = sorted(BEHAVIOR[BEHAVIOR["patient"] == 1]["subID"].unique().tolist())   # Patients (n = 55)
# Defining columns
TASK_COLUMNS         = ['consumption_score', 'devaluation_ratio',
                        'rate', 'slope_run_1', 'slope_run_2', 
                        'task-ordering', 
                        'devalued-coin', 'devalued-direction', 
                        'coin-contingency', 'stim-contingency',
                        'slowMouseNoBug', 'slowMouseCodeBug', 'fastMouseCodeBug', 'mouseBug']
PSYCHOMETRIC_COLUMNS = ['OCI-R_Total', 
                        'COHS_Routine', 'COHS_Automaticity', 
                        'SRS_Total', 'OLIFE_Total', 'STAI_Total',
                        'LSPS_Total', 'AUDIT_Total', 'EAT_Total', 'PDSS_Total', 'PSWQ_Total',
                        'BDI_Total', 'Apath_Total', 'BISBAS-BIS_Total', 'BIS_Total']
DEMOGRAPHIC_COLUMNS  = ['subID', 'age', 'sex', 'hand', 'patient']
ALL_COLUMNS          = DEMOGRAPHIC_COLUMNS + TASK_COLUMNS + PSYCHOMETRIC_COLUMNS


def get_subids(subjects: List["Subject"]) -> List[int]:
    """Return numeric subject IDs from a sequence of `Subject` objects.

    Filters out objects missing the `subid` attribute.
    """
    return [int(sub.subid) for sub in subjects if getattr(sub, 'subid', None) is not None]


def read_trial_info(subid: int, usecols: Optional[list] = None) -> pd.DataFrame:
    """Read trial CSV for a subject.

    Expects files named like ``R01_0002-trial_info.csv`` in
    the `TRIAL_INFO_DIRECTORY`. Raises ``FileNotFoundError`` with a
    helpful message if the file is missing.
    """
    filepath = os.path.join(TRIAL_INFO_DIRECTORY, f'R01_{subid:04d}-trial_info.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trial info not found for subject {subid}: {filepath}")
    # Use simple in-memory cache to avoid repeated disk reads
    if subid not in TRIAL_INFO_CACHE:
        TRIAL_INFO_CACHE[subid] = pd.read_csv(filepath)
    df = TRIAL_INFO_CACHE[subid]
    if usecols is None:
        return df.copy()
    return df.loc[:, usecols].copy()


def get_response_direction_mapping(subid: int) -> dict:
    """Return mapping {'resp1': resp_value, 'resp2': resp_value} for subject.

    If mapping cannot be determined, returns an empty dict.
    """
    try:
        df = read_trial_info(subid, usecols=['resp', 'condition', 'wasRnf'])
    except FileNotFoundError:
        return {}
    df  = df[df['wasRnf'] == 1]
    out = {}
    for name in ('resp1', 'resp2'):
        vals = df.loc[df['condition'] == name, 'resp']
        if not vals.empty:
            out[name] = vals.iloc[0]
    return out


def get_consumption_score(subid: int) -> float:
    """Calculates the consumption score for a given subject ID."""
    df = read_trial_info(subid)
    df = df[(df['run'] == 1) & (df['phase'] == 'consumption')]

    if df.empty:
        raise ValueError(f"No consumption-phase trials found for subject {subid} in run 1.")

    devalued = df['devalued_coin'].iat[0]
    valid    = df['selected_coin'].isin(['silver', 'gold']) & (df['selected_coin'] != devalued)
    return float(int(valid.sum()))


def get_block(subid: int, runid: int, blockid: int) -> pd.DataFrame:
    """Return the DataFrame slice corresponding to a training block.

    The function is tolerant to run identifiers that are 0-based or 1-based
    in the CSV: it will prefer an exact match for ``runid`` and fall back to
    ``runid - 1`` if necessary.
    """
    df   = read_trial_info(subid)
    runs = set(df['run'].unique())
    if runid in runs:
        run_mask = df['run'] == runid
    elif (runid - 1) in runs:
        run_mask = df['run'] == (runid - 1)
    else:
        raise ValueError(f"Run {runid} not present for subject {subid}")

    df_run = df[run_mask].reset_index()
    events = df_run[df_run['event'].isin(['training_period_start', 'ITI_start'])].reset_index()

    if len(events) % 2 != 0:
        raise ValueError("Uneven number of training block events — check the data.")
    n_blocks = len(events) // 2
    if not (1 <= blockid <= n_blocks):
        raise ValueError(f"blockid {blockid} is out of range (1..{n_blocks}).")

    start_idx = events.loc[(blockid - 1) * 2, 'index']
    end_idx   = events.loc[(blockid - 1) * 2 + 1, 'index']
    return df_run[(df_run['index'] > start_idx - 1) & (df_run['index'] < end_idx + 1)].reset_index(drop=True)


def get_block_rate(subid: int, runid: int, blockid: int) -> float:
    """Calculates the rate of correct responses for a specific block."""
    df       = get_block(subid, runid, blockid)
    correct  = int((df['corrResp'] == df['resp']).sum())
    duration = float(df['globalClock_t'].iat[-1] - df['globalClock_t'].iat[0])
    if duration <= 0:
        raise ValueError("Non-positive duration for block — cannot compute rate.")
    return correct / duration


def get_block_reward(subid: int, runid: int, blockid: int) -> int:
    """Calculates the total reward for a specific block."""
    return get_block(subid, runid, blockid)['wasRnf'].sum()


def get_run_rate(subid: int, runid: int) -> float:
    """Calculates the rate of correct swiping responses for a specific run."""
    # avoid loading all blocks individually if many are missing — iterate safely
    correct = 0
    time    = 0.0
    for b in range(1, 11):
        try:
            dfb = get_block(subid, runid, b)
        except ValueError:
            continue
        correct += int((dfb['corrResp'] == dfb['resp']).sum())
        time += float(dfb['globalClock_t'].iat[-1] - dfb['globalClock_t'].iat[0])
    if time <= 0:
        raise ValueError("Non-positive total duration for run — cannot compute rate.")
    return correct / time


def get_mean_rate(subid: int) -> float:
    """Calculates the mean rate of correct swiping responses across all runs."""
    correct = 0
    time    = 0.0
    for r in (1, 2):
        for b in range(1, 11):
            try:
                dfb = get_block(subid, r, b)
            except ValueError:
                continue
            correct += int((dfb['corrResp'] == dfb['resp']).sum())
            time += float(dfb['globalClock_t'].iat[-1] - dfb['globalClock_t'].iat[0])
    if time <= 0:
        raise ValueError("Non-positive total duration across runs — cannot compute mean rate.")
    return correct / time


def estimate_devaluation_ratio(subid: int) -> float:
    """Estimate devaluation ratio from the subject's choice responses.

    Falls back gracefully if expected columns are missing.
    """
    df         = read_trial_info(subid)
    train_rows = df[df['phase'] == 'training']
    if train_rows.empty:
        raise ValueError(f"No training rows for subject {subid}")
    first = train_rows.iloc[0]

    corr  = first.get('corrResp', '')
    if isinstance(corr, str) and 'left' in corr:
        silver_response = 'left'
        gold_response = 'right'
    else:
        silver_response = 'right'
        gold_response = 'left'

    correct = gold_response if first.get('devalued_coin') == 'silver' else silver_response
    choices = df.loc[df['event'] == 'choice_resp', 'resp']
    if choices.empty:
        return 0.0
    return float(choices.eq(correct).sum()) / float(len(choices))


def get_devaluation_ratio(subid: int, filepath: Optional[str] = None) -> float:
    """Retrieves the devaluation ratio for a given subject ID, optionally from a specified file."""
    if filepath is None:
        filepath = os.path.join(DATA_DIRECTORY, 'csvs', 'choice_test_devaluation_ratio.csv')
    try:
        return pd.read_csv(filepath, index_col=0).loc[subid, 'devaluation_ratio']
    except Exception:
        return estimate_devaluation_ratio(subid)


def get_contingencies(subid: int) -> Tuple[int, int]:
    """Calculates the number of correct responses for coin and stim contingencies."""
    df   = read_trial_info(subid, usecols=['event', 'corrResp', 'resp'])
    df   = df[df['event'] == 'contingency_resp']
    # coin: first two contingency responses; stim: subsequent responses
    coin = int(sum((df.iloc[i]['corrResp'] == df.iloc[i]['resp']) for i in range(min(2, len(df)))))
    stim = int(sum((df.iloc[i]['corrResp'] == df.iloc[i]['resp']) for i in range(2, len(df))))
    return coin, stim


def get_task_ordering(subid: int) -> int:
    """Retrieves the task ordering for a given subject ID."""
    val = read_trial_info(subid, usecols=['task_ordering'])['task_ordering'].iat[0]
    return int(val) - 1


def get_devalued_coin(subid: int) -> int:
    """Determines if the devalued coin is gold for a given subject ID."""
    val = read_trial_info(subid, usecols=['devalued_coin'])['devalued_coin'].iat[0]
    return int(str(val).lower() == 'gold')


def get_devalued_direction(subid: int) -> int:
    """Determines the response direction for the devalued coin."""
    df   = read_trial_info(subid, usecols=['coin_img', 'devalued_coin', 'corrResp'])
    img  = f'stim/{df["devalued_coin"].iat[0]}_coin.png'
    vals = df.loc[df['coin_img'] == img, 'corrResp']
    if vals.empty:
        raise ValueError(f"Could not find coin image row for subject {subid}")
    return int(vals.iat[0] == 'left')

def get_rate_slope(subid: int, runid: int) -> float:
    """Calculates the slope of correct response rates across blocks in a run."""
    y        = np.asarray([get_block_rate(subid, runid, b) for b in range(1, 11)])
    X        = sm.add_constant(np.arange(1, 11))
    fit      = sm.OLS(y, X).fit()
    _, slope = fit.params
    return slope
    
def main():
    # Validating all the functions
    random_subid = SUBJECTS[0]
    print(f"Consumption score for subject {random_subid}: {get_consumption_score(random_subid)}")
    print(f"Devaluation ratio for subject {random_subid}: {get_devaluation_ratio(random_subid)}")
    print(f"Block 1 rate for subject {random_subid}, run 1: {get_block_rate(random_subid, 1, 1)}")
    print(f"Run 1 rate for subject {random_subid}: {get_run_rate(random_subid, 1)}")
    print(f"Mean rate for subject {random_subid}: {get_mean_rate(random_subid)}")
    coin_contingency, stim_contingency = get_contingencies(random_subid)
    print(f"Contingencies for subject {random_subid}: coin={coin_contingency}, stim={stim_contingency}")
    print(f"Task ordering for subject {random_subid}: {get_task_ordering(random_subid)}")
    print(f"Devalued coin for subject {random_subid}: {get_devalued_coin(random_subid)}")
    print(f"Devalued direction for subject {random_subid}: {get_devalued_direction(random_subid)}")
    print(f"Rate slope for subject {random_subid}, run 1: {get_rate_slope(random_subid, 1)}")

if __name__ == "__main__":
    main()