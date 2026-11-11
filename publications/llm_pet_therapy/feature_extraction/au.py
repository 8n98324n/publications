"""Presence proportions on qualified, physically timed OpenFace frames."""
import numpy as np
import pandas as pd

def from_openface(csv_path,clock_path,start_physical_s,end_physical_s,*,target_source_confirmed):
    if target_source_confirmed is not True:raise ValueError('Confirmed target source and response boundary required')
    frame=pd.read_csv(csv_path);frame.columns=frame.columns.str.strip()
    clock=pd.read_csv(clock_path);clock.columns=clock.columns.str.strip()
    if len(frame)!=len(clock):raise ValueError('OpenFace and physical frame clock differ')
    times=clock.source_physical_pts_s.to_numpy(float)
    if len(times)==0 or not np.isfinite(times).all() or not np.allclose(np.diff(times),1/15,rtol=0,atol=1.5e-6):raise ValueError('Expected contiguous physical 15 fps frames')
    if start_physical_s<times[0]-.067 or end_physical_s>times[-1]+1/15+.067:raise ValueError('Requested answer boundary is not covered')
    if not start_physical_s<end_physical_s:raise ValueError('Invalid answer boundary')
    selected=(times>=start_physical_s-1e-8)&(times<end_physical_s-1e-8)
    chosen=frame.loc[selected];cols={f'au{a}_prevalence':chosen[f'AU{a}_c'].tolist() for a in ('05','07','12')}
    if not chosen.success.isin([0,1]).all() or not chosen.confidence.between(0,1).all():raise ValueError('Invalid face tracking fields')
    return summarize_selected(cols,((chosen.success==1)&(chosen.confidence>=.8)).tolist())

def summarize_selected(presence_columns, good):
    """A small pure helper exposes denominator and exact 80% gate for checks."""
    n, valid_n = len(good), sum(bool(value) for value in good)
    if n == 0:
        return 'invalid_empty_window', 0, 0, None, {}
    if any(len(values) != n or any(value not in (0, 1) for value in values) for values in presence_columns.values()):
        raise ValueError('Presence values must be finite binary with a shared denominator')
    coverage = valid_n / n
    if valid_n * 5 < n * 4:
        return 'qc_fail', n, valid_n, coverage, {}
    values = {fid: sum(value for value, keep in zip(column, good) if keep) / valid_n
              for fid, column in presence_columns.items()}
    return 'ok', n, valid_n, coverage, values
