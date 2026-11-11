"""RR correction/qualification and exact-name blood-pressure mapping."""
import math,unicodedata
import numpy as np
from .rr_trace import trace_correction,RATE,_metrics

def rr_features(rr_seconds,*,source_verified,unresolved_structure_reason):
    """Explicit source and structural assessment accompany each selected stage.

    unresolved_structure_reason is a documented string (empty only when the
    structural assessment has no unresolved issue), not a Time-column veto.
    Missing whole stages are never reconstructed.
    """
    if type(source_verified) is not bool or not isinstance(unresolved_structure_reason,str):raise ValueError('Explicit stage-source and interval-structure assessment required')
    features=('mean_hr','rmssd','sdnn');minimum=dict(mean_hr=10,rmssd=30,sdnn=60)
    variants=[('full_response','primary_quality',features),('full_response','strict_c5',features),('full_response','hrnormal_c10',('mean_hr',)),('first60s','primary_quality',features)]
    if len(rr_seconds)==0:
        return {'trace':None,'features':[dict(feature_id=f,window_variant=w,quality_variant=q,usable=False,value=None,reason='empty_stage') for w,q,ff in variants for f in ff]}
    trace=trace_correction(rr_seconds);a=np.asarray(trace['original_ticks']);b=np.asarray(trace['corrected_ticks'])
    support=bool(trace['endpoints_preserved'] and trace['span_preserved'] and trace['wrapper_matches_public_implementation'])
    for event in trace['events']:
        if not event['actual_edit']:continue
        local=event['local_original_interval_ids'];lo=event['local_support_start_s'];hi=event['local_support_end_s']
        support &= bool(local) and all(1<=j<=len(rr_seconds) for j in local) and 0<=lo<hi<=trace['corrected_support_duration_s'] and not event['touches_record_endpoint']
        if event['new_tick'] is not None:support &= lo<=event['new_tick']/RATE<=hi
    for step in trace['operation_steps']:
        support &= step['matches_backend'] and step['before_ticks'][0]==step['after_ticks'][0]==a[0] and step['before_ticks'][-1]==step['after_ticks'][-1]==a[-1]
    counts=trace['edit_counts'];special=None
    if support and not unresolved_structure_reason:
        if counts['insertion']==counts['deletion']==0 and trace['pure_internal_relocation_history']:special='HR_count_preserved_internal_relocation_only'
        elif counts['insertion'] or counts['deletion']:special='HR_local_count_correction_automated_estimate'
    rows=[];affected=set(trace['final_affected_original_interval_ids']);duration=math.fsum(rr_seconds)
    for window,quality,fs in variants:
        upper=60*RATE if window=='first60s' else math.inf
        raw_take=(a[:-1]>=0)&(a[1:]<=upper);nn_take=(b[:-1]>=0)&(b[1:]<=upper)
        ids=set((np.flatnonzero(raw_take)+1).tolist());den=len(ids);num=len(ids&affected)
        nn=np.diff(b)[nn_take]/RATE*1000;values=_metrics(nn)
        limit=5 if quality=='strict_c5' or window=='first60s' else 10
        fraction_ok=den>0 and num*100<=den*limit
        for f in fs:
            reason=[];path='regular_c'+str(limit)
            if not source_verified:reason.append('source_stage_not_verified')
            if unresolved_structure_reason:reason.append(unresolved_structure_reason)
            if not support:reason.append('correction_support_or_endpoint_failure')
            if window=='first60s' and duration<60:reason.append('raw_record_does_not_cover_first60s')
            if window=='full_response' and duration<minimum[f]:reason.append('below_minimum_duration')
            if not fraction_ok:
                if window=='full_response' and quality=='primary_quality' and f=='mean_hr' and special:path=special
                else:reason.append('affected_fraction_exceeds_'+str(limit)+'_percent')
            if not den or values[f] is None or not np.isfinite(values[f]):reason.append('insufficient_complete_intervals')
            rows.append(dict(feature_id=f,window_variant=window,quality_variant=quality,usable=not reason,value=values[f] if not reason else None,original_duration_s=duration,affected_interval_n=num,original_interval_n=den,admission_path=path if not reason else None,reason=';'.join(reason),human_review_status='automated_check_only'))
    return {'trace':trace,'features':rows}

def blood_pressure(workbook,mapping_workbook,scope_ids,*,data_sheet='dogspss',mapping_sheet='Sheet1'):
    """Local private inputs only; returns study IDs and four numeric readings."""
    from openpyxl import load_workbook
    def norm(x):return unicodedata.normalize('NFKC',str(x)).strip()
    book=load_workbook(mapping_workbook,read_only=True,data_only=True)
    raw=list(book[mapping_sheet].values);book.close();head=list(raw[0]);names={};ids=set()
    for row in raw[1:]:
        name,pid=row[head.index('participant_id')],row[head.index('study_id')]
        if name is None or pid is None:continue
        name=norm(name);pid=f'{int(float(pid)):03d}'
        if name in names or pid in ids:raise ValueError('Duplicate name or study ID in mapping')
        names[name]=pid;ids.add(pid)
    book=load_workbook(workbook,read_only=True,data_only=True);raw=list(book[data_sheet].values);book.close()
    expected={0:'participant_id',8:'sbp_pre',9:'dbp_pre',11:'sbp_post',12:'dbp_post'}
    if any(raw[0][k]!=v for k,v in expected.items()):raise ValueError('Blood-pressure column headers differ')
    found={}
    for row in raw[1:]:
        if row[0] is None:continue
        name=norm(row[0])
        if name not in names:raise ValueError('An exact normalized name has no study mapping')
        pid=names[name]
        if pid in found:raise ValueError('Duplicate mapped blood-pressure reading')
        values={k:row[j] for k,j in [('sbp_pre',8),('dbp_pre',9),('sbp_post',11),('dbp_post',12)]}
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in values.values()):raise ValueError('Blood-pressure readings must be finite positive numeric values')
        found[pid]={'participant_id':pid,**values}
    if not set(scope_ids)<=set(found):raise ValueError('Blood-pressure source does not cover the requested scope')
    return [found[pid] for pid in sorted(scope_ids)]
