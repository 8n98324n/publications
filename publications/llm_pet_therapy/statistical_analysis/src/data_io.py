"""Read qualified measurements without population imputation."""
from pathlib import Path
import csv,json,math
import numpy as np
import pandas as pd
from .models.registry import REGISTRY,FEATURES,CONDITIONS

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def boolean(x):
    if str(x).lower() not in ('true','false','1','0'):raise ValueError('An explicit boolean is required')
    return str(x).lower() in ('true','1')
def number(x):
    if x in ('',None):return np.nan
    v=float(x)
    if not math.isfinite(v):raise ValueError('Use an empty cell for missingness, not infinity/NaN text')
    return v
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(rows,pd.DataFrame):frame=rows.copy()
    else:frame=pd.DataFrame(clean(rows))
    for col in frame:
        if col in ('seed_boot','seed_perm'):frame[col]=frame[col].map(lambda x:str(x) if pd.notna(x) else '')
        if frame[col].map(lambda v:isinstance(v,(list,dict))).any():frame[col]=frame[col].map(lambda v:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict)) else v)
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g',na_rep='')

def load_inputs(directory,*,demo=False):
    directory=Path(directory);scopes=read_json(directory/'scopes.json')
    for key in ('study_main100','specified_avrr92','independent_thermal79'):
        ids=scopes[key]
        if len(ids)!=len(set(ids)) or any(not isinstance(i,str) or not i for i in ids):raise ValueError('Invalid scope identifiers')
        if not set(ids)<=set(scopes['study_main100']):raise ValueError('Scope is not a subset of the study roster')
    if not demo and [len(scopes[k]) for k in ('study_main100','specified_avrr92','independent_thermal79')]!=[100,92,79]:raise ValueError('Study scope sizes must be 100/92/79; use --demo only for synthetic inputs')
    participants=read_csv(directory/'participants.csv');p={r['participant_id']:r for r in participants}
    if len(p)!=len(participants) or set(p)!=set(scopes['study_main100']):raise ValueError('Participant roster mismatch')
    for r in p.values():
        for k in ('Petting','Feeding','STAI_pre','STAI_post','STAI_improvement'):r[k]=number(r[k])
        if CONDITIONS.get(r['condition'])!=(r['Petting'],r['Feeding']):raise ValueError('Condition and factor coding disagree')
        if not all(20<=r[k]<=80 for k in ('STAI_pre','STAI_post')):raise ValueError('STAI missing/out of range; do not impute targets')
        if r['STAI_improvement']!=r['STAI_pre']-r['STAI_post']:raise ValueError('STAI improvement direction is pre minus post')
        if boolean(r['in_prediction_scope'])!=(r['participant_id'] in scopes['specified_avrr92']):raise ValueError('Prediction scope mismatch')
    frames={f:[] for f in FEATURES};seen=set()
    for r in read_csv(directory/'features_paired.csv'):
        pid,f=r['participant_id'],r['feature_id'];key=(pid,f)
        if key in seen or pid not in p or f not in REGISTRY:raise ValueError('Duplicate/unknown feature pair')
        seen.add(key);reg=REGISTRY[f];inside=pid in scopes[reg['scope_id']]
        if r['scope_id']!=reg['scope_id'] or boolean(r['in_analysis_scope'])!=inside:raise ValueError('Feature scope violation')
        if r['window_variant']!=reg['window_variant']:raise ValueError('Measurement window identifier changed')
        r.update(p[pid])
        for k in ('in_analysis_scope','pre_usable','post_usable','eligible_factorial','eligible_correlation'):r[k]=boolean(r[k])
        for t in ('pre','post'):
            r[t+'_value']=number(r[t+'_value']);r[t+'_duration_s']=number(r.get(t+'_duration_s',''))
            usable=r[t+'_usable'];v=r[t+'_value']
            if usable and (not inside or not np.isfinite(v)):raise ValueError('Eligible stage needs in-scope finite value')
            if not usable and np.isfinite(v):raise ValueError('Excluded source value must not enter the formal feature column')
            if usable and ((reg['lower_bound'] is not None and v<reg['lower_bound']) or (reg['upper_bound'] is not None and v>reg['upper_bound'])):raise ValueError('Feature outside measurement bounds')
        r['delta_feature']=number(r['delta_feature'])
        if r['eligible_factorial'] and not (r['pre_usable'] and r['post_usable']):raise ValueError('Invalid paired eligibility')
        if r['eligible_correlation'] and not r['eligible_factorial']:raise ValueError('Correlation cannot restore an excluded pair')
        if r['eligible_factorial']:
            if not np.isclose(r['delta_feature'],r['post_value']-r['pre_value'],rtol=0,atol=1e-12):raise ValueError('Feature delta must be post minus pre')
        elif np.isfinite(r['delta_feature']):raise ValueError('Excluded delta must be blank')
        frames[f].append(r)
    if len(seen)!=len(p)*19:raise ValueError('Complete participant by feature skeleton required')
    frames={f:pd.DataFrame(rs).sort_values('participant_id').reset_index(drop=True) for f,rs in frames.items()}
    stages=read_csv(directory/'features_long.csv');stage_seen=set()
    for r in stages:
        key=(r['participant_id'],r['feature_id'],r['timepoint'])
        if key in stage_seen or key[2] not in ('pre','post'):raise ValueError('Duplicate/invalid long stage')
        stage_seen.add(key);row=frames[key[1]].set_index('participant_id').loc[key[0]]
        if boolean(r['analysis_usable'])!=row[key[2]+'_usable']:raise ValueError('Long and paired eligibility conflict')
        a,b=number(r['analysis_value']),row[key[2]+'_value']
        if not (np.isnan(a) and np.isnan(b)) and a!=b:raise ValueError('Long and paired values conflict')
    if len(stage_seen)!=len(p)*38:raise ValueError('Complete long-stage skeleton required')
    rr=pd.DataFrame(read_csv(directory/'rr_variants.csv'))
    for col in ('value','original_duration_s'):rr[col]=rr[col].map(number)
    rr['usable']=rr.usable.map(boolean)
    keys=['participant_id','feature_id','timepoint','window_variant','quality_variant']
    if rr.duplicated(keys).any() or not set(rr.participant_id)<=set(scopes['specified_avrr92']):raise ValueError('RR variant scope/uniqueness failure')
    if (rr.usable&~np.isfinite(rr.value)).any() or (~rr.usable&np.isfinite(rr.value)).any():raise ValueError('RR value and eligibility disagree')
    for r in rr.itertuples():
        minimum=60 if r.window_variant=='first60s' else {'mean_hr':10,'rmssd':30,'sdnn':60}[r.feature_id]
        if r.usable and r.original_duration_s<minimum:raise ValueError('RR variant support below minimum duration')
    repeats=pd.DataFrame(read_csv(directory/'smile_repeats.csv'))
    repeats['repeat_id']=repeats.repeat_id.astype(int);repeats['value']=repeats.value.map(number)
    if repeats.duplicated(['participant_id','timepoint','feature_id','repeat_id']).any():raise ValueError('Duplicate smile repeat')
    if not repeats['value'].between(0,4).all() or not (repeats.value==np.floor(repeats.value)).all():raise ValueError('Smile repeats must be original integers 0..4')
    for f in FEATURES[:2]:
        fr=frames[f];rep=repeats[repeats.feature_id.eq(f)]
        for r in fr[fr.eligible_factorial].itertuples():
            for t in ('pre','post'):
                vals=rep[rep.participant_id.eq(r.participant_id)&rep.timepoint.eq(t)]
                if sorted(vals.repeat_id)!=[1,2] or vals.value.mean()!=getattr(r,t+'_value'):raise ValueError('Smile feature must equal two valid original calls')
    return dict(frames=frames,participants=p,scopes=scopes,rr=rr,repeats=repeats,demo=demo)

def prediction_inputs(data):
    ids=sorted(data['scopes']['specified_avrr92']);features=data['frames']
    delta=np.column_stack([features[f].set_index('participant_id').loc[ids,'delta_feature'].to_numpy(float) for f in FEATURES])
    post=np.column_stack([features[f].set_index('participant_id').loc[ids,'post_value'].to_numpy(float) for f in FEATURES])
    targets=pd.DataFrame([data['participants'][p] for p in ids]);p=targets.Petting.to_numpy(float);f=targets.Feeding.to_numpy(float)
    basic=np.column_stack([targets.STAI_pre,p,f,p*f])
    labels=list(CONDITIONS);classes=targets.condition.map({x:i for i,x in enumerate(labels)}).to_numpy(int)
    subsets={'main92':np.arange(len(ids)),'delta_cc':np.flatnonzero(np.isfinite(delta).all(axis=1)),'post_cc':np.flatnonzero(np.isfinite(post).all(axis=1))}
    for v,pos in subsets.items():
        if min(np.bincount(classes[pos],minlength=4))<5:raise ValueError('Each prediction subset needs at least five persons per class: '+v)
    return dict(ids=np.array(ids),delta=delta,post=post,basic=basic,extended=np.column_stack([basic,post]),y_class=classes,y_reg=targets.STAI_post.to_numpy(float),subsets=subsets)

def check(data):
    ml=prediction_inputs(data)
    return {'participants':len(data['participants']),'scope_counts':{k:len(v) for k,v in data['scopes'].items()},'features':{f:{'pre':int(x.pre_usable.sum()),'post':int(x.post_usable.sum()),'paired':int(x.eligible_factorial.sum()),'correlation':int(x.eligible_correlation.sum())} for f,x in data['frames'].items()},'complete_cases':{k:len(v) for k,v in ml['subsets'].items()},'synthetic':data['demo']}
