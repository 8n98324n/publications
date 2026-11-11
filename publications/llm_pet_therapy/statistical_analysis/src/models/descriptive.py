import hashlib,json
import numpy as np
from .registry import CONDITIONS

def sample_hash(ids):return hashlib.sha256(json.dumps(sorted(map(str,ids)),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def describe(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,M=None,SD=None,median=None,P25=None,P75=None,min=None,max=None)
    q=np.quantile(a,[.25,.5,.75],method='linear')
    return dict(n=len(a),M=float(a.mean()),SD=float(a.std(ddof=1)) if len(a)>1 else None,median=float(q[1]),P25=float(q[0]),P75=float(q[2]),min=float(a.min()),max=float(a.max()))

def make_descriptives(paired,meta):
    rows=[]
    for scope in ('paired','single_stage'):
        for tp in ('pre','post','delta') if scope=='paired' else ('pre','post'):
            usable=paired.eligible_factorial if scope=='paired' else paired[tp+'_usable']
            part=paired.loc[usable]
            column='delta_feature' if tp=='delta' else tp+'_value'
            for condition in ['all',*CONDITIONS]:
                group=part if condition=='all' else part.loc[part.condition.eq(condition)]
                rows.append({**meta,'model_sample_id':meta.get('sample_id'),
                    'sample_id':sample_hash(group.participant_id),'sample_id_role':'actual_descriptive_members',
                    'sample_scope':scope,'condition':condition,'timepoint':tp,**describe(group[column])})
    return rows
