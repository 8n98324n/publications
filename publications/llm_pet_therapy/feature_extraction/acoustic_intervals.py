import csv,gzip,hashlib,json,math
from pathlib import Path
def rows(p):
    opener=gzip.open if str(p).endswith('.gz') else open
    with opener(p,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def union(ii):
    out=[]
    for a,b in sorted(ii):
        if b<=a:continue
        if out and a<=out[-1][1]+1e-9:out[-1][1]=max(b,out[-1][1])
        else:out.append([float(a),float(b)])
    return out

def minus(ii,jj):
    out=[]
    for a,b in union(ii):
        t=a
        for x,y in union(jj):
            if y<=t or x>=b:continue
            if x>t:out.append([t,min(x,b)])
            t=max(t,y)
            if t>=b:break
        if t<b:out.append([t,b])
    return out

def clip(ii,a,b):return union([(max(a,x),min(b,y)) for x,y in ii if y>a and x<b])

def dur(ii):return sum(b-a for a,b in union(ii))

def contains(ii,a,b):return any(a>=x-1e-7 and b<=y+1e-7 for x,y in ii)

def truth(x):return x is True or str(x).lower()=='true'

def silence_measure(stage,domains,protected,energy_path,vad_path):
    import numpy as np
    asset=stage['audio_asset_id'];ep=Path(energy_path);vp=Path(vad_path)
    with np.load(ep,allow_pickle=False) as data:
        es,ee,db=[data[k].copy() for k in ['cell_start_sec','cell_end_sec','rms_dbfs']]
    vr=rows(vp);vs=np.array([float(r['start_sec']) for r in vr]);ve=np.array([float(r['end_sec']) for r in vr]);pr=np.array([float(r['speech_probability']) for r in vr])
    padded=np.array([truth(r['contains_terminal_model_padding']) or str(r['contains_terminal_model_padding'])=='1' for r in vr],dtype=bool)
    for s,e in [(es,ee),(vs,ve)]:
        if len(s)!=len(e) or np.any(e<=s) or np.any(s[1:]<e[:-1]-1e-8):raise ValueError('Overlapping/invalid cache timeline')
    valid=np.isfinite(db);qmask=np.zeros(len(db),bool)
    for a,b in domains:qmask|=(es>=a-1e-9)&(ee<=b+1e-9)
    usable=db[valid&qmask]
    if not len(usable):raise ValueError('No finite energy values for Q99')
    if not np.any(usable>-240.+1e-8):raise ValueError('All reference energy is numerical zero floor; relative threshold not measurable')
    q99=float(np.quantile(usable,.99,method='linear'));theta=q99-25.
    atoms=[]
    for domain_index,(a,b) in enumerate(domains):
        bounds=np.unique(np.r_[a,b,es[(es>a)&(es<b)],ee[(ee>a)&(ee<b)],vs[(vs>a)&(vs<b)],ve[(ve>a)&(ve<b)],
            [x for ab in protected for x in ab if a<x<b]])
        mids=(bounds[:-1]+bounds[1:])/2
        ei=np.searchsorted(es,mids,side='right')-1;vi=np.searchsorted(vs,mids,side='right')-1
        for lo,hi,m,eidx,vidx in zip(bounds[:-1],bounds[1:],mids,ei,vi):
            support=eidx>=0 and vidx>=0 and m<ee[eidx] and m<ve[vidx] and math.isfinite(float(db[eidx])) and math.isfinite(float(pr[vidx])) and 0<=pr[vidx]<=1
            support=bool(support and not padded[vidx] and es[eidx]>=a-1e-9 and ee[eidx]<=b+1e-9 and vs[vidx]>=a-1e-9 and ve[vidx]<=b+1e-9)
            speech=contains(protected,float(lo),float(hi))
            cand=bool(support and db[eidx]<theta and pr[vidx]<.5 and not speech)
            atoms.append({'start':float(lo),'end':float(hi),'domain':domain_index,'valid':bool(support),'candidate':cand,'speech_protected':speech})
    events=[];i=0
    while i<len(atoms):
        if not atoms[i]['candidate']:i+=1;continue
        first=i
        while i+1<len(atoms) and atoms[i+1]['candidate'] and atoms[i+1]['domain']==atoms[first]['domain'] and abs(atoms[i+1]['start']-atoms[i]['end'])<1e-8:i+=1
        last=i;a=atoms[first]['start'];b=atoms[last]['end'];domain=atoms[first]['domain']
        left=first==0 or atoms[first-1]['domain']!=domain or not atoms[first-1]['valid']
        right=last+1==len(atoms) or atoms[last+1]['domain']!=domain or not atoms[last+1]['valid']
        truncated=left or right
        events.append({'participant_id':stage['participant_id'],'timepoint':stage['timepoint'],'source_index':asset,
            'event_id':f'{asset}:silent_candidate:{len(events):04d}','start_s':a,'end_s':b,'duration_s':b-a,
            'theta_s_dbfs':theta,'candidate_before_duration_filter':True,'left_truncated':left,'right_truncated':right,
            'complete':not truncated,'qualifies_ge_200ms':not truncated and b-a>=.2-1e-9,
            'entire_event_removed_from_D_s':truncated})
        i+=1
    known=union([[x['start'],x['end']] for x in atoms if x['valid']])
    trunc=union([[e['start_s'],e['end_s']] for e in events if not e['complete']])
    den=minus(known,trunc);ds=dur(den)
    num=sum(e['duration_s'] for e in events if e['qualifies_ge_200ms'])
    if not -1e-8<=num<=ds+1e-8:raise ValueError('Silence numerator exceeds denominator')
    return {'value_candidate':num/ds if ds>0 else None,'D_s_s':ds,'silence_numerator_s':num,
        'energy_Q99_dbfs':q99,'theta_s_dbfs':theta,'silence_raw_technical_unknown_s':dur(domains)-dur(known),
        'silence_truncated_candidate_s':dur(trunc),'D_s_intervals':den,'events':events,
        'energy_cache_sha256':sha(ep),'vad_cache_sha256':sha(vp),'error':None}

def periodic_measure(frame_path,domains):
    p=Path(frame_path)
    rr=rows(p);intervals=[];voiced=[];seen=set()
    for r in rr:
        a=float(r['frame_start_sec']);b=float(r['cell_end_sec']);end=float(r['estimation_support_end_sec'])
        if a in seen:raise ValueError('Duplicate periodic output timestamp')
        seen.add(a)
        # F0 zero is a normal unvoiced observation. Nonfinite output is missing.
        finite=all(math.isfinite(float(r[k])) for k in ['F0_final_Hz','voicing_final_score','gaussian_window_RMS'])
        if not finite or not truth(r['waveform_support_complete']) or not contains(domains,a,end):continue
        if abs(end-a-.06)>1e-7 or b-a>.0100001:raise ValueError('Unexpected periodic support/weight')
        expected=float(r['F0_final_Hz'])>0 and float(r['gaussian_window_RMS'])>=.001
        if expected!=truth(r['algorithm_periodic_voiced']):raise ValueError('Periodicity cache predicate changed')
        iv=clip(domains,a,b);intervals+=iv
        if expected:voiced+=iv
    if abs(dur(intervals)-sum(b-a for a,b in intervals))>1e-6:raise ValueError('Periodic time overlap')
    dv=dur(intervals);num=dur(voiced)
    return {'value_candidate':num/dv if dv>0 else None,'D_v_s':dv,'periodic_numerator_s':num,
        'D_v_intervals':union(intervals),'periodic_intervals':union(voiced),'cache_sha256':sha(p),
        'semantic_P_filter_applied':False,'extra_final_score_threshold_applied':False,'error':None}
