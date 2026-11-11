"""Participant-level rank correlation inference; all changes recomputed per draw."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.stats import rankdata

def stable_seed(label: str, master_seed: int = 20260921) -> int:
    return int.from_bytes(hashlib.sha256(f'{master_seed}:{label}'.encode()).digest()[:8],'big')

def rng_for(label, master_seed=20260921):
    seed=stable_seed(label,master_seed)
    return np.random.Generator(np.random.PCG64(seed)),seed

def row_pearson(x,y):
    x=np.atleast_2d(np.asarray(x,float)); y=np.atleast_2d(np.asarray(y,float))
    xc=x-x.mean(axis=1,keepdims=True); yc=y-y.mean(axis=1,keepdims=True)
    ssx=(xc*xc).sum(axis=1); ssy=(yc*yc).sum(axis=1)
    norm=np.sqrt(ssx*ssy)
    # Regression can leave roundoff-sized residuals for theoretically constant
    # vectors. These are unestimable, not spurious near-unit correlations.
    valid=(ssx>1e-20)&(ssy>1e-20)&np.isfinite(norm)
    out=np.full(len(x),np.nan)
    out[valid]=(xc[valid]*yc[valid]).sum(axis=1)/norm[valid]
    return np.clip(out,-1,1)

def rank_statistic(x,y,design=None):
    """Rows are draws. Each row is reranked, then both ranks are refitted to X."""
    rx=rankdata(np.atleast_2d(x),axis=1,method='average')
    ry=rankdata(np.atleast_2d(y),axis=1,method='average')
    if design is not None:
        # Batched least-squares coefficient calculation performs two separate
        # regressions per draw. X is fixed under stratified draw arrangement.
        pinv=np.linalg.pinv(design)
        rx=rx-(rx@pinv.T)@design.T
        ry=ry-(ry@pinv.T)@design.T
    return row_pearson(rx,ry)

def correlation_inference(frame, label, partial=False, B_boot=20000, B_perm=20000,
                          master_seed=20260921, chunk=256,max_degenerate_fraction=.1):
    """Frame contains paired participant rows; input rows never act as frames/beats.

    Bootstrap each participant's pre/post feature/STAI record together within
    each of four cells. Overall permutation moves whole STAI participant records
    globally; partial permutation moves them only within their original cell.
    """
    n=len(frame)
    out={'n':n,'rho':None,'CI_low':None,'CI_high':None,'p_perm_raw':None,
         'B_boot_attempted':0,'B_boot_valid':0,'B_boot_degenerate':0,
         'B_perm_attempted':0,'B_perm_valid':0,'B_perm_degenerate':0,'B_perm_extreme':None,
         'B_boot_planned':B_boot,'B_perm_planned':B_perm,
         'CI_method':'percentile_condition_stratified_participant_bootstrap',
         'p_method':'within_condition_pairing_permutation' if partial else 'global_pairing_permutation',
         'quantile_method':'linear','rank_method':'average','rng':'numpy.PCG64',
         'master_seed':master_seed,'controlled_variables':'Intercept,Petting,Feeding,Petting:Feeding' if partial else '',
         'resampling_batch_size':chunk,'max_degenerate_bootstrap_fraction':max_degenerate_fraction,
         'status':'not_estimable','reason':'','warning':'',
         'permutation_equality_atol':1e-12,'permutation_equality_rtol':1e-12}
    mode='partial' if partial else 'spearman'
    boot_rng,boot_seed=rng_for(label+':'+mode+':bootstrap',master_seed)
    perm_rng,perm_seed=rng_for(label+':'+mode+':permutation',master_seed)
    out.update(seed_boot=boot_seed,seed_perm=perm_seed)
    if n<3: out['reason']='fewer_than_three_participants';return out
    # Sorting by factors groups bootstrap positions consistently. It changes
    # neither estimand nor sample, and makes identical sample order reproducible.
    f=frame.sort_values(['Petting','Feeding','participant_id']).reset_index(drop=True)
    pre=f.pre_value.to_numpy(float); post=f.post_value.to_numpy(float)
    stpre=f.STAI_pre.to_numpy(float); stpost=f.STAI_post.to_numpy(float)
    p=f.Petting.to_numpy(float); feeding=f.Feeding.to_numpy(float)
    design=np.column_stack([np.ones(n),p,feeding,p*feeding]) if partial else None
    if partial and (np.linalg.matrix_rank(design)<4 or n<=4):
        out['reason']='partial_design_rank_deficient_or_no_residual_df';return out
    delta=post-pre; improvement=stpre-stpost
    rho=rank_statistic(delta,improvement,design)[0]
    if not np.isfinite(rho): out['reason']='constant_vector_or_no_within_condition_rank_variation';return out
    out['rho']=float(rho)
    groups=[np.flatnonzero((p==pv)&(feeding==fv)) for pv,fv in [(0,0),(0,1),(1,0),(1,1)]]
    groups=[g for g in groups if len(g)]
    samples=[]
    for offset in range(0,B_boot,chunk):
        m=min(chunk,B_boot-offset)
        idx=np.empty((m,n),int)
        for g in groups: idx[:,g]=boot_rng.choice(g,size=(m,len(g)),replace=True)
        # Recompute differences from jointly drawn records, then redo midranks
        # (sampling original rank vectors is wrong in the presence of ties).
        rs=rank_statistic(post[idx]-pre[idx],stpre[idx]-stpost[idx],design)
        samples.append(rs)
    bs=np.concatenate(samples) if samples else np.empty(0)
    valid=bs[np.isfinite(bs)]
    out.update(B_boot_attempted=len(bs),B_boot_valid=len(valid),B_boot_degenerate=int(np.isnan(bs).sum()))
    if len(valid)>=2:
        lo,hi=np.quantile(valid,[.025,.975],method='linear')
        out.update(CI_low=float(lo),CI_high=float(hi))
    if len(valid)!=B_boot: out['warning']='degenerate_bootstrap_draws_recorded'
    if B_boot and len(valid)<(1-max_degenerate_fraction)*B_boot:
        out['warning']='degenerate_bootstrap_fraction_exceeds_locked_threshold_CI_unavailable'
        out.update(CI_low=None,CI_high=None)
    extreme=0; perm_valid=0
    for offset in range(0,B_perm,chunk):
        m=min(chunk,B_perm-offset)
        idx=np.tile(np.arange(n),(m,1))
        if partial:
            for g in groups:
                for j in range(m): idx[j,g]=perm_rng.permutation(g)
        else:
            for j in range(m): idx[j]=perm_rng.permutation(n)
        rs=rank_statistic(np.broadcast_to(delta,(m,n)),stpre[idx]-stpost[idx],design)
        finite=np.isfinite(rs)
        tol=1e-12+1e-12*abs(rho)
        extreme+=int(np.sum(np.abs(rs[finite])>=abs(rho)-tol))
        perm_valid+=int(finite.sum())
    out.update(B_perm_attempted=B_perm,B_perm_valid=perm_valid,B_perm_degenerate=B_perm-perm_valid,B_perm_extreme=extreme)
    if perm_valid==B_perm and B_perm>0: out['p_perm_raw']=(1+extreme)/(B_perm+1)
    else: out['warning']=(out['warning']+';invalid_or_no_permutation_draws_p_unavailable').strip(';')
    out['status']='ok' if out['CI_low'] is not None and out['p_perm_raw'] is not None else 'partial_estimation'
    return out
