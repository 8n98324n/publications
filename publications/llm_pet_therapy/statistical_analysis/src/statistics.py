"""All 19 feature models and the specified measurement sensitivities."""
import copy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .data_io import write_json,write_csv
from .models.registry import REGISTRY,FEATURES,CONDITIONS,CONTRASTS
from .models.ancova import fit_ancova,contrast_result
from .models.correlation import correlation_inference
from .models.descriptive import sample_hash,make_descriptives
from .models.sensitivity import fit_smile_ordinal,fit_hrv_log_ancova
from .models.duration import duration_in_minutes
from .models.reliability import icc_absolute,participant_bootstrap

def au05_slopes(frame):
    n=len(frame);baseline=float(frame.pre_value.mean());x=frame.pre_value.to_numpy(float)-baseline
    p=frame.Petting.to_numpy(float);f=frame.Feeding.to_numpy(float);y=frame.post_value.to_numpy(float)
    X=np.column_stack([np.ones(n),x,p,f,p*f,x*p,x*f,x*p*f])
    fit=sm.OLS(y,X).fit();robust=fit.get_robustcov_results(cov_type='HC3',use_t=True)
    beta=np.asarray(fit.params);covs={'OLS':np.asarray(fit.cov_params()),'HC3':np.asarray(robust.cov_params())}
    rank=int(np.linalg.matrix_rank(X));df=n-rank;effects=[];means=[];coefficients=[]
    names=['Intercept','pre_centered','Petting','Feeding','Petting:Feeding','pre_centered:Petting','pre_centered:Feeding','pre_centered:Petting:Feeding']
    for kind,cov in covs.items():
        for effect,c in CONTRASTS.items():effects.append({'effect':effect,'covariance_type':kind,**contrast_result(np.r_[c,[0,0,0]],beta,cov,X,df)})
        for condition,(P,F) in CONDITIONS.items():
            r=contrast_result([1,0,P,F,P*F,0,0,0],beta,cov,X,df);r['mean']=r.pop('b')
            means.append({'condition':condition,'covariance_type':kind,'baseline_reference':baseline,**r})
        for j,name in enumerate(names):coefficients.append({'parameter':name,'covariance_type':kind,**contrast_result(np.eye(8)[j],beta,cov,X,df)})
    return {'effects':effects,'means':means,'coefficients':coefficients,'covariances':{'parameter_names':names,**covs,'rank':rank,'residual_df':df},'diagnostics':{'n':n,'rank':rank,'residual_df':df,'baseline_reference':baseline,'condition_number':float(np.linalg.cond(X)),'posthoc':True,'status':'ok' if rank==8 else 'partial_estimation'}}

def rr_frame(data,feature,window,quality):
    frame=data['frames'][feature].copy()
    frame=frame[frame.participant_id.isin(data['scopes']['specified_avrr92'])].copy()
    rr=data['rr'];rows=rr[rr.feature_id.eq(feature)&rr.window_variant.eq(window)&rr.quality_variant.eq(quality)]
    idx=rows.set_index(['participant_id','timepoint'],verify_integrity=True)
    for t in ('pre','post'):
        part=idx.loc[[(p,t) for p in frame.participant_id]]
        frame[t+'_value']=part.value.to_numpy();frame[t+'_usable']=part.usable.to_numpy()
        frame[t+'_duration_s']=part.original_duration_s.to_numpy()
    frame['eligible_factorial']=frame.pre_usable&frame.post_usable
    frame['eligible_correlation']=frame.eligible_factorial&frame.STAI_pre.between(20,80)&frame.STAI_post.between(20,80)
    frame['delta_feature']=np.where(frame.eligible_factorial,frame.post_value-frame.pre_value,np.nan)
    return frame

def planned_models(data):
    jobs=[]
    def add(f,comparison,role,frame,mode='raw',association=True,seed=None):
        fid=f+':'+comparison
        jobs.append(dict(feature_id=f,comparison=comparison,role=role,frame=frame,mode=mode,association=association,model_id=fid,seed_label=seed or fid))
    for f in FEATURES:
        add(f,'primary','primary',data['frames'][f],seed=f+':'+REGISTRY[f]['window_variant'])
    for f in ('sbp','dbp'):
        add(f,'specified92','BP92',data['frames'][f].loc[lambda x:x.participant_id.isin(data['scopes']['specified_avrr92'])].copy(),seed=f+':'+REGISTRY[f]['window_variant'])
    for f in ('mean_hr','rmssd','sdnn'):
        full=rr_frame(data,f,'full_response','primary_quality');fixed=rr_frame(data,f,'first60s','primary_quality');strict=rr_frame(data,f,'full_response','strict_c5')
        fullids=set(full.loc[full.eligible_factorial,'participant_id']);fixedids=set(fixed.loc[fixed.eligible_factorial,'participant_id']);inter=fullids&fixedids
        if not set(strict.loc[strict.eligible_factorial,'participant_id'])<=fullids:raise ValueError('Strict quality sample not nested within full sample')
        formal=data['frames'][f].set_index('participant_id')
        for r in full.itertuples():
            for t in ('pre','post'):
                if getattr(r,t+'_usable')!=formal.loc[r.participant_id,t+'_usable']:raise ValueError('Primary RR and variant eligibility differ')
                if getattr(r,t+'_usable') and not np.isclose(getattr(r,t+'_value'),formal.loc[r.participant_id,t+'_value'],atol=1e-12,rtol=0):raise ValueError('Primary RR and variant value differ')
        for comparison,frame in [('S5_full_strict',strict),('I_full',full[full.participant_id.isin(inter)]),('I_first60',fixed[fixed.participant_id.isin(inter)]),('S60_first60_separately',fixed)]:add(f,comparison,'RR',frame.copy())
        if f=='mean_hr':
            normal=rr_frame(data,f,'full_response','hrnormal_c10')
            if set(normal.loc[normal.eligible_factorial,'participant_id'])!=fullids:add(f,'S10_full_regular_subset','RR',normal,seed=f+':S10_full_quality_extension')
        else:
            add(f,'S_full_duration_minutes','RR',full,'duration_minutes',False)
            add(f,'S_full_natural_log','RR',full,'natural_log',False)
    add('au05_prevalence','posthoc_group_specific_slopes','AU05',data['frames']['au05_prevalence'],'group_slopes',False)
    return jobs

def run(data,config,out,*,demo=False):
    out.mkdir(parents=True,exist_ok=False);jobs=planned_models(data)
    tables={};summaries=[]
    def emit(role,table,rows):tables.setdefault((role,table),[]).extend(rows)
    for number,job in enumerate(jobs,1):
        f=job['feature_id'];frame=job['frame'];part=frame[frame.eligible_factorial].copy();reg=REGISTRY[f]
        meta={k:job[k] for k in ('feature_id','comparison','role','mode','model_id','seed_label')}
        meta.update(n=len(part),sample_id=sample_hash(part.participant_id),unit=reg['unit'],window_variant=reg['window_variant'])
        if job['mode']=='duration_minutes':model=duration_in_minutes(part,f)
        elif job['mode']=='natural_log':model=fit_hrv_log_ancova(part,f,hrv_duration_rule_confirmed=True)
        elif job['mode']=='group_slopes':model=au05_slopes(part)
        else:model=fit_ancova(part,reg)
        for source,target in [('effects','factorial_effects'),('means','adjusted_means'),('coefficients','model_coefficients'),('cases','case_diagnostics'),('baseline_support','baseline_support')]:emit(job['role'],target,[{**meta,**row} for row in model.get(source,[])])
        emit(job['role'],'model_diagnostics',[{**meta,**model.get('diagnostics',{})}])
        if job['role'] in ('primary','BP92'):emit(job['role'],'descriptives',make_descriptives(frame,{k:v for k,v in meta.items() if k!='n'}))
        associations={}
        if job['association']:
            cf=part[part.eligible_correlation].copy();args=dict(config['correlation'])
            if demo:args.update(B_boot=100,B_perm=100)
            for partial,name in [(False,'change_correlations'),(True,'partial_rank_correlations')]:
                result=correlation_inference(cf,job['seed_label'],partial=partial,**args)
                associations[name]=result;emit(job['role'],name,[{**meta,**result}])
        write_json(out/'models'/(job['model_id'].replace(':','__')+'.json'),{'metadata':meta,'participant_ids':part.participant_id.tolist(),'model':model,'associations':associations})
        emit(job['role'],'analysis_samples',[{**meta,'participant_id':r.participant_id,'eligible_factorial':r.eligible_factorial,'eligible_correlation':r.eligible_correlation} for r in frame.itertuples()])
        summaries.append(meta)
        print(f'Statistics {number}/{len(jobs)}: {f} {job["comparison"]}',flush=True)
    for (role,name),rows in tables.items():
        if rows:write_csv(out/role/(name+'.csv'),rows)
    # Both stage measurements of a sampled person stay together for ICC.
    reliability={};ordinal=[]
    for f in FEATURES[:2]:
        fr=data['frames'][f];part=fr[fr.eligible_factorial].copy();ids=part.participant_id.tolist()
        calls=data['repeats'][data['repeats'].feature_id.eq(f)].set_index(['participant_id','timepoint','repeat_id'])['value']
        clusters={p:[tuple(calls.loc[p,t,k] for k in (1,2)) for t in ('pre','post')] for p in ids}
        flat=[pair for p in sorted(clusters) for pair in clusters[p]]
        reliability[f]=icc_absolute(flat)
        if not demo:reliability[f]['bootstrap']=participant_bootstrap(clusters,**config['icc'])
        else:reliability[f]['bootstrap_status']='not_run_in_demo'
        for k in (1,2):
            sub=part.copy()
            for t in ('pre','post'):sub[t+'_value']=[calls.loc[p,t,k] for p in ids]
            result=fit_smile_ordinal(sub,f,k,main_sample_ids=ids)
            ordinal.append(result)
    write_json(out/'smile/reliability.json',reliability);write_json(out/'smile/ordinal_models.json',ordinal)
    for name in ('effects','coefficients','thresholds','probabilities'):
        write_csv(out/'smile'/('ordinal_'+name+'.csv'),[{'feature_id':r['feature_id'],'repeat_id':r['repeat_id'],**v} for r in ordinal for v in r.get(name,[])])
    write_csv(out/'smile/ordinal_model_status.csv',[{k:r.get(k) for k in ('feature_id','repeat_id','status','reason','n')} for r in ordinal])
    from scipy.stats import spearmanr
    for suffix,column in [('pre','pre_value'),('post','post_value'),('delta','delta_feature')]:
        wide=pd.DataFrame({f:x.set_index('participant_id')[column] for f,x in data['frames'].items()});records=[]
        for f in FEATURES:
            for g in FEATURES:
                a=wide[[f]] if f==g else wide[[f,g]];a=a.dropna();n=len(a)
                ok=n>=3 and a.iloc[:,0].nunique()>1 and a.iloc[:,-1].nunique()>1
                records.append({'feature1':f,'feature2':g,'n':n,'rho':float(spearmanr(a.iloc[:,0],a.iloc[:,-1]).statistic) if ok else None,'status':'ok' if ok else 'not_estimable'})
        write_csv(out/('feature_correlations_'+suffix+'.csv'),records)
    write_json(out/'run_summary.json',{'synthetic':demo,'model_count':len(jobs),'primary_features':19,'ordinal_models':4,'ICC_bootstrap_ran':not demo,'models':summaries})
    return summaries
