"""OLS ANCOVA, estimability-aware contrasts, fixed HC3 t-reference, diagnostics."""
from __future__ import annotations
import warnings
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from .registry import PARAMETERS,CONTRASTS,CONDITIONS

def contrast_result(c,beta,cov,X,df):
    c=np.asarray(c,float)
    out={'b':None,'SE':None,'statistic':None,'df':float(df),'CI_low':None,'CI_high':None,'p_raw':None,
         'status':'not_estimable','reason':'','reference_distribution':'t'}
    rowspace=np.linalg.pinv(X)@X
    if not np.allclose(c@rowspace,c,rtol=1e-9,atol=1e-10):
        out['reason']='contrast_not_in_design_row_space';return out
    out['b']=float(c@beta)
    if df<=0: out['reason']='no_residual_degrees_of_freedom';return out
    if cov is None or not np.isfinite(cov).all():out['reason']='covariance_unavailable';return out
    variance=float(c@cov@c)
    if variance<=1e-24:out['reason']='zero_or_degenerate_variance';return out
    se=np.sqrt(variance);t=out['b']/se;q=stats.t.ppf(.975,df)
    out.update(SE=float(se),statistic=float(t),CI_low=float(out['b']-q*se),CI_high=float(out['b']+q*se),p_raw=float(2*stats.t.sf(abs(t),df)),status='ok')
    return out

def fit_ancova(frame,registry_entry):
    n=len(frame);effects=[];means=[];coefficients=[];cases=[]
    diag={'n':n,'status':'not_estimable','reason':'','rank':None,'condition_number':None,
          'residual_df':None,'residual_sd':None,'r_squared':None,'adjusted_r_squared':None,
          'baseline_reference':None,'missing_cells':[],'diagnostic_p_values_are_not_selection_rules':True}
    group_counts={f'n_{name}':int(frame.condition.eq(name).sum()) for name in CONDITIONS}
    if n==0:
        for covname in ('OLS','HC3'):
            for name in CONTRASTS:effects.append({'effect':name,'covariance_type':covname,'status':'not_estimable','reason':'no_eligible_pairs',**group_counts})
            for name in CONDITIONS:means.append({'condition':name,'covariance_type':covname,'status':'not_estimable','reason':'no_eligible_pairs','mean':None,'SE':None,'CI_low':None,'CI_high':None})
            for name in PARAMETERS:coefficients.append({'parameter':name,'covariance_type':covname,'status':'not_estimable','reason':'no_eligible_pairs','b':None,'SE':None,'CI_low':None,'CI_high':None})
        return {'effects':effects,'means':means,'coefficients':coefficients,'covariances':{},'diagnostics':diag,'cases':cases,'baseline_support':[]}
    baseline=float(frame.pre_value.mean())
    pre=frame.pre_value.to_numpy(float)-baseline;p=frame.Petting.to_numpy(float);f=frame.Feeding.to_numpy(float)
    X=np.column_stack([np.ones(n),pre,p,f,p*f]);y=frame.post_value.to_numpy(float)
    rank=int(np.linalg.matrix_rank(X));df=n-rank
    diag.update(rank=rank,condition_number=float(np.linalg.cond(X)),residual_df=df,baseline_reference=baseline,
                missing_cells=[k for k in CONDITIONS if group_counts['n_'+k]==0],
                pre_post_pearson=float(stats.pearsonr(frame.pre_value,y).statistic) if n>=3 and np.std(pre)>0 and np.std(y)>0 else None)
    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter('always')
        fit=sm.OLS(y,X).fit()
        influence=fit.get_influence()
        h=np.asarray(influence.hat_matrix_diag)
        standard_cov=np.asarray(fit.cov_params()) if df>0 else None
        hc3=None
        if df>0 and np.all(h<1-1e-12):
            robust=fit.get_robustcov_results(cov_type='HC3',use_t=True)
            assert robust.use_t and np.array_equal(fit.params,robust.params)
            hc3=np.asarray(robust.cov_params())
        if df>0:
            diag.update(residual_sd=float(np.sqrt(fit.ssr/df)),r_squared=float(fit.rsquared),adjusted_r_squared=float(fit.rsquared_adj))
        diag.update(status='ok' if rank==5 and df>0 else 'partial_estimation',reason='' if rank==5 and df>0 else 'design_rank_deficiency_or_no_residual_df',
                    hc3_status='available' if hc3 is not None else 'unavailable_leverage_one_or_no_df',
                    outcome_zero_variance=bool(np.std(y)<=1e-14),
                    perfect_or_degenerate_residual_fit=bool(fit.ssr<=1e-20),
                    high_cook_threshold=4/n,leverage_max=float(h.max()))
        residual=np.asarray(fit.resid)
        if 3<=n<=5000 and np.std(residual)>1e-12:
            sw=stats.shapiro(residual);diag.update(shapiro_W=float(sw.statistic),shapiro_p_diagnostic=float(sw.pvalue))
        if rank==5 and df>0 and np.std(residual)>1e-12:
            bp=het_breuschpagan(residual,X)
            diag.update(breusch_pagan_LM=float(bp[0]),breusch_pagan_p_diagnostic=float(bp[1]))
        # Fixed joint test for all 3 baseline-by-condition terms. Preserve a
        # diagnostic failure if the extended design has no available rank/df.
        extended=np.column_stack([X,pre*p,pre*f,pre*p*f]);ext_rank=np.linalg.matrix_rank(extended)
        if rank==5 and ext_rank==8 and n>8:
            larger=sm.OLS(y,extended).fit();fstat,pval,dfdiff=larger.compare_f_test(fit)
            diag.update(slope_homogeneity_F_diagnostic=float(fstat),slope_homogeneity_p_diagnostic=float(pval),slope_homogeneity_df_num=float(dfdiff),slope_homogeneity_df_den=float(larger.df_resid))
        else:diag['slope_homogeneity_status']='not_estimable_rank_or_df'
        lo=registry_entry['lower_bound'];hi=registry_entry['upper_bound']
        beyond=np.zeros(n,bool)
        if lo is not None:beyond|=fit.fittedvalues<lo
        if hi is not None:beyond|=fit.fittedvalues>hi
        diag['fitted_outside_feature_bounds_n']=int(beyond.sum())
        # Order-indexed theoretical normal quantiles make the saved QQ source
        # data exactly reproducible; they are diagnostics, not exclusion gates.
        order=np.argsort(residual);qq=np.empty(n);qq[order]=stats.norm.ppf((np.arange(n)+.5)/n)
        try:student=np.asarray(influence.resid_studentized_external)
        except (ValueError,np.linalg.LinAlgError):student=np.full(n,np.nan)
        cooks=np.asarray(influence.cooks_distance[0])
        for i,pid in enumerate(frame.participant_id):
            cases.append({'participant_id':pid,'pre_value':float(frame.pre_value.iloc[i]),'post_value':float(y[i]),'fitted_value':float(fit.fittedvalues[i]),'residual':float(residual[i]),'leverage':float(h[i]),'cook_distance':float(cooks[i]),'externally_studentized_residual':float(student[i]),'cook_above_4_over_n':bool(cooks[i]>4/n),'qq_theoretical_normal':float(qq[i]),'fitted_outside_feature_bounds':bool(beyond[i]),'removed_by_diagnostic':False})
    diag['warnings']=list(dict.fromkeys(str(w.message) for w in warnings_seen))
    covs={'parameter_names':PARAMETERS,'OLS':standard_cov,'HC3':hc3,'residual_df':df,'inference':'t, including HC3','rank':rank}
    for covname,cov in [('OLS',standard_cov),('HC3',hc3)]:
        for name,c in CONTRASTS.items():
            effects.append({'effect':name,'covariance_type':covname,**contrast_result(c,fit.params,cov,X,df),**group_counts})
        for j,name in enumerate(PARAMETERS):
            c=np.eye(5)[j]
            coefficients.append({'parameter':name,'covariance_type':covname,**contrast_result(c,fit.params,cov,X,df)})
        for name,(pv,fv) in CONDITIONS.items():
            c=[1,0,pv,fv,pv*fv]
            result=contrast_result(c,fit.params,cov,X,df)
            result['mean']=result.pop('b')
            means.append({'condition':name,'covariance_type':covname,'baseline_reference':baseline,**result})
    support=[]
    for name in CONDITIONS:
        part=frame.loc[frame.condition.eq(name)]
        support.append({'condition':name,'n':len(part),'pre_min':float(part.pre_value.min()) if len(part) else None,'pre_max':float(part.pre_value.max()) if len(part) else None,'reference_inside_observed_baseline_range':bool(part.pre_value.min()<=baseline<=part.pre_value.max()) if len(part) else False})
    return {'effects':effects,'means':means,'coefficients':coefficients,'covariances':covs,'diagnostics':diag,'cases':cases,'baseline_support':support}
