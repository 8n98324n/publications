"""Ordinal and HRV measurement sensitivity models."""
import hashlib,warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog as _linprog
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from .ancova import contrast_result
from .registry import CONDITIONS,CONTRASTS
SMILE_FEATURES=('smile_prevalence_score','smile_intensity_score')
REQUIRED=('participant_id','pre_value','post_value','Petting','Feeding','condition')
def linprog(*args,**kwargs):
    options=dict(kwargs.pop('options',{}) or {});options['threads']=1
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
        return _linprog(*args,options=options,**kwargs)


def _sample(frame):
    ids=sorted(frame.participant_id.astype(str).tolist())
    return dict(n=len(ids),sample_ids=ids,
        sample_id_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
        **{f'n_{name}':int(frame.condition.eq(name).sum()) for name in CONDITIONS})

def _validate_frame(frame):
    missing=set(REQUIRED)-set(frame.columns)
    if missing:raise ValueError(f'Missing paired-frame columns: {sorted(missing)}')
    f=frame.copy().reset_index(drop=True)
    if f.participant_id.isna().any() or f.participant_id.duplicated().any():
        raise ValueError('Participant pairs must be unique and have nonmissing IDs')
    f['participant_id']=f.participant_id.astype(str)
    for col in ('pre_value','post_value','Petting','Feeding'):
        f[col]=pd.to_numeric(f[col],errors='raise')
        if not np.isfinite(f[col]).all():raise ValueError(f'Nonfinite paired input: {col}')
    for row in f.itertuples(index=False):
        if row.condition not in CONDITIONS or CONDITIONS[row.condition]!=(row.Petting,row.Feeding):
            raise ValueError('Condition labels and 0/1 factor values disagree')
    return f

def _empty(reason,status='not_estimable',frame=None,**metadata):
    return dict(status=status,reason=reason,effects=[],coefficients=[],covariances={},
        means=[],thresholds=[],probabilities=[],diagnostics={},
        **(_sample(frame) if frame is not None else {}),**metadata)

def _exp_finite(value):
    if value is None or not np.isfinite(value) or value>np.log(np.finfo(float).max):return None
    answer=float(np.exp(value))
    return answer if answer>0 else None

def _normal_contrast(c,beta,cov):
    c=np.asarray(c,float);b=float(c@beta);v=float(c@cov@c)
    out=dict(b=b,SE=None,statistic=None,df=None,CI_low=None,CI_high=None,p_raw=None,
        reference_distribution='asymptotic_normal',status='not_estimable',reason='nonpositive_or_nonfinite_variance')
    if np.isfinite(v) and v>1e-24:
        se=float(np.sqrt(v));z=b/se;cut=float(stats.norm.ppf(.975))
        out.update(SE=se,statistic=z,CI_low=b-cut*se,CI_high=b+cut*se,
            p_raw=float(2*stats.norm.sf(abs(z))),status='ok',reason='')
    return out

def _ordered_separation_direction(X,y,k):
    """Find a sufficient likelihood-improving recession direction by linear programming.

    Each observed response's upper latent bound may only move outward and its
    lower bound may only move outward. At least one must improve strictly.
    This detects complete/quasi separation without changing or combining grades.
    No arbitrary coefficient or p-value cutoff selects a model.
    """
    d=X.shape[1];rows=[]
    for xi,grade in zip(X,y):
        if grade<k-1:
            r=np.r_[-xi,np.zeros(k-1)];r[d+grade]=1;rows.append(r)
        if grade>0:
            r=np.r_[xi,np.zeros(k-1)];r[d+grade-1]=-1;rows.append(r)
    A=np.asarray(rows,float);constraints=[-A]
    if k>2:
        order=np.zeros((k-2,d+k-1))
        for j in range(k-2):order[j,d+j]=1;order[j,d+j+1]=-1
        constraints.append(order)
    C=np.vstack(constraints)
    answer=linprog(-A.sum(axis=0),A_ub=C,b_ub=np.zeros(len(C)),
        bounds=[(-1,1)]*(d+k-1),method='highs')
    if not answer.success:return None
    return bool(-answer.fun>1e-7)

def fit_smile_ordinal(frame,feature_id,repeat_id,*,main_sample_ids,maxiter=1000):
    """One original 0--4 repeat, on exactly the main smile model's paired sample.

    Input uses REQUIRED paired columns. main_sample_ids is mandatory and must
    match exactly; missing calls never silently create a smaller sensitivity set.
    Unobserved grades are reported as not estimable, never collapsed or imputed.
    """
    if feature_id not in SMILE_FEATURES:raise ValueError('Ordinal sensitivity is specified only for the two smile grades')
    if repeat_id not in (1,2):raise ValueError('repeat_id must be 1 or 2')
    f=_validate_frame(frame)
    meta=dict(feature_id=feature_id,repeat_id=repeat_id,model='proportional_odds_logit',
        unit='log_odds',positive_direction='higher_original_smile_grade',
        reference_distribution='asymptotic_normal',df=None,
        additional_intercept=False,categories_merged=False,
        probability_standardization='same empirical baseline-grade distribution in all four conditions')
    if len(set(map(str,main_sample_ids)))!=len(main_sample_ids):raise ValueError('Duplicate main sample IDs')
    if set(f.participant_id)!=set(map(str,main_sample_ids)):
        return _empty('repeat_pairs_do_not_match_locked_main_sample',frame=f,**meta)
    if len(f)==0:return _empty('no_eligible_main_pairs',frame=f,**meta)
    grades=np.r_[f.pre_value,f.post_value]
    if not np.all((grades>=0)&(grades<=4)&(grades==np.floor(grades))):
        return _empty('original_integer_0_to_4_grades_required_no_rounding_of_means',frame=f,**meta)
    observed=sorted(f.post_value.astype(int).unique().tolist())
    absent=sorted(set(range(5))-set(observed))
    meta.update(observed_post_grades=observed,unobserved_post_grades=absent,
        post_grade_counts={str(i):int(f.post_value.eq(i).sum()) for i in range(5)})
    if len(observed)<2:return _empty('fewer_than_two_observed_post_grades',frame=f,**meta)
    baseline=float(f.pre_value.mean())
    X=pd.DataFrame({'pre_centered':f.pre_value-baseline,'Petting':f.Petting,
        'Feeding':f.Feeding,'Petting:Feeding':f.Petting*f.Feeding},dtype=float)
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(f)),X]))<5:
        return _empty('covariate_rank_deficiency_or_implicit_intercept',frame=f,**meta)
    y=f.post_value.astype(int)
    encoded=np.searchsorted(observed,y.to_numpy())
    separated=_ordered_separation_direction(X.to_numpy(),encoded,len(observed))
    if separated is None:return _empty('separation_diagnostic_numerical_failure',frame=f,**meta)
    if separated:return _empty('complete_or_quasi_separation_direction_detected_no_grade_merging',frame=f,**meta)
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            model=OrderedModel(y,X,distr='logit')
            fitted=model.fit(method='bfgs',disp=False,maxiter=maxiter,gtol=1e-8)
            converged=bool(fitted.mle_retvals.get('converged',False))
            beta=np.asarray(fitted.params,float)
            cov=np.asarray(fitted.cov_params(),float)
        diag=dict(converged=converged,optimizer='bfgs',maxiter=maxiter,gradient_tolerance=1e-8,
            warnings=list(dict.fromkeys(str(w.message) for w in seen)),
            log_likelihood=float(fitted.llf),baseline_reference=baseline,
            covariance_condition_number=float(np.linalg.cond(cov)),
            separation_direction_detected=False,probability_missing_grades_not_estimated=absent)
        if not converged or not np.isfinite(beta).all() or not np.isfinite(cov).all() or np.linalg.eigvalsh((cov+cov.T)/2).min()<=0:
            out=_empty('nonconvergence_or_nonpositive_covariance',frame=f,**meta);out['diagnostics']=diag;return out
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        return _empty('fit_failure:'+str(exc),frame=f,**meta)
    names=list(fitted.params.index);effects=[];coefficients=[]
    for name,c5 in CONTRASTS.items():
        c=np.zeros(len(beta))
        for term,weight in zip(('pre_centered','Petting','Feeding','Petting:Feeding'),c5[1:]):
            c[names.index(term)]=weight
        result=_normal_contrast(c,beta,cov)
        effects.append(dict(effect=name,contrast=dict(zip(names,c.tolist())),**result,
            odds_ratio=_exp_finite(result['b']),OR_CI_low=_exp_finite(result['CI_low']),OR_CI_high=_exp_finite(result['CI_high']),
            ratio_interpretation='exponentiated average conditional log-odds contrast; not a marginal probability effect'))
    for j,name in enumerate(names):
        coefficients.append(dict(parameter=name,parameter_type='regression' if j<4 else 'threshold_internal',
            parameterization='slope' if j<4 else 'first_cutpoint_then_log_positive_increment',
            **_normal_contrast(np.eye(len(beta))[j],beta,cov)))
    cutpoints=model.transform_threshold_params(beta)[1:-1]
    thresholds=[]
    for j,cutpoint in enumerate(cutpoints):
        jac=np.zeros(len(beta));jac[4]=1
        for l in range(1,j+1):jac[4+l]=np.exp(beta[4+l])
        se=float(np.sqrt(jac@cov@jac))
        thresholds.append(dict(lower_observed_grade=observed[j],upper_observed_grade=observed[j+1],
            cutpoint=float(cutpoint),SE_delta_method=se,CI_low=float(cutpoint-1.959963984540054*se),
            CI_high=float(cutpoint+1.959963984540054*se),reference_distribution='asymptotic_normal',
            unit='latent_logit_cutpoint',intervening_unobserved_grade_thresholds_not_separately_identified=observed[j+1]-observed[j]>1))
    probabilities=[]
    for condition,(pv,fv) in CONDITIONS.items():
        design=X.copy();design['Petting']=pv;design['Feeding']=fv;design['Petting:Feeding']=pv*fv
        predicted=np.asarray(model.predict(beta,exog=design)).mean(axis=0)
        if not np.isclose(predicted.sum(),1):raise AssertionError('Predicted category probabilities do not sum to 1')
        for grade in range(5):
            probabilities.append(dict(condition=condition,grade=grade,
                probability=float(predicted[observed.index(grade)]) if grade in observed else None,
                status='ok' if grade in observed else 'not_estimable_unobserved_grade',
                standardized_over_n=len(f),reference_sample_sha256=_sample(f)['sample_id_sha256']))
    return dict(status='ok',reason='',**meta,**_sample(f),effects=effects,coefficients=coefficients,
        covariances=dict(parameter_names=names,model_covariance=cov,reference_distribution='asymptotic_normal'),
        means=[],thresholds=thresholds,probabilities=probabilities,diagnostics=diag)

def _fit_linear_sensitivity(f,feature_id,*,duration_adjust=False,log_scale=False):
    n=len(f)
    if n==0:return _empty('no_eligible_pairs',frame=f,feature_id=feature_id)
    pre=np.log(f.pre_value.to_numpy()) if log_scale else f.pre_value.to_numpy(float)
    y=np.log(f.post_value.to_numpy()) if log_scale else f.post_value.to_numpy(float)
    names=['Intercept','pre_centered','Petting','Feeding','Petting:Feeding']
    X=np.column_stack([np.ones(n),pre-pre.mean(),f.Petting,f.Feeding,f.Petting*f.Feeding])
    reference={'pre_value_on_fitted_scale':float(pre.mean())}
    if duration_adjust:
        names+=['pre_duration_s_centered','post_duration_s_centered']
        a=f.pre_duration_s.to_numpy(float);b=f.post_duration_s.to_numpy(float)
        X=np.column_stack([X,a-a.mean(),b-b.mean()])
        reference.update(pre_duration_s=float(a.mean()),post_duration_s=float(b.mean()))
    rank=int(np.linalg.matrix_rank(X));df=n-rank
    if duration_adjust and rank<len(names):
        return _empty('duration_adjustment_design_rank_deficient_or_constant_collinear_lengths',frame=f,feature_id=feature_id)
    if df<=0:return _empty('no_residual_degrees_of_freedom',frame=f,feature_id=feature_id)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        model=sm.OLS(y,X).fit()
        cov_ols=np.asarray(model.cov_params())
        h=model.get_influence().hat_matrix_diag
        cov_hc3=np.asarray(model.get_robustcov_results(cov_type='HC3',use_t=True).cov_params()) if np.all(h<1-1e-12) else None
    effects=[];coefficients=[];means=[]
    for covname,cov in [('OLS',cov_ols),('HC3',cov_hc3)]:
        for name,c5 in CONTRASTS.items():
            c=np.r_[c5,np.zeros(len(names)-5)]
            result=contrast_result(c,model.params,cov,X,df)
            if log_scale:result.update(geometric_ratio=_exp_finite(result['b']),ratio_CI_low=_exp_finite(result['CI_low']),ratio_CI_high=_exp_finite(result['CI_high']),ratio_interpretation='exp of average conditional log contrast; not arithmetic mean difference')
            effects.append(dict(effect=name,covariance_type=covname,unit='natural_log_ms' if log_scale else 'ms',**result))
        for j,name in enumerate(names):
            result=contrast_result(np.eye(len(names))[j],model.params,cov,X,df)
            coefficients.append(dict(parameter=name,covariance_type=covname,**result))
        for condition,(pv,fv) in CONDITIONS.items():
            c=np.r_[[1,0,pv,fv,pv*fv],np.zeros(len(names)-5)]
            result=contrast_result(c,model.params,cov,X,df);result['mean']=result.pop('b')
            if log_scale:result.update(geometric_mean_ms=_exp_finite(result['mean']),geometric_CI_low_ms=_exp_finite(result['CI_low']),geometric_CI_high_ms=_exp_finite(result['CI_high']))
            means.append(dict(condition=condition,covariance_type=covname,**result))
    return dict(status='ok' if rank==len(names) else 'partial_estimation',reason='' if rank==len(names) else 'rank_deficiency',
        feature_id=feature_id,**_sample(f),effects=effects,coefficients=coefficients,means=means,
        covariances=dict(parameter_names=names,OLS=cov_ols,HC3=cov_hc3,residual_df=df,rank=rank,reference_distribution='t_including_HC3'),
        diagnostics=dict(rank=rank,residual_df=df,reference=reference,condition_number=float(np.linalg.cond(X)),
            residual_sd=float(np.sqrt(model.ssr/df)),r_squared=float(model.rsquared),adjusted_r_squared=float(model.rsquared_adj),
            warnings=list(dict.fromkeys(str(w.message) for w in seen))),
        thresholds=[],probabilities=[],unit='natural_log_ms' if log_scale else 'ms',
        formula='log(post_ms) ~ 1 + centered(log(pre_ms)) + Petting + Feeding + Petting:Feeding' if log_scale else 'post_ms ~ 1 + centered(pre_ms) + Petting + Feeding + Petting:Feeding + centered(pre_duration_s) + centered(post_duration_s)',
        interpretation='measurement sensitivity; post-answer duration may be post-treatment; no claim of improved causal adjustment' if duration_adjust else 'log-scale sensitivity does not replace the raw-ms primary result')

def fit_hrv_duration_adjusted(frame,feature_id,*,hrv_duration_rule_confirmed=False):
    if feature_id not in ('rmssd','sdnn'):raise ValueError('Length covariates prespecified only for RMSSD/SDNN')
    f=_validate_frame(frame)
    if not hrv_duration_rule_confirmed:return _empty('HRV_minimum_duration_not_confirmed',status='pending_researcher_decision',frame=f,feature_id=feature_id)
    for col in ('pre_duration_s','post_duration_s'):
        if col not in f:
            return _empty('missing_or_invalid_actual_duration_no_automatic_imputation',frame=f,feature_id=feature_id)
        values=pd.to_numeric(f[col],errors='coerce')
        if not np.isfinite(values).all() or not values.gt(0).all():
            return _empty('missing_or_invalid_actual_duration_no_automatic_imputation',frame=f,feature_id=feature_id)
        f[col]=values
    return _fit_linear_sensitivity(f,feature_id,duration_adjust=True)

def fit_hrv_log_ancova(frame,feature_id,*,hrv_duration_rule_confirmed=False):
    if feature_id not in ('rmssd','sdnn'):raise ValueError('Log sensitivity prespecified only for RMSSD/SDNN')
    f=_validate_frame(frame)
    if not hrv_duration_rule_confirmed:return _empty('HRV_minimum_duration_not_confirmed',status='pending_researcher_decision',frame=f,feature_id=feature_id)
    invalid=f.pre_value.le(0)|f.post_value.le(0)
    if invalid.any():
        return _empty('nonpositive_values_need_source_review_no_added_constant',status='pending_source_review',frame=f,
            feature_id=feature_id,nonpositive_ids=f.loc[invalid,'participant_id'].tolist())
    return _fit_linear_sensitivity(f,feature_id,log_scale=True)
