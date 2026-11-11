"""Supported acoustic features from actual component and period diagnostics."""
import math
import numpy as np
import pandas as pd
from .acoustic_intervals import union,dur,contains,silence_measure,periodic_measure

def load_diagnostic_segment(directory,first_source_sample,last_source_sample_exclusive,sample_rate_hz=44100):
    """Parse diagnostics and attach final pitch/energy on matching timestamps."""
    from pathlib import Path
    from .jitter_diagnostics import assess
    from .jitter_support import audit_support
    directory=Path(directory);assessment,frames=assess(directory/'engine.log',directory/'Jitter.csv');support=audit_support(directory/'engine.log')
    if not assessment['parser_checks_pass'] or not support['support_contract_pass']:raise ValueError('Diagnostic period-support contract failed')
    pitch=pd.read_csv(directory/'PitchFinal.csv',sep=';');energy=pd.read_csv(directory/'Energy.csv',sep=';')
    pf={round(float(r.frameTime),7):r for r in pitch.itertuples()};ef={round(float(r.frameTime),7):r for r in energy.itertuples()}
    for row in frames:
        t=round(row['time_sec'],7)
        if t not in pf or t not in ef:raise ValueError('Diagnostic/final-pitch/energy timeline mismatch')
        row['new_final_f0_hz']=float(pf[t].F0final);row['new_RMS']=float(ef[t].pcm_RMSenergy)
    return dict(first_source_sample=first_source_sample,last_source_sample_exclusive=last_source_sample_exclusive,sample_rate_hz=sample_rate_hz,parser_pass=True,support_contract_pass=True,frames=frames)

def pitch_and_hnr(frame,target_support,*,segments_succeeded=True):
    """target_support covers each full 60 ms window, from the supplied source map."""
    f0=frame.F0_final_Hz.to_numpy();hn=frame.HNR_dB_raw.to_numpy();target=np.asarray(target_support,bool)
    if len(target)!=len(frame):raise ValueError('Target support mask length mismatch')
    base=frame.waveform_support_complete.to_numpy(bool)&frame.algorithm_periodic_voiced.to_numpy(bool)&np.isfinite(f0)&(f0>=55)&(f0<=1000)
    good=base&target;hb=base&np.isfinite(hn)&~frame.HNR_limit_flag.to_numpy(bool);hg=hb&target
    fcover=float(good.sum()/base.sum()) if base.sum() else 0.;hcover=float(hg.sum()/hb.sum()) if hb.sum() else 0.
    st=12*np.log2(f0[good]/27.5)
    specs=[('f0_median_st',int(good.sum()),fcover,100,float(np.median(st)) if len(st) else None),('f0_iqr_st',int(good.sum()),fcover,200,float(np.quantile(st,.75,method='linear')-np.quantile(st,.25,method='linear')) if len(st) else None),('hnr',int(hg.sum()),hcover,100,float(np.mean(hn[hg])) if hg.any() else None)]
    return [dict(feature_id=f,value=v if segments_succeeded and n>=minimum and coverage>=.8 else None,usable=bool(segments_succeeded and n>=minimum and coverage>=.8),supported_frames=n,target_supported_periodic_fraction=coverage) for f,n,coverage,minimum,v in specs]

def ratio_qualification(result,reference_duration,kind):
    denominator=result['D_s_s' if kind=='silence' else 'D_v_s']
    usable=reference_duration>0 and denominator>=1 and denominator/reference_duration>=.8
    return dict(usable=usable,value=result['value_candidate'] if usable else None,denominator_s=denominator,reference_duration_s=reference_duration)

def jitter_from_diagnostics(segments,domains,target_intervals):
    """Each segment supplies PCM sample bounds/rate and parsed diagnostic rows.

    Rows must include new_final_f0_hz/new_RMS from matching component timestamps.
    A parser/support error must be resolved before this function is called.
    """
    accepted=[];allrows=[]
    for segment in sorted(segments,key=lambda r:int(r['first_source_sample'])):
        if segment['parser_pass'] is not True or segment['support_contract_pass'] is not True:raise ValueError('Jitter diagnostic contract failed')
        rate=int(segment['sample_rate_hz']);offset=int(segment['first_source_sample'])/rate;end=int(segment['last_source_sample_exclusive'])/rate
        for r in segment['frames']:
            t=float(r['time_sec'])+offset;cellend=min(t+.01,end)
            a=min(t,offset+float(r['all_support_start'])/rate);b=max(t+.06,offset+float(r['all_support_end_exclusive'])/rate)
            tech=str(r['technical_current_adjacent_pair_support']).lower() in ('true','1')
            waveform=b<=end+1e-7 and a>=offset-1e-7 and contains(domains,a,b)
            source=contains(target_intervals,a,b);f0=float(r['new_final_f0_hz']);rms=float(r['new_RMS'])
            keep=tech and waveform and source and 55<=f0<=1000 and rms>=.001;val=float(r['output_jitter_ratio'])
            if keep and (not math.isfinite(val) or not 0<=val<=1):raise ValueError('Supported jitter outside engine range')
            record={'frame_start_s':t,'cell_end_s':cellend,'accepted':keep,'jitter_ratio':val};allrows.append(record)
            if keep:accepted.append(record)
    intervals=[[r['frame_start_s'],r['cell_end_s']] for r in accepted];support=dur(intervals)
    if abs(support-sum(b-a for a,b in intervals))>1e-6:raise ValueError('Jitter frame weights overlap')
    numerator=sum((r['cell_end_s']-r['frame_start_s'])*r['jitter_ratio'] for r in accepted)
    usable=len(accepted)>=100 and support>=1
    return dict(value=100*numerator/support if usable else None,usable=usable,supported_estimates=len(accepted),supported_time_s=support,frames=allrows)

def component_frames(directory,start_s,end_s):
    """Join unsmoothed openSMILE sinks of one contiguous PCM segment."""
    from pathlib import Path
    d=Path(directory)
    def read(name):
        p=d/(name+'.csv');return pd.read_csv(p if p.exists() else d/(name+'.csv.gz'),sep=';')
    f=read('PitchFinal');h=read('Harmonics');e=read('Energy')
    for x in (h,e):
        if len(x)!=len(f) or not np.allclose(x.frameTime,f.frameTime,atol=1e-7):raise ValueError('Component time axes differ')
    t=f.frameTime.to_numpy()+start_s;fin=f.F0final.to_numpy();energy=e.pcm_RMSenergy.to_numpy();hn=h.HarmonicsToNoiseRatioACFLogdB.to_numpy();voice=f.voicingFinalUnclipped.to_numpy()
    return pd.DataFrame(dict(frame_start_sec=t,cell_end_sec=np.minimum(t+.01,end_s),estimation_support_end_sec=t+.06,F0_final_Hz=fin,voicing_final_score=voice,gaussian_window_RMS=energy,algorithm_periodic_voiced=(fin>0)&(energy>=.001)&np.isfinite(voice)&np.isfinite(fin),waveform_support_complete=t+.06<=end_s+1e-7,HNR_dB_raw=hn,HNR_limit_flag=np.isclose(hn,-90,atol=1e-5)|np.isclose(hn,110,atol=1e-5)))
