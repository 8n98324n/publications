"""Build qualified, unimputed statistical inputs from identified stage outputs."""
import argparse,shutil
from pathlib import Path
import numpy as np
from ..data_io import read_csv,read_json,boolean,number,write_csv,write_json,load_inputs,check
from ..models.registry import FEATURES,REGISTRY

def assemble(source,output,*,demo=False):
    """source contains participants/scopes, stage_values, pair_exclusions,
    rr_variants and smile_repeats. An exclusion can remove, never add a pair.
    Stage source values outside a feature's scope remain excluded.
    """
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new output directory')
    roster=read_csv(source/'participants.csv');scopes=read_json(source/'scopes.json')
    ids=[r['participant_id'] for r in roster];stage={}
    for r in read_csv(source/'stage_values.csv'):
        key=tuple(r[k] for k in ('participant_id','feature_id','timepoint'))
        if key in stage or key[0] not in ids or key[1] not in FEATURES or key[2] not in ('pre','post'):raise ValueError('Duplicate or unknown stage')
        stage[key]=r
    exclusions={}
    for r in read_csv(source/'pair_exclusions.csv'):
        key=(r['participant_id'],r['feature_id'])
        if key in exclusions or key[0] not in ids or key[1] not in FEATURES or not r['reason']:raise ValueError('Invalid documented paired exclusion')
        exclusions[key]=r
    long=[];pairs=[]
    for pid in sorted(ids):
        for fid in FEATURES:
            reg=REGISTRY[fid];inside=pid in scopes[reg['scope_id']]
            r=dict(participant_id=pid,feature_id=fid,scope_id=reg['scope_id'],window_variant=reg['window_variant'],in_analysis_scope=inside)
            for tp in ('pre','post'):
                raw=stage.get((pid,fid,tp));usable=bool(inside and raw and boolean(raw['analysis_usable']))
                value=number(raw['analysis_value']) if usable else np.nan
                if usable and not np.isfinite(value):raise ValueError('Usable stage requires a finite qualified value')
                reason='outside_feature_scope' if not inside else 'no_stage_output' if raw is None else raw.get('reason','')
                r[tp+'_usable']=usable;r[tp+'_value']=value;r[tp+'_duration_s']=number(raw.get('duration_s','')) if raw else np.nan;r[tp+'_reason']=reason
                long.append(dict(participant_id=pid,feature_id=fid,timepoint=tp,analysis_usable=usable,analysis_value=value,reason=reason))
            exclusion=exclusions.get((pid,fid),{})
            ok=r['pre_usable'] and r['post_usable']
            r['eligible_factorial']=ok and not boolean(exclusion.get('exclude_factorial','false'))
            r['eligible_correlation']=r['eligible_factorial'] and not boolean(exclusion.get('exclude_correlation','false'))
            r['delta_feature']=r['post_value']-r['pre_value'] if r['eligible_factorial'] else np.nan
            r['pair_reason']=exclusion.get('reason','')
            pairs.append(r)
    output.mkdir(parents=True)
    write_csv(output/'participants.csv',roster);write_json(output/'scopes.json',scopes)
    write_csv(output/'features_long.csv',long);write_csv(output/'features_paired.csv',pairs)
    for name in ('rr_variants.csv','smile_repeats.csv'):shutil.copyfile(source/name,output/name)
    result=check(load_inputs(output,demo=demo));write_json(output/'assembly_check.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--demo',action='store_true');a=p.parse_args()
    assemble(a.source,a.output,demo=a.demo)
