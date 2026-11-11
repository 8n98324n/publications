"""Serial aggregation of supplied tool outputs, with explicit source decisions."""
import argparse,os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from ..data_io import read_csv,read_json,write_json,write_csv,boolean

def process(manifest,output):
    import pandas as pd
    from . import acoustic,acoustic_intervals,au,smile,thermal,cardiovascular
    manifest=Path(manifest);base=manifest.resolve().parent;output=Path(output)
    output.mkdir(parents=True,exist_ok=False);stages=[];rr=[];repeats=[]
    def file(p):return base/p
    for index,item in enumerate(read_json(manifest)):
        kind=item['type'];result=None
        def record(feature,value,usable,reason='',duration=None):
            stages.append(dict(participant_id=item['participant_id'],timepoint=item['timepoint'],feature_id=feature,analysis_usable=usable,analysis_value=value if usable else None,reason=reason,duration_s=duration))
        if kind=='smile':
            result,reps=smile.aggregate(read_json(file(item['accepted_json']))['accepted'],read_csv(file(item['identity_csv'])))
            stages.extend(result);repeats.extend(reps)
        elif kind=='au':
            result=au.from_openface(file(item['openface_csv']),file(item['frame_clock_csv']),item['start_s'],item['end_s'],target_source_confirmed=item['target_source_confirmed'])
            status,_,_,_,values=result
            for fid in ('au05_prevalence','au07_prevalence','au12_prevalence'):record(fid,values.get(fid),status=='ok',status,item['end_s']-item['start_s'])
        elif kind=='pitch_hnr':
            frame=pd.read_csv(file(item['frame_csv']))
            mask=[acoustic_intervals.contains(item['target_intervals'],a,b) for a,b in zip(frame.frame_start_sec,frame.estimation_support_end_sec)]
            result=acoustic.pitch_and_hnr(frame,mask,segments_succeeded=item['segments_succeeded'])
            for row in result:record(row['feature_id'],row['value'],row['usable'],'' if row['usable'] else 'acoustic_support_below_threshold')
        elif kind=='silence':
            result=acoustic_intervals.silence_measure(item,item['domains'],item['protected_speech_intervals'],file(item['energy_npz']),file(item['vad_csv']))
            q=acoustic.ratio_qualification(result,item['reference_duration_s'],'silence');record('auto_silent_interval_proportion',q['value'],q['usable'],'' if q['usable'] else 'acoustic_known_duration_below_threshold')
        elif kind=='periodic':
            result=acoustic_intervals.periodic_measure(file(item['frame_csv']),item['domains'])
            q=acoustic.ratio_qualification(result,item['reference_duration_s'],'periodic');record('auto_periodic_vocalization_proportion',q['value'],q['usable'],'' if q['usable'] else 'acoustic_known_duration_below_threshold')
        elif kind=='jitter':
            segments=read_json(file(item['segments_json']))
            segments=[acoustic.load_diagnostic_segment(file(r['directory']),r['first_source_sample'],r['last_source_sample_exclusive'],r['sample_rate_hz']) if 'directory' in r else r for r in segments]
            result=acoustic.jitter_from_diagnostics(segments,item['domains'],item['target_intervals'])
            record('jitter_local_pct',result['value'],result['usable'],'' if result['usable'] else 'jitter_current_period_support_below_threshold')
        elif kind=='thermal':
            frames=read_csv(file(item['roi_frames_csv']));result={}
            for roi,fid in [('nasal_tip','nose_tip_temperature'),('left_cheek','left_cheek_temperature'),('right_cheek','right_cheek_temperature')]:
                take=[r for r in frames if r['roi_key']==roi]
                if len({r['frame_index'] for r in take})!=30:raise ValueError('Thermal stage needs 30 unique selected frames per ROI')
                result[roi]=q=thermal.stage_mean([float(r['temperature_c']) if r['temperature_c'] else float('nan') for r in take],[boolean(r['valid']) for r in take],roi)
                record(fid,q['value'],q['usable'],'' if q['usable'] else 'insufficient_valid_thermal_frames')
        elif kind=='rr':
            seconds=[float(x.strip()) for x in file(item['rr_file']).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
            result=cardiovascular.rr_features(seconds,source_verified=item['source_verified'],unresolved_structure_reason=item['unresolved_structure_reason'])
            for row in result['features']:
                rr.append(dict(participant_id=item['participant_id'],timepoint=item['timepoint'],**row))
                if (row['window_variant'],row['quality_variant'])==('full_response','primary_quality'):record(row['feature_id'],row['value'],row['usable'],row['reason'],row.get('original_duration_s',0))
        elif kind=='blood_pressure':
            result=cardiovascular.blood_pressure(file(item['workbook']),file(item['name_mapping_workbook']),item['scope_ids'],data_sheet=item.get('data_sheet','dogspss'),mapping_sheet=item.get('mapping_sheet','Sheet1'))
            for row in result:
                for tp in ('pre','post'):
                    for fid in ('sbp','dbp'):stages.append(dict(participant_id=row['participant_id'],timepoint=tp,feature_id=fid,analysis_value=row[fid+'_'+tp],analysis_usable=True,reason=''))
        else:raise ValueError('Unknown processing type: '+kind)
        write_json(output/f'evidence_{index:04d}.json',result)
    keys=[(r['participant_id'],r['feature_id'],r['timepoint']) for r in stages]
    if len(keys)!=len(set(keys)):raise ValueError('Multiple outputs for the same study stage; resolve the supplied source mapping')
    write_csv(output/'stage_values.csv',stages)
    if rr:write_csv(output/'rr_variants.csv',rr)
    if repeats:write_csv(output/'smile_repeats.csv',repeats)
    return stages

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();process(a.manifest,a.output)
