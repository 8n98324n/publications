"""Explicit, serial calls to caller-supplied media and acoustic tools."""
from pathlib import Path
import csv,json,os,subprocess
import numpy as np

def execute(argv,*,cwd=None,log=None,env_extra=None):
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):env[k]='1'
    env.update(env_extra or {})
    result=subprocess.run([str(x) for x in argv],cwd=cwd,env=env,check=True,capture_output=True)
    if log:Path(log).write_bytes(result.stdout+b'\n'+result.stderr)
    return result.stdout

def ffmpeg_prefix(executable):
    return [executable,'-hide_banner','-loglevel','error','-n','-filter_threads','1','-filter_complex_threads','1','-threads','1']

def compress_smile(source,output,ffmpeg='ffmpeg'):
    """Input must already be at the correct speed and have its response boundary."""
    return execute(ffmpeg_prefix(ffmpeg)+['-i',source,'-map','0:v:0','-vf','fps=2','-an','-c:v','libx264','-crf','32','-preset','medium','-threads','1',output])

def raw_video_to_openface(source,output_dir,start_physical_s,end_physical_s,*,ffmpeg,ffprobe,openface):
    """Only the original CFR30 container with the documented 2x clock relation.

    Do not call on clips whose playback clock has already been corrected.
    """
    import pandas as pd
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=False);source=Path(source).resolve()
    probe=json.loads(execute([ffprobe,'-v','error','-threads','1','-select_streams','v:0','-show_streams','-show_frames','-show_entries','stream=pix_fmt:frame=best_effort_timestamp_time,pkt_duration_time,duration_time','-of','json',source]))
    if probe['streams'][0]['pix_fmt']!='yuv420p':raise ValueError('Source pixels are not the specified yuv420p format')
    pts=np.array([float(r['best_effort_timestamp_time']) for r in probe['frames']]);physical=2*pts
    if not np.allclose(np.diff(pts),1/30,atol=1.5e-6,rtol=0):raise ValueError('Original video must have contiguous CFR30 timestamps')
    idx=np.flatnonzero((physical>=start_physical_s-1e-8)&(physical<end_physical_s-1e-8))
    if not len(idx) or not np.array_equal(idx,np.arange(idx[0],idx[-1]+1)):raise ValueError('Response frame selection is empty or discontinuous')
    first,last=int(idx[0]),int(idx[-1])+1;n=len(idx);vf=f'trim=start_frame={first}:end_frame={last},setpts=2*(PTS-STARTPTS)'
    output=output_dir/'clip.mkv'
    execute(ffmpeg_prefix(ffmpeg)+['-i',source,'-map','0:v:0','-vf',vf,'-frames:v',n,'-an','-c:v','ffv1','-level','3','-g','1','-pix_fmt','yuv420p','-r','15','-fps_mode','cfr','-threads','1',output])
    def md5(path,filters=None):
        cmd=ffmpeg_prefix(ffmpeg)+['-i',path,'-map','0:v:0']
        if filters:cmd+=['-vf',filters,'-frames:v',n]
        text=execute(cmd+['-an','-pix_fmt','yuv420p','-fps_mode','passthrough','-threads','1','-f','framemd5','-']).decode()
        return [s.split(',')[-1].strip() for s in text.splitlines() if s and not s.startswith('#')]
    if md5(source,vf)!=md5(output):raise ValueError('Decoded pixels were not preserved')
    execute([Path(openface).resolve(),'-f',output,'-out_dir',output_dir/'openface','-of','clip.csv','-au_static','-aus','-2Dfp'],cwd=output_dir,log=output_dir/'openface.log')
    log=(output_dir/'openface.log').read_text(errors='replace')
    if 'main_static_svms.txt' not in log:raise ValueError('Static AU model was not reported loaded')
    frame=pd.read_csv(output_dir/'openface/clip.csv');frame.columns=frame.columns.str.strip()
    if len(frame)!=n or not np.array_equal(frame.frame,np.arange(n)+1):raise ValueError('OpenFace frame count or order changed')
    if not np.allclose(frame.timestamp,physical[idx]-physical[idx[0]],rtol=0,atol=.00051):raise ValueError('OpenFace physical clock mismatch')
    pd.DataFrame(dict(openface_frame=np.arange(n)+1,source_frame_zero_based=idx,source_physical_pts_s=physical[idx],source_container_pts_s=pts[idx])).to_csv(output_dir/'frame_clock.csv',index=False)
    return output_dir/'openface/clip.csv',output_dir/'frame_clock.csv'

def acoustic_segment(source,output_dir,first_sample,last_sample_exclusive,*,opensmile,sample_rate=44100,diagnostic=False):
    import soundfile as sf
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=False)
    with sf.SoundFile(source) as f:
        if f.samplerate!=sample_rate or f.subtype!='PCM_16':raise ValueError('Expected native-rate PCM16 source')
        f.seek(first_sample);samples=f.read(last_sample_exclusive-first_sample,dtype='int16',always_2d=True)
    if len(samples)!=last_sample_exclusive-first_sample:raise ValueError('PCM interval incomplete')
    wav=output_dir/'segment.wav';sf.write(wav,samples[:,0],sample_rate,subtype='PCM_16')
    config=Path(__file__).resolve().parents[2]/'config/acoustic_components.conf'
    command=[opensmile,'-C',config,'-I',wav,'-periodfile',output_dir/'periods.csv']
    for name in ('PitchFinal','PitchBeforeEnergy','Shs','Jitter','Harmonics','Energy'):command+=['-out'+name,output_dir/(name+'.csv')]
    execute(command,cwd=config.parent,log=output_dir/'engine.log',env_extra={'OPENSMILE_JITTER_DIAGNOSTICS_V2':'1' if diagnostic else '0'})
    return output_dir

def vad_cache(source,energy_path,probability_path,model_path,*,ffmpeg='ffmpeg'):
    import onnxruntime as ort
    from .vad import make_session,probabilities
    from .audio_energy import energy_scan
    energy=energy_scan(source)
    starts=np.arange(len(energy['db']))*.01;ends=np.minimum(starts+.01,energy['duration'])
    np.savez_compressed(energy_path,cell_start_sec=starts,cell_end_sec=ends,rms_dbfs=energy['db'])
    raw=execute(ffmpeg_prefix(ffmpeg)+['-i',source,'-vn','-af','pan=mono|c0=c0,aresample=16000','-ac','1','-ar','16000','-c:a','pcm_f32le','-threads','1','-f','f32le','pipe:1'])
    audio=np.frombuffer(raw,dtype='<f4').copy()
    if abs(len(audio)/16000-energy['duration'])>1/16000+1e-9:raise ValueError('VAD resampling changed the source duration')
    prob,padding=probabilities(audio,make_session(ort,model_path),np)
    with Path(probability_path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['start_sec','end_sec','speech_probability','contains_terminal_model_padding'])
        for i,v in enumerate(prob):writer.writerow([i*512/16000,min((i+1)*512/16000,len(audio)/16000),float(v),bool(padding and i==len(prob)-1)])

def prepare_jitter_source(opensmile_source):
    """Copy the provided licensed diagnostic source into a user-owned v3.0.2 tree."""
    import shutil
    root=Path(opensmile_source);target=root/'src/dspcore/pitchJitter.cpp'
    if not target.is_file() or not (root/'CMakeLists.txt').is_file():raise ValueError('Expected openSMILE source checkout')
    backup=target.with_suffix('.cpp.original')
    if backup.exists():raise ValueError('Backup exists; inspect the source tree before applying again')
    shutil.copyfile(target,backup)
    shutil.copyfile(Path(__file__).resolve().parents[2]/'vendor/opensmile/pitchJitter_v3.0.2_diagnostics_v2.cpp',target)
    return target

def thermal_stage(matrix_files,output_dir,*,model_path=None,landmarks_npz=None,manual_exclusions):
    """Thirty ordered Celsius matrices and supplied frame/ROI exclusion records.

    manual_exclusions contains {frame_index: 1..30, roi_key, reason}. An empty
    list is explicit; this function does not infer previously reviewed masks.
    landmarks_npz, when supplied, contains landmarks[30,478,2], scores[30,478].
    """
    from . import thermal
    from ..data_io import write_json,write_csv
    if len(matrix_files)!=30 or not isinstance(manual_exclusions,list):raise ValueError('Thirty selected frames and explicit exclusions required')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    cache=np.load(landmarks_npz,allow_pickle=False) if landmarks_npz else None
    if cache is not None and (cache['landmarks'].shape!=(30,478,2) or cache['scores'].shape!=(30,478)):raise ValueError('Unexpected thermal landmark-cache shape')
    if cache is None and model_path is None:raise ValueError('Provide cached landmarks or the exact model weights')
    model=None if cache is not None else thermal.build_tfan_model(model_path)
    observations=[]
    for i,file in enumerate(matrix_files):
        matrix=thermal.read_temperature_matrix(Path(file))
        observations.append(thermal.infer_frame(model,matrix,cache['landmarks'][i] if cache is not None else None,cache['scores'][i] if cache is not None else None))
    if cache is not None:cache.close()
    thermal.apply_phase_qc(observations)
    for e in manual_exclusions:
        if type(e['frame_index']) is not int or not 1<=e['frame_index']<=30 or e['roi_key'] not in thermal.ROI_ORDER or not e['reason']:raise ValueError('Invalid supplied exclusion')
        roi=observations[e['frame_index']-1].get('rois',{}).get(e['roi_key'])
        if roi:roi['valid']=False;roi['reasons'].append(e['reason'])
    rows=[]
    for i,obs in enumerate(observations,1):
        for key in thermal.ROI_ORDER:
            roi=obs.get('rois',{}).get(key,{})
            rows.append(dict(frame_index=i,roi_key=key,temperature_c=roi.get('temperature_c'),valid=bool(obs['global_valid'] and roi.get('valid',False)),reason=';'.join(obs['global_reasons']+roi.get('reasons',[]))))
    write_csv(output_dir/'roi_frames.csv',rows);write_json(output_dir/'frame_qc.json',observations)
    return output_dir/'roi_frames.csv'

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path)
    args=p.parse_args();items=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    base=args.manifest.resolve().parent
    pathkeys={'source','output','output_dir','energy_path','probability_path','model_path','opensmile_source','landmarks_npz'}
    calls={'compress_smile':compress_smile,'raw_video_to_openface':raw_video_to_openface,'acoustic_segment':acoustic_segment,'vad_cache':vad_cache,'prepare_jitter_source':prepare_jitter_source,'thermal_stage':thermal_stage}
    for spec in items:
        operation=spec['operation'];kwargs=spec['arguments'].copy()
        for key in pathkeys&kwargs.keys():kwargs[key]=base/kwargs[key]
        if 'matrix_files' in kwargs:kwargs['matrix_files']=[base/z for z in kwargs['matrix_files']]
        calls[operation](**kwargs)
if __name__=='__main__':main()
