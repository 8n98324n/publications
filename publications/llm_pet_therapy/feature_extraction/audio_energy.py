import math
import numpy as np
import soundfile as sf

def energy_scan(path):
    x,sr=sf.read(path,dtype='float32',always_2d=True)
    if not np.all(np.isfinite(x)): raise ValueError('nonfinite audio samples')
    # Fixed first-channel screening; no mixing/normalisation or research extraction.
    mono=x[:,0]; hop=max(1,round(sr*.01)); n=math.ceil(len(mono)/hop)
    pad=np.pad(mono,(0,n*hop-len(mono)))
    frames=pad.reshape(n,hop)
    rms=np.sqrt(np.mean(frames.astype('float64')**2,axis=1))
    db=20*np.log10(np.maximum(rms,1e-12))
    floor=float(np.percentile(db,20)); threshold=float(np.clip(floor+10,-60,-25))
    active=np.flatnonzero(db>=threshold)
    length=len(mono)/sr
    start=max(0,float(active[0])*.01-.1) if len(active) else 0
    end=min(length,float(active[-1]+1)*.01+.1) if len(active) else length
    peak=float(np.max(np.abs(x)))
    clipn=int(np.count_nonzero(np.abs(x)>=.999))
    # Persist every 10ms energy cell for audit; no threshold-based deletion.
    return dict(sr=sr,channels=x.shape[1],samples=len(x),duration=length,peak=peak,
                clipping_fraction=clipn/x.size,db=db,noise_proxy_dbfs=floor,
                candidate_threshold_dbfs=threshold,energy_candidate_start=start,
                energy_candidate_end=end,screen_channel=0)
