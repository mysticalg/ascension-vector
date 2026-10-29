"""Offline beat/section analysis -> beats.json.   Usage: python3 analyse.py [music.wav]
Needs numpy + ffmpeg. Tempo: onset-flux autocorrelation (hop 5.8 ms), refined with a comb
search; then a half/double-tempo test (if the off-beat comb is >=80% of the on-beat comb,
the faster tempo is used)."""
import numpy as np, json, subprocess, sys
SR=11025; src=sys.argv[1] if len(sys.argv)>1 else 'music.wav'
raw=subprocess.run(['ffmpeg','-loglevel','error','-i',src,'-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout
x=np.frombuffer(raw,np.float32); dur=len(x)/SR
N,H=1024,64; fps=SR/H; nf=(len(x)-N)//H
fr=np.lib.stride_tricks.as_strided(x,(nf,N),(x.strides[0]*H,x.strides[0]))
S=np.abs(np.fft.rfft(fr*np.hanning(N),axis=1)).astype(np.float32); f=np.fft.rfftfreq(N,1/SR)
band=lambda a,b:S[:,(f>=a)&(f<b)].sum(1)
bass,mid,high=band(20,150),band(150,2000),band(2000,5500)
L=np.log1p(S*10); fl=np.r_[0,np.maximum(0,np.diff(L,axis=0)).sum(1)]
fl-=np.convolve(fl,np.ones(171)/171,'same'); fl=np.maximum(fl,0)
def comb(bpm):
    p=60*fps/bpm; s=[fl[np.arange(ph,nf-1,p).astype(int)].mean() for ph in np.linspace(0,p,64,endpoint=False)]
    return max(s), np.linspace(0,p,64,endpoint=False)[int(np.argmax(s))]
h=len(fl)//2; ac=np.correlate(fl[:h],fl[:h],'full')[h-1:]
lags=np.arange(int(fps*60/200),int(fps*60/60)); acb=60*fps/lags[int(np.argmax(ac[lags]))]
base=acb/2 if acb>110 else acb; print(f'autocorr peak {acb:.2f} BPM')
cands=np.arange(base-1.5,base+1.5,.01); sc=[comb(b)[0] for b in cands]; slow=cands[int(np.argmax(sc))]
on,ph=comb(slow); p=60*fps/slow
off=fl[np.arange(ph+p/2,nf-1,p).astype(int)].mean()
bpm=slow*2 if off>=0.8*on else slow
print(f'slow {slow:.2f} on {on:.1f} off {off:.1f} -> bpm {bpm:.2f}')
step=60/bpm; t0=(ph/fps)%step
beats=np.arange(t0,dur,step)
k=24; on_=[i for i in range(k,nf-k,4) if fl[i]==fl[i-k:i+k].max() and fl[i]>fl.mean()+2*fl.std()]
def env(v,st=0.25):
    n=int(dur/st); r=np.array([v[int(i*st*fps):int((i+1)*st*fps)].mean() for i in range(n)]); return r/(np.percentile(r,99)+1e-9)
E=env(S.sum(1),0.5); sm=np.convolve(E,np.ones(16)/16,'same')
nov=np.abs(np.convolve(sm,np.r_[np.ones(16),-np.ones(16)],'same')); secs=[]
for c in np.argsort(-nov):
    t=c*0.5
    if 20<t<dur-20 and all(abs(t-s)>40 for s in secs): secs.append(t)
    if len(secs)>=7: break
# snap sections to nearest bar (4 beats)
bar=4*step; secs=sorted(round(t0+round((s-t0)/bar)*bar,3) for s in secs)
r=lambda a:[round(float(v),3) for v in np.clip(a,0,1.5)]
json.dump(dict(duration=round(dur,3),bpm=round(float(bpm),3),beats=[round(float(b),3) for b in beats],
  onsets=[round(i/fps,3) for i in on_],sections=[0]+secs,envStep=0.25,
  energy=r(env(S.sum(1))),bass=r(env(bass)),mid=r(env(mid)),high=r(env(high))),open('beats.json','w'))
print(f'dur {dur:.1f}s beats {len(beats)} onsets {len(on_)} sections {[0]+secs}')
