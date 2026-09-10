"""Aggregate corrected-run scalars at matched epochs; run in hlr_env from repo root."""
import json,glob,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out={}
for a in ['a0','a1','a2','a3']:
 p=glob.glob('isaacgymenvs/runs/imitation_corrected_%s_*/summaries/*'%a)[0]
 e=EventAccumulator(p,size_guidance={'scalars':0}).Reload()
 ep=e.Scalars('info/epochs'); steps=np.array([v.step for v in ep]); epochs=np.array([v.value for v in ep])
 tags=[t for t in e.Tags()['scalars'] if t.startswith(('latent_align/','losses/')) or t in ['rewards/iter','episode_lengths/iter']]
 result={'source':p,'max_epoch':max(epochs),'metrics':{}}
 for t in tags:
  vs=e.Scalars(t); xs=np.array([v.step for v in vs]); xs=xs if t.endswith('/iter') else np.interp(xs,steps,epochs); ys=np.array([v.value for v in vs]); bins={}
  for lo,hi in [(500,2500),(6500,7500),(12500,15000),(19000,20000)]:
   sel=ys[(xs>lo)&(xs<=hi)]
   if len(sel):bins[str(hi)]={'mean':float(sel.mean()),'n':len(sel),'min':float(sel.min()),'max':float(sel.max())}
  result['metrics'][t]=bins
 out[a]=result
 print(a, result['max_epoch'],flush=True)
json.dump(out,open('artifacts/imitation_evaluation/curve_audit_20000.json','w'),indent=2,allow_nan=False)
