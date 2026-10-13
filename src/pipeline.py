"""Reproducible exploratory ground-motion directionality pipeline.
Target: absolute geographic orientation of maximum 5%-damped SDOF pseudo-acceleration at T=1s.
This is an implementation choice, not a verified reproduction of the source paper.
"""
import argparse, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COLS = {'Record Sequence Number':'rsn', 'Earthquake Name':'event',
        'Earthquake Magnitude':'magnitude', 'Joyner-Boore Dist. (km)':'rjb_km',
        'Vs30 (m/s) selected for analysis':'vs30', 'PGA (g)':'pga', 'PGV (cm/sec)':'pgv',
        'Station Latitude':'lat', 'Station Longitude':'lon'}

def read_at2(path):
    lines = Path(path).read_text(errors='replace').splitlines()
    header = ' '.join(lines[:8]); m = re.search(r'NPTS\s*=\s*(\d+)\s*,?\s*DT\s*=\s*([\d.Ee+-]+)', header, re.I)
    if not m:
        n = re.search(r'NPTS\s*=\s*(\d+)', header, re.I)
        d = re.search(r'DT\s*=\s*([\d.Ee+-]+)', header, re.I)
        if not (n and d): raise ValueError('AT2 header missing NPTS or DT')
        npts, dt = int(n.group(1)), float(d.group(1))
    else: npts, dt = int(m.group(1)), float(m.group(2))
    start = next((i+1 for i,line in enumerate(lines) if 'NPTS' in line.upper() and 'DT' in line.upper()), 4)
    vals = np.fromstring(' '.join(lines[start:]).replace('D+','E+').replace('D-','E-'), sep=' ')
    if len(vals)<npts or dt<=0: raise ValueError('Invalid acceleration series')
    return vals[:npts],dt

def parse_name(p):
    m=re.match(r'RSN(\d+)_',p.name,re.I)
    if not m:return None
    # NGA-West2 filenames end in channel azimuth (000-359); non-numeric endings are ignored.
    a=re.search(r'(\d{3})(?:\.AT2)$',p.name,re.I)
    if not a:return None
    deg=int(a.group(1))
    return (int(m.group(1)),deg) if deg<360 else None

def sdof(acc,dt,period=1.,damping=.05):
    w=2*np.pi/period
    A=np.array([[0,1],[-w*w,-2*damping*w]])
    B=np.array([[0],[-1.]])
    Ad,Bd,Cd,Dd,_=signal.cont2discrete((A,B,np.array([[1.,0.]]),np.array([[0.]])),dt)
    b,a=signal.ss2tf(Ad,Bd,Cd,Dd)
    return signal.lfilter(b[0],a,acc)

def orientations(folder,period=1.,step=1):
    byrsn={}
    for p in Path(folder).rglob('*'):
        if p.suffix.lower()!='.at2':continue
        info=parse_name(p)
        if info:byrsn.setdefault(info[0],[]).append((info[1],p))
    rows=[]; failures=[]
    for rsn,entries in sorted(byrsn.items()):
        choices=[(abs(abs((a-b+180)%360-180)-90),a,b,p,q)
                 for i,(a,p) in enumerate(entries) for b,q in entries[i+1:]]
        if not choices:continue
        _,az1,az2,p1,p2=min(choices,key=lambda x:x[0])
        separation=abs((az1-az2+180)%360-180)
        if abs(separation-90)>15:continue
        try:
            a,dt=read_at2(p1); b,dt2=read_at2(p2)
            if abs(dt-dt2)>1e-8:raise ValueError('Component sample intervals differ')
            n=min(len(a),len(b)); u1=sdof(a[:n],dt,period);u2=sdof(b[:n],dt,period)
            # Earth-fixed orientation, using each channel's compass azimuth.
            th=np.deg2rad(np.arange(0,180,step)); a1=np.deg2rad(az1);a2=np.deg2rad(az2)
            # Horizontal components are orthogonal; opposite channel sign is handled by azimuth.
            response=np.max(np.abs(np.cos(th-a1)[:,None]*u1+np.cos(th-a2)[:,None]*u2),axis=1)
            k=int(np.argmax(response))
            rows.append(dict(rsn=rsn,orientation_deg=float(np.rad2deg(th[k])),psa_m_s2=float((2*np.pi/period)**2*response[k]),component_1=p1.name,component_2=p2.name,azimuth_1=az1,azimuth_2=az2,dt=dt))
        except Exception as exc:failures.append({'rsn':rsn,'error':str(exc)})
    return pd.DataFrame(rows),failures,len(byrsn)

def metadata(path):
    df=pd.read_excel(path,engine='openpyxl',usecols=lambda c:c in COLS)
    df=df.rename(columns=COLS)
    if 'rsn' not in df:raise ValueError('Flatfile missing Record Sequence Number')
    df=df.drop_duplicates('rsn').copy()
    for c in ['rsn','magnitude','rjb_km','vs30','pga','pgv','lat','lon']:
        if c in df:
            df[c]=pd.to_numeric(df[c],errors='coerce');df.loc[df[c]<=-999,c]=np.nan
    return df

def haversine(lat1,lon1,lat2,lon2):
    x1,x2=np.radians([lat1,lat2]);dl=x2-x1;do=np.radians(lon2-lon1)
    a=np.sin(dl/2)**2+np.cos(x1)*np.cos(x2)*np.sin(do/2)**2
    return 6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))

def pairs(df,max_km=100,cap=200):
    out=[]
    required=['event','lat','lon','orientation_deg']
    d=df.dropna(subset=required)
    for event,g in d.groupby('event'):
        if str(event).strip() in ('','-999'):continue
        g=g.sort_values('rsn').reset_index(drop=True);count=0
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                x,y=g.iloc[i],g.iloc[j]
                dist=haversine(x.lat,x.lon,y.lat,y.lon)
                if dist>max_km:continue
                diff=abs(x.orientation_deg-y.orientation_deg)%180
                row={'event':event,'rsn1':int(x.rsn),'rsn2':int(y.rsn),'station_distance_km':float(dist),
                     'angle_difference_deg':float(min(diff,180-diff))}
                for c in ['magnitude','rjb_km','vs30','pga','pgv']:
                    row['mean_'+c]=float(np.nanmean([x.get(c,np.nan),y.get(c,np.nan)])) if not (pd.isna(x.get(c,np.nan)) and pd.isna(y.get(c,np.nan))) else np.nan
                    row['absdiff_'+c]=float(abs(x.get(c,np.nan)-y.get(c,np.nan)))
                out.append(row);count+=1
                if count>=cap:break
            if count>=cap:break
    return pd.DataFrame(out)

def evaluate(df,out,seed=42):
    cols=[c for c in df if c.startswith('mean_') or c.startswith('absdiff_')]+['station_distance_km']
    if df.empty or df.event.nunique()<5 or len(df)<30:
        return {'status':'insufficient data for credible event-held-out evaluation','pairs':len(df),'events':int(df.event.nunique()) if not df.empty else 0}
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed)
    tr,te=next(split.split(df,groups=df.event))
    xtr,xte=df.iloc[tr][cols],df.iloc[te][cols]; ytr,yte=df.iloc[tr].angle_difference_deg,df.iloc[te].angle_difference_deg
    models={'constant_median':None,'ridge':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=10)),
            'mlp':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,16),alpha=10,early_stopping=False,max_iter=1000,random_state=seed))}
    results={}
    for name,model in models.items():
        try:
            if model is None:pred=np.full(len(yte),ytr.median())
            else:model.fit(xtr,ytr);pred=model.predict(xte)
            pred=np.clip(pred,0,90)
            results[name]={'mae_deg':float(mean_absolute_error(yte,pred)), 'rmse_deg':float(np.sqrt(mean_squared_error(yte,pred))), 'r2':float(r2_score(yte,pred)) if len(yte)>1 else None}
        except Exception as exc:results[name]={'error':str(exc)}
    return {'status':'evaluated','train_pairs':len(tr),'test_pairs':len(te),'train_events':int(df.iloc[tr].event.nunique()),'test_events':int(df.iloc[te].event.nunique()),'features':cols,'metrics':results}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--at2-dir',required=True);ap.add_argument('--flatfile',required=True);ap.add_argument('--out',default='outputs');ap.add_argument('--period',type=float,default=1.);ap.add_argument('--radius',type=float,default=100.);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    orient,fail,nrsn=orientations(args.at2_dir,args.period)
    if orient.empty:raise RuntimeError('No usable horizontal AT2 pairs. Check filenames and data path.')
    orient.to_csv(out/'orientations.csv',index=False)
    meta=metadata(args.flatfile);joined=orient.merge(meta,on='rsn',how='left')
    joined.to_csv(out/'records_with_metadata.csv',index=False)
    paired=pairs(joined,args.radius);paired.to_csv(out/'same_event_pairs.csv',index=False)
    close=pairs(joined,5);close.to_csv(out/'same_event_pairs_within_5km.csv',index=False)
    result=evaluate(paired,out)
    summary={'definition':'earth-fixed max 5%-damped SDOF PSA orientation, 0-179 deg; interstation axial angular difference 0-90 deg',
             'period_s':args.period,'radius_km':args.radius,'at2_rsns_with_numeric_azimuths':nrsn,'usable_records':len(orient),
             'metadata_matched':int(joined.event.notna().sum()) if 'event' in joined else 0,
             'pairs_within_5km':len(close),'pairs_within_radius':len(paired),'events_within_radius':int(paired.event.nunique()) if not paired.empty else 0,
             'failed_waveforms':fail,'evaluation':result,
             'limitations':['Small non-random PEER RSN subset','1-second period is a chosen setting; verify against assigned paper','Not an exact paper reproduction','AT2 filename azimuth convention should be checked against metadata','Pairs from same event are dependent; event-held-out split used']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('failed_waveforms','limitations')},indent=2))
    print('Saved:',out.resolve())
if __name__=='__main__':main()
