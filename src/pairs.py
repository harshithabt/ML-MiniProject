import numpy as np
import pandas as pd
from .waveforms import angular_difference

def haversine(lat1,lon1,lat2,lon2):
    p1,p2=np.deg2rad([lat1,lat2]); dp=p2-p1; dl=np.deg2rad(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return float(6371*2*np.arcsin(np.sqrt(a)))

def build_pairs(df, max_km=100):
    rows=[]
    df=df.dropna(subset=["event_id","lat","lon","orientation_deg","magnitude"])
    for event, group in df.groupby("event_id"):
        recs=group.to_dict("records")
        for i,a in enumerate(recs):
            for b in recs[i+1:]:
                if a["rsn"]==b["rsn"]: continue
                km=haversine(a["lat"],a["lon"],b["lat"],b["lon"])
                if km>max_km or km<0.001: continue
                row={"event_id":event,"rsn_1":int(a["rsn"]),"rsn_2":int(b["rsn"]),
                     "station_separation_km":km,"delta_orientation_deg":angular_difference(a["orientation_deg"],b["orientation_deg"]),
                     "magnitude":(a["magnitude"]+b["magnitude"])/2}
                for col in ["rjb_km","vs30","pga_g","pgv_cm_s"]:
                    row[f"mean_{col}"]=np.nanmean([a[col],b[col]]) if not(pd.isna(a[col]) and pd.isna(b[col])) else np.nan
                    row[f"diff_{col}"]=abs(a[col]-b[col]) if pd.notna(a[col]) and pd.notna(b[col]) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)
