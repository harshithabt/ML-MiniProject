import numpy as np
import pandas as pd

def load_metadata(path):
    df = pd.read_excel(path, sheet_name=0)
    cols = {"Record Sequence Number":"rsn", "EQID":"event_id", "Earthquake Name":"event_name",
            "Earthquake Magnitude":"magnitude", "Joyner-Boore Dist. (km)":"rjb_km",
            "Vs30 (m/s) selected for analysis":"vs30", "PGA (g)":"pga_g",
            "PGV (cm/sec)":"pgv_cm_s", "Station Latitude":"lat", "Station Longitude":"lon"}
    missing = set(cols)-set(df.columns)
    if missing: raise ValueError(f"Flatfile missing columns: {sorted(missing)}")
    df = df[list(cols)].rename(columns=cols)
    for c in df.columns:
        if c not in ("event_name",):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] == -999, c] = np.nan
    return df.dropna(subset=["rsn"]).drop_duplicates("rsn")
