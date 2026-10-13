import re
from pathlib import Path
import pandas as pd
from .waveforms import read_at2, orientation

# PEER AT2 names normally end with a 3-digit sensor azimuth (e.g., HMC180).
def get_azimuth(name):
    stem = Path(name).stem.upper()
    m = re.search(r"(\d{3})$", stem)
    if not m: return None
    value = int(m.group(1))
    return value if 0 <= value <= 360 else None

def process_folder(folder, period=1., damping=.05):
    groups = {}
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() != ".at2": continue
        m = re.match(r"RSN(\d+)_", path.name, re.I)
        if not m: continue
        az = get_azimuth(path.name)
        if az is not None: groups.setdefault(int(m.group(1)), []).append((path, az))
    rows, issues = [], []
    for rsn, items in groups.items():
        # Find orthogonal component pair, avoid mixing vertical or arbitrary traces.
        choices = []
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                a, b = items[i][1], items[j][1]
                sep = abs(a-b) % 180
                error = abs(sep - 90)
                if error <= 5: choices.append((error, items[i], items[j]))
        if not choices:
            issues.append({"rsn":rsn, "reason":"No azimuth-verified orthogonal pair"}); continue
        _, (f1,a1), (f2,a2) = sorted(choices, key=lambda x:x[0])[0]
        try:
            x, dt1 = read_at2(f1); y, dt2 = read_at2(f2)
            if abs(dt1-dt2)>1e-8: raise ValueError("Sample intervals differ")
            angle, psa = orientation(x,y,dt1,a1,a2,period,damping)
            rows.append({"rsn":rsn,"component_1":f1.name,"component_2":f2.name,
                         "azimuth_1":a1,"azimuth_2":a2,"orientation_deg":angle,
                         "psa_m_s2_if_input_m_s2":psa,"dt":dt1,"npts":min(len(x),len(y))})
        except Exception as e: issues.append({"rsn":rsn,"reason":str(e)})
    return pd.DataFrame(rows), pd.DataFrame(issues)
