import re
import numpy as np
from scipy import signal

def read_at2(path):
    lines = path.read_text(errors="ignore").splitlines()
    header = " ".join(lines[:12])
    match = re.search(r"NPTS\s*=\s*(\d+)\s*,?\s*DT\s*=\s*([0-9.eE+-]+)", header, re.I)
    if not match:
        raise ValueError(f"Missing NPTS/DT: {path.name}")
    n, dt = int(match.group(1)), float(match.group(2))
    start = next(i for i, line in enumerate(lines) if "NPTS" in line.upper() and "DT" in line.upper()) + 1
    tokens = " ".join(lines[start:]).replace("D+", "E+").replace("D-", "E-")
    values = np.fromstring(tokens, sep=" ")
    if dt <= 0 or len(values) < n: raise ValueError(f"Invalid waveform: {path.name}")
    return values[:n], dt

def displacement(acc, dt, period=1., damping=.05):
    w = 2 * np.pi / period
    A = np.array([[0., 1.], [-w*w, -2*damping*w]])
    B = np.array([[0.], [-1.]])
    C = np.array([[1., 0.]])
    D = np.array([[0.]])
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dt, method="zoh")
    b, a = signal.ss2tf(Ad, Bd, Cd, Dd)
    return signal.lfilter(b[0], a, acc)

def orientation(acc1, acc2, dt, az1, az2, period=1., damping=.05):
    n = min(len(acc1), len(acc2))
    u1 = displacement(acc1[:n], dt, period, damping)
    u2 = displacement(acc2[:n], dt, period, damping)
    # Geographic axes, clockwise from north. Valid only for near-orthogonal horizontal sensors.
    theta = np.arange(180, dtype=float)
    c1 = np.cos(np.deg2rad(theta - az1))
    c2 = np.cos(np.deg2rad(theta - az2))
    # Chunk angle calculation to avoid huge temporary arrays.
    peak = np.array([np.max(np.abs(c1[i]*u1 + c2[i]*u2)) for i in range(180)])
    k = int(peak.argmax())
    return float(theta[k]), float((2*np.pi/period)**2 * peak[k])

def angular_difference(a, b):
    d = abs(float(a) - float(b)) % 180
    return min(d, 180-d)
