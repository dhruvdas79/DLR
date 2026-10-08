import numpy as np

D, EMIS, ABSORP = 27.5e-3, 0.7, 0.8
ALT, R_AC, SIGMA = 32.5, 1.4085e-4, 5.670374419e-8
SURF = np.pi * D

def ampacity(Ta, V, wdir, ghi, azimuth=None, Tc=None):
    """IEEE 738 steady-state ampacity (vectorised). Copied verbatim from the source notebook."""
    if azimuth is None:
        azimuth = 90.0  # default for standalone use
    if Tc is None:
        Tc = 75.0
    Ta = np.asarray(Ta, float); V = np.clip(np.asarray(V, float), 0, None); ghi = np.clip(np.asarray(ghi, float), 0, None)
    delta = np.mod(np.asarray(wdir, float) - azimuth, 180.0)
    phi = np.radians(np.minimum(delta, 180.0 - delta))    # angle between wind and line axis, 0..90 deg
    Tf = (Ta + Tc) / 2
    mu = 1.458e-6 * (Tf + 273.0) ** 1.5 / (Tf + 383.4)
    rho = (1.293 - 1.525e-4 * ALT + 6.379e-9 * ALT ** 2) / (1 + 0.00367 * Tf)
    k = 2.424e-2 + 7.477e-5 * Tf - 4.407e-9 * Tf ** 2
    Re = D * rho * V / mu
    K = 1.194 - np.cos(phi) + 0.194 * np.cos(2 * phi) + 0.368 * np.sin(2 * phi)
    dT = Tc - Ta
    q_nat = 3.645 * np.sqrt(rho) * D ** 0.75 * np.clip(dT, 0, None) ** 1.25
    q_low = K * (1.01 + 1.35 * Re ** 0.52) * k * dT
    q_high = K * 0.754 * Re ** 0.6 * k * dT
    qc = np.maximum.reduce([q_nat, q_low, q_high])
    qr = SIGMA * EMIS * SURF * ((Tc + 273.15) ** 4 - (Ta + 273.15) ** 4)
    qs = ABSORP * ghi * D
    return np.sqrt(np.clip(qc + qr - qs, 0, None) / R_AC)
