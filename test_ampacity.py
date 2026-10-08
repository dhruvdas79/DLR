import numpy as np
from dlr_planb.physics import ampacity

def test_ampacity_regression():
    cases = [
        ((27, 1.5, 150, 0), 919),
        ((27, 1.5, 150, 800), 848),
        ((27, 0.0, 150, 0), 625),
        ((27, 5, 150, 0), 1252),
        ((35, 1, 180, 900), 701),
        ((27, 1.5, 180, 300), 927),
        ((27, 1.5, 90, 300), 635),
        ((27, 0.0, 180, 300), 586),
        ((27, 3, 180, 300), 1116),
        ((27, 3, 90, 300), 743),
    ]
    for args, expected in cases:
        result = ampacity(*args, azimuth=90, Tc=75)
        assert np.isclose(result, expected, atol=1.0), f"Failed for {args}: expected {expected}, got {result}"

def test_ampacity_monotonicity():
    # ampacity decreases when ghi increases
    amp_ghi0 = ampacity(27, 1.5, 150, 0, azimuth=90, Tc=75)
    amp_ghi800 = ampacity(27, 1.5, 150, 800, azimuth=90, Tc=75)
    assert amp_ghi800 < amp_ghi0
    
    # ampacity increases with wind speed
    amp_v1 = ampacity(27, 1.0, 150, 0, azimuth=90, Tc=75)
    amp_v5 = ampacity(27, 5.0, 150, 0, azimuth=90, Tc=75)
    assert amp_v5 > amp_v1
    
    # ampacity is lowest when wind is parallel to the line
    amp_perp = ampacity(27, 1.5, 180, 0, azimuth=90, Tc=75) # 180 - 90 = 90 deg (perpendicular)
    amp_para = ampacity(27, 1.5, 90, 0, azimuth=90, Tc=75) # 90 - 90 = 0 deg (parallel)
    assert amp_para < amp_perp
