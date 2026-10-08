import os

LAT = float(os.environ.get('DLR_LAT', '11.758'))
LON_str = os.environ.get('DLR_LON', '')
if not LON_str:
    raise RuntimeError('Set DLR_LON environment variable (longitude of the line).')
LON = float(LON_str)
LINE_AZIMUTH_DEG = float(os.environ.get('DLR_LINE_AZIMUTH_DEG', '90'))
MODEL = os.environ.get('DLR_MODEL', 'ecmwf_ifs025')
START_DATE = os.environ.get('DLR_START_DATE', '2024-02-01')
END_DATE = os.environ.get('DLR_END_DATE', '')
if not END_DATE:
    import datetime as dt
    END_DATE = (dt.date.today() - dt.timedelta(days=7)).isoformat()
LOCAL_UTC_OFFSET = int(os.environ.get('DLR_LOCAL_UTC_OFFSET', '7'))
LEADS = [1, 2, 3]
T_MAX = 75.0
REFRESH = os.environ.get('DLR_REFRESH', 'false').lower() == 'true'
SEED = 42
