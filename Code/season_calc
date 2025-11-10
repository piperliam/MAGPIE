
"""
MAGPIE Mars Arrival Season Calculator (v1.0)
--------------------------------------------
Given:
    depart_dt_seed : datetime (UTC)
    tof_days       : float

Outputs:
    arrival_dt (UTC)
    Mars areocentric solar longitude Ls [deg]
    Northern hemisphere season string

Dependencies:
    pip install marstime



This is to support other simulators and calcs

Liam Piper 
2025
"""

import datetime
import marstime  # type: ignore


# ----------------------------------------------------------
# Mars season helpers
# ----------------------------------------------------------

def j2000_offset_days(dt: datetime.datetime) -> float:
    """Convert datetime (UTC) to days since J2000 epoch."""
    j2000_epoch = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    delta = dt - j2000_epoch
    return delta.total_seconds() / 86400.0


def mars_ls(dt: datetime.datetime) -> float:
    """Compute Mars areocentric solar longitude Ls [deg] for given datetime."""
    return float(marstime.Mars_Ls(j2000_offset_days(dt)))


def mars_north_season(ls_deg: float) -> str:
    """Map Ls to northern hemisphere season."""
    ls = ls_deg % 360.0
    if 0 <= ls < 90:
        return "Northern Spring (Ls 0-90°)"
    elif 90 <= ls < 180:
        return "Northern Summer (Ls 90-180°)"
    elif 180 <= ls < 270:
        return "Northern Autumn (Ls 180-270°)"
    else:
        return "Northern Winter (Ls 270-360°)"


def get_arrival_ls_and_season(depart_dt_seed: datetime.datetime, tof_days: float):
    """Compute arrival datetime, Mars Ls, and northern season."""
    arrival_dt = depart_dt_seed + datetime.timedelta(days=tof_days)
    ls_deg = mars_ls(arrival_dt)
    season = mars_north_season(ls_deg)
    return arrival_dt, ls_deg, season


# ----------------------------------------------------------
# Example usage (for MAGPIE scripts) 
# ----------------------------------------------------------

if __name__ == "__main__":
    # Example inputs
    depart_dt_seed = datetime.datetime(2028, 11, 10, tzinfo=datetime.timezone.utc)
    tof_days = 210.0  # days

    arrival_dt, ls_deg, season = get_arrival_ls_and_season(depart_dt_seed, tof_days)

    print("=== Mars Arrival Season Report ===")
    print(f"Departure (UTC): {depart_dt_seed.isoformat()}")
    print(f"Time of Flight : {tof_days:.1f} days")
    print(f"Arrival (UTC)  : {arrival_dt.isoformat()}")
    print(f"Mars Ls        : {ls_deg:7.3f}°")
    print(f"Season (north) : {season}")
