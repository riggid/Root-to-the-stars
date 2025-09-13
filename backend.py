from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Sun, Earth, Mars, Venus, Mercury, Jupiter, Saturn, Uranus, Neptune
from poliastro.maneuver import Maneuver
# Old version fallback
try:
    from poliastro.iod.lambert import izzo
except ImportError:
    from poliastro.maneuver import lambert as izzo

from poliastro.ephem import Ephem
from poliastro.util import time_range
from poliastro.core.elements import rv2coe
from poliastro import iod
from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.constants import J2000
from poliastro.util import norm

from skyfield.api import load

# Load planetary ephemerides
planets = load("de421.bsp")
ts = load.timescale()

# Map planet names to Skyfield keys
PLANET_MAP = {
    "Mercury": "mercury barycenter",
    "Venus": "venus barycenter",
    "Earth": "earth barycenter",
    "Mars": "mars barycenter",
    "Jupiter": "jupiter barycenter",
    "Saturn": "saturn barycenter",
    "Uranus": "uranus barycenter",
    "Neptune": "neptune barycenter",
}

# Map to poliastro bodies
POLIASTRO_BODIES = {
    "Mercury": Mercury,
    "Venus": Venus,
    "Earth": Earth,
    "Mars": Mars,
    "Jupiter": Jupiter,
    "Saturn": Saturn,
    "Uranus": Uranus,
    "Neptune": Neptune,
}

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ---
class PorkchopRequest(BaseModel):
    origin: str
    target: str
    dep_start: str
    dep_end: Optional[str] = "2050-01-01"   # <-- Default if missing
    n_dep: int = 20
    tof_min: int = 120
    tof_max: int = 400
    n_tof: int = 20

class BestTransferResult(BaseModel):
    dep_date: str
    tof_days: float
    dv: float

class TrajectoryRequest(BaseModel):
    origin: str
    target: str
    dep_date: str
    tof_days: float
    n_samples: int = 50

class TrajectoryResult(BaseModel):
    dep_date: str
    tof_days: float
    trajectory_km: List[List[float]]
    origin_pos_km: List[float]
    target_pos_km: List[float]

# --- Helper functions ---
def get_body_state(body_name: str, t):
    """Return position, velocity vectors for a body at time t (Skyfield → km)."""
    planet = planets[PLANET_MAP[body_name]]
    r, v = planet.at(t).ecliptic_position().km, planet.at(t).ecliptic_velocity().km_per_s
    return np.array(r), np.array(v)

# --- Endpoints ---

@app.get("/api/planet-positions", response_model=Dict[str, float])
def get_planet_positions(date: str = Query(None, description="ISO date (YYYY-MM-DD)")):
    """Return heliocentric longitudes of planets at given date (default now)."""
    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            t = ts.utc(dt.year, dt.month, dt.day)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        t = ts.now()

    positions = {}#!/usr/bin/env python3
"""
backend.py - FastAPI backend for "Space Trajectory Planner"
Uses DE421 ephemerides for trajectory calculations.
"""

# ---------- Imports ----------
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

from astropy.time import Time
from astropy import units as u
from skyfield.api import load
from poliastro.iod import lambert
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.bodies import Sun, Earth as pEarth, Mars as pMars

# ---------- Create FastAPI app ----------
app = FastAPI(title="Space Trajectory Backend")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load ephemerides ----------
ts = load.timescale()
planets = load("de421.bsp")

# Planet mapping
PLANET_MAP = {
    "Mercury": "mercury barycenter",
    "Venus": "venus barycenter",
    "Earth": "earth barycenter",
    "Mars": "mars barycenter",
    "Jupiter": "jupiter barycenter",
    "Saturn": "saturn barycenter",
    "Uranus": "uranus barycenter",
    "Neptune": "neptune barycenter",
    "Pluto": "pluto barycenter",
}

# ---------- Models ----------
class PorkchopRequest(BaseModel):
    origin: str
    target: str
    dep_start: str
    dep_end: str
    n_dep: int = 20
    tof_min: int = 120
    tof_max: int = 400
    n_tof: int = 20

class PorkchopResult(BaseModel):
    dep_dates: List[str]
    tof_days: List[float]
    dv_grid: List[List[float]]

class BestTransferResult(BaseModel):
    dep_date: str
    tof_days: float
    dv: float

class EphemerisResult(BaseModel):
    body: str
    date: str
    position_km: List[float]
    velocity_kms: List[float]

class TrajectoryRequest(BaseModel):
    origin: str
    target: str
    dep_date: str
    tof_days: float
    n_samples: int = 50

class TrajectoryResult(BaseModel):
    dep_date: str
    tof_days: float
    trajectory_km: List[List[float]]
    origin_pos_km: List[float]
    target_pos_km: List[float]

class TransferComparisonResult(BaseModel):
    hohmann: dict
    lambert_best: dict
    lambert_realistic: dict

# ---------- Helpers ----------
def get_rv(body, t_astropy):
    """Return position and velocity (km, km/s) of body at given astropy time"""
    dt = t_astropy.to_datetime(timezone=None)
    ts_sf = ts.utc(dt.year, dt.month, dt.day,
                   dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
    obj = body.at(ts_sf)
    return np.array(obj.position.km), np.array(obj.velocity.km_per_s)

# ---------- Routes ----------
@app.get("/api/planet-positions", response_model=Dict[str, float])
def get_planet_positions(date: str = Query(None, description="ISO date (YYYY-MM-DD) to view planets at")):
    """
    Get heliocentric longitudes for all major planets.
    If `date` is provided, get positions at that date, else use current time.
    """
    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            t = ts.utc(dt.year, dt.month, dt.day)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        t = ts.now()

    positions = {}
    for name, key in PLANET_MAP.items():
        planet_obj = planets[key]
        heliocentric = planets["sun"].at(t).observe(planet_obj)
        _, lon, _ = heliocentric.ecliptic_latlon()
        positions[name] = float(lon.degrees)
    return positions

@app.post("/porkchop", response_model=PorkchopResult)
def porkchop(req: PorkchopRequest):
    origin = planets[PLANET_MAP[req.origin]]
    target = planets[PLANET_MAP[req.target]]

    dep_times = Time(np.linspace(Time(req.dep_start).jd,
                                 Time(req.dep_end).jd, req.n_dep), format="jd")

    tof_days = np.linspace(req.tof_min, req.tof_max, req.n_tof)
    dv_matrix = np.full((req.n_dep, req.n_tof), np.nan)

    for i, dep_t in enumerate(dep_times):
        r1, v1 = get_rv(origin, dep_t)
        for j, tof in enumerate(tof_days * 86400):
            arr_t = Time(dep_t.jd + tof / 86400.0, format="jd", scale="utc")
            r2, v2 = get_rv(target, arr_t)
            try:
                (v_dep, v_arr), = lambert(Sun.k, r1 * u.km, r2 * u.km, tof * u.s)
                dv_dep = np.linalg.norm(v_dep.to(u.km/u.s).value - v1)
                dv_arr = np.linalg.norm(v_arr.to(u.km/u.s).value - v2)
                dv_matrix[i, j] = dv_dep + dv_arr
            except Exception:
                dv_matrix[i, j] = np.nan

    return PorkchopResult(
        dep_dates=[t.utc.iso for t in dep_times],
        tof_days=tof_days.tolist(),
        dv_grid=dv_matrix.tolist()
    )

@app.post("/best_transfer", response_model=BestTransferResult)
def best_transfer(req: PorkchopRequest):
    result = porkchop(req)
    dv_array = np.array(result.dv_grid)
    if np.all(np.isnan(dv_array)):
        return BestTransferResult(dep_date="N/A", tof_days=0.0, dv=0.0)
    idx = np.nanargmin(dv_array)
    i, j = np.unravel_index(idx, dv_array.shape)
    return BestTransferResult(
        dep_date=result.dep_dates[i],
        tof_days=float(result.tof_days[j]),
        dv=float(dv_array[i, j])
    )

@app.post("/trajectory", response_model=TrajectoryResult)
def trajectory(req: TrajectoryRequest):
    origin = planets[PLANET_MAP[req.origin]]
    target = planets[PLANET_MAP[req.target]]

    dep_t = Time(req.dep_date, scale="utc")
    tof = req.tof_days * 86400.0
    arr_t = Time(dep_t.jd + req.tof_days, format="jd", scale="utc")

    r1, v1 = get_rv(origin, dep_t)
    r2, v2 = get_rv(target, arr_t)

    (v_dep, v_arr), = lambert(Sun.k, r1 * u.km, r2 * u.km, tof * u.s)
    orb = Orbit.from_vectors(Sun, r1 * u.km, v_dep)

    times = dep_t + np.linspace(0, req.tof_days, req.n_samples) * u.day
    traj = []
    for t in times:
        r = orb.propagate(t - dep_t).r.to(u.km).value
        traj.append(r.tolist())

    return TrajectoryResult(
        dep_date=req.dep_date,
        tof_days=req.tof_days,
        trajectory_km=traj,
        origin_pos_km=r1.tolist(),
        target_pos_km=r2.tolist()
    )

@app.get("/compare_transfers", response_model=TransferComparisonResult)
def compare_transfers(dep_date: str = "2029-01-01", alt_leo_km: float = 400.0, alt_lmo_km: float = 300.0):
    dep_t = Time(dep_date, scale="utc")

    # Hohmann
    earth_orbit = Orbit.circular(Sun, pEarth.a, epoch=dep_t)
    mars_orbit = Orbit.circular(Sun, pMars.a, epoch=dep_t)
    hohmann = Maneuver.hohmann(earth_orbit, mars_orbit)
    dv_hohmann = sum(m.delta_v.norm().to(u.km/u.s).value for m in hohmann.impulses)
    tof_hohmann = hohmann.get_total_time().to(u.day).value

    # Lambert best
    pork_req = PorkchopRequest(
        origin="Earth",
        target="Mars",
        dep_start=dep_date,
        dep_end="2050-01-01",
        n_dep=20
    )
    best = best_transfer(pork_req)
    dv_lambert = best.dv
    tof_lambert = best.tof_days
    dep_lambert = best.dep_date

    # Realistic Lambert
    R_earth = (pEarth.R + alt_leo_km * u.km).to(u.km).value
    v_circ_leo = np.sqrt(pEarth.k.to(u.km**3/u.s**2).value / R_earth)
    R_mars = (pMars.R + alt_lmo_km * u.km).to(u.km).value
    v_circ_lmo = np.sqrt(pMars.k.to(u.km**3/u.s**2).value / R_mars)
    dv_realistic = dv_lambert + v_circ_leo + v_circ_lmo

    return TransferComparisonResult(
        hohmann={
            "dv_kms": dv_hohmann,
            "tof_days": tof_hohmann,
            "dep_date": dep_date
        },
        lambert_best={
            "dv_kms": dv_lambert,
            "tof_days": tof_lambert,
            "dep_date": dep_lambert
        },
        lambert_realistic={
            "dv_kms": dv_realistic,
            "tof_days": tof_lambert,
            "dep_date": dep_lambert,
            "leo_alt_km": alt_leo_km,
            "lmo_alt_km": alt_lmo_km
        }
    )

    for name, key in PLANET_MAP.items():
        planet_obj = planets[key]
        heliocentric = planets["sun"].at(t).observe(planet_obj)
        _, lon, _ = heliocentric.ecliptic_latlon()
        positions[name] = float(lon.degrees)
    return positions

@app.post("/best_transfer", response_model=BestTransferResult)
def best_transfer(req: PorkchopRequest):
    """Compute the best transfer window with lowest Δv."""
    try:
        dep_start = Time(req.dep_start)
        dep_end = Time(req.dep_end) if req.dep_end else Time("2050-01-01")

        dep_times = Time(np.linspace(dep_start.jd, dep_end.jd, req.n_dep), format="jd")
        tof_days = np.linspace(req.tof_min, req.tof_max, req.n_tof)

        best_dv = 1e9
        best_dep = None
        best_tof = None

        for dep in dep_times:
            for tof in tof_days:
                arr = dep + tof * u.day
                try:
                    r1, v1 = get_body_state(req.origin, ts.tt_jd(dep.jd))
                    r2, v2 = get_body_state(req.target, ts.tt_jd(arr.jd))
                    (v1_lam, v2_lam), = izzo(Sun.k, r1 * u.km, r2 * u.km, tof * u.day)
                    dv = (np.linalg.norm((v1_lam - v1 * u.km/u.s).to(u.km/u.s).value) +
                          np.linalg.norm((v2_lam - v2 * u.km/u.s).to(u.km/u.s).value))
                    if dv < best_dv:
                        best_dv = dv
                        best_dep = dep
                        best_tof = tof
                except Exception:
                    continue

        if best_dep is None:
            raise HTTPException(status_code=500, detail="No valid transfer found")

        return BestTransferResult(
            dep_date=best_dep.utc.iso.split()[0],
            tof_days=float(best_tof),
            dv=float(best_dv),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transfer computation failed: {str(e)}")

@app.post("/trajectory", response_model=TrajectoryResult)
def trajectory(req: TrajectoryRequest):
    """Generate a transfer trajectory between two planets."""
    try:
        dep_time = Time(req.dep_date)
        arr_time = dep_time + req.tof_days * u.day

        r1, v1 = get_body_state(req.origin, ts.tt_jd(dep_time.jd))
        r2, v2 = get_body_state(req.target, ts.tt_jd(arr_time.jd))

        (v1_lam, v2_lam), = izzo(Sun.k, r1 * u.km, r2 * u.km, req.tof_days * u.day)

        # Sample trajectory
        times = np.linspace(dep_time.jd, arr_time.jd, req.n_samples)
        traj_points = []
        for jd in times:
            frac = (jd - dep_time.jd) / (arr_time.jd - dep_time.jd)
            pos = r1 + frac * (r2 - r1)
            traj_points.append(pos.tolist())

        return TrajectoryResult(
            dep_date=req.dep_date,
            tof_days=req.tof_days,
            trajectory_km=traj_points,
            origin_pos_km=r1.tolist(),
            target_pos_km=r2.tolist(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trajectory computation failed: {str(e)}")

# --- Run with: uvicorn backend:app --reload ---
