#!/usr/bin/env python3
"""
backend.py - FastAPI backend for "Space Trajectory Planner"
Uses DE440s ephemerides (planet barycenters) for trajectory calculations.
"""

# ---------- Imports ----------
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

from astropy.time import Time
from astropy import units as u
from skyfield.api import load
from poliastro.iod import lambert
from poliastro.twobody import Orbit
from poliastro.bodies import Sun

# ---------- Create FastAPI app ----------
app = FastAPI(title="Space Trajectory Backend (DE440s)")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load DE440s ephemerides ----------
ts = load.timescale()
planets = load("de440s.bsp")  # ~120 MB, 1550–2650

# Planet barycenter mapping
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

sun_skyfield = planets["sun"]

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

class TrajectoryRequest(BaseModel):
    origin: str
    target: str
    dep_date: str
    tof_days: float
    n_samples: int = 100

class TrajectoryResult(BaseModel):
    dep_date: str
    tof_days: float
    trajectory_km: List[List[float]]
    origin_pos_km: List[float]
    target_pos_km: List[float]

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
def get_planet_positions():
    """
    Get current heliocentric longitudes for all major planets
    """
    t = ts.now()
    positions = {}
    for name, key in PLANET_MAP.items():
        planet_obj = planets[key]
        heliocentric = sun_skyfield.at(t).observe(planet_obj)
        _, lon, _ = heliocentric.ecliptic_latlon()
        positions[name] = lon.degrees
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
        return BestTransferResult(dep_date="N/A", tof_days=0, dv=0)
    idx = np.nanargmin(dv_array)
    i, j = np.unravel_index(idx, dv_array.shape)
    return BestTransferResult(
        dep_date=result.dep_dates[i],
        tof_days=result.tof_days[j],
        dv=dv_array[i, j]
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

    (v_dep, _), = lambert(Sun.k, r1 * u.km, r2 * u.km, tof * u.s)
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
