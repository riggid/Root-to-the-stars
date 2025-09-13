#!/usr/bin/env python3
"""
Enhanced backend.py - FastAPI backend for "Advanced Space Mission Planner"
Includes spacecraft analysis, fuel calculations, and mission feasibility assessment
"""
import os
import json
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

from astropy.time import Time
from astropy import units as u
from skyfield.api import load
from poliastro.iod import lambert
from poliastro.twobody import Orbit
from poliastro.bodies import Sun

# ---------- Create FastAPI app ----------
app = FastAPI(title="Advanced Space Mission Backend", version="2.0.1") # Version bump

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load ephemerides ----------
ts = load.timescale()

planets = load(os.path.join("data/de421.bsp"))

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
# ✨ FIX: Renamed variable and added Uranus and Neptune orbital periods
ORBITAL_PERIODS_DAYS = {
    "Mercury": 88, "Venus": 225, "Earth": 365, "Mars": 687,
    "Jupiter": 4333, "Saturn": 10759, "Uranus": 30687, "Neptune": 60190,
}

# ---------- Enhanced Models ----------
class PorkchopRequest(BaseModel):
    origin: str
    target: str
    dep_start: str
    dep_end: str = "2050-01-01"
    n_dep: int = 20
    tof_min: int = 120
    tof_max: int = 400
    n_tof: int = 20

class SpacecraftSpecs(BaseModel):
    name: str
    dry_mass_kg: float
    fuel_capacity_kg: float
    specific_impulse_s: float
    thrust_n: float
    payload_capacity_kg: float
    cost_per_launch_usd: float = 0
    reusable: bool = False

class MissionRequest(BaseModel):
    origin: str
    target: str
    dep_date: str
    tof_days: float
    spacecraft_name: str
    payload_mass_kg: float = 10000
    n_samples: int = 50

class FuelAnalysis(BaseModel):
    required_fuel_kg: float
    fuel_efficiency_percent: float
    mass_ratio: float
    feasible: bool
    fuel_margin_percent: float

class CostAnalysis(BaseModel):
    launch_cost_usd: float
    fuel_cost_usd: float
    total_mission_cost_usd: float
    cost_per_kg_payload: float

class MissionAnalysis(BaseModel):
    trajectory: dict
    fuel_analysis: FuelAnalysis
    cost_analysis: CostAnalysis
    spacecraft_utilization: dict
    recommendations: List[str]

class SpacecraftComparison(BaseModel):
    spacecraft_options: List[dict]
    recommended_spacecraft: str
    comparison_metrics: dict

# ---------- Load Spacecraft Database from JSON ----------
def load_spacecraft_db(filepath: str = "data/spacecraft.json") -> dict:
    
    """Loads the spacecraft database from a JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # This provides a more helpful error if the file is missing
        raise RuntimeError(f"FATAL: The spacecraft database file was not found at '{filepath}'. Please make sure it exists.")
    except json.JSONDecodeError:
        raise RuntimeError(f"FATAL: The file '{filepath}' contains invalid JSON.")

# This line calls the function and creates the database variable
SPACECRAFT_DATABASE = load_spacecraft_db()

# ---------- Helpers ----------
def get_rv(body, t_astropy):
    t_skyfield = ts.from_astropy(t_astropy)
    obj = body.at(t_skyfield)
    return np.array(obj.position.km), np.array(obj.velocity.km_per_s)

# ✨ FIX 3: Centralized trajectory calculation logic into a helper function
def _calculate_lambert_transfer(origin_name: str, target_name: str, dep_date_str: str, tof_days: float):
    """Calculates a Lambert transfer and returns key trajectory parameters."""
    try:
        origin_body = planets[PLANET_MAP[origin_name]]
        target_body = planets[PLANET_MAP[target_name]]

        dep_t = Time(dep_date_str, scale="utc")
        tof_s = tof_days * 86400.0
        arr_t = Time(dep_t.jd + tof_days, format="jd", scale="utc")

        r1, v1 = get_rv(origin_body, dep_t)
        r2, v2 = get_rv(target_body, arr_t)

        (v_dep, v_arr), = lambert(Sun.k, r1 * u.km, r2 * u.km, tof_s * u.s)
        
        dv_dep = np.linalg.norm(v_dep.to(u.km/u.s).value - v1)
        dv_arr = np.linalg.norm(v_arr.to(u.km/u.s).value - v2)
        total_dv_kms = dv_dep + dv_arr

        return {
            "r1": r1, "v1": v1, "r2": r2, "v2": v2,
            "v_dep": v_dep, "v_arr": v_arr, "total_dv_kms": total_dv_kms,
            "dep_t": dep_t
        }
    except Exception as e:
        # Raise a specific exception that can be caught by endpoints
        raise ValueError(f"Lambert calculation failed: {e}")

def calculate_fuel_requirements(delta_v_ms, spacecraft_specs, payload_kg):
    g0 = 9.81
    dry_mass, fuel_capacity = spacecraft_specs["dry_mass_kg"], spacecraft_specs["fuel_capacity_kg"]
    isp, max_payload = spacecraft_specs["specific_impulse_s"], spacecraft_specs["payload_capacity_kg"]
    
    ve = isp * g0
    m_initial = dry_mass + payload_kg
    mass_ratio = np.exp(delta_v_ms / ve)
    required_fuel = m_initial * (mass_ratio - 1)
    
    feasible = bool(required_fuel <= fuel_capacity and payload_kg <= max_payload)
    fuel_efficiency = (delta_v_ms * payload_kg) / (required_fuel * ve) * 100 if required_fuel > 0 else 0
    fuel_margin = ((fuel_capacity - required_fuel) / fuel_capacity) * 100 if feasible else 0
    
    return {
        "required_fuel_kg": float(required_fuel), "fuel_efficiency_percent": float(fuel_efficiency),
        "mass_ratio": float(mass_ratio), "feasible": feasible, "fuel_margin_percent": float(fuel_margin)
    }

def calculate_mission_cost(spacecraft_specs, fuel_analysis, payload_kg):
    launch_cost = spacecraft_specs["cost_per_launch_usd"]
    fuel_cost = fuel_analysis["required_fuel_kg"] * 1.0 # Simplified cost
    total_cost = launch_cost + fuel_cost
    cost_per_kg = total_cost / payload_kg if payload_kg > 0 else 0
    
    return {
        "launch_cost_usd": float(launch_cost), "fuel_cost_usd": float(fuel_cost),
        "total_mission_cost_usd": float(total_cost), "cost_per_kg_payload": float(cost_per_kg)
    }

def get_mission_recommendations(fuel_analysis, cost_analysis, spacecraft_specs):
    recs = []
    if not fuel_analysis["feasible"]: recs.append("⚠️ Mission not feasible: Consider reducing payload or using a more powerful vehicle.")
    if fuel_analysis["fuel_margin_percent"] < 10 and fuel_analysis["feasible"]: recs.append("⚠️ Low fuel margin: Consider a 10-15% safety buffer.")
    if fuel_analysis["fuel_efficiency_percent"] < 20: recs.append("💡 Low fuel efficiency: Consider optimizing trajectory or propulsion.")
    if cost_analysis["cost_per_kg_payload"] > 10000: recs.append("💰 High cost per kg: Consider bulk payload or a reusable vehicle.")
    if spacecraft_specs["reusable"]: recs.append("♻️ Reusable vehicle provides significant cost savings on multiple missions.")
    if fuel_analysis["fuel_margin_percent"] > 30: recs.append("✅ Excellent fuel margin provides a high safety buffer.")
    return recs if recs else ["✅ Mission profile looks optimal!"]


# ---------- Routes ----------
@app.get("/")
def root(): return {"message": "Advanced Space Mission Planner API v2.0"}

@app.get("/api/spacecraft", response_model=Dict[str, dict])
def get_spacecraft(): return SPACECRAFT_DATABASE

@app.get("/api/spacecraft/{spacecraft_name}")
def get_spacecraft_specs(spacecraft_name: str):
    if spacecraft_name not in SPACECRAFT_DATABASE:
        raise HTTPException(status_code=404, detail="Spacecraft not found")
    return SPACECRAFT_DATABASE[spacecraft_name]

@app.get("/api/planet-positions", response_model=Dict[str, list])
def get_planet_positions(date: str = Query(None, description="ISO date (YYYY-MM-DD) to view planets at")):
    """Get heliocentric 3D positions (x,y,z in km) for all major planets""" # Updated docstring
    t = ts.now()
    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            t = ts.utc(dt.year, dt.month, dt.day)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    positions = {}
    for name, key in PLANET_MAP.items():
        if name == "Pluto": continue
        planet_obj = planets[key]
        
        # ✨ FIX: Get the full 3D position vector instead of just the longitude
        position_vector = planets["sun"].at(t).observe(planet_obj).position
        positions[name] = position_vector.km.tolist() # Returns [x, y, z] in km
    
    return positions

@app.get("/api/orbit-path/{planet_name}", response_model=List[List[float]])
@app.get("/api/orbit-path/{planet_name}", response_model=List[List[float]])
def get_orbit_path(planet_name: str, samples: int = 200):
    """Calculates a list of 3D coordinates representing a planet's orbit."""
    if planet_name not in PLANET_MAP or planet_name not in ORBITAL_PERIODS_DAYS:
        raise HTTPException(status_code=404, detail="Planet not found")

    period = ORBITAL_PERIODS_DAYS[planet_name]
    planet_body = planets[PLANET_MAP[planet_name]]

    # ✨ FIX: Use astropy.time.Time for calculations to ensure unit compatibility
    t0 = Time.now()
    # This line now works correctly as it's a pure astropy operation
    times = t0 + np.linspace(0, period, samples) * u.day

    positions = []
    for t in times:
        # Convert from astropy time to skyfield time just before getting position
        t_skyfield = ts.from_astropy(t)
        position_vector = planets["sun"].at(t_skyfield).observe(planet_body).position
        positions.append(position_vector.km.tolist())
    
    return positions

@app.post("/porkchop")
def porkchop(req: PorkchopRequest):
    origin = planets[PLANET_MAP[req.origin]]
    target = planets[PLANET_MAP[req.target]]
    dep_times = Time(np.linspace(Time(req.dep_start).jd, Time(req.dep_end).jd, req.n_dep), format="jd")
    tof_days = np.linspace(req.tof_min, req.tof_max, req.n_tof)
    dv_matrix = np.full((req.n_dep, req.n_tof), np.nan)

    for i, dep_t in enumerate(dep_times):
        r1, v1 = get_rv(origin, dep_t)
        for j, tof in enumerate(tof_days):
            try:
                arr_t = Time(dep_t.jd + tof, format="jd", scale="utc")
                r2, v2 = get_rv(target, arr_t)
                (v_dep, v_arr), = lambert(Sun.k, r1 * u.km, r2 * u.km, tof * u.day)
                dv_dep = np.linalg.norm(v_dep.to(u.km/u.s).value - v1)
                dv_arr = np.linalg.norm(v_arr.to(u.km/u.s).value - v2)
                dv_matrix[i, j] = dv_dep + dv_arr
            except (ValueError, RuntimeError):
                dv_matrix[i, j] = np.nan
    
    return {"dep_dates": [t.utc.iso for t in dep_times], "tof_days": tof_days.tolist(), "dv_grid": dv_matrix.tolist()}

@app.post("/best_transfer")
def best_transfer(req: PorkchopRequest):
    result = porkchop(req)
    dv_array = np.array(result["dv_grid"])
    
    if np.all(np.isnan(dv_array)):
        raise HTTPException(status_code=400, detail="No valid transfers found for the given date and TOF range.")
    
    idx = np.nanargmin(dv_array)
    i, j = np.unravel_index(idx, dv_array.shape)
    
    return {"dep_date": result["dep_dates"][i], "tof_days": float(result["tof_days"][j]), "dv": float(dv_array[i, j])}

@app.post("/trajectory", response_model=MissionAnalysis)
def trajectory(req: MissionRequest):
    try:
        transfer_data = _calculate_lambert_transfer(req.origin, req.target, req.dep_date, req.tof_days)
        
        orb = Orbit.from_vectors(Sun, transfer_data["r1"] * u.km, transfer_data["v_dep"])
        times = transfer_data["dep_t"] + np.linspace(0, req.tof_days, req.n_samples) * u.day
        traj_km = [orb.propagate(t - transfer_data["dep_t"]).r.to(u.km).value.tolist() for t in times]

        if req.spacecraft_name not in SPACECRAFT_DATABASE:
            raise HTTPException(status_code=400, detail="Unknown spacecraft")
        
        specs = SPACECRAFT_DATABASE[req.spacecraft_name]
        
        fuel_analysis = calculate_fuel_requirements(transfer_data["total_dv_kms"] * 1000, specs, req.payload_mass_kg)
        cost_analysis = calculate_mission_cost(specs, fuel_analysis, req.payload_mass_kg)
        recommendations = get_mission_recommendations(fuel_analysis, cost_analysis, specs)
        
        payload_util = (req.payload_mass_kg / specs["payload_capacity_kg"]) * 100 if specs["payload_capacity_kg"] > 0 else 0
        fuel_util = (fuel_analysis["required_fuel_kg"] / specs["fuel_capacity_kg"]) * 100 if specs["fuel_capacity_kg"] > 0 else 0
        
        spacecraft_util = {
            "payload_utilization_percent": float(payload_util),
            "fuel_utilization_percent": float(fuel_util),
            "thrust_to_weight_ratio": float(specs["thrust_n"] / (specs["dry_mass_kg"] * 9.81))
        }

        return MissionAnalysis(
            trajectory={
                "dep_date": req.dep_date, "tof_days": req.tof_days, "trajectory_km": traj_km,
                "origin_pos_km": transfer_data["r1"].tolist(), "target_pos_km": transfer_data["r2"].tolist(),
                "total_dv_kms": float(transfer_data["total_dv_kms"])
            },
            fuel_analysis=fuel_analysis, cost_analysis=cost_analysis,
            spacecraft_utilization=spacecraft_util, recommendations=recommendations
        )
        
    except (ValueError, HTTPException) as e:
        raise HTTPException(status_code=500, detail=f"Mission analysis failed: {str(e)}")

@app.post("/compare_spacecraft", response_model=SpacecraftComparison)
def compare_spacecraft(origin: str, target: str, dep_date: str, tof_days: float, payload_kg: float = 10000):
    try:
        # ✨ FIX 4: Use the internal helper function directly for efficiency
        transfer_data = _calculate_lambert_transfer(origin, target, dep_date, tof_days)
        delta_v_kms = transfer_data["total_dv_kms"]
        
        spacecraft_analysis = []
        for name, specs in SPACECRAFT_DATABASE.items():
            fuel_analysis = calculate_fuel_requirements(delta_v_kms * 1000, specs, payload_kg)
            cost_analysis = calculate_mission_cost(specs, fuel_analysis, payload_kg)
            
            feasibility = 100 if fuel_analysis["feasible"] else 0
            efficiency = min(fuel_analysis["fuel_efficiency_percent"], 100)
            cost = max(0, 100 - (cost_analysis["cost_per_kg_payload"] / 1000))
            margin = min(fuel_analysis["fuel_margin_percent"], 100)
            
            score = (feasibility * 0.4 + efficiency * 0.2 + cost * 0.2 + margin * 0.2)
            
            spacecraft_analysis.append({
                "name": name, "feasible": fuel_analysis["feasible"],
                "fuel_efficiency": fuel_analysis["fuel_efficiency_percent"],
                "total_cost": cost_analysis["total_mission_cost_usd"],
                "cost_per_kg": cost_analysis["cost_per_kg_payload"],
                "fuel_margin": fuel_analysis["fuel_margin_percent"],
                "overall_score": score, "reusable": specs["reusable"]
            })
        
        spacecraft_analysis.sort(key=lambda x: x["overall_score"], reverse=True)
        recommended = next((s for s in spacecraft_analysis if s["feasible"]), spacecraft_analysis[0])
        
        return SpacecraftComparison(
            spacecraft_options=spacecraft_analysis,
            recommended_spacecraft=recommended["name"],
            comparison_metrics={
                "delta_v_requirement_kms": delta_v_kms,
                "payload_mass_kg": payload_kg,
                "mission_duration_days": tof_days
            }
        )
    except (ValueError, HTTPException) as e:
        raise HTTPException(status_code=500, detail=f"Spacecraft comparison failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)