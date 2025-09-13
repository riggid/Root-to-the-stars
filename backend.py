#!/usr/bin/env python3
"""
Enhanced backend.py - FastAPI backend for "Advanced Space Mission Planner"
Includes spacecraft analysis, fuel calculations, and mission feasibility assessment
"""

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
app = FastAPI(title="Advanced Space Mission Backend", version="2.0.0")

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

# ---------- Spacecraft Database ----------
SPACECRAFT_DATABASE = {
    "Falcon 9": {
        "dry_mass_kg": 25600,
        "fuel_capacity_kg": 411000,
        "specific_impulse_s": 311,
        "thrust_n": 7607000,
        "payload_capacity_kg": 22800,
        "cost_per_launch_usd": 62000000,
        "reusable": True
    },
    "Falcon Heavy": {
        "dry_mass_kg": 64000,
        "fuel_capacity_kg": 1400000,
        "specific_impulse_s": 311,
        "thrust_n": 22819000,
        "payload_capacity_kg": 63800,
        "cost_per_launch_usd": 90000000,
        "reusable": True
    },
    "Saturn V": {
        "dry_mass_kg": 130000,
        "fuel_capacity_kg": 2300000,
        "specific_impulse_s": 421,
        "thrust_n": 34020000,
        "payload_capacity_kg": 140000,
        "cost_per_launch_usd": 1230000000,
        "reusable": False
    },
    "Starship": {
        "dry_mass_kg": 120000,
        "fuel_capacity_kg": 1200000,
        "specific_impulse_s": 378,
        "thrust_n": 72000000,
        "payload_capacity_kg": 150000,
        "cost_per_launch_usd": 10000000,
        "reusable": True
    },
    "SLS Block 1": {
        "dry_mass_kg": 85000,
        "fuel_capacity_kg": 970000,
        "specific_impulse_s": 452,
        "thrust_n": 39140000,
        "payload_capacity_kg": 95000,
        "cost_per_launch_usd": 2000000000,
        "reusable": False
    }
}

# ---------- Helpers ----------
def get_rv(body, t_astropy):
    """Return position and velocity (km, km/s) of body at given astropy time"""
    t_skyfield = ts.from_astropy(t_astropy)
    obj = body.at(t_skyfield)
    return np.array(obj.position.km), np.array(obj.velocity.km_per_s)

def calculate_fuel_requirements(delta_v_ms, spacecraft_specs, payload_kg):
    """Calculate fuel requirements using the rocket equation"""
    # Constants
    g0 = 9.81  # m/s²
    
    # Extract spacecraft parameters
    dry_mass = spacecraft_specs["dry_mass_kg"]
    fuel_capacity = spacecraft_specs["fuel_capacity_kg"]
    isp = spacecraft_specs["specific_impulse_s"]
    max_payload = spacecraft_specs["payload_capacity_kg"]
    
    # Calculate exhaust velocity
    ve = isp * g0  # m/s
    
    # Initial mass (dry + payload + fuel)
    m0 = dry_mass + payload_kg
    
    # Required mass ratio from rocket equation: ΔV = ve * ln(m0/mf)
    mass_ratio = np.exp(delta_v_ms / ve)
    
    # Required fuel mass
    required_fuel = m0 * (mass_ratio - 1)
    
    # Check feasibility
    feasible = bool(required_fuel <= fuel_capacity and payload_kg <= max_payload)
    
    # Fuel efficiency (higher is better)
    if required_fuel > 0:
        fuel_efficiency = (delta_v_ms * payload_kg) / (required_fuel * ve) * 100
    else:
        fuel_efficiency = 0
    
    # Fuel margin
    fuel_margin = ((fuel_capacity - required_fuel) / fuel_capacity) * 100 if feasible else 0
    
    return {
        "required_fuel_kg": float(required_fuel),
        "fuel_efficiency_percent": float(fuel_efficiency),
        "mass_ratio": float(mass_ratio),
        "feasible": feasible,
        "fuel_margin_percent": float(fuel_margin)
    }

def calculate_mission_cost(spacecraft_specs, fuel_analysis, payload_kg):
    """Calculate total mission cost"""
    launch_cost = spacecraft_specs["cost_per_launch_usd"]
    
    # Fuel cost (simplified - $1 per kg of fuel)
    fuel_cost = fuel_analysis["required_fuel_kg"] * 1.0
    
    # Total cost
    total_cost = launch_cost + fuel_cost
    
    # Cost per kg of payload
    cost_per_kg = total_cost / payload_kg if payload_kg > 0 else 0
    
    return {
        "launch_cost_usd": float(launch_cost),
        "fuel_cost_usd": float(fuel_cost),
        "total_mission_cost_usd": float(total_cost),
        "cost_per_kg_payload": float(cost_per_kg)
    }

def get_mission_recommendations(fuel_analysis, cost_analysis, spacecraft_specs):
    """Generate mission recommendations"""
    recommendations = []
    
    if not fuel_analysis["feasible"]:
        recommendations.append("⚠️ Mission not feasible with current spacecraft - consider reducing payload or selecting more powerful vehicle")
    
    if fuel_analysis["fuel_margin_percent"] < 10:
        recommendations.append("⚠️ Low fuel margin - consider adding 10-15% safety margin")
    
    if fuel_analysis["fuel_efficiency_percent"] < 20:
        recommendations.append("💡 Low fuel efficiency - consider optimizing trajectory or using more efficient propulsion")
    
    if cost_analysis["cost_per_kg_payload"] > 10000:
        recommendations.append("💰 High cost per kg - consider bulk payload or reusable vehicle")
    
    if spacecraft_specs["reusable"]:
        recommendations.append("♻️ Using reusable vehicle - significant cost savings on multiple missions")
    
    if fuel_analysis["fuel_margin_percent"] > 30:
        recommendations.append("✅ Excellent fuel margin - mission has high safety buffer")
    
    if not recommendations:
        recommendations.append("✅ Mission profile looks optimal!")
    
    return recommendations

# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "Advanced Space Mission Planner API v2.0"}

@app.get("/api/spacecraft", response_model=Dict[str, dict])
def get_spacecraft():
    """Get all available spacecraft specifications"""
    return SPACECRAFT_DATABASE

@app.get("/api/spacecraft/{spacecraft_name}")
def get_spacecraft_specs(spacecraft_name: str):
    """Get specifications for a specific spacecraft"""
    if spacecraft_name not in SPACECRAFT_DATABASE:
        raise HTTPException(status_code=404, detail="Spacecraft not found")
    return SPACECRAFT_DATABASE[spacecraft_name]

@app.get("/api/planet-positions", response_model=Dict[str, float])
def get_planet_positions(date: str = Query(None, description="ISO date (YYYY-MM-DD) to view planets at")):
    """Get heliocentric longitudes for all major planets"""
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
        try:
            planet_obj = planets[key]
            heliocentric = planets["sun"].at(t).observe(planet_obj)
            _, lon, _ = heliocentric.ecliptic_latlon()
            positions[name] = float(lon.degrees)
        except Exception as e:
            print(f"Error calculating position for {name}: {e}")
            positions[name] = 0.0
    
    return positions

@app.post("/porkchop")
def porkchop(req: PorkchopRequest):
    """Generate porkchop plot data"""
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

    return {
        "dep_dates": [t.utc.iso for t in dep_times],
        "tof_days": tof_days.tolist(),
        "dv_grid": dv_matrix.tolist()
    }

@app.post("/best_transfer")
def best_transfer(req: PorkchopRequest):
    """Find the optimal transfer window with lowest delta-V"""
    try:
        result = porkchop(req)
        dv_array = np.array(result["dv_grid"])
        
        if np.all(np.isnan(dv_array)):
            raise HTTPException(status_code=400, detail="No valid transfers found")
        
        idx = np.nanargmin(dv_array)
        i, j = np.unravel_index(idx, dv_array.shape)
        
        return {
            "dep_date": result["dep_dates"][i],
            "tof_days": float(result["tof_days"][j]),
            "dv": float(dv_array[i, j])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transfer computation failed: {str(e)}")

@app.post("/trajectory")
def trajectory(req: MissionRequest):
    """Generate trajectory with full mission analysis"""
    try:
        origin = planets[PLANET_MAP[req.origin]]
        target = planets[PLANET_MAP[req.target]]

        dep_t = Time(req.dep_date, scale="utc")
        tof = req.tof_days * 86400.0
        arr_t = Time(dep_t.jd + req.tof_days, format="jd", scale="utc")

        r1, v1 = get_rv(origin, dep_t)
        r2, v2 = get_rv(target, arr_t)

        # Calculate Lambert solution
        (v_dep, v_arr), = lambert(Sun.k, r1 * u.km, r2 * u.km, tof * u.s)
        
        # Calculate total delta-V
        dv_dep = np.linalg.norm(v_dep.to(u.km/u.s).value - v1)
        dv_arr = np.linalg.norm(v_arr.to(u.km/u.s).value - v2)
        total_dv = dv_dep + dv_arr
        
        # Generate trajectory points
        orb = Orbit.from_vectors(Sun, r1 * u.km, v_dep)
        times = dep_t + np.linspace(0, req.tof_days, req.n_samples) * u.day
        traj = []
        for t in times:
            r = orb.propagate(t - dep_t).r.to(u.km).value
            traj.append(r.tolist())

        # Get spacecraft specs
        if req.spacecraft_name not in SPACECRAFT_DATABASE:
            raise HTTPException(status_code=400, detail="Unknown spacecraft")
        
        spacecraft_specs = SPACECRAFT_DATABASE[req.spacecraft_name]
        
        # Calculate fuel and cost analysis
        fuel_analysis = calculate_fuel_requirements(total_dv * 1000, spacecraft_specs, req.payload_mass_kg)
        cost_analysis = calculate_mission_cost(spacecraft_specs, fuel_analysis, req.payload_mass_kg)
        
        # Generate recommendations
        recommendations = get_mission_recommendations(fuel_analysis, cost_analysis, spacecraft_specs)
        
        # Spacecraft utilization metrics
        payload_utilization = (req.payload_mass_kg / spacecraft_specs["payload_capacity_kg"]) * 100
        fuel_utilization = (fuel_analysis["required_fuel_kg"] / spacecraft_specs["fuel_capacity_kg"]) * 100
        
        spacecraft_utilization = {
            "payload_utilization_percent": float(payload_utilization),
            "fuel_utilization_percent": float(fuel_utilization),
            "thrust_to_weight_ratio": float(spacecraft_specs["thrust_n"] / (spacecraft_specs["dry_mass_kg"] * 9.81))
        }

        return {
            "trajectory": {
                "dep_date": req.dep_date,
                "tof_days": req.tof_days,
                "trajectory_km": traj,
                "origin_pos_km": r1.tolist(),
                "target_pos_km": r2.tolist(),
                "total_dv_kms": float(total_dv)
            },
            "fuel_analysis": fuel_analysis,
            "cost_analysis": cost_analysis,
            "spacecraft_utilization": spacecraft_utilization,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mission analysis failed: {str(e)}")

@app.post("/compare_spacecraft", response_model=SpacecraftComparison)
def compare_spacecraft(
    origin: str,
    target: str, 
    dep_date: str,
    tof_days: float,
    payload_kg: float = 10000
):
    """Compare all spacecraft options for a given mission"""
    try:
        # First get the delta-V requirement
        best_transfer_req = PorkchopRequest(
            origin=origin,
            target=target,
            dep_start=dep_date,
            dep_end=dep_date,
            n_dep=1,
            tof_min=int(tof_days),
            tof_max=int(tof_days),
            n_tof=1
        )
        
        transfer_result = best_transfer(best_transfer_req)
        delta_v_kms = transfer_result["dv"]
        
        # Analyze each spacecraft
        spacecraft_analysis = []
        for name, specs in SPACECRAFT_DATABASE.items():
            fuel_analysis = calculate_fuel_requirements(delta_v_kms * 1000, specs, payload_kg)
            cost_analysis = calculate_mission_cost(specs, fuel_analysis, payload_kg)
            
            # Calculate overall score (0-100)
            feasibility_score = 100 if fuel_analysis["feasible"] else 0
            efficiency_score = min(fuel_analysis["fuel_efficiency_percent"], 100)
            cost_score = max(0, 100 - (cost_analysis["cost_per_kg_payload"] / 1000))
            margin_score = min(fuel_analysis["fuel_margin_percent"], 100)
            
            overall_score = (feasibility_score * 0.4 + efficiency_score * 0.2 + 
                           cost_score * 0.2 + margin_score * 0.2)
            
            spacecraft_analysis.append({
                "name": name,
                "feasible": fuel_analysis["feasible"],
                "fuel_efficiency": fuel_analysis["fuel_efficiency_percent"],
                "total_cost": cost_analysis["total_mission_cost_usd"],
                "cost_per_kg": cost_analysis["cost_per_kg_payload"],
                "fuel_margin": fuel_analysis["fuel_margin_percent"],
                "overall_score": overall_score,
                "reusable": specs["reusable"]
            })
        
        # Sort by overall score
        spacecraft_analysis.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Find recommended spacecraft (best feasible option)
        recommended = next((s for s in spacecraft_analysis if s["feasible"]), spacecraft_analysis[0])
        
        return {
            "spacecraft_options": spacecraft_analysis,
            "recommended_spacecraft": recommended["name"],
            "comparison_metrics": {
                "delta_v_requirement_kms": delta_v_kms,
                "payload_mass_kg": payload_kg,
                "mission_duration_days": tof_days
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spacecraft comparison failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)