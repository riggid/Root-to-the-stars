# generate_orbits.py

import json
import numpy as np
from astropy.time import Time
from astropy import units as u
from skyfield.api import load
from skyfield.errors import EphemerisRangeError

print("Starting orbit data generation...")

# --- Configuration (should match your backend.py) ---
ts = load.timescale()
planets = load("../data/de421.bsp")

PLANET_MAP = {
    "Mercury": "mercury barycenter", "Venus": "venus barycenter", "Earth": "earth barycenter",
    "Mars": "mars barycenter", "Jupiter": "jupiter barycenter", "Saturn": "saturn barycenter",
    "Uranus": "uranus barycenter", "Neptune": "neptune barycenter",
}

ORBITAL_PERIODS_DAYS = {
    "Mercury": 88, "Venus": 225, "Earth": 365, "Mars": 687,
    "Jupiter": 4333, "Saturn": 10759, "Uranus": 30687, "Neptune": 60190,
}

# --- Main Logic ---
def generate_all_orbits(samples_per_orbit=150):
    """Calculates and returns a dictionary of orbital paths for all planets."""
    all_orbits = {}
    print("Calculating orbits for all supported planets...")

    for name, period in ORBITAL_PERIODS_DAYS.items():
        # ✨ FIX: Wrap the calculation in a try...except block
        try:
            print(f"  - Processing {name}...")
            planet_body = planets[PLANET_MAP[name]]
            
            start_jd = ts.now().tdb
            day_offsets = np.linspace(0, period, samples_per_orbit)
            orbit_jds = start_jd + day_offsets
            times = ts.tdb_jd(orbit_jds)

            position_vectors = planets["sun"].at(times).observe(planet_body).position
            positions = position_vectors.km.T.tolist()
            
            all_orbits[name] = positions
        
        except EphemerisRangeError:
            # If an error occurs, print a warning and skip this planet
            print(f"    - WARNING: Full orbit for {name} is outside the ephemeris range. Skipping.")
            continue
            
    return all_orbits


if __name__ == "__main__":
    # Generate the data
    orbital_data = generate_all_orbits()
    
    # Save the data to a JSON file
    output_path = "../data/orbital_data.json"
    with open(output_path, "w") as f:
        json.dump(orbital_data, f, indent=2)
        
    print(f"\n✅ Success! Orbital data has been saved to '{output_path}'")