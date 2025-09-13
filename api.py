# You'll need to install Flask, Flask-CORS, and skyfield:
# pip install flask flask_cors skyfield
from flask import Flask, jsonify
from skyfield.api import load
from flask_cors import CORS

# Initialize the Flask web server
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) is crucial to allow 
# your HTML file to request data from this server.
CORS(app) 

# --- Skyfield Setup ---
# Load the ephemeris data once when the server starts for efficiency.
ts = load.timescale()
planets_ephem = load('de421.bsp') # JPL DE421 ephemeris
sun = planets_ephem['sun']
# Define the planet names as skyfield knows them.
# The barycenters are used for gas giants for better accuracy.
planet_names = {
    'Mercury': 'mercury',
    'Venus': 'venus',
    'Earth': 'earth',
    'Mars': 'mars',
    'Jupiter': 'jupiter barycenter',
    'Saturn': 'saturn barycenter',
    'Uranus': 'uranus barycenter',
    'Neptune': 'neptune barycenter'
}

@app.route('/api/planet-positions')
def get_planet_positions():
    """
    This is the API endpoint. When accessed, it calculates the current
    heliocentric (sun-centered) longitude for each planet and returns it.
    """
    # Get the current time for the calculation.
    t = ts.now()
    
    positions = {}
    
    for name, skyfield_name in planet_names.items():
        planet = planets_ephem[skyfield_name]
        
        # Calculate the position of the planet as seen from the Sun.
        heliocentric = sun.at(t).observe(planet)
        
        # Get the ecliptic longitude, which is the angle around the sun
        # on the plane of the solar system.
        _, lon, _ = heliocentric.ecliptic_latlon()
        
        # Store the angle in degrees in our results dictionary.
        positions[name] = lon.degrees
        
    # Return the dictionary as a JSON response.
    return jsonify(positions)

# This part runs the server when you execute "python api.py"
if __name__ == '__main__':
    # host='0.0.0.0' makes the server accessible from your Windows browser
    # while it's running inside WSL.
    app.run(host='0.0.0.0', port=5000, debug=True)

