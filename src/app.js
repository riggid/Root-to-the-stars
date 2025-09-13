// --- CONFIGURATION ---
const API_BASE_URL = 'http://127.0.0.1:8000';

// --- GLOBAL VARIABLES ---
let scene, camera, renderer, controls;
let planetMeshes = {};
let planetPositions = {};
let trajectoryLine = null;
let spacecraftData = {};

// --- INITIALIZATION ---
function initScene() {
  scene = new THREE.Scene();
  // ✨ FIX 1: Increased camera's far clipping plane to see the whole solar system
  camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 100000);
  renderer = new THREE.WebGLRenderer({ antialias:true, canvas: document.getElementById('spaceCanvas') });
  renderer.setSize(window.innerWidth, window.innerHeight);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 1;
  controls.maxDistance = 20000; // Increased max zoom out

  camera.position.set(20, 8, 20);
  camera.lookAt(0, 0, 0);

  // Add stars
  const starGeo = new THREE.BufferGeometry();
  const starVertices = [];
  for(let i=0;i<10000;i++){
    starVertices.push((Math.random()-0.5)*120000, (Math.random()-0.5)*120000, (Math.random()-0.5)*120000);
  }
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starVertices,3));
  const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color:0xffffff, size:1.2 }));
  scene.add(stars);

  // Add Sun
  const sun = new THREE.Mesh(new THREE.SphereGeometry(2,32,32), new THREE.MeshBasicMaterial({ color:0xffdd33 }));
  scene.add(sun);

  // Add planets
  const AU = 149597870;
  const planetData = [
    {name:'Mercury', radius:2.44, orbit:0.387*AU, color:0x909090},
    {name:'Venus', radius:6.05, orbit:0.723*AU, color:0xe6c27a},
    {name:'Earth', radius:6.37, orbit:1*AU, color:0x3a90ff},
    {name:'Mars', radius:3.39, orbit:1.524*AU, color:0xff4500},
    {name:'Jupiter', radius:69.91, orbit:5.203*AU, color:0xd4a06a},
    {name:'Saturn', radius:58.23, orbit:9.537*AU, color:0xf1d17b},
    {name:'Uranus', radius:25.36, orbit:19.191*AU, color:0x7fe6ff},
    {name:'Neptune', radius:24.62, orbit:30.07*AU, color:0x2e70c1},
  ];

  // A consistent scale for everything 3D
  const scaleFactor = 0.00001;

  planetData.forEach(p=>{
    // ✨ FIX: Exaggerate planet sizes for visibility
    const planetSize = Math.log2(p.radius) * 100;
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(planetSize,32,32),
      new THREE.MeshStandardMaterial({color:p.color})
    );
    scene.add(mesh);
    planetMeshes[p.name] = {mesh, orbit: p.orbit * scaleFactor};
  });

  scene.add(new THREE.AmbientLight(0x404040,1.0));
  const sunLight = new THREE.PointLight(0xffffff, 2);
  scene.add(sunLight);
}

// --- API & DATA HANDLING ---
async function loadSpacecraftData() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/spacecraft`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    spacecraftData = await response.json();
    
    const spacecraftSelect = document.getElementById('spacecraft');
    spacecraftSelect.innerHTML = '<option value="">Select Spacecraft...</option>';
    
    for (const name in spacecraftData) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      spacecraftSelect.appendChild(option);
    }
  } catch (error) {
    console.error('Failed to load spacecraft data:', error);
    document.getElementById('status').textContent = 'Error: Could not load spacecraft data.';
  }
}

async function updatePlanetPositions(date = null) {
  const statusEl = document.getElementById('positionStatus');
  try {
    statusEl.textContent = 'Updating planet positions...';
    let url = `${API_BASE_URL}/api/planet-positions`;
    if (date) url += `?date=${date}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Server error fetching positions`);
    planetPositions = await response.json();
    statusEl.textContent = `Displaying positions for: ${date || 'today'}`;
    statusEl.className = 'text-green-400 text-xs';
  } catch (error) {
    console.error('Failed to fetch planet positions:', error);
    statusEl.textContent = 'Error: Failed to load planet positions!';
    statusEl.className = 'text-red-400 text-xs';
  }
}

// --- UI & VISUALIZATION UPDATES ---
function updateSpacecraftInfo(rocketName) {
  const rocket = spacecraftData[rocketName];
  if (!rocket) return;
  
  const infoDiv = document.getElementById('spacecraftInfo');
  const specsDiv = document.getElementById('rocketSpecs');
  
  specsDiv.innerHTML = `
    <div class="grid grid-cols-2 gap-2 text-xs">
      <div><strong>Dry Mass:</strong> ${(rocket.dry_mass_kg || 0).toLocaleString()} kg</div>
      <div><strong>Payload:</strong> ${(rocket.payload_capacity_kg || 0).toLocaleString()} kg</div>
      <div><strong>Thrust:</strong> ${(rocket.thrust_n / 1000).toLocaleString()} kN</div>
      <div><strong>Isp:</strong> ${rocket.specific_impulse_s} s</div>
      <div><strong>Fuel Capacity:</strong> ${(rocket.fuel_capacity_kg || 0).toLocaleString()} kg</div>
      <div><strong>Reusable:</strong> ${rocket.reusable ? 'Yes' : 'No'}</div>
    </div>
  `;
  
  infoDiv.classList.remove('hidden');
}

function updatePerformanceChart(missionData, rocketName) {
  const ctx = document.getElementById('performanceChart').getContext('2d');
  
  if (window.missionChart) window.missionChart.destroy();
  
  const fuel = missionData.fuel_analysis;
  const cost = missionData.cost_analysis;
  const util = missionData.spacecraft_utilization;
  
  window.missionChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Fuel Efficiency', 'Payload Usage', 'Cost Effectiveness', 'Fuel Margin', 'Mission Feasible'],
      datasets: [{
        label: rocketName,
        data: [
          Math.min(fuel.fuel_efficiency_percent, 100),
          util.payload_utilization_percent,
          Math.max(0, 100 - (cost.cost_per_kg_payload / 1000)),
          Math.min(fuel.fuel_margin_percent, 100),
          fuel.feasible ? 100 : 0
        ],
        backgroundColor: 'rgba(59, 130, 246, 0.2)', borderColor: 'rgba(59, 130, 246, 1)',
        pointBackgroundColor: 'rgba(59, 130, 246, 1)', pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff', pointHoverBorderColor: 'rgba(59, 130, 246, 1)'
      }]
    },
    options: { responsive: true, plugins: { legend: { labels: { color: 'white' } } },
      scales: { r: { angleLines: { color: 'rgba(255, 255, 255, 0.1)' }, grid: { color: 'rgba(255, 255, 255, 0.1)' },
          pointLabels: { color: 'white', font: { size: 10 } }, ticks: { color: 'white', backdropColor: 'transparent' }
        }
      }
    }
  });
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  const scaleFactor = 0.00001; // Ensure scaleFactor is accessible here

  Object.keys(planetMeshes).forEach(planetName => {
    const planetInfo = planetMeshes[planetName];
    // planetPositions now contains { Earth: [x, y, z], Mars: [x, y, z], ... }
    if (planetPositions[planetName]) {
      const pos = planetPositions[planetName]; // pos is the [x, y, z] array from the backend

      // ✨ FIX: Use the precise 3D coordinates from the backend
      // We still swap y and z because Skyfield's coordinate system (z-up)
      // is different from Three.js's default (y-up).
      const x = pos[0] * scaleFactor;
      const y = pos[2] * scaleFactor; // Use z from backend for y in three.js
      const z = pos[1] * scaleFactor; // Use y from backend for z in three.js
      
      planetInfo.mesh.position.set(x, y, z);
    }
  });

  renderer.render(scene, camera);
}


// --- EVENT LISTENERS ---
document.getElementById('spacecraft').addEventListener('change', function() {
  if (this.value !== '') updateSpacecraftInfo(this.value);
  else document.getElementById('spacecraftInfo').classList.add('hidden');
});

document.getElementById('refreshPositions').addEventListener('click', () => {
  const date = document.getElementById('viewDate').value;
  updatePlanetPositions(date);
});

document.getElementById('computeBtn').addEventListener('click', async () => {
  const origin = document.getElementById('origin').value;
  const target = document.getElementById('target').value;
  const dep_start = document.getElementById('dep_start').value;
  const tof_min = parseFloat(document.getElementById('tof_min').value);
  const tof_max = parseFloat(document.getElementById('tof_max').value);
  const spacecraftName = document.getElementById('spacecraft').value;
  const payloadMass = parseFloat(document.getElementById('payloadMass').value) || 10000;
  const statusEl = document.getElementById('status');

  if (!spacecraftName) {
    statusEl.textContent = "Please select a spacecraft first!";
    return;
  }

  statusEl.textContent = "Computing optimal mission profile...";
  statusEl.className = 'text-yellow-400 text-sm mt-1';
  
  try {
    // 1. Get best transfer window
    const bestRes = await fetch(`${API_BASE_URL}/best_transfer`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin, target, dep_start, n_dep: 50, tof_min, tof_max, n_tof: 50 })
    }).then(r => r.json());

    if(bestRes.detail) throw new Error(bestRes.detail);

    // ✨ FIX 2: Update 3D view to show planets at departure
    const departureDate = bestRes.dep_date.split('T')[0];
    await updatePlanetPositions(departureDate);

    // 2. Get full mission analysis
    const missionRes = await fetch(`${API_BASE_URL}/trajectory`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin, target, dep_date: bestRes.dep_date, tof_days: bestRes.tof_days,
        spacecraft_name: spacecraftName, payload_mass_kg: payloadMass
      })
    }).then(r => r.json());

    if(missionRes.detail) throw new Error(missionRes.detail);
    
    // Update trajectory visualization
    if (trajectoryLine) scene.remove(trajectoryLine);
    const scaleFactor = 0.00001;
    // Note: Swapping y/z is correct (Skyfield z-up vs Three.js y-up)
    const points = missionRes.trajectory.trajectory_km.map(p => 
      new THREE.Vector3(p[0] * scaleFactor, p[2] * scaleFactor, p[1] * scaleFactor)
    );
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    trajectoryLine = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 }));
    scene.add(trajectoryLine);

    // Update results display
    document.getElementById('resultOrigin').textContent = `Origin: ${origin}`;
    document.getElementById('resultTarget').textContent = `Target: ${target}`;
    document.getElementById('resultDeparture').textContent = `Departure: ${departureDate}`;
    document.getElementById('resultTOF').textContent = `Flight Time: ${bestRes.tof_days.toFixed(1)} days`;
    document.getElementById('resultDeltaV').textContent = `Delta-V: ${missionRes.trajectory.total_dv_kms.toFixed(2)} km/s`;

    const fuel = missionRes.fuel_analysis;
    document.getElementById('fuelRequired').textContent = `Fuel Required: ${(fuel.required_fuel_kg / 1000).toFixed(1)} tonnes`;
    document.getElementById('fuelEfficiency').textContent = `Fuel Efficiency: ${fuel.fuel_efficiency_percent.toFixed(1)}%`;
    document.getElementById('payloadCapacity').textContent = `Payload: ${payloadMass.toLocaleString()} / ${spacecraftData[spacecraftName].payload_capacity_kg.toLocaleString()} kg`;
    document.getElementById('missionFeasibility').textContent = `Mission Status: ${fuel.feasible ? '✅ FEASIBLE' : '❌ NOT FEASIBLE'}`;
    document.getElementById('missionFeasibility').className = `font-bold ${fuel.feasible ? 'text-green-400' : 'text-red-400'}`;
    
    const cost = missionRes.cost_analysis;
    document.getElementById('launchCost').textContent = `Launch Cost: $${(cost.launch_cost_usd / 1000000).toFixed(1)}M`;
    document.getElementById('fuelCost').textContent = `Fuel Cost: $${(cost.fuel_cost_usd / 1000).toFixed(0)}K`;
    document.getElementById('totalCost').textContent = `Total Cost: $${(cost.total_mission_cost_usd / 1000000).toFixed(1)}M`;

    updatePerformanceChart(missionRes, spacecraftName);

    statusEl.textContent = `Mission computed successfully!`;
    statusEl.className = 'text-green-400 text-sm mt-1';
    document.getElementById('resultsPanel').classList.remove('hidden');

  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err.message}`;
    statusEl.className = 'text-red-400 text-sm mt-1';
  }
});

document.getElementById('closeResults').addEventListener('click', () => {
  document.getElementById('resultsPanel').classList.add('hidden');
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- MAIN EXECUTION ---
async function init() {
  initScene();
  await loadSpacecraftData();
  await updatePlanetPositions(document.getElementById('viewDate').value);
  animate();
}

init();