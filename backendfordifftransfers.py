import math

class InterplanetaryTransfer:
    def __init__(self):
        # Solar gravitational parameter (km³/s²)
        self.MU_SUN = 132712440018

        # Planetary orbital data (mean distances from Sun in km, orbital period in days)
        self.planet_data = {
            "mercury": {"a": 57909050, "period": 87.969},
            "venus":   {"a": 108208000, "period": 224.701},
            "earth":   {"a": 149598023, "period": 365.256},
            "mars":    {"a": 227939200, "period": 686.980},
            "jupiter": {"a": 778299000, "period": 4332.59},
            "saturn":  {"a": 1433530000, "period": 10759.22}
        }

    # Hohmann Transfer
    def calculate_hohmann_transfer(self, planet1, planet2):
        r1 = self.planet_data[planet1]["a"]
        r2 = self.planet_data[planet2]["a"]

        a_transfer = (r1 + r2) / 2
        v1_planet = math.sqrt(self.MU_SUN / r1)
        v2_planet = math.sqrt(self.MU_SUN / r2)

        v1_transfer = math.sqrt(self.MU_SUN * (2/r1 - 1/a_transfer))
        v2_transfer = math.sqrt(self.MU_SUN * (2/r2 - 1/a_transfer))

        delta_v1 = abs(v1_transfer - v1_planet)
        delta_v2 = abs(v2_planet - v2_transfer)
        total_delta_v = delta_v1 + delta_v2

        transfer_time = math.pi * math.sqrt(a_transfer**3 / self.MU_SUN)
        transfer_time_days = transfer_time / (24 * 3600)

        return {
            "method": "Hohmann Transfer",
            "departure": planet1,
            "arrival": planet2,
            "deltaV1": delta_v1,
            "deltaV2": delta_v2,
            "totalDeltaV": total_delta_v,
            "transferTimeYears": transfer_time_days / 365.25
        }

    # Bi-elliptic Transfer
    def calculate_bielliptic_transfer(self, planet1, planet2, intermediate_radius=None):
        r1 = self.planet_data[planet1]["a"]
        r2 = self.planet_data[planet2]["a"]

        r3 = intermediate_radius or max(r1, r2) * 2
        a1 = (r1 + r3) / 2
        a2 = (r2 + r3) / 2

        v1_planet = math.sqrt(self.MU_SUN / r1)
        v1_transfer1 = math.sqrt(self.MU_SUN * (2/r1 - 1/a1))
        delta_v1 = abs(v1_transfer1 - v1_planet)

        v3_transfer1 = math.sqrt(self.MU_SUN * (2/r3 - 1/a1))
        v3_transfer2 = math.sqrt(self.MU_SUN * (2/r3 - 1/a2))
        delta_v2 = abs(v3_transfer2 - v3_transfer1)

        v2_planet = math.sqrt(self.MU_SUN / r2)
        v2_transfer2 = math.sqrt(self.MU_SUN * (2/r2 - 1/a2))
        delta_v3 = abs(v2_planet - v2_transfer2)

        total_delta_v = delta_v1 + delta_v2 + delta_v3

        time1 = math.pi * math.sqrt(a1**3 / self.MU_SUN)
        time2 = math.pi * math.sqrt(a2**3 / self.MU_SUN)
        total_time_days = (time1 + time2) / (24 * 3600)

        return {
            "method": "Bi-elliptic Transfer",
            "departure": planet1,
            "arrival": planet2,
            "totalDeltaV": total_delta_v,
            "transferTimeYears": total_time_days / 365.25
        }

    # Lambert Transfer (simplified)
    def calculate_lambert_transfer(self, planet1, planet2, time_of_flight):
        r1 = self.planet_data[planet1]["a"]
        r2 = self.planet_data[planet2]["a"]
        tof = time_of_flight * 24 * 3600

        c = math.sqrt(r1*r1 + r2*r2 - 2*r1*r2*math.cos(math.pi/3))
        s = (r1 + r2 + c) / 2
        a_min = s / 2
        tof_min = math.pi * math.sqrt(a_min**3 / self.MU_SUN)
        tof_parabolic = (1/3) * math.sqrt(2/self.MU_SUN) * (s**1.5 - (s-c)**1.5)

        if tof < tof_parabolic:
            a_transfer = -s / 2
        else:
            a_transfer = a_min * (tof / tof_min)**(2/3)

        # ✅ Guard check
        val1 = (2 / r1 - 1 / a_transfer)
        val2 = (2 / r2 - 1 / a_transfer)

        if val1 <= 0 or val2 <= 0:
            raise ValueError(
                f"Invalid Lambert transfer: TOF={time_of_flight} days requires hyperbolic trajectory"
            )

        v1_transfer = math.sqrt(self.MU_SUN * val1)
        v2_transfer = math.sqrt(self.MU_SUN * val2)

        v1_planet = math.sqrt(self.MU_SUN / r1)
        v2_planet = math.sqrt(self.MU_SUN / r2)

        delta_v1 = abs(v1_transfer - v1_planet)
        delta_v2 = abs(v2_planet - v2_transfer)
        total_delta_v = delta_v1 + delta_v2

        return {
            "method": "Lambert Transfer",
            "departure": planet1,
            "arrival": planet2,
            "deltaV1": delta_v1,
            "deltaV2": delta_v2,
            "totalDeltaV": total_delta_v,
            "transferTimeDays": time_of_flight,
            "trajectory": "Elliptical" if a_transfer > 0 else "Hyperbolic"
        }

    # Compare all transfers
    def compare_transfers(self, planet1, planet2, time_of_flight=None):
        results = []

        results.append(self.calculate_hohmann_transfer(planet1, planet2))

        r1 = self.planet_data[planet1]["a"]
        r2 = self.planet_data[planet2]["a"]
        ratio = max(r1, r2) / min(r1, r2)

        if ratio > 11.94:
            results.append(self.calculate_bielliptic_transfer(planet1, planet2))

        if time_of_flight:
            try:
                results.append(self.calculate_lambert_transfer(planet1, planet2, time_of_flight))
            except ValueError as e:
                print(f"Lambert transfer skipped: {e}")

        results.sort(key=lambda x: x["totalDeltaV"])
        return results
from fastapi import FastAPI

app = FastAPI()   # 👈 This is what uvicorn is looking for

interplanetary = InterplanetaryTransfer()

@app.get("/transfer/{from_planet}/{to_planet}")
def transfer(from_planet: str, to_planet: str, tof: int = None):
    return interplanetary.compare_transfers(from_planet.lower(), to_planet.lower(), tof)
