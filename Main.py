
'''
Dataframe i x j: showing:
    q_ij    : travel demand between airport i and j
    d_ij    : distance between airports i and j
    Yield_ij: (function of d_ij) revenue per Revenue Passenger Kilometers (RPK) flown (average yield)


Sets:
N: set of airports

Decision Variables:
xij: direct flow from airport i to airport j
zij: number of flights from airport i to airport j
wij: flow from airport i to airport j that transfers at the hub


Parameters:
- A/C matrix-
s: number of seats per aircraft
Speed [km/h]
Seats
Average TAT [mins]
Maximum range [km]
Runway required [m]

Weekly lease cost [€]
Fixed operating cost C_X [€]
Time cost parameter C_T [€/hr]
Fuel cost parameter C_F

- Constants - 
CASK: unit operation cost per per available seat per kilometre (ASK) flown
sp: speed of the aircraft
LF: average load factor
AC: number of aircraft
LTO: landing and take-off time
BT: aircraft avg. utilisation time
gk: 0 if a hub is located at airport k; 1 otherwise

Workflow for dictionary
import pandas as pd
import itertools

I = ["AC1", "AC2"]
J = ["Gate1", "Gate2"]
K = ["scenario1", "scenario2"]   # or time periods, routes, whatever k is

rows = []
for i, j, k in itertools.product(I, J, K):
    # replace these with your real values:
    distance = ...
    demand   = ...
    yield_   = ...

    rows.append({
        "i": i,
        "j": j,
        "k": k,
        "distance": distance,
        "demand": demand,
        "yield":  yield_
    })

df = pd.DataFrame(rows).set_index(["i", "j", "k"]).sort_index()
print(df)

'''
import pandas as pd
import itertools
import gurobipy as gp
import numpy as np



def build_distance_matrix(path):
    # --- Load the Excel file as-is (wide format) ---
    df = pd.read_excel(path, header=None)

    # Row mapping based on your sheet:
    # 0: City names (ignored for distance matrix)
    # 1: ICAO codes (column labels we want)
    # 2: Latitude
    # 3: Longitude

    icao = df.iloc[1, 1:].tolist()
    lat  = df.iloc[2, 1:].to_numpy(dtype=float)
    lon  = df.iloc[3, 1:].to_numpy(dtype=float)
    
    # Initialize matrix (empty)
    D = pd.DataFrame(index=icao, columns=icao, dtype=float)

    # Iterate over all i, j combinations
    for i, j in itertools.product(range(len(icao)), repeat=2):

        lat_i = lat[i]
        lon_i = lon[i]
        lat_j = lat[j]
        lon_j = lon[j]

        d_ij = 2 * np.arcsin(np.sqrt(np.sin(np.radians((lat_i - lat_j) / 2))**2 +
                            np.cos(np.radians(lat_i)) * np.cos(np.radians(lat_j)) * np.sin(np.radians((lon_i - lon_j) / 2))**2))

        d_ij = d_ij * 6371  # Radius of Earth in kilometers
        D.iloc[i, j] = d_ij
    return D

# Build distance matrix
D_matrix = build_distance_matrix('C:\\Users\\jimru\\OneDrive\\Documenten\\python\\Airline_planning\\Problem 1 - Data\\airport_data.xlsx')
    
AC = pd.read_excel('C:\\Users\\jimru\\OneDrive\\Documenten\\python\\Airline_planning\\Problem 1 - Data\\AircraftData.xlsx', index_col=0)

print(AC)




I = ["Origin", "Origin"]                # To be filled in
J = ["Destination", "Destination"]      # To be filled in
rows = []
for i, j in itertools.product(I, J, K):
    distance = ...
    demand   = ...
    yield_   = ...

    rows.append({
        "i": i,
        "j": j,
        "distance": distance,
        "demand": demand,
        "yield":  yield_
    })

OD = OD.DataFrame(rows).set_index(["i", "j"]).sort_index()
print(OD)

# Initialize gubropyi model
model = gp.Model("Airline_Planning")

# Create decision variables
I = OD.index.get_level_values(0)
J = OD.index.get_level_values(1)
x = model.addVars(I, J, name="xij", vtype=gp.GRB.CONTINUOUS, lb=0)
z = model.addVars(I, J, name="zij", vtype=gp.GRB.INTEGER, lb=0)
w = model.addVars(I, J, name="wij", vtype=gp.GRB.CONTINUOUS, lb=0)


# Objective function

model.setObjective(obj, gp.MINIMIZE)



