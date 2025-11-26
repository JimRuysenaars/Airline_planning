
'''
OD matrix includes: dataframe i x j: showing:
    q_ij    : travel demand between airport i and j
    d_ij    : distance between airports i and j
    Yield_ij: (function of d_ij) revenue per Revenue Passenger Kilometers (RPK) flown (average yield)

Airport dataframe: TODO

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

'''
import pandas as pd
import itertools
import gurobipy as gp
import numpy as np

def get_ICAO(path):
    df = pd.read_excel(path, header=None)
    icao = df.iloc[1, 1:].tolist()
    return icao

def get_demand(path):
    df = pd.read_excel(path, header=None)

    # Row 0: labels (empty cell then ICAO codes)
    icao = df.iloc[0, 1:].tolist()

    # Row 1: ICAO again (same order as row/column labels)
    # Demand matrix starts at row 1, col 1
    demand_df = df.iloc[1:, 1:]
    demand_df.index = icao     # set row labels
    demand_df.columns = icao   # set column labels

    demand_df = demand_df.astype(float)

    return demand_df

def build_distance_matrix(path):
    df = pd.read_excel(path, header=None)

    # Row mapping based on sheet:
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

def create_OD(path_distance, path_demand):
    Distance_matrix = build_distance_matrix(path_distance)
    Demand_matrix  = get_demand(path_demand)
    I = get_ICAO(path_distance)
    J = get_ICAO(path_distance)
    rows = []
    for i, j in itertools.product(I, J):
        distance = Distance_matrix.loc[i, j]  
        demand   = Demand_matrix.loc[i, j]
        yield_   = 5.9 * distance ** -0.76 + 0.043 if distance > 0 else 0  # Avoid division by zero

        rows.append({
            "i": i,
            "j": j,
            "distance": distance,
            "demand": demand,
            "yield":  yield_
        })

    OD = pd.DataFrame(rows).set_index(["i", "j"]).sort_index()
    return OD
   

# Build dataframes
AC = pd.read_excel('C:\\Users\\jimru\\OneDrive\\Documenten\\python\\Airline_planning\\Problem 1 - Data\\AircraftData.xlsx', index_col=0)
OD = create_OD('C:\\Users\\jimru\\OneDrive\\Documenten\\python\\Airline_planning\\Problem 1 - Data\\airport_data.xlsx', 'C:\\Users\\jimru\\OneDrive\\Documenten\\python\\Airline_planning\\Problem 1 - Data\\demand_per_week.xlsx')
print(AC)
print(OD.head())


# Initialize gubropyi model
model = gp.Model("Airline_Planning")

# Create decision variables
I = OD.index.get_level_values(0).unique().to_list()                         # set of airports i       
J = OD.index.get_level_values(1).unique().to_list()                         # set of airports j
K = AC.index                                                                # set of aircraft types k 



x = model.addVars(I, J, name="xij", vtype=gp.GRB.CONTINUOUS, lb=0)          # Direct flow from airport i to airport j
y = model.addVars(K, name="y", vtype=gp.GRB.CONTINUOUS, lb=0)               # number of aircraft of type k
z = model.addVars(I, J, K, name="zijk", vtype=gp.GRB.INTEGER, lb=0)         # number of flights from airport i to airport j with aircraft type k
w = model.addVars(I, J, name="wij", vtype=gp.GRB.CONTINUOUS, lb=0)          # flow from airport i to airport j that transfers at the hub


# Objective function
revenue = gp.quicksum( ( 5.9 * OD.loc[(i, j),"distance"] ** -0.76 + 0.043 ) * (x[(i, j)] + w[(i, j)]) 
                      for i in I 
                      for j in J )

costs = gp.quicksum( z[(i, j), k] * (AC.loc[k, "fixed_operating_cost"] + 
                                   AC.loc[k, "time_cost_parameter"] * (OD.loc[(i, j), "distance"] / AC.loc[k, "speed"]) +
                                   AC.loc[k, "fuel_cost_parameter"] * fuel_cost * OD.loc[(i, j),"distance"] / 1.5
                                   for i in I
                                   for j in J
                                   for k in K ))
obj = revenue - costs
model.setObjective(obj, gp.MAXIMIZE)


# Contstraints

# C1: number of transported passengers lower than demand
gp.addConstrs( ( x[(i, j)] + w[(i, j)] <= gp.quicksum( OD.loc[(i, j), "demand"])          
                for i in I  
                for j in J ), name="C1_Capacity")

# C2: w lower than demand
gp.addConstrs( (w[i, j] <= OD.loc[(i, j), "demand"] * g[i] * g[j]
                for i in I
                for j in J ), name="C2_Transfer_via_hub")

