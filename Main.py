
'''
OD matrix includes: dataframe i x j: showing:
    q_ij    : travel demand between airport i and j
    d_ij    : distance between airports i and j
    Yield_ij: (function of d_ij) revenue per Revenue Passenger Kilometers (RPK) flown (average yield)

Airport dataframe:
    Variable G: 0 if a hub is located at airport k; 1 otherwise:
        -> new column with G, only 0 for hub airport Madrid
    runway length
    available slots 

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
from matplotlib.patches import FancyArrowPatch
import geopandas as gpd
from matplotlib.lines import Line2D
import openpyxl


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

def create_AC(path):
    df = pd.read_excel(path)

    # First column is parameter names
    df = df.set_index(df.columns[0]).T

    # Remove any "Unit" column
    if "Unit" in df.columns:
        df = df.drop(columns=["Unit"])

    # Remove any index row called "Unit"
    df = df[~df.index.str.contains("Unit", case=False)]

    # Convert everything else to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    return df


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

def ICAO_coordinates(path):
    df = pd.read_excel(path, header=None)

    # Row mapping based on sheet:
    # 1: ICAO codes (column labels we want)
    # 2: Latitude
    # 3: Longitude

    icao = df.iloc[1, 1:].tolist()
    lat  = df.iloc[2, 1:].to_numpy(dtype=float)
    lon  = df.iloc[3, 1:].to_numpy(dtype=float)

    coordinates = pd.DataFrame(index=icao, columns=['latitude', 'longitude'], dtype=float)
    for i in range(len(icao)):
        coordinates.loc[icao[i], 'latitude'] = lat[i]
        coordinates.loc[icao[i], 'longitude'] = lon[i]
    return coordinates



def create_OD(path_distance, path_demand):
    Distance_matrix = build_distance_matrix(path_distance)
    Demand_matrix  = get_demand(path_demand)
    I = get_ICAO(path_distance)
    J = get_ICAO(path_distance)
    rows = []
    for i, j in itertools.product(I, J):
        if i == j:
            continue  # Skip same airport pairs
        else:
            distance = Distance_matrix.loc[i, j]  
            demand   = Demand_matrix.loc[i, j]
            yield_   = 5.9 * distance ** -0.76 + 0.043 if distance > 0 else 0  # Avoid division by zero 5.9

            rows.append({
                "i": i,
                "j": j,
                "distance": distance,
                "demand": demand,
                "yield":  yield_
            })

    OD = pd.DataFrame(rows).set_index(["i", "j"]).sort_index()
    return OD

def create_AP(path):
    df = pd.read_excel(path, header=None)
    icao = df.iloc[1, 1:].tolist()
    runway_length_raw = df.iloc[4, 1:]
    runway_length = pd.to_numeric(runway_length_raw, errors='coerce').fillna(0).to_numpy()
    available_slots_raw = df.iloc[5, 1:]
    available_slots = pd.to_numeric(available_slots_raw, errors='coerce').fillna(0).to_numpy()

    
    AP = pd.DataFrame(index=icao, columns=['runway_length', 'available_slots'], dtype=float)
    for i in range(len(icao)):
        AP.loc[icao[i], 'runway_length'] = runway_length[i]
        if icao[i] == 'LEMF':  # Paris (LFPG) or Madrid (LEMF) airport as hub
            AP.loc[icao[i], 'G'] = 0
            AP.loc[icao[i], 'available_slots'] = 9999  # Infinite slots for hub
        else:
            AP.loc[icao[i], 'G'] = 1
            AP.loc[icao[i], 'available_slots'] = available_slots[i]   
    return AP

def build_a_ijk(OD, AC):
    I = OD.index.get_level_values(0).unique().to_list()                         
    J = OD.index.get_level_values(1).unique().to_list()                         
    K = AC.index                                                                
    a_ijk = {}
    for i in I:
        for j in J:
            for k in K:
                if i == j:
                    continue
                else:
                    if OD.loc[(i, j), "distance"] <= AC.loc[k, "maximum_range"]:
                        a_ijk[(i, j, k)] = 10000
                    else:
                        a_ijk[(i, j, k)] = 0
    return a_ijk

# Build dataframes
AC = create_AC('Problem 1 - Data\\AircraftData.xlsx')
OD = create_OD('Problem 1 - Data\\airport_data.xlsx', 'Problem 1 - Data\\demand_per_week.xlsx')
AP = create_AP('Problem 1 - Data\\airport_data.xlsx')
a_ijk = build_a_ijk(OD, AC)


# Initialize gurobipy model
model = gp.Model("Airline_Planning")

# Parameters
fuel_cost = 1.42  # €/gallon
utilization_time = 10 * 7   # 10 hours of operstions per day, 7 days a week -> 70 hours per week
LF = 0.75  # Load factor


# Create decision variables
pairs = OD.index.tolist()       # List of (i,j) pairs
K = AC.index                                                                # set of aircraft types k 

# Decision variables
x = model.addVars(pairs, name="xij", vtype=gp.GRB.INTEGER, lb=0)          # Direct flow from airport i to airport j
w = model.addVars(pairs, name="wij", vtype=gp.GRB.INTEGER, lb=0)          # flow from airport i to airport j that transfers at the hub
y = model.addVars(K, name="y", vtype=gp.GRB.INTEGER, lb=0)               # number of aircraft of type k
z = model.addVars( [(i, j, k) for (i, j) in pairs for k in K], name="zijk", vtype=gp.GRB.INTEGER, lb=0)         # number of flights from airport i to airport j with aircraft type k


# Objective function: 
eps = 1e-6   # small number to avoid 0**negative

# Revenue: sum over actual OD pairs only
revenue = gp.quicksum(
    (5.9 * (OD.loc[p, "distance"] + eps) ** (-0.76) + 0.043) * (x[p] + w[p]) * OD.loc[p, "distance"]
    for p in pairs
)

# Costs: sum over pairs and aircraft types that exist in z
costs = gp.quicksum(
    z[i, j, k] * (
        AC.loc[k, "fixed_operating_cost"]
        + AC.loc[k, "time_cost_parameter"] * (OD.loc[(i, j), "distance"] / AC.loc[k, "speed"])
        + AC.loc[k, "fuel_cost_parameter"] * fuel_cost * OD.loc[(i, j), "distance"] / 1.5
    ) * 0.7 
    for (i, j) in pairs
    for k in K
) + gp.quicksum(y[k] * AC.loc[k, "weekly_lease_cost"] for k in K)

# Set objective
model.setObjective(revenue - costs, gp.GRB.MAXIMIZE)
model.setParam('MIPGap', 0.005)


# Contstraints

# C1: number of transported passengers lower than demand
model.addConstrs(
    ( x[p] + w[p] <= OD.loc[p, "demand"] for p in pairs ),
    name="C1_Capacity"
)

# C1*: w lower than demand
model.addConstrs(
    ( w[p] <= OD.loc[p, "demand"] * AP.loc[p[1], "G"] * AP.loc[p[0], "G"] for p in pairs ),
    name="C1*_Transfer_via_hub"
)

# C2: Capacity per flight
for (i, j) in pairs:
    model.addConstr(
        x[(i, j)] + 
            gp.quicksum( w[(i, m)] * (1 - AP.loc[j, "G"]) for m in AP.index if m != j if m != i)
             + gp.quicksum( w[(m, j)] * ( 1- AP.loc[i, "G"]) for m in AP.index if m != j if m != i)
        <= gp.quicksum(z[i, j, k] * AC.loc[k, "seats"] * LF for k in K),
        name=f"C2_Capacity_{i}_{j}"
    )

# C3: Every flight goes back and forth
#for (i,j) in pairs:
for i in AP.index:
    for k in K:
        model.addConstr(
                gp.quicksum(z[i, j, k] for j in AP.index if j != i)
                ==
                gp.quicksum(z[j, i, k] for j in AP.index if j != i),
                name=f"C3_RoundTrip_{i}_{k}"
            )
        
# C4: Fleet availability: Operating time is smaller than available time
for k in K:
    model.addConstr(
        gp.quicksum(
            z[i, j, k] * (
                OD.loc[(i, j), "distance"] / AC.loc[k, "speed"] +
                AC.loc[k, "average_tat"] / 60 * (1 + 0.5 * (1 - AP.loc[j, "G"]))
            )
            for (i, j) in pairs
        )
        <= utilization_time * y[k],
        name=f"C4_Fleet_{k}"
    )

# C5: Range constraint
for (i, j) in pairs:
    for k in K:
        model.addConstr(
            z[i, j, k] <= a_ijk[(i, j, k)],
            name=f"C5_Range_{i}_{j}_{k}"
        )

# C6: All flights start or end at hub
for (i, j) in pairs:
    for k in K:
        model.addConstr(
            z[i, j, k] * AP.loc[i, "G"] * AP.loc[j, "G"] <= 0.001,
            name=f"C6_Hub_{i}_{j}_{k}"
        )

# C7 : Runway length constraint
for (i, j) in pairs:
    for k in K:
        model.addConstr(
            z[i, j, k] * AC.loc[k, "runway_required"] <= AP.loc[j, "runway_length"] *  z[i, j, k]  ,
            name=f"C7_Runway_{i}_{j}_{k}"
        )

# C8: Available slots for landing constraint
for j in AP.index:
    model.addConstr(
        gp.quicksum(z[i, j, k] for i in AP.index if i != j for k in K) <= AP.loc[j, "available_slots"],
        name=f"C8_Available_slots_{j}")

# Optimize model
model.optimize()
# Retrieve and print results
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal objective value: {model.objVal}")
    for v in model.getVars():
        if v.X > 0:
            # if v.VarName.startswith("wij"):
                
            print(f"{v.VarName}: {v.X}")

model.write("model.lp")  
# After model.optimize()

eps = 1e-12

print("\n=== Binding / relevant constraints for non-zero z variables ===")

z_values = {}
for z in model.getVars():
    if not z.VarName.startswith("z"):
        continue
    if abs(z.X) < eps:
        continue   # skip z == 0



    for constr in model.getConstrs():
        coef = model.getCoeff(constr, z)
        z_values[z.VarName] = z.X

print(z_values)


def make_map(z_values):
    coordinates_df = ICAO_coordinates('Problem 1 - Data\\airport_data.xlsx')
    import matplotlib.pyplot as plt
    world = gpd.read_file('https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip')

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot all airports
    ax.scatter(coordinates_df['longitude'], coordinates_df['latitude'], 
               color='black', s=30, zorder=5, label='Airports')


        
    oceans = gpd.read_file(
    "https://naciscdn.org/naturalearth/110m/physical/ne_110m_ocean.zip"
).to_crs("EPSG:4326")

    # Load world map from Natural Earth data
    # --- WATER ---
    oceans.plot(
        ax=ax,
        color='lightblue',
        edgecolor='none',
        zorder=0
    )

    # --- LAND / COUNTRIES ---
    world.plot(
        ax=ax,
        column= None,          # or 'has_airport' or None
        color='#f7c97f',
        edgecolor='black',
        linewidth=0.5,
        legend=True,
        zorder=1
    )


    # Plot flight routes where z > 0.8
    for z_var_name, z_value in z_values.items():
        if z_value > 0.8:
            # Parse z variable name: zijk[i,j,k]
            parts = z_var_name.split('[')[1].rstrip(']').split(',')
            airport_i = parts[0].strip()
            airport_j = parts[1].strip()
            
            if airport_i in coordinates_df.index and airport_j in coordinates_df.index:
                lon_i = coordinates_df.loc[airport_i, 'longitude']
                lat_i = coordinates_df.loc[airport_i, 'latitude']
                lon_j = coordinates_df.loc[airport_j, 'longitude']
                lat_j = coordinates_df.loc[airport_j, 'latitude']
                
                # Extract aircraft type from z variable name
                parts = z_var_name.split('[')[1].rstrip(']').split(',')
                aircraft_type = parts[2].strip()
                
                # Define color map for aircraft types
                color_map = {
                    'AC1': 'indigo',
                    'AC2': 'orange',
                    'AC3': 'royalblue',
                    'AC4': 'green'
                }
                
                line_color = color_map.get(aircraft_type, 'gray')
                
                arrow = FancyArrowPatch((lon_i, lat_i), (lon_j, lat_j),
                                        arrowstyle='-', mutation_scale=20, 
                                        color=line_color, alpha=0.6, linewidth=3, zorder=4)
                ax.add_patch(arrow)

            # Add airport labels
    for airport, row in coordinates_df.iterrows():
        ax.annotate(airport, (row['longitude'], row['latitude']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Create legend for aircraft types
        legend_elements = [Line2D([0], [0], color='indigo', lw=3, label='AC1'),
                           Line2D([0], [0], color='orange', lw=3, label='AC2'),
                           Line2D([0], [0], color='royalblue', lw=3, label='AC3'),
                           Line2D([0], [0], color='green', lw=3, label='AC4')]
        ax.legend(handles=legend_elements, loc='upper left')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Southern Star Airlines network map')
        # Longitude (x-axis)
    ax.set_xlim(-30, 35)

    # Latitude (y-axis)
    ax.set_ylim(30, 70)


    # Plot the world map as background
    world.boundary.plot(ax=ax, linewidth=1, color='black')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def write_schedule(z_values, filename="flight_schedule.xlsx"):
    
    # Parse z_values into structured data
    schedule_data = []
    for z_var_name, z_value in z_values.items():
        parts = z_var_name.split('[')[1].rstrip(']').split(',')
        airport_i = parts[0].strip()
        airport_j = parts[1].strip()
        aircraft_type = parts[2].strip()
        
        schedule_data.append({
            'Origin': airport_i,
            'Destination': airport_j,
            'Aircraft': aircraft_type,
            'Flights': z_value
        })
    
    # Create DataFrame and write to Excel
    df = pd.DataFrame(schedule_data)
    df.to_excel(filename, index=False, sheet_name='Schedule')
    
    # Format Excel file
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    

    
    wb.save(filename)

def make_map_frequency(z_values):
    linewidth = 3
    coordinates_df = ICAO_coordinates('Problem 1 - Data\\airport_data.xlsx')
    import matplotlib.pyplot as plt
    world = gpd.read_file('https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip')

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot all airports
    ax.scatter(coordinates_df['longitude'], coordinates_df['latitude'], 
               color='black', s=30, zorder=5, label='Airports')

    oceans = gpd.read_file(
    "https://naciscdn.org/naturalearth/110m/physical/ne_110m_ocean.zip"
).to_crs("EPSG:4326")

    # Load world map from Natural Earth data
    # --- WATER ---
    oceans.plot(
        ax=ax,
        color='lightblue',
        edgecolor='none',
        zorder=0
    )

    # --- LAND / COUNTRIES ---
    world.plot(
        ax=ax,
        column= None,
        color='#f7c97f',
        edgecolor='black',
        linewidth=0.5,
        legend=True,
        zorder=1
    )

    # Plot flight routes where z > 0.8, color by frequency
    for z_var_name, z_value in z_values.items():
        if z_value > 0.8:
            parts = z_var_name.split('[')[1].rstrip(']').split(',')
            airport_i = parts[0].strip()
            airport_j = parts[1].strip()
            
            if airport_i in coordinates_df.index and airport_j in coordinates_df.index:
                lon_i = coordinates_df.loc[airport_i, 'longitude']
                lat_i = coordinates_df.loc[airport_i, 'latitude']
                lon_j = coordinates_df.loc[airport_j, 'longitude']
                lat_j = coordinates_df.loc[airport_j, 'latitude']
                
                # Color based on frequency (z_value)
                if z_value <= 1.5:
                    line_color = 'royalblue'
                    linewidth = linewidth
                elif z_value <= 2.5:
                    line_color = 'darkgreen'
                    linewidth = linewidth
                elif z_value <= 3.5:
                    line_color = 'violet'
                    linewidth = linewidth
                else:
                    line_color = 'bisque'
                    linewidth = linewidth
                
                arrow = FancyArrowPatch((lon_i, lat_i), (lon_j, lat_j),
                                        arrowstyle='-', mutation_scale=20, 
                                        color=line_color, alpha=0.6, linewidth=linewidth, zorder=4)
                ax.add_patch(arrow)

    # Add airport labels
    for airport, row in coordinates_df.iterrows():
        ax.annotate(airport, (row['longitude'], row['latitude']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Create legend for frequencies
    legend_elements = [Line2D([0], [0], color='royalblue', lw=linewidth, label='1 time/week'),
                       Line2D([0], [0], color='darkgreen', lw=linewidth, label='2 times/week'),
                       Line2D([0], [0], color='violet', lw=linewidth, label='3 times/week'),
                       Line2D([0], [0], color='bisque', lw=linewidth, label='4+ times/week')]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Southern Star Airlines network map')
    ax.set_xlim(-30, 35)
    ax.set_ylim(30, 70)

    world.boundary.plot(ax=ax, linewidth=1, color='black')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


#write_schedule(z_values)

make_map_frequency(z_values)

make_map(z_values)

# for var in model.getVars():
#     if not var.VarName.startswith("z"):
#         continue
#     if abs(var.X) < eps:
#         continue   # skip z == 0

#     print(f"\nVariable {var.VarName} = {var.X}")

    # for constr in model.getConstrs():
    #     coef = model.getCoeff(constr, var)
    #     if abs(coef) < eps:
    #         continue  # variable not in this constraint

    #     slack = constr.Slack
    #     print(f"  - {constr.ConstrName}: coef={coef}, slack={slack}")


'''
When changing the Hub to EGLL, more than one flight per week is operated for some routes and AC2, 3 and 4 all have one aircraft.
WHen increasing the yield for the madrid airport by a lot, more routes are served and with more AC. 
WHen changing the 5.9 in the yield funtion to 200. THis is the result:
Also add 70 hours operation constrain is a simplification.

y[AC2]: 1.0
y[AC3]: 4.0
y[AC4]: 4.0
zijk[BIKF,LEMF,AC3]: 2.0
zijk[EDDF,LEMF,AC4]: 6.0
zijk[EDDM,LEMF,AC2]: 1.0
zijk[EDDM,LEMF,AC3]: 5.0
zijk[EDDM,LEMF,AC4]: 4.0
zijk[EDDT,LEMF,AC4]: 1.0
zijk[EFHK,LEMF,AC3]: 1.0
zijk[EFHK,LEMF,AC4]: 2.0
zijk[EGLL,LEMF,AC3]: 5.0
zijk[EGLL,LEMF,AC4]: 10.0
zijk[EGPH,LEMF,AC3]: 2.0
zijk[EHAM,LEMF,AC4]: 1.0
zijk[EIDW,LEMF,AC4]: 3.0
zijk[EPWA,LEMF,AC3]: 3.0
zijk[ESSA,LEMF,AC4]: 3.0
zijk[LEBL,LEMF,AC4]: 3.0
zijk[LEMF,BIKF,AC3]: 2.0
zijk[LEMF,EDDF,AC4]: 6.0
zijk[LEMF,EDDM,AC2]: 1.0
zijk[LEMF,EDDM,AC3]: 5.0
zijk[LEMF,EDDM,AC4]: 4.0
zijk[LEMF,EDDT,AC4]: 1.0
zijk[LEMF,EFHK,AC3]: 1.0
zijk[LEMF,EFHK,AC4]: 2.0
zijk[LEMF,EGLL,AC3]: 5.0
zijk[LEMF,EGLL,AC4]: 10.0
zijk[LEMF,EGPH,AC3]: 2.0
zijk[LEMF,EHAM,AC4]: 1.0
zijk[LEMF,EIDW,AC4]: 3.0
zijk[LEMF,EPWA,AC3]: 3.0
zijk[LEMF,ESSA,AC4]: 3.0
zijk[LEMF,LEBL,AC4]: 3.0
zijk[LEMF,LFPG,AC3]: 1.0
zijk[LEMF,LFPG,AC4]: 9.0
zijk[LEMF,LGIR,AC2]: 1.0
zijk[LEMF,LGIR,AC3]: 1.0
zijk[LEMF,LICJ,AC3]: 3.0
zijk[LEMF,LIRF,AC3]: 8.0
zijk[LEMF,LIRF,AC4]: 2.0
zijk[LEMF,LPMA,AC3]: 2.0
zijk[LEMF,LPPT,AC3]: 2.0
zijk[LEMF,LPPT,AC4]: 1.0
zijk[LEMF,LROP,AC3]: 3.0
zijk[LFPG,LEMF,AC3]: 1.0
zijk[LFPG,LEMF,AC4]: 9.0
zijk[LGIR,LEMF,AC2]: 1.0
zijk[LGIR,LEMF,AC3]: 1.0
zijk[LICJ,LEMF,AC3]: 3.0
zijk[LIRF,LEMF,AC3]: 8.0
zijk[LIRF,LEMF,AC4]: 2.0
zijk[LPMA,LEMF,AC3]: 2.0
zijk[LPPT,LEMF,AC3]: 2.0
zijk[LPPT,LEMF,AC4]: 1.0
zijk[LROP,LEMF,AC3]: 3.0
'''