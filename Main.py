
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
AC = pd.read_excel('C:\\Users\\jimru\\OneDrive\\Documenten\\python\\Airline_planning\\Problem 1 - Data\\AircraftData.xlsx', index_col=0)

print(AC)




I = ["Origin", "Origin"]           # To be filled in
J = ["Destination", "Destination"] # To be filled in
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



