import pandas as pd
import numpy as np
import gurobipy as gp
import itertools


path_flights = 'Problem 2 - Data/flights.xlsx'
path_itineraries = 'Problem 2 - Data/itineraries.xlsx'
path_recapture = 'Problem 2 - Data/recapture.xlsx'

"""
def get_data(path_flights, path_itineraries, path_recapture):
    df_flights = pd.read_excel(path_flights, index_col=0, header=0)
    df_itineraries = pd.read_excel(path_itineraries, index_col=0, header=0)
    df_recapture = pd.read_excel(path_recapture, header=0)

    return df_flights, df_itineraries, df_recapture
"""

# ---------- DATA READING FUNCTIONS ----------

def create_FLIGHTS(path):
    """
    flights.xlsx:
        columns: ['flight_id', 'capacity', ...]
    """
    FL = pd.read_excel(path, index_col="Flight No.", header=0)
    return FL

def create_ITINERARIES(path, flights_index):
    """
    itineraries.xlsx:
        columns: ['itinerary_id', 'fare', 'demand', <one column per flight>]
        flight columns should have the same labels as flights_index
        and contain 0/1 (δ_pi).
    Returns:
        IT  : dataframe with fare, demand (index = itinerary id)
        DEL : dataframe δ_pi (rows = itineraries, cols = flights)
    """
    df = pd.read_excel(path)
    # Set the index to the 'Itinerary' column and make them strings
    df = pd.read_excel(path, dtype={'Itinerary': str})
    df = df.set_index('Itinerary')

    # parameter tables
    IT = df[['Price [EUR]', 'Demand']].copy()

    # Create empty incidence matrix
    DEL = pd.DataFrame(0, index=df.index, columns=flights_index)

    # For each itinerary, set 1 for the flights in 'Flight 1' and 'Flight 2'
    for it in df.index:
        f1 = df.loc[it, "Flight 1"]
        f2 = df.loc[it, "Flight 2"]

        if isinstance(f1, str) and f1 in flights_index:
            DEL.loc[it, f1] = 1
        if isinstance(f2, str) and f2 in flights_index:
            DEL.loc[it, f2] = 1

    return IT, DEL

def create_RECAPTURE(path, P):
    """
    recapture.xlsx:
        columns: ['p', 'r', 'b_rp']
        p = original itinerary, r = itinerary used, b_rp = recapture rate
    """

    df = pd.read_excel(path, dtype={'From Itinerary': str, 'To Itinerary': str, 'Recapture Rate': float})

    B = pd.DataFrame(index=P, columns=P, dtype=float)
    for rows in df.itertuples(index=False):
        p = rows[0]  # From Itinerary
        r = rows[1]  # To Itinerary
        b_rp = rows[2]  # Recapture Rate
        B.loc[p, r] = b_rp

    for p, r in itertools.product(P, P):
        if p == r:
            B.loc[p, r] = 1.0  # ensure b_pp = 1

    B = B.fillna(0.0)  # fill NaN with 0    

    return B


# ---------- BUILD DATAFRAMES ----------


# Sets
FL = create_FLIGHTS(path_flights)
L = FL.index.tolist()       # flights i
IT, DEL = create_ITINERARIES(path_itineraries, flights_index=FL.index)
P = IT.index.tolist()       # itineraries p

# Recapture pairs (p, r)
B = create_RECAPTURE(path_recapture, P)


PR = [(p, r) for p in P for r in P if B.loc[p, r] == 1.0]

print(PR)


# ---------- GUROBI MODEL ----------

model = gp.Model("Passenger_Mix_Flow")

# Decision variables: x_rp = pax originally from p, flown on r
# Only create vars for recapture pairs listed in PR
x = model.addVars(PR, name="x_rp", lb=0.0, vtype=gp.GRB.CONTINUOUS)

# ---------- Objective: max total revenue ----------

# revenue = sum_{(p,r)} fare_r * x_rp
revenue = gp.quicksum(
    IT.loc[r, 'fare'] * x[p, r]
    for (p, r) in PR
)

model.setObjective(revenue, gp.GRB.MAXIMIZE)

# ---------- Constraints ----------

# C1: seat capacity on each flight i
for i in L:
    model.addConstr(
        gp.quicksum(
            DEL.loc[r, i] * x[p, r]
            for (p, r) in PR
        ) <= FL.loc[i, 'capacity'],
        name=f"C1_Capacity_{i}"
    )

# C2: cannot reallocate more pax from itinerary p than its demand
for p in P:
    model.addConstr(
        gp.quicksum(
            x[p, r] / B[(p, r)]
            for (pp, r) in PR if pp == p
        ) <= IT.loc[p, 'demand'],
        name=f"C2_Demand_{p}"
    )

# (C3: x_rp >= 0 is already enforced by lb=0)


model.optimize()

if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal objective (revenue) = {model.objVal:.2f}")
    print("\nNon-zero flows x_rp:")
    for (p, r) in PR:
        if x[p, r].X > 1e-6:
            print(f"x[{p},{r}] = {x[p,r].X:.2f}")
