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


import pandas as pd
import gurobipy as gp

# ---------- DATA READING FUNCTIONS ----------

def create_FLIGHTS(path):
    """
    flights.xlsx:
        columns: ['flight_id', 'capacity', ...]
    """
    df = pd.read_excel(path)
    # Make sure 'flight_id' and 'capacity' match your sheet
    df = df.set_index('Flight No.')
    # Keep only capacity (you can keep more cols if useful)
    FL = df[['Capacity']].copy()
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
    df = df.set_index('Itinerary')

    # parameter tables
    IT = df[['Price [EUR]', 'Demand']].copy()

    # incidence matrix δ_pi: use the flight IDs as columns
    DEL = df[flights_index].astype(float)
    return IT, DEL


def create_RECAPTURE(path):
    """
    recapture.xlsx:
        columns: ['p', 'r', 'b_rp']
        p = original itinerary, r = itinerary used, b_rp = recapture rate
    Returns:
        B    : Series with MultiIndex (p,r) -> b_rp
        PR   : list of (p,r) tuples (valid recapture pairs)
    """
    df = pd.read_excel(path)
    B = df.set_index(['From Itinerary', 'To Itinerary'])['Recapture Rate'].astype(float)
    PR = list(B.index)    # list of (p,r) tuples
    return B, PR


# ---------- BUILD DATAFRAMES ----------

FL = create_FLIGHTS(path_flights)
IT, DEL = create_ITINERARIES(path_itineraries, flights_index=FL.index)
B, PR = create_RECAPTURE(path_recapture)

# Sets
L = FL.index.tolist()       # flights i
P = IT.index.tolist()       # itineraries p

# Make sure DEL has rows P and columns L
assert list(DEL.index) == P
assert list(DEL.columns) == L


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
