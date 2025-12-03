import pandas as pd
import numpy as np
import gurobipy as gp
import itertools

path_flights = 'Problem 2 - Data/flights.xlsx'
path_itineraries = 'Problem 2 - Data/itineraries.xlsx'
path_recapture = 'Problem 2 - Data/recapture.xlsx'

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
FLi = FL.index.tolist()       # flights i
IT, DEL = create_ITINERARIES(path_itineraries, flights_index=FL.index)
P = IT.index.tolist()       # itineraries p

# Recapture pairs (p, r)
B = create_RECAPTURE(path_recapture, P)

PR0 = [(p, r) for p in P for r in P if B.loc[p, r] == 1.0]

print(PR0)


# ---------- Loop generating columns ----------

while running:

    # solve function that solves model with given columns

    # calculate reduced costs based on duals

    # select new columns wi
h




    
    pass



def solve_model(PR=PR0):

    # ---------- GUROBI MODEL ----------

    model = gp.Model("Passenger_Mix_Flow")

    # Decision variables: t_rp = pax originally from p, flown on r
    # Only create vars for recapture pairs listed in PR
    t = model.addVars(PR, name="t_rp", lb=0.0, vtype=gp.GRB.CONTINUOUS)

    # ---------- Objective: max total revenue ----------

    # revenue = sum_{(p,r)} fare_r * t_rp
    revenue = gp.quicksum(
        IT.loc[r, 'fare'] * t[p, r]
        for (p, r) in PR
    )

    model.setObjective(revenue, gp.GRB.MAXIMIZE)

    # ---------- Constraints ----------

    # C1: seat capacity on each flight i
    for i in FLi:
        model.addConstr(
            gp.quicksum(
                DEL.loc[i, p] * t[p, r] for (p, r) in PR) - gp.quicksum(DEL.loc[i, p] * B.loc[r, p] * t[r, p] for (r, p) in PR)
                <= IT.loc[i, 'Demand'] - FL.loc[i, 'Capacity'],
            name=f"C1_Capacity_{i}"
        )

    # C2: number of passengers is lower than the demand for each itinerary p
    for p in P:
        model.addConstr(
            gp.quicksum(t[p, r] for r in P if (p, r) in PR) <= IT.loc[p, 'Demand'],
            name=f"C2_Demand_{p}"
        )

    # C3: number of passengers is positive
    for (p, r) in PR:
        model.addConstr(
            t[p, r] >= 0.0,
            name=f"C3_NonNegativity_{p}_{r}"
        )

    # ---------- Optimize model ----------
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        print(f"Optimal objective (revenue) = {model.objVal:.2f}")
        print("\nNon-zero flows x_rp:")
        for (p, r) in PR:
            if t[p, r].X > 1e-6:
                print(f"x[{p},{r}] = {t[p,r].X:.2f}")