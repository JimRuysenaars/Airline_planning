import pandas as pd
import numpy as np
import gurobipy as gp
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import json


path_flights = 'Problem 2 - Data/flights.xlsx'
path_itineraries = 'Problem 2 - Data/itineraries.xlsx'
path_recapture = 'Problem 2 - Data/recapture.xlsx'

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
        columns: ['p', 'r', 'b_pr']
        p = original itinerary, r = itinerary used, b_pr = recapture rate
    """

    df = pd.read_excel(path, dtype={'From Itinerary': str, 'To Itinerary': str, 'Recapture Rate': float})

    index = pd.MultiIndex.from_product([P, P], names=['p', 'r'])
    B = pd.DataFrame(index=index, columns=['b_pr', 'reduced_cost'] , dtype=float)

    B['b_pr'] = 0.0
    B['reduced_cost'] = np.nan

    for rows in df.itertuples(index=False):
        p = rows[0]  # From Itinerary
        r = rows[1]  # To Itinerary
        b_pr = rows[2]  # Recapture Rate
        if p in P and r in P:
            B.loc[(p, r), 'b_pr'] = b_pr
        # B.loc[p, r] = {"b_pr" : b_pr, "reduced_cost": None}  # store recapture rate

    for p, r in itertools.product(P, P):
        if p == r:
            B.loc[(p, r), 'b_pr'] = 1.0  # ensure b_pp = 1
        # artificial → r recapture allowed with rate 1.0
    for r in P:
        B.loc[("artificial", r), "b_pr"] = 1.0


    B['b_pr'] = B['b_pr'].fillna(0.0)   # only for b_pr, leave reduced_cost NaN

    return B

def create_Q(DEL, IT):
    # Create an empty Q dataframe indexed by flight IDs
    Q = pd.DataFrame(0.0, index=DEL.columns, columns=["Q"])

    # Loop over each flight (column of DEL)
    for flight in DEL.columns:

        total_demand = 0.0

        # Loop over each itinerary (row of DEL)
        for it in DEL.index:

            delta = DEL.loc[it, flight]      # 0 or 1
            demand = IT.loc[it, "Demand"]    # D_p

            total_demand += delta * demand

        # Store in Q
        Q.loc[flight, "Q"] = total_demand

    return Q

def make_PR0_list(P):
    PR = [(p, p) for p in P]
    PR += [(p, "artificial") for p in P]
    PR += [("artificial", p) for p in P]
    PR0 = list(dict.fromkeys(PR))
    return PR0

def make_PR_total_list(P):
    PR = [(p, r) for p in P for r in P]
    PR += [(p, "artificial") for p in P]
    PR += [("artificial", r) for r in P]
    PR_total = list(dict.fromkeys(PR))
    return PR_total
# ---------- BUILD DATAFRAMES ----------

# Sets
FL = create_FLIGHTS(path_flights)
L = FL.index.tolist()       # flights i
IT, DEL = create_ITINERARIES(path_itineraries, flights_index=FL.index)
P = IT.index.tolist()       # itineraries p
R = list(P) + ["artificial"]
Qi = create_Q(DEL, IT)

# Recapture pairs (p, r)
B = create_RECAPTURE(path_recapture, P)

# TODO: Add all columns (p,r) where p,r in P plus artificial ones
PR0 = make_PR_total_list(P)


# ---------- GUROBI MODEL ----------

def solve_model(PR):

    # ---------- GUROBI MODEL ----------
    model = gp.Model("Passenger_Mix_Flow_Original")
    # Decision variables: t_pr = pax originally from p, flown on r
    # Only create vars for recapture pairs listed in PR
    x = model.addVars(PR, name="x_pr", lb=0.0, vtype=gp.GRB.CONTINUOUS)
    
    # Objective
    revenue = gp.quicksum(
        (IT.loc[r, 'Price [EUR]'] * x[p, r]
        for (p, r) in PR
    ))

    model.setObjective(revenue, gp.GRB.MAXIMIZE)

    # ---------- Constraints ----------
    
    # C1: seat capacity on each flight i
    for i in L:
        model.addConstr(
            gp.quicksum(
                DEL.loc[r,i] * x[p, r] for (p, r) in PR)
                <= FL.loc[i, 'Capacity'],
            name=f"C1_Capacity_{i}"
        )

    # C2: number of passengers is lower than the demand for each itinerary p
    for p in P:
        model.addConstr(
            gp.quicksum(x[p, r] / B.loc[(p, r), "b_pr"] for r in P if (p, r) in PR if B.loc[(p, r), "b_pr"] != 0) <= IT.loc[p, 'Demand'],
            name=f"C2_Demand_{p}"
        )

    # TODO : iterate in the constraint above over P_p and D_p

    # C3: number of passengers is positive
    for (p, r) in PR:
        model.addConstr(
            x[p, r] >= 0.0,
            name=f"C3_NonNegativity_{p}_{r}"
        )

    # ---------- Optimize model ----------

    t0 = time.perf_counter()
    model.optimize()
    t1 = time.perf_counter()

    result = {
        "status": model.status,
        "runtime": t1 - t0,
        "obj": None,
        "x_values": {}
    }


    if model.status == gp.GRB.OPTIMAL:
        result["obj"] = model.objVal
        model.write("P2_Original_model.lp")
        # capture decision variable values (only for created PR)
        for (p, r) in PR:
            # Gurobi returns .X for var
            val = x[p, r].X if x[p, r] is not None else 0.0
            result["x_values"][(p, r)] = float(val)

    # optional: write the lp for debugging
    model.write("P2_Original_model.lp")

    return result


dictionary = solve_model(PR0)

for key, value in dictionary["x_values"].items():
    if abs(value) > 1e-9:   # handles floating point noise
        print(key, value)
