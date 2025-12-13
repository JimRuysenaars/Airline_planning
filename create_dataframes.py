import pandas as pd
import numpy as np
import gurobipy as gp
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


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

def save_x_values_to_excel(x_values, filename="x_values_final.xlsx"):
    df = pd.DataFrame(
        [(p, r, val) for (p, r), val in x_values.items()],
        columns=["p", "r", "x_value"]
    )
    df.to_excel(filename, index=False)
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