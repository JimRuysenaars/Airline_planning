import pandas as pd
import numpy as np
import gurobipy as gp
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- FILE PATHS ----------

path_flights =      'Problem 2 - Data/flights.xlsx'
path_itineraries = 'Problem 2 - Data/itineraries.xlsx'
path_recapture =    'Problem 2 - Data/recapture.xlsx'

# ---------- DATA READING FUNCTIONS ----------

def create_FLIGHTS(path):
    """
    This function reads the flights.xlsx file and returns a DataFrame
    with flight information.

    -input-
    path : str : path to flights.xlsx file

    -output-
    FL   : DataFrame : dataframe with flight information
    """
    FL = pd.read_excel(path, index_col="Flight No.", header=0)
    return FL

def create_ITINERARIES(path, flights_index):
    """
    This function reads the itineraries.xlsx file and returns two DataFrames:
    - IT : DataFrame : dataframe with itinerary parameters (price, demand)
    - DEL: DataFrame : incidence matrix of itineraries and flights

    -input-
    path          : str : path to itineraries.xlsx file
    flights_index : list: list of flight IDs

    -output-
    IT   : DataFrame : dataframe with itinerary parameters (price, demand)
    DEL  : DataFrame : incidence matrix of itineraries and flights
    """

    df = pd.read_excel(path, dtype={'Itinerary': str})
    df = df.set_index('Itinerary')

    # Extract itinerary parameters
    IT = df[['Price [EUR]', 'Demand']].copy()

    # Create empty incidence matrix DEL
    DEL = pd.DataFrame(0, index=df.index, columns=flights_index)

    # For each itinerary, set delta to 1 for both flights in the itinerary
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
    This function reads the recapture.xlsx file and returns a DataFrame B
    with recapture rates b_pr for each pair of itineraries (p, r).
    
    -input-
    path : str : path to recapture.xlsx file
    P    : list: list of itinerary IDs

    -output-
    B    : DataFrame : dataframe with recapture rates b_pr for each (p, r)
    """

    df =    pd.read_excel(path, dtype={'From Itinerary': str, 'To Itinerary': str, 'Recapture Rate': float})
    index = pd.MultiIndex.from_product([P, P], names=['p', 'r'])
    B =     pd.DataFrame(index=index, columns=['b_pr', 'reduced_cost'] , dtype=float)

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
    """
    This function computes the total demand Q_i for each flight i
    based on the incidence matrix DEL and the itinerary demand IT.

    -input-
    DEL : DataFrame : incidence matrix of itineraries and flights
    IT  : DataFrame : dataframe with itinerary parameters (price, demand)

    -output-
    Q   : DataFrame : dataframe with total demand Q_i for each flight i
    """

    # Create an empty Q dataframe indexed by flight IDs
    Q = pd.DataFrame(0.0, index=DEL.columns, columns=["Q"])

    # Loop over each flight and add up demand from all itineraries
    for flight in DEL.columns:
        total_demand = 0.0

        for it in DEL.index:
            delta = DEL.loc[it, flight]
            demand = IT.loc[it, "Demand"]
            total_demand += delta * demand

        # Store in Q
        Q.loc[flight, "Q"] = total_demand

    return Q

def make_PR_total_list(P):
    '''
    This function creates a list of all (p, r) pairs where recapture is possible
    based on the recapture DataFrame B.

    -input-
    P : list : list of itinerary IDs

    -output-
    PR_total : list : list of (p, r) pairs with possible recapture
    '''

    PR = [(p, r) for p in P for r in P if B.loc[(p, r), "b_pr"] != 0]
    PR_total = list(dict.fromkeys(PR))

    return PR_total

def save_x_values_to_excel(x_values, filename="x_values_final.xlsx"):
    """
    This function saves the decision variable values x_values to an Excel file.

    -input-
    x_values : dict : dictionary with keys as (p, r) and values as x_pr
    filename  : str  : name of the output Excel file
    """

    df = pd.DataFrame(
        [(p, r, val) for (p, r), val in x_values.items()],
        columns=["p", "r", "x_value"]
    )
    df.to_excel(filename, index=False)

# ---------- BUILD DATAFRAMES ----------

FL =        create_FLIGHTS(path_flights)                                        # Flight data
L =         FL.index.tolist()                                                   # List of flight IDs
IT, DEL =   create_ITINERARIES(path_itineraries, flights_index=FL.index)        # Itinerary data + incidence matrix
P =         IT.index.tolist()                                                   # List of itinerary IDs
R =         list(P) + ["artificial"]                                            # List of recapture itineraries + artificial
Qi =        create_Q(DEL, IT)                                                   # Total demand per flight

# Recapture pairs (p, r)
B = create_RECAPTURE(path_recapture, P)
PR0 = make_PR_total_list(P)

# ---------- GUROBI MODEL ----------

start_total = time.perf_counter()

def solve_model(PR):

    # ---------- GUROBI MODEL ----------
    model =     gp.Model("Passenger_Mix_Flow_Original")
    x =         model.addVars(PR, name="x_pr", lb=0.0, vtype=gp.GRB.INTEGER)
    
    # Objective function: maximize total revenue
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
            gp.quicksum(x[p, r] / B.loc[(p, r), "b_pr"] for r in P if (p, r) in PR and r != 'artificial') <= IT.loc[p, 'Demand'],
            name=f"C2_Demand_{p}"
        )

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
    model.write("P2_Original_model.lp")


    # Save results
    result = {
        "status": model.status,
        "runtime_s": t1 - t0,
        "num_columns": len(PR), 
        "obj": model.objVal if model.status == gp.GRB.OPTIMAL else None,
        "x_values": {}
    }

    # Write variable values if optimal
    if model.status == gp.GRB.OPTIMAL:
        result["obj"] = model.objVal
        model.write("P2_Original_model.lp")
        for (p, r) in PR:
            val = x[p, r].X if x[p, r] is not None else 0.0
            result["x_values"][(p, r)] = float(val)

    return result



dictionary = solve_model(PR0)
end_total = time.perf_counter()
total_runtime = end_total - start_total

# Show results
print(f"Status: {dictionary['status']}")
print(f"Runtime (s): {dictionary['runtime_s']:.2f}")
print(f"Total runtime (s): {total_runtime}")
print(f"Number of columns in model: {dictionary['num_columns']}")
print(f"Optimal objective: {dictionary['obj']:.2f}")
save_x_values_to_excel(dictionary["x_values"])
