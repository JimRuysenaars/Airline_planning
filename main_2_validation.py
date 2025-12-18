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


path_flights = 'Problem 2 - Data/flights_test.xlsx'
path_itineraries = 'Problem 2 - Data/itineraries_test.xlsx'
path_recapture = 'Problem 2 - Data/recapture_test.xlsx'

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

        if not pd.isna(f1):
            f1 = str(int(f1)) if isinstance(f1, (int, float)) else str(f1)
            if f1 in flights_index:
                DEL.loc[it, f1] = 1

        if not pd.isna(f2):
            f2 = str(int(f2)) if isinstance(f2, (int, float)) else str(f2)
            if f2 in flights_index:
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
            B.loc[(p, r), 'b_pr'] = 0.0  # ensure b_pp = 0 -> can't transfer from itinerary 1 to 1
        # artificial → r recapture allowed with rate 1.0
    # for r in P:
    #     B.loc[("artificial", r), "b_pr"] = 1.0


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
            print(f"Flight {flight}, Itinerary {it}: δ = {delta}, Demand = {demand}")
            total_demand += delta * demand

        # Store in Q
        Q.loc[flight, "Q"] = total_demand
        print(f"Flight {flight}: Total Demand Q = {total_demand}")

    return Q

# def make_PR0_list(P):
#     # PR = [(p, p) for p in P]
#     PR0 = [(p, "artificial") for p in P if p != "artificial"]
#     #PR += [("artificial", p) for p in P]
#     # PR0 = list(dict.fromkeys(PR))
#     return PR0

# def make_PR_total_list(P):
#     PR = [(p, r) for p in P for r in P]
#     # PR += [(p, "artificial") for p in P]
#     PR_total = list(dict.fromkeys(PR))
#     return PR_total

# ---------- BUILD DATAFRAMES ----------

# Sets
FL = create_FLIGHTS(path_flights)
FL.index = FL.index.astype(str)
L = FL.index.tolist()       # flights i
IT, DEL = create_ITINERARIES(path_itineraries, flights_index=FL.index)
P = IT.index.tolist()       # itineraries p
R = list(P) + ["artificial"]
Qi = create_Q(DEL, IT)

# Recapture pairs (p, r)
B = create_RECAPTURE(path_recapture, P)


# ---------- GUROBI MODEL ----------

def solve_model(PR, integer=False):

    # ---------- GUROBI MODEL ----------
    model = gp.Model("Passenger_Mix_Flow")

    vtype = gp.GRB.INTEGER if integer else gp.GRB.CONTINUOUS
    t = model.addVars(PR, name="t_pr", lb=0.0, vtype=vtype)

    
    # Objective
    lost_revenue = gp.quicksum(
        (IT.loc[p, 'Price [EUR]'] - B.loc[(p, r), "b_pr"] * IT.loc[r, 'Price [EUR]']) * t[p,r]
        for (p, r) in PR )

    model.setObjective(lost_revenue, gp.GRB.MINIMIZE)

    # ---------- Constraints ----------
    
    # C1: seat capacity on each flight i
    # for i in L:
    #     model.addConstr(
    #         gp.quicksum(
    #             DEL.loc[p,i] * t[p, r] for (p, r) in PR) - gp.quicksum(DEL.loc[p, i] * B.loc[(r, p), "b_pr"] * t[r, p] for (p, r) in PR if r != 'artificial' )
    #             >= Qi.loc[i, "Q"] - FL.loc[i, 'Capacity'],
    #         name=f"C1_Capacity_{i}"
    #     )

    for i in L:
        model.addConstr(
            gp.quicksum(
                DEL.loc[p,i] * t[p, r] for (p, r) in PR) - gp.quicksum(DEL.loc[p, i] * B.loc[(r, p), "b_pr"] * t[r, p] for (r, p) in PR if r != 'artificial' )
                >= Qi.loc[i, "Q"] - FL.loc[i, 'Capacity'],
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

    t0 = time.perf_counter()
    model.optimize()
    t1 = time.perf_counter()

    result = {
        "status": model.status,
        "runtime": t1 - t0,
        "obj": None,
        "duals": {"pi": {}, "sigma": {}},
        "t_values": {}
    }

    model.write("model_part2_test.lp")
    if model.status == gp.GRB.OPTIMAL:
        result["obj"] = model.objVal
        # model.write("model_part2_test.lp")
        # capture decision variable values (only for created PR)
        for (p, r) in PR:
            # Gurobi returns .X for var
            val = t[p, r].X if t[p, r] is not None else 0.0
            result["t_values"][(p, r)] = float(val)

        if integer:
            return result
        
    # ---------- Return dual variables ----------
    for constr in model.getConstrs():
        name = constr.ConstrName
        if name.startswith("C1_Capacity_"):
            i = name[len("C1_Capacity_"):]
            result["duals"]["pi"][i] = constr.Pi
            result.setdefault("slack", {})[name] = constr.Slack
        elif name.startswith("C2_Demand_"):
            p = name[len("C2_Demand_"):]
            result["duals"]["sigma"][p] = constr.Pi
            result.setdefault("slack", {})[name] = constr.Slack

    print(result["duals"]["pi"])
    return result

"""

TODO: Check that all variablies are t(p,r) where p is the low!! and r the high in notation in the slide
                                    t(p,r) = t(low,high)

"""

# ---------- Loop generating columns ----------

columns = [(p, "artificial") for p in P if p != "artificial"]
initial_column_count = len(columns)
iteration = 0

start_total = time.perf_counter()

with open("column_generation_log_test.txt", "a") as f:
    f.write(f"===== New Run =====\n")

obj_values = []
recaptured_values = []
iteration_numbers = []

running = True
while running:
    iteration += 1
    # solve function that solves model with given columns
    # assert all((r,p) not in columns for (p,r) in columns), "Reverse arc detected!"
    res = solve_model(columns)
    duals = res["duals"]

    # Track iteration number
    iteration_numbers.append(iteration)

    # A. Save objective value
    obj_values.append(res["obj"])

    # B. Save recaptured artificial → real passengers
    tvals = res["t_values"]
    recaptured = sum(tvals.get(("artificial", p), 0.0) for p in P)
    recaptured_values.append(recaptured)


    # calculate reduced costs based on duals
    for (p, r), b_pr in B['b_pr'].items():
        if b_pr == 0:
            continue

        reduced_cost = (sum( (DEL.loc[p, i] - DEL.loc[r,i] * B.loc[(p, r), "b_pr"])  *  duals["pi"][i] 
                            for i in L) 
        + duals["sigma"][p] 
        - (IT.loc[p, 'Price [EUR]'] - B.loc[(p, r), "b_pr"] * IT.loc[r, 'Price [EUR]'] ))
        
        B.loc[(p, r), "reduced_cost"] = -reduced_cost
    
    # select new columns with the most negative reduced cost
    reduced_costs = [
    {'pair': (p, r), 'reduced_cost': B.loc[(p, r), 'reduced_cost']}
    for (p, r), b_pr in B['b_pr'].items()
    if b_pr != 0]
    
    reduced_costs = sorted(reduced_costs, key=lambda x: x['reduced_cost'])
    pr_min_red_cost = reduced_costs[0]['pair']      # ('p','r') with lowest reduced cost
    print("Reduced costs sorted: ", reduced_costs)
    pr_min_red_cost_reverse = (pr_min_red_cost[1], pr_min_red_cost[0])


    # Add selected columns to columns
    searching = True
    while searching:
        if pr_min_red_cost not in columns:
            # columns.extend([pr_min_red_cost, pr_min_red_cost_reverse])
            columns.extend([pr_min_red_cost])
            searching = False

        else:
            print(f"Pair not selected: {pr_min_red_cost} already in columns. Searching for next best.")
            reduced_costs.pop(0)
            if len(reduced_costs) == 0:
                searching = False
                break
            else:
                pr_min_red_cost = reduced_costs[0]['pair']
                pr_min_red_cost_reverse = (pr_min_red_cost[1], pr_min_red_cost[0])

    #print("Hier komt res: ", res["slack"])
   
    # Print info
    print(f"Selected column (p,r) = {pr_min_red_cost} with reduced cost = {B.loc[pr_min_red_cost, 'reduced_cost']:.2f}")

    # Check stopping criteria
    if B.loc[pr_min_red_cost, "reduced_cost"] >= -0.001:
        running = False




end_total = time.perf_counter()
total_runtime = end_total - start_total
print(f"COLUMNS: {columns}")
# final solve to get final t_values & duals (ensure final RMP optimal)
final_res = solve_model(columns, integer=False)

# compute total spilled passengers:
t_vals = final_res["t_values"]
total_spilled = 0.0
for p in P:
    flown = sum(t_vals.get((p, r), 0.0) for r in P if (p, r) in t_vals)
    spilled = max(0.0, IT.loc[p, 'Demand'] - flown)
    total_spilled += spilled

# prepare concise outputs
initial_cols = initial_column_count
final_cols = len(columns)
iterations = iteration
optimal_obj = final_res["obj"]

# first 5 itineraries (show t for all r)
first5_itins = P[:5]
first5_t = {p: {r: t_vals.get((p, r), 0.0) for r in P if (p, r) in t_vals} for p in first5_itins}

# first 5 flights duals
first5_flights = L[:5]
first5_duals = {i: final_res["duals"]["pi"].get(i, 0.0) for i in first5_flights}

# print summary
print("\n===== FINAL SUMMARY =====")
print(f"Optimal objective (airline cost) = {optimal_obj:.2f}")
print(f"Total passengers spilled = {total_spilled:.2f}")
print(f"Number of columns (RMP) before = {initial_cols}, after = {final_cols}")
print(f"Number of iterations = {iterations}")
print(f"Total column-generation runtime (s) = {total_runtime:.2f}")
print("\nDecision variables for first 5 itineraries:")
nonzero_t = [(p, r, v) for (p, r), v in t_vals.items() if abs(v) > 1e-9]
# take first 5
for p, r, v in nonzero_t[:5]:
    print(f"t[{p},{r}] = {v:.2f}")

print("\nDuals (pi) for first 5 flights:")
nonzero_duals = [(i, v) for i, v in final_res["duals"]["pi"].items() if abs(v) > 1e-9]
for i, v in nonzero_duals[:5]:
    print(f"Flight {i}: pi = {v:.4f}")



summary = {
    "optimal_obj": optimal_obj,
    "total_spilled": total_spilled,
    "initial_columns": initial_cols,
    "final_columns": final_cols,
    "iterations": iterations,
    "total_runtime_s": total_runtime,
    "first5_itineraries_t": first5_t,
    "first5_flights_duals": first5_duals
}



t_df = (
    pd.DataFrame(
        [(p, r, val) for (p, r), val in t_vals.items()],
        columns=["p", "r", "t_value"]
    )
)

t_df.to_excel("t_values_validation_test.xlsx", index=False)
print("Saved t_values to t_values_validation_test.xlsx")



