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
            B.loc[(p, r), 'b_pr'] = 0.0  # ensure b_pp = 0 -> can't transfer from itinerary 1 to 1
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
    # PR = [(p, p) for p in P]
    PR = [(p, "artificial") for p in P]
    #PR += [("artificial", p) for p in P]
    PR0 = list(dict.fromkeys(PR))
    return PR0

def make_PR_total_list(P):
    PR = [(p, r) for p in P for r in P]
    # PR += [(p, "artificial") for p in P]
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

# TODO: Pls fix
PR0 = make_PR0_list(P)


# ---------- GUROBI MODEL ----------

def solve_model(PR):

    # ---------- GUROBI MODEL ----------
    model = gp.Model("Passenger_Mix_Flow")
    # Decision variables: t_pr = pax originally from p, flown on r
    # Only create vars for recapture pairs listed in PR
    t = model.addVars(PR, name="t_pr", lb=0.0, vtype=gp.GRB.CONTINUOUS)
    
    # Objective
    lost_revenue = gp.quicksum(
        (IT.loc[p, 'Price [EUR]'] - B.loc[(p, r), "b_pr"] * IT.loc[r, 'Price [EUR]']) * t[p,r]
        for (p, r) in PR
    )

    model.setObjective(lost_revenue, gp.GRB.MINIMIZE)

    # ---------- Constraints ----------
    
    # C1: seat capacity on each flight i
    for i in L:
        model.addConstr(
            gp.quicksum(
                DEL.loc[p,i] * t[p, r] for (p, r) in PR) - gp.quicksum(DEL.loc[p, i] * B.loc[(r, p), "b_pr"] * t[r, p] for (p, r) in PR if r != 'artificial')
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


    if model.status == gp.GRB.OPTIMAL:
        result["obj"] = model.objVal
        model.write("model_part2.lp")
        # capture decision variable values (only for created PR)
        for (p, r) in PR:
            # Gurobi returns .X for var
            val = t[p, r].X if t[p, r] is not None else 0.0
            result["t_values"][(p, r)] = float(val)

    # ---------- Return dual variables ----------
    for constr in model.getConstrs():
        name = constr.ConstrName
        if name.startswith("C1_Capacity_"):
            i = name[len("C1_Capacity_"):]
            result["duals"]["pi"][i] = constr.Pi
        elif name.startswith("C2_Demand_"):
            p = name[len("C2_Demand_"):]
            result["duals"]["sigma"][p] = constr.Pi

    # optional: write the lp for debugging
    model.write("model_part2.lp")

    return result

"""
TODO: don't forget to check that the initial PR given to the solve_model function includes the artificial itineraties
        -> using the function written for it: extend_DV nogwat

TODO: Finish the solve_model function by havin it return dual values in correct format

TODO: Check that all variablies are t(p,r) where p is the low!! and r the high in notation in the slide
                                    t(p,r) = t(low,high)

"""

# ---------- Loop generating columns ----------

columns = PR0.copy()
initial_column_count = len(columns)
iteration = 0

start_total = time.perf_counter()

with open("column_generation_log.txt", "a") as f:
    f.write(f"===== New Run =====\n")

running = True
while running:
    iteration += 1
    # solve function that solves model with given columns
    res = solve_model(columns)
    duals = res["duals"]

    # calculate reduced costs based on duals
    for (p, r), b_pr in B['b_pr'].items():
        if b_pr == 0:
            continue

        reduced_cost = (sum( (DEL.loc[p, i] - DEL.loc[r,i] * B.loc[(p, r), "b_pr"])  *  duals["pi"][i] 
                            for i in L) 
        + duals["sigma"][p] 
        - (IT.loc[p, 'Price [EUR]'] - B.loc[(p, r), "b_pr"] * IT.loc[r, 'Price [EUR]'] ))
        
        B.loc[(p, r), "reduced_cost"] = reduced_cost
    
    # select new columns with the most negative reduced cost
    reduced_costs = [
    {'pair': (p, r), 'reduced_cost': B.loc[(p, r), 'reduced_cost']}
    for (p, r), b_pr in B['b_pr'].items()
    if b_pr != 0]
    
    reduced_costs = sorted(reduced_costs, key=lambda x: x['reduced_cost'])
    pr_min_red_cost = reduced_costs[0]['pair']      # ('p','r') with lowest reduced cost
    pr_min_red_cost_reverse = (pr_min_red_cost[1], pr_min_red_cost[0])


    # Add selected columns to columns
    searching = True
    while searching:
        if pr_min_red_cost not in columns:
            columns.extend([pr_min_red_cost, pr_min_red_cost_reverse])
            searching = False

        else:
            # print(f"Pair not selected: {pr_min_red_cost} already in columns. Searching for next best.")
            reduced_costs.pop(0)
            pr_min_red_cost = reduced_costs[0]['pair']
            pr_min_red_cost_reverse = (pr_min_red_cost[1], pr_min_red_cost[0])

   
    # Print info
    print(f"Selected column (p,r) = {pr_min_red_cost} with reduced cost = {B.loc[pr_min_red_cost, 'reduced_cost']:.2f}")

    # Write to file
    # with open("column_generation_log.txt", "a") as f:
    #     f.write(f"Iteration: {iteration}\t"
    #             f"Selected column: {pr_min_red_cost}\t"
    #             f"Reduced cost selected dec variable: {B.loc[pr_min_red_cost, 'reduced_cost']:.2f}\t"
    #             # f"Columns for next iteration: {columns}\t"
    #             # f"Duals: {duals}\t"
    #             f"reduced_costs: {B['reduced_cost'].to_dict()}\n")

    # Check stopping criteria
    if B.loc[pr_min_red_cost, "reduced_cost"] >= -0.001:
        running = False


end_total = time.perf_counter()
total_runtime = end_total - start_total

# final solve to get final t_values & duals (ensure final RMP optimal)
final_res = solve_model(columns)

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
first5_itins = P[:]
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
for p, d in first5_t.items():
    print(f" Itinerary {p}:")
    for r, val in d.items():
        print(f"   t[{p},{r}] = {val:.2f}")

print("\nDuals (pi) for first 5 flights:")
for i, val in first5_duals.items():
    print(f" Flight {i}: pi = {val:.4f}")


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

import json
with open("cg_summary.json", "w") as f:
    json.dump(summary, f, indent=2)




def heatmap_t_values(t_values, P):
    # Create a 2D array for heatmap
    heatmap_data = np.zeros((len(P), len(P)))

    p_index = {p: idx for idx, p in enumerate(P)}
    r_index = {r: idx for idx, r in enumerate(P)}

    for (p, r), val in t_values.items():
        heatmap_data[p_index[p], r_index[r]] = val



    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=P, yticklabels=P, cmap="RdYlGn_r")
    plt.xlabel("Itinerary r")
    plt.ylabel("Itinerary p")
    plt.title("Heatmap of t[p,r] values")
    plt.show()

flight_info = {flight_number : {"demand": 0, "capacity" : 0} for flight_number in FL.index}

for itinerary in DEL.index:
    for flight in DEL.columns:
        if DEL.loc[itinerary, flight] == 1:
            flight_info[flight]["demand"] += IT.loc[itinerary, "Demand"].item()
            flight_info[flight]["capacity"] = FL.loc[flight, "Capacity"].item()


def plot_spilled_passengers_heatmap(FL, flight_info):
    """
    Heatmap:
      - y-axis: origin of a flight with flight number in parentheses
      - x-axis: destination airport
      - color: number of spilled passengers on that flight
               (demand - capacity, floored at 0)
    """
    # Adjust these if your column names are different
    origin_col = "Origin"
    destination_col = "Destination"

    # Unique destinations for x-axis
    destinations = FL[destination_col].unique().tolist()

    # Flights for y-axis
    flights = FL.index.tolist()

    # Matrix: rows = flights, cols = destinations
    heatmap_data = np.zeros((len(flights), len(destinations)))

    for row_idx, flight in enumerate(flights):
        origin = FL.loc[flight, origin_col]
        dest = FL.loc[flight, destination_col]

        # spilled passengers = max(demand - capacity, 0)
        demand = flight_info[flight]["demand"]
        capacity = flight_info[flight]["capacity"]
        spilled = demand - capacity

        # place in matrix at (flight row, destination column)
        col_idx = destinations.index(dest)
        heatmap_data[row_idx, col_idx] = spilled

    # Build y-tick labels: "Origin (FlightNo)"
    yticklabels = [f"{FL.loc[f, origin_col]} ({f})" for f in flights]
    yticklabels = [f"{f}" for f in flights]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data.transpose(),  # transpose for correct orientation
        annot=False,
        xticklabels=yticklabels,
        yticklabels=destinations,
        cmap = "Reds"
    )
    plt.xticks(fontsize=8)

    plt.ylabel("Origin (Flight number)")
    plt.title("Spilled passengers per flight (demand - capacity)")
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    sns.set(font_scale=0.1)
    plt.yticks(rotation=0)      # horizontal labels
    plt.xticks(rotation=45)     # angled columns
    plt.xlabel("Destination")
    plt.show()

plot_spilled_passengers_heatmap(FL, flight_info)


