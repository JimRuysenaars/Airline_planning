import pandas as pd
import numpy as np
import gurobipy as gp
import itertools


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
Qi = create_Q(DEL, IT)

# Recapture pairs (p, r)
B = create_RECAPTURE(path_recapture, P)

# TODO: Pls fix
PR0 = make_PR0_list(P)
#print(PR0)


# ---------- GUROBI MODEL ----------

def solve_model(PR):

    # ---------- GUROBI MODEL ----------

    model = gp.Model("Passenger_Mix_Flow")

    # Decision variables: t_pr = pax originally from p, flown on r
    # Only create vars for recapture pairs listed in PR
    t = model.addVars(PR, name="t_pr", lb=0.0, vtype=gp.GRB.CONTINUOUS)
    # print("PR: ", PR)
    
    # ---------- Objective: max total revenue ----------

    # revenue = sum_{(p,r)} fare_r * t_pr
    lost_revenue = gp.quicksum(
        (IT.loc[p, 'Price [EUR]'] - B.loc[(p, r), "b_pr"] * IT.loc[r, 'Price [EUR]']) * t[p, r]
        for (p, r) in PR
    )

    model.setObjective(lost_revenue, gp.GRB.MINIMIZE)

    # ---------- Constraints ----------
    
    # C1: seat capacity on each flight i
    for i in L:
        model.addConstr(
            gp.quicksum(
                DEL.loc[p, i] * t[p, r] for (p, r) in PR) - gp.quicksum(DEL.loc[p, i] * B.loc[(r, p), "b_pr"] * t[r, p] for (p, r) in PR if r != 'artificial')
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
    model.optimize()
    sum = 0
    if model.status == gp.GRB.OPTIMAL:
        print(f"Optimal objective (revenue) = {model.objVal:.2f}")
        # print("\nNon-zero flows t_rp:")
        # for (p, r) in PR:
        #     if t[p, r].X > 1e-6:
        #         sum += t[p, r].X
        #         print(f"t[{p},{r}] = {t[p,r].X:.2f}")

    # ---------- Return dual variables ----------
    duals = {"pi": {}, "sigma": {}}
    for constr in model.getConstrs():
        name = constr.ConstrName
        if name.startswith("C1_Capacity_"):
            i = name[len("C1_Capacity_"):]
            duals["pi"][i] = constr.Pi
        elif name.startswith("C2_Demand_"):
            p = name[len("C2_Demand_"):]
            duals["sigma"][p] = constr.Pi

    model.write("model_part2.lp")

    return duals



# print(solve_model(make_PR_total_list(P)))

"""
TODO: don't forget to check that the initial PR given to the solve_model function includes the artificial itineraties
        -> using the function written for it: extend_DV nogwat

TODO: Finish the solve_model function by havin it return dual values in correct format
"""

# ---------- Loop generating columns ----------

history = []
columns = PR0.copy()
running = True
iteration = 0
with open("column_generation_log.txt", "a") as f:
    f.write(f"===== New Run =====\n")

while running:
    iteration += 1
    # solve function that solves model with given columns
    duals = solve_model(columns)

    # calculate reduced costs based on duals
    for (p, r), b_pr in B['b_pr'].items():
        if b_pr == 0:
            continue

        reduced_cost = (sum( (DEL.loc[p, i] - DEL.loc[r, i] * B.loc[(p, r), "b_pr"])  *  duals["pi"][i] 
                            for i in L) 
        + duals["sigma"][p] 
        - (IT.loc[p, 'Price [EUR]'] - B.loc[(p, r), "b_pr"] * IT.loc[r, 'Price [EUR]'] )
        )
        
        B.loc[(p, r), "reduced_cost"] = reduced_cost
    # select new columns with the most negative reduced cost

    reduced_costs = [
    {'pair': (p, r), 'reduced_cost': B.loc[(p, r), 'reduced_cost']}
    for (p, r), b_pr in B['b_pr'].items()
    if b_pr != 0]
    

    value = [item['reduced_cost'] for item in reduced_costs if item['pair'] == ('15', '16')]
    print("Reduced cost:", value)

    reduced_costs = sorted(reduced_costs, key=lambda x: x['reduced_cost'])
    print("Reduced costs: ", reduced_costs[0:6])
    pr_min_red_cost = reduced_costs[0]['pair']      # ('p','r') with lowest reduced cost
    pr_min_red_cost_reverse = (pr_min_red_cost[1], pr_min_red_cost[0])


    # Add selected columns to columns
    searching = True
    while searching:
        if pr_min_red_cost not in columns:
            columns.extend([pr_min_red_cost, pr_min_red_cost_reverse])
            searching = False

        else:
            print(f"Pair not selected: {pr_min_red_cost} already in columns. Searching for next best.")
            reduced_costs.pop(0)
            pr_min_red_cost = reduced_costs[0]['pair']
            pr_min_red_cost_reverse = (pr_min_red_cost[1], pr_min_red_cost[0])


    history.append({
        "iteration": iteration,
        "columns_in_model": list(columns),
        "reduced_costs": { (p,r): B.loc[(p,r),"reduced_cost"] for (p,r) in B.index },
        "chosen_column": pr_min_red_cost,
        "chosen_column_reverse": pr_min_red_cost_reverse,
    })      
        
    # Print info
    print(f"Selected column (p,r) = {pr_min_red_cost} with reduced cost = {B.loc[pr_min_red_cost, 'reduced_cost']:.2f}")

    # Write to file
    with open("column_generation_log.txt", "a") as f:
        f.write(f"Iteration: {iteration}\t"
                f"Selected column: {pr_min_red_cost}\t"
                f"Reduced cost selected dec variable: {B.loc[pr_min_red_cost, 'reduced_cost']:.2f}\t"
                f"Columns for next iteration: {columns}\t"
                # f"Duals: {duals}\t"
                f"reduced_costs: {B['reduced_cost'].to_dict()}\n")

    # Check stopping criteria
    if B.loc[pr_min_red_cost, "reduced_cost"] >= -0.001:
        running = False


df_history = pd.DataFrame(history)
df_history.to_excel("cg_iterations.xlsx", index=False)



