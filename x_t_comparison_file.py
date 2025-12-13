import pandas as pd
from create_dataframes import create_RECAPTURE, create_ITINERARIES, create_FLIGHTS

path_flights = 'Problem 2 - Data/flights.xlsx'
path_itineraries = 'Problem 2 - Data/itineraries.xlsx'
path_recapture = 'Problem 2 - Data/recapture.xlsx'

FL = create_FLIGHTS(path_flights)
IT, DEL = create_ITINERARIES(path_itineraries, flights_index=FL.index)
P = IT.index.tolist()       # itineraries p
B = create_RECAPTURE(path_recapture, P)

# Load the saved t-values
t_df = pd.read_excel("t_values_final.xlsx")

t_values = {
    (row["p"], row["r"]): row["t_value"]
    for _, row in t_df.iterrows()
}

x_df = pd.read_excel("x_values_final.xlsx")

x_values = {
    (row["p"], row["r"]): row["x_value"]
    for _, row in x_df.iterrows()
}

# print(t_values[("0", "artificial")])
# print(x_values[("0", "1")])


# Converting t_values to x_values
rows = []
for p in P:
    t_out = sum(
        t_values.get((p, r), 0.0)
        for r in P if r != p)
    
    x_pp = IT.loc[p, "Demand"] - t_out

    rows.append({
        "p": p,
        "r": p,
        "x_value": x_pp
    })


for p in P:
    for r in P:
        if r == p:
            continue

        b_pr = B.loc[(p, r), "b_pr"]
        t_pr = t_values.get((p, r), 0.0)

        x_pr = b_pr * t_pr

        if abs(x_pr) > 1e-9:  # optional sparsity filter
            rows.append({
                "p": p,
                "r": r,
                "x_value": x_pr
            })

x_df_computed = pd.DataFrame(rows)

x_df.to_excel("x_values_computed.xlsx", index=False)


# Objective values
# Original revenue
revenue_OG = 0
for _, row in x_df.iterrows():
    p, r, x_val = row["p"], row["r"], row["x_value"]
    revenue_OG += IT.loc[r, "Price [EUR]"] * x_val

# Computed revenue
revenue_computed = 0
for _, row in x_df_computed.iterrows():
    p, r, x_val = row["p"], row["r"], row["x_value"]
    revenue_computed += IT.loc[r, "Price [EUR]"] * x_val

# print(revenue_OG, revenue_computed)

# Merge the original and computed x-values
comparison_df = x_df.merge(
    x_df_computed,
    on=["p", "r"],
    how="outer",
    suffixes=("_orig", "_computed")
)

# Filter rows where the values are different (allowing for floating point tolerance)
tolerance = 1e-6
diff_df = comparison_df[
    (comparison_df["x_value_orig"] - comparison_df["x_value_computed"]).abs() > tolerance
]

# Show the differences
# print(diff_df)

diff_df.to_excel("x_values_differences.xlsx", index=False)