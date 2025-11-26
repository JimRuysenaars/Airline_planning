
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

path_demand = 'Problem 1 - Data/demand_per_week.xlsx'
path_airports = 'Problem 1 - Data/airport_data.xlsx'
path_aircraft_data = 'Problem 1 - Data/AircraftData.xlsx'
path_combined_data = 'Problem 1 - Data/combined_data.xlsx'

earth_radius = 6371  # in kilometers
fuel_cost = 1.42 # EUR per gallon


def load_data(path_combined_data, path_demand, path_airports):
    pop_GDP_data = pd.read_excel(path_combined_data, header=0, index_col=0).dropna(how='all', axis=1)

    demand_data = pd.read_excel(path_demand, index_col=0)
    airport_data = pd.read_excel(path_airports, index_col=0)

    return pop_GDP_data, demand_data, airport_data


def build_distance_matrix(path_airports, earth_radius):
    df = pd.read_excel(path_airports, header=None)

    icao = df.iloc[1, 1:].tolist()
    lat  = df.iloc[2, 1:].to_numpy(dtype=float)
    lon  = df.iloc[3, 1:].to_numpy(dtype=float)
    
    distances = pd.DataFrame(index=icao, columns=icao, dtype=float)

    for i, j in itertools.product(range(len(icao)), repeat=2):
        lat_i = lat[i]
        lon_i = lon[i]
        lat_j = lat[j]
        lon_j = lon[j]

        d_ij = 2 * np.arcsin(np.sqrt(np.sin(np.radians((lat_i - lat_j) / 2))**2 +
                            np.cos(np.radians(lat_i)) * np.cos(np.radians(lat_j)) * np.sin(np.radians((lon_i - lon_j) / 2))**2))

        d_ij = d_ij * earth_radius
        distances.iloc[i, j] = d_ij
    return distances


def OLS(combined_data, distances, path_demand):    
    df = pd.read_excel(combined_data, index_col=0)
    observed_demand = pd.read_excel(path_demand, index_col=0)

    pairwise_dataframe = pd.DataFrame(columns=['log_observed_demand', 'log_pop_product', 'log_gdp_product', 'neg_log_distance_cost'])
    dummy = 0

    for i, j in itertools.product(df.index, repeat=2):
        if i != j:
            pop_i = df.loc[i, 'Population 2021']
            pop_j = df.loc[j, 'Population 2021']
            gdp_i = df.loc[i, 'GDP 2021']
            gdp_j = df.loc[j, 'GDP 2021']
        
            observed_demand_ij = observed_demand.loc[i, j]
            d_ij = distances.loc[i, j]

            y_ij = np.log(observed_demand_ij)
            x1_ij = np.log(pop_i * pop_j)
            x2_ij = np.log(gdp_i * gdp_j)
            x3_ij = -np.log(d_ij * fuel_cost)
            pairwise_dataframe.loc[dummy] = [y_ij, x1_ij, x2_ij, x3_ij]

        dummy += 1

    X = pairwise_dataframe[['log_pop_product', 'log_gdp_product', 'neg_log_distance_cost']]
    y = pairwise_dataframe['log_observed_demand']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    results = model.summary()

    ln_k = model.params['const']
    beta_1 = model.params['log_pop_product']
    beta_2 = model.params['log_gdp_product']
    beta_3 = model.params['neg_log_distance_cost']

    k = np.exp(ln_k)


    predicted_demand_dataframe = pd.DataFrame(index=df.index, columns=df.index, dtype=int)
    for i, j in itertools.product(df.index, repeat=2):
        if i != j:
            pop_i = df.loc[i, 'Population 2021']
            pop_j = df.loc[j, 'Population 2021']
            gdp_i = df.loc[i, 'GDP 2021']
            gdp_j = df.loc[j, 'GDP 2021']
            d_ij = distances.loc[i, j]

            predicted_demand_ij = k * (pop_i ** beta_1) * (pop_j ** beta_1) * (gdp_i ** beta_2) * (gdp_j ** beta_2) * ((d_ij * fuel_cost) ** (-beta_3))
            predicted_demand_dataframe.loc[i, j] = int(round(predicted_demand_ij,0))
        else:
            predicted_demand_dataframe.loc[i, j] = int(0)

    return results, k, beta_1, beta_2, beta_3, predicted_demand_dataframe


def forecast_population_GDP(pop_GDP_data):
    forecasted_population_GDP = np.array([['Code', 'Forecasted Population 2026', 'Forecasted GDP 2026']])
    for code in pop_GDP_data.index:
        pop_2021 = pop_GDP_data.loc[code, 'Population 2021']
        pop_2024 = pop_GDP_data.loc[code, 'Population 2024']
        growth_rate_pop = (pop_2024 - pop_2021)/3
        gdp_2021 = pop_GDP_data.loc[code, 'GDP 2021']
        gdp_2024 = pop_GDP_data.loc[code, 'GDP 2024']
        growth_rate_gdp = (gdp_2024 - gdp_2021)/3
    
        forecasted_population_GDP = np.vstack((forecasted_population_GDP, [code, pop_2024 + growth_rate_pop * 2, gdp_2024 + growth_rate_gdp * 2]))
    forecasted_population_GDP = pd.DataFrame(forecasted_population_GDP[1:], columns=forecasted_population_GDP[0], index=forecasted_population_GDP[1:,0])

    return forecasted_population_GDP

def future_demand_forecast(k, beta_1, beta_2, beta_3, pop_GDP_data, distances):
    future_demand_matrix = pd.DataFrame(index=distances.index, columns=distances.columns, dtype=float)

    for i, j in itertools.product(distances.index, repeat=2):
        if i != j:
            pop_i_2026 = float(pop_GDP_data.loc[i, 'Forecasted Population 2026'])
            pop_j_2026 = float(pop_GDP_data.loc[j, 'Forecasted Population 2026'])
            gdp_i_2026 = float(pop_GDP_data.loc[i, 'Forecasted GDP 2026'])
            gdp_j_2026 = float(pop_GDP_data.loc[j, 'Forecasted GDP 2026'])
            d_ij = distances.loc[i, j]

            demand_ij = k * (pop_i_2026 ** beta_1) * (pop_j_2026 ** beta_1) * (gdp_i_2026 ** beta_2) * (gdp_j_2026 ** beta_2) * ((d_ij * fuel_cost) ** (-beta_3))
            future_demand_matrix.loc[i, j] = int(round(demand_ij,0))
        else:
            future_demand_matrix.loc[i, j] = int(0)

    return future_demand_matrix


def compare_demand_matrices(observed_demand, predicted_demand):
    """
    # observed_demand = observed (or baseline)
    # predicted_demand = predicted (or comparison)

    # --- signed difference (non-normalised) ---
    signed_diff = predicted_demand - observed_demand

    # Create percentual (relative) normalised difference
    # Avoid division by zero by replacing zeros temporarily
    den = observed_demand.replace(0, np.nan)
    percent_diff = (signed_diff / den).abs()

    # If any NaNs appear (because observed_demand was 0), fill with 0 or keep NaN
    percent_diff = percent_diff.fillna(0)

    # --- Custom centered colormap ---
    green_white_pink = LinearSegmentedColormap.from_list(
        "GWP", ["pink", "white", "green"]
    )

    # --- Center colormap at zero using TwoSlopeNorm ---
    norm_signed = TwoSlopeNorm(vcenter=0)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ===== Heatmap 1: percentual difference =====
    im1 = axes[0].imshow(percent_diff.values, cmap=green_white_pink, aspect="equal")

    axes[0].set_title("Percentual difference: (predicted_demand - observed_demand) / observed_demand")
    axes[0].set_xticks(range(len(observed_demand.columns)))
    axes[0].set_yticks(range(len(observed_demand.index)))
    axes[0].set_xticklabels(observed_demand.columns, rotation=90)
    axes[0].set_yticklabels(observed_demand.index)
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_label("Relative difference (fraction)")

    # ===== Heatmap 2: signed non-normalised difference =====
    im2 = axes[1].imshow(signed_diff.values, cmap=green_white_pink,
                        norm=norm_signed, aspect="equal")

    axes[1].set_title("Signed difference: predicted_demand - observed_demand")
    axes[1].set_xticks(range(len(observed_demand.columns)))
    axes[1].set_yticks(range(len(observed_demand.index)))
    axes[1].set_xticklabels(observed_demand.columns, rotation=90)
    axes[1].set_yticklabels(observed_demand.index)
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label("Difference")

    plt.tight_layout()
    plt.show()

    """
    df1 = observed_demand
    df2 = predicted_demand

    # --- df1 = observed (or base), df2 = predicted (or comparison) ---

    # Signed difference
    signed_diff = df2 - df1

    # Percentual difference: (df2 - df1) / df1
    den = df1.replace(0, np.nan)
    percent_diff = signed_diff / den
    percent_diff = percent_diff.fillna(0)  # avoid NaN after division

    # --- Custom seaborn colormap: pink → white → green ---
    green_white_pink = LinearSegmentedColormap.from_list(
        "GWP", ["pink", "white", "green"]
    )

    # Centered normalization at 0 for both heatmaps
    norm_signed = TwoSlopeNorm(vcenter=0)

    # --- Plot seaborn heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ===== Heatmap 1: Percentual difference =====
    sns.heatmap(
        percent_diff,
        cmap='Reds',
        ax=axes[0],
        linewidths=1,        # grid line thickness
        linecolor='white',    # grid line color
        square=True,
        cbar_kws={"label": "Relative difference (fraction)"}
    )
    axes[0].set_title("Percentual Difference: (df2 - df1) / df1")
    axes[0].set_xlabel("Airport j")
    axes[0].set_ylabel("Airport i")

    # ===== Heatmap 2: Signed non-normalised difference =====
    sns.heatmap(
        signed_diff,
        cmap='seismic',
        norm=norm_signed,
        ax=axes[1],
        linewidths=1,
        linecolor='white',
        annot=True,
        square=True,
        cbar_kws={"label": "Signed Difference"}
    )
    axes[1].set_title("Signed Difference: df2 - df1")
    axes[1].set_xlabel("Airport j")
    axes[1].set_ylabel("Airport i")

    plt.tight_layout()
    plt.show()


    return None


pop_GDP_data, demand_data, airport_data = load_data(path_combined_data, path_demand, path_airports)
distances = build_distance_matrix(path_airports, earth_radius)
results, k, beta_1, beta_2, beta_3, predicted_demand_dataframe = OLS(path_combined_data, distances, path_demand)
forecasted_population_GDP = forecast_population_GDP(pop_GDP_data)
future_demand_matrix = future_demand_forecast(k, beta_1, beta_2, beta_3, forecasted_population_GDP, distances)

compare_demand_matrices(demand_data, predicted_demand_dataframe)

