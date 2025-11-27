import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

######## PARAMETERS & PATHS #######

path_demand = 'Problem 1 - Data/demand_per_week.xlsx'       # weekly demand data between airports
path_airports = 'Problem 1 - Data/airport_data.xlsx'        # airport data with ICAO codes, latitudes and longitudes
path_aircraft_data = 'Problem 1 - Data/AircraftData.xlsx'   # aircraft data including seats, speed, range, costs
path_combined_data = 'Problem 1 - Data/combined_data.xlsx'  # contains, ICAO code, population and GDP data in one file

earth_radius = 6371  # in kilometers
fuel_cost = 1.42 # EUR per gallon

##### FUNCTIONS #######

def load_data(path_combined_data, path_demand, path_airports):
    """
    Loads the data from the different excel files and returns them as dataframes
    """

    pop_GDP_data = pd.read_excel(path_combined_data, header=0, index_col=0).dropna(how='all', axis=1)

    demand_data = pd.read_excel(path_demand, index_col=0)
    airport_data = pd.read_excel(path_airports, index_col=0)

    return pop_GDP_data, demand_data, airport_data


def build_distance_matrix(path_airports, earth_radius):
    """
    Using the same indices as in the demand dataframe, this function builds a distance matrix between all airports according to appendix C
    """
    df = pd.read_excel(path_airports, header=None)

    icao = df.iloc[1, 1:].tolist()
    lat  = df.iloc[2, 1:].to_numpy(dtype=float)
    lon  = df.iloc[3, 1:].to_numpy(dtype=float)
    
    distances = pd.DataFrame(index=icao, columns=icao, dtype=float)

    for i, j in itertools.product(range(len(icao)), repeat=2):          # generates all combinations of airports i, j
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
    """
    Uses the ordinary least squares method to calibrate the parameters of the gravity model using the population and GDP data from 2021.
    In order to executre the regression, a dataframe is first constructed containing all pairwise cominations of airports i, j and ommits 
    same Origin-Destination pairs to avoid biases in the regression.
    The model is then linearized by taking the logarithm of both sides for each origin destination pair and assigns the variables y_ij and x_ij such that the paramters
    to be optimized are coefficients in a liear function. y = ln(k) + beta_1 * x1 + beta_2 * x2 + beta_3 * x3. Using the statsmodels package, the OLS regression is
    performed and the results are returned along with the predicted demand matrix according to the calibrated gravity model, for later comparison.
    """    
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

            y_ij = np.log(observed_demand_ij)       # log of observed demand
            x1_ij = np.log(pop_i * pop_j)           # log of population product
            x2_ij = np.log(gdp_i * gdp_j)           # log of GDP product
            x3_ij = -np.log(d_ij * fuel_cost)       # negative log of distance times constant fuel cost
            pairwise_dataframe.loc[dummy] = [y_ij, x1_ij, x2_ij, x3_ij]

        dummy += 1          # dummy varible to assign row index to origin destination pairs.

    X = pairwise_dataframe[['log_pop_product', 'log_gdp_product', 'neg_log_distance_cost']]
    y = pairwise_dataframe['log_observed_demand']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()          # Ordinary Least Squares regression

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

    return k, beta_1, beta_2, beta_3, predicted_demand_dataframe


def forecast_population_GDP(pop_GDP_data):
    """
    This function aims to answer subquestion 2. of 1A by forecasting the population and GDP values for 2026 based on the data from 2021 and 2024.
    The forecast assumes a constant growth rate.
    """
    forecasted_population_GDP = np.array([['Code', 'Forecasted Population 2026', 'Forecasted GDP 2026']])
    # Iterate over all airports (cities) given in the population data
    for code in pop_GDP_data.index:                                 
        pop_2021 = pop_GDP_data.loc[code, 'Population 2021']
        pop_2024 = pop_GDP_data.loc[code, 'Population 2024']
        growth_rate_pop = (pop_2024 - pop_2021)/3
        gdp_2021 = pop_GDP_data.loc[code, 'GDP 2021']
        gdp_2024 = pop_GDP_data.loc[code, 'GDP 2024']
        growth_rate_gdp = (gdp_2024 - gdp_2021)/3
    
        forecasted_population_GDP = np.vstack((forecasted_population_GDP, [code, pop_2024 + growth_rate_pop * 2, gdp_2024 + growth_rate_gdp * 2]))

    # Transform the numpy array into a dataframe for easier comparison
    forecasted_population_GDP = pd.DataFrame(forecasted_population_GDP[1:], columns=forecasted_population_GDP[0], index=forecasted_population_GDP[1:,0])

    return forecasted_population_GDP

def future_demand_forecast(k, beta_1, beta_2, beta_3, pop_GDP_data, distances):
    """
    Takes the forecasted population and GDP for 2026 and computes the forecasted demand using the calibrated gravity model and parameters from the OLS regression.
    Same origin-destination pairs have a value of zero.
    """
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
    The previous function generated the forecasted demand matrix for 2026 in the same format as the observed demand matrix, with origin and destination codes
    as rows and columns. This function compares both matrices to verify the accuracy of the parameters obtained from the ordinary least squares method.
    This function create two heatmaps: one for the percentual difference between the observed and predicted demand, and one for the absolute difference.
    """ 
    signed_diff = predicted_demand - observed_demand            # Absolute difference between predicted and observed demand

    den = observed_demand.replace(0, np.nan)                    # Replace zeros with NaN to avoid division by zero for same origin-destination pairs
    percent_diff = (signed_diff / den).abs()                    # Absolute percentual difference
    percent_diff = percent_diff.fillna(0)

    norm_signed = TwoSlopeNorm(vcenter=0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # First heatmap for the absolutepercentual difference
    sns.heatmap(
        percent_diff,
        cmap='Reds',
        ax=axes[0],
        linewidths=1,        # grid line thickness
        linecolor='white',    # grid line color
        square=True,
        cbar_kws={"label": "Percentual difference"}
    )
    axes[0].set_title("Percentual difference between predicted and observed demand")
    axes[0].set_xlabel("Airport i")
    axes[0].set_ylabel("Airport j")

    # Second heatmap for the relative difference between predicted and observed demand
    sns.heatmap(
        signed_diff,
        cmap='seismic',
        norm=norm_signed,
        ax=axes[1],
        linewidths=1,
        linecolor='white',
        annot=True,
        square=True,
        cbar_kws={"label": "Relative difference"}
    )
    axes[1].set_title("Relative difference between predicted and observed demand")
    axes[1].set_xlabel("Airport i")
    axes[1].set_ylabel("Airport j")

    plt.tight_layout()
    plt.show()


    return None

def __main__():
    """
    Main execution of all functions. Returns the future demand matrix for 2026.
    """
    pop_GDP_data, demand_data, airport_data = load_data(path_combined_data, path_demand, path_airports)
    distances = build_distance_matrix(path_airports, earth_radius)
    k, beta_1, beta_2, beta_3, predicted_demand_dataframe = OLS(path_combined_data, distances, path_demand)
    forecasted_population_GDP = forecast_population_GDP(pop_GDP_data)
    future_demand_matrix = future_demand_forecast(k, beta_1, beta_2, beta_3, forecasted_population_GDP, distances)
    compare_demand_matrices(demand_data, predicted_demand_dataframe)

    future_demand_matrix.to_excel("future_demand_matrix_2026.xlsx")

    return future_demand_matrix

__main__()