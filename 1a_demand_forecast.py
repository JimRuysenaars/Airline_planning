
from importlib.resources import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

path_pop = 'Problem 1 - Data/pop.xlsx'
path_demand = 'Problem 1 - Data/demand_per_week.xlsx'
path_airports = 'Problem 1 - Data/airport_data.xlsx'
path_aircraft_data = 'Problem 1 - Data/AircraftData.xlsx'

earth_radius = 6371  # in kilometers
fuel_cost = 1.42 # EUR per gallon


def load_data(path_pop, path_demand, path_airports):
    pop_GDP_data = pd.read_excel(path_pop, header=2, index_col=0).dropna(how='all', axis=1)

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


def OLS(pop_GDP_data, demand_data, airport_data, distances):
    df = pd.DataFrame(columns=["Origin", "Destination", 
                                "Origin Population", "Destination Population",
                                "Origin GDP per capita", "Destination GDP per capita",
                                "Distance"])
    
    codes = airport_data['ICAO'].tolist()

    for i in range(len(codes)):
        demand_ij = demand_data.loc[i, j]

    for i, j in itertools.product(codes, repeat=2):
        code_i = codes[i]
        code_j = codes[j]
        
        demand_ij = demand_data.loc[code_i, code_j]

        pop_i = pop_GDP_data.loc[i, '2021']
        pop_j = pop_GDP_data.loc[j, '2021']

        distance_ij = distances.loc[code_i, code_j]


    return df


def log_demand(pop_GDP_data, airport_data, distances, fuel_cost):
    
    return None

pop_GDP_data, demand_data, airport_data = load_data(path_pop, path_demand, path_airports)
distances = build_distance_matrix(path_airports, earth_radius)

print(log_demand(pop_GDP_data, airport_data, distances, fuel_cost))



