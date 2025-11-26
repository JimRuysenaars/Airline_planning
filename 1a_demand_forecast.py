
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_pop = 'Problem 1 - Data/pop.xlsx'
path_demand = 'Problem 1 - Data/demand_per_week.xlsx'
path_airports = 'Problem 1 - Data/airport_data.xlsx'
path_aircraft_data = 'Problem 1 - Data/AircraftData.xlsx'

earth_radius = 6371  # in kilometers

def load_data():
    pop_GDP_data = pd.read_excel(path_pop, header=2, index_col=0).dropna(how='all', axis=1)

    demand_data = pd.read_excel(path_demand, index_col=0)
    airport_data = pd.read_excel(path_airports)


    return pop_GDP_data, demand_data, airport_data

print(load_data()[0])


    