
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_pop = 'Problem 1 - Data/pop.xlsx'
path_demand = 'Problem 1 - Data/DemandGroup17.xlsx'
path_aircraft_data = 'Problem 1 - Data/AircraftData.xlsx'

earth_radius = 6371  # in kilometers

def load_data():
    pop_data = pd.read_excel(path_pop, header=2).dropna(how='all', axis=1).to_numpy()
    demand_data = pd.read_excel(path_demand).dropna(how='all', axis=0).to_numpy()
    aircraft_data = pd.read_excel(path_aircraft_data).to_numpy()

    airport_data = 

    return pop_data, demand_data, aircraft_data

print(load_data()[1])


#def convert_distance(path_demand, radius=earth_radius):
    