# GOOGLE CVRP :D
import pandas as pd
import numpy as np
from tqdm import tqdm

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
cluster1 = {"cost_matrix": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cost_matrix_good/cost_matrix_clust1.csv",
            "coordinates": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster1.csv"}


cluster2 = {"cost_matrix": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cost_matrix_good/cost_matrix_clust2.csv",
            "coordinates": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster2.csv"
            }


cluster3 = {"cost_matrix": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cost_matrix_good/cost_matrix_clust3.csv",
            "coordinates": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster3.csv"}


cluster4 = {"cost_matrix": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cost_matrix_good/cost_matrix_clust4.csv",
            "coordinates": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster4.csv"}


cluster5 = {"cost_matrix": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cost_matrix_good/cost_matrix_clust5.csv",
            "coordinates": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster5.csv"}


cluster6 = {"cost_matrix": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cost_matrix_good/cost_matrix_clust6.csv",
            "coordinates": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster6.csv"}


cluster7 = {"cost_matrix": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cost_matrix_good/cost_matrix_clust7.csv",
            "coordinates": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster7.csv"}


cluster8 = {"cost_matrix": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cost_matrix_good/cost_matrix_clust8.csv",
            "coordinates": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster8.csv"}


first_echelon = {"cost_matrix": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/1-e/cost_matrix/cost_matrix_1e.csv",
                 "coordinates": "Survey/vrp-nanostores/vrp-nanostores/food_deserts/outputs/data.csv"}




def create_data_matrix():
    locations = []
    # dataframe
    file = pd.read_csv(cluster8["cost_matrix"],
                       sep=",", header=None)
    tspSize = len(file)

    # 0 matrix
    distances = [[0] * (tspSize) for i in range(tspSize)]

    # coord (lat,lon)
    coordenadas = pd.read_csv(cluster8["coordinates"])[
        ["latitude", "longitude"]]  # depends on cluster needed
    # coordenadas = coordenadas.iloc[1:,:]    #FOR FIRST ECHELON ONLY :D

    for fila in range(len(coordenadas.index)):
        locations.append(coordenadas.iloc[fila])

    # new matrix have calc. measures (only infierior triangle)
    for i in range(tspSize):
        for j in range(i+1, tspSize):
            # poner la distancia del dataframe a la nueva matriz
            distance = int(round(100*file[i][j]))
            distances[i][j] = distance
            distances[j][i] = distance  # DISTANCE = COST
    return distances


# file = pd.read_csv("Survey/vrp-nanostores/vrp-nanostores/food_deserts/outputs/data.csv",
# #                    sep=",", header=None)
#
# len(file)


def create_data_model(num_vehicles):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = create_data_matrix()  # REALLY IS A COST MATRIX

    # CANT CARGA FOR FIRST_ECHELON, DEMANDA FOR ALL THE CLUSTERS
    data['demands'] = [int(i) for i in pd.read_csv(
        cluster8["coordinates"])["demanda"].fillna(0)]
    data['vehicle_capacities'] = [550 for i in range(num_vehicles)]
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        # print(plan_output)
        total_distance += route_distance
        total_load += route_load
        with open(f"Survey/vrp-nanostores/vrp-nanostores/food_deserts/outputs/2-e/clust8/route/route_vehicle{vehicle_id}.txt", "w") as file:
            file.write(plan_output)
            file.close()
        print("aaa")
    print('Total cost for all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))
    with open(f"Survey/vrp-nanostores/vrp-nanostores/food_deserts/outputs/2-e/clust8/load_dist_{data['num_vehicles']}vehicles.txt", "w") as file:
        out_file = ""
        out_file += str(total_load) + "," + str(total_distance)
        file.write(out_file)
        file.close()  # OPEN AND ANALYZE LATER WITH PANDAS


def solucionar(num_vehicles):
    data = create_data_model(num_vehicles)
    # Create the routing index manager.

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """cost between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.seconds = 60 * 10 * 18  # x hours
    # search_parameters.log_search = True

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    # Print solution on console.
    if assignment:
        print_solution(data, manager, routing, assignment)
    else:
        print("nada")

num = [i for i in range(1, len(pd.read_csv(cluster8['coordinates'])))]
pd.read_csv(cluster8['coordinates'])['demanda'].sum()


|#
# [pd.read_csv(i["cost_matrix"], header=None).shape for i in c]
# [len(pd.read_csv(i["coordinates"])) for i in c]



for i in tqdm(num):
    solucionar(i)

# solucionar(1000)

# max([int(i) for i in pd.read_csv(cluster5["coordinates"])["demanda"].fillna(0)])


print(a)
