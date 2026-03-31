import osmnx as ox
import networkx as nx
import random
import math
import json

def generate_graph_data(min_arcs=30, max_arcs=200, max_attempts=20):
    """
    Generate dataset based on 7-step logic for Arc Routing Problems.
    """
    print("Step 1: Graph Extraction...")
    G = None
    # Try random locations until we find a graph that matches the size constraints
    for attempt in range(max_attempts):
        # Pick random coordinates, leaning towards urban areas globally
        lat = random.uniform(30.0, 45.0)  # Roughly US / Europe
        lon = random.uniform(-120.0, 10.0)
        
        try:
            # Get a small spatial graph
            G_temp = ox.graph_from_point((lat, lon), dist=800, network_type='drive')
            
            # Extract strongly connected components
            strong_components = sorted(nx.strongly_connected_components(G_temp), key=len, reverse=True)
            if not strong_components:
                continue
                
            # Take the largest strongly connected component
            largest_scc = G_temp.subgraph(strong_components[0]).copy()
            num_arcs = largest_scc.number_of_edges()
            
            if min_arcs <= num_arcs <= max_arcs:
                print(f" -> Found valid graph with {num_arcs} arcs at ({lat:.4f}, {lon:.4f})")
                G = largest_scc
                break
        except Exception as e:
            # OSMnx may fail if no data is present at location
            continue

    if G is None:
        print(" -> Falling back to a predefined location (Piedmont, CA) due to random search exhaustion.")
        G_temp = ox.graph_from_place("Piedmont, California, USA", network_type='drive')
        strong_components = sorted(nx.strongly_connected_components(G_temp), key=len, reverse=True)
        G = G_temp.subgraph(strong_components[0]).copy()
        print(f" -> Fallback graph has {G.number_of_edges()} arcs.")
        
    # Project the graph to UTM so that coordinates are in meters for correct Euclidean distance
    G_proj = ox.project_graph(G)
    
    # Convert node labels to integers
    G_proj = nx.convert_node_labels_to_integers(G_proj)
    
    print("\nStep 2: Vehicle Fleet...")
    num_vehicles = 5
    print(f" -> Number of vehicles (|M|) set to: {num_vehicles}")
    
    print("\nStep 3: Required Arcs (Ar)...")
    total_arcs = G_proj.number_of_edges()
    edges = list(G_proj.edges(keys=True))
    
    if total_arcs >= 80:
        num_required = random.randint(60, 70)
        # Ensure we don't request more required arcs than total arcs if bound is broken by fallback
        num_required = min(num_required, total_arcs)
    else:
        num_required = int(0.75 * total_arcs)
        
    required_arcs = set(random.sample(edges, num_required))
    print(f" -> Total arcs (|A|): {total_arcs}")
    print(f" -> Required arcs (|Ar|): {num_required}")
    
    print("\nStep 4: Priority Classes...")
    priority_classes = [1, 2, 3]
    for u, v, k in edges:
        if (u, v, k) in required_arcs:
            G_proj[u][v][k]['is_required'] = True
            G_proj[u][v][k]['priority'] = random.choice(priority_classes)
        else:
            G_proj[u][v][k]['is_required'] = False
            G_proj[u][v][k]['priority'] = 0  # 0 indicates no priority/non-required
            
    print("\nStep 5: Traversal and Servicing Time...")
    # Compute Euclidean distance d_a'
    for u, v, k in edges:
        x_u, y_u = G_proj.nodes[u]['x'], G_proj.nodes[u]['y']
        x_v, y_v = G_proj.nodes[v]['x'], G_proj.nodes[v]['y']
        # Distance calculation
        d_a_prime = math.sqrt((x_u - x_v)**2 + (y_u - y_v)**2)
        G_proj[u][v][k]['d_a_prime'] = d_a_prime
        
    # Find d_max'
    d_max_prime = max(G_proj[u][v][k]['d_a_prime'] for u, v, k in edges)
    
    # Normalize to get traversal time (d_a) and calculate service time (s_a)
    for u, v, k in edges:
        d_a_prime = G_proj[u][v][k]['d_a_prime']
        d_a = d_a_prime / d_max_prime if d_max_prime > 0 else 0
        G_proj[u][v][k]['traversal_time'] = d_a
        
        if (u, v, k) in required_arcs:
            # Service time is twice traversal time
            s_a = 2 * d_a
            G_proj[u][v][k]['service_time'] = s_a
        else:
            G_proj[u][v][k]['service_time'] = 0.0

    print("\nStep 6: Demand Calculation...")
    # Calculate demand for each arc
    for u, v, k in edges:
        d_a = G_proj[u][v][k]['traversal_time']
        q_a = (d_a * 0.5) + 0.5
        G_proj[u][v][k]['demand'] = q_a

    print("\nStep 7: Vehicle Capacity (Q)...")
    # Capacity is calculated using the sum of demands from required arcs
    capacity_sum = 0
    for u, v, k in edges:
        if (u, v, k) in required_arcs:
            q_a = G_proj[u][v][k]['demand']
            capacity_sum += (q_a / 3) + 0.5
            
    Q = capacity_sum
    print(f" -> Calculated Vehicle Capacity (Q): {Q:.4f}")
    
    # Store processed information into a structured data dictionary
    dataset = {
        'num_vehicles': num_vehicles,
        'total_arcs': total_arcs,
        'required_arcs_count': num_required,
        'vehicle_capacity': Q,
        'nodes': {},
        'arcs': []
    }
    
    for n, data_dict in G_proj.nodes(data=True):
        dataset['nodes'][n] = {'x': data_dict['x'], 'y': data_dict['y']}
        
    for u, v, k, data_dict in G_proj.edges(keys=True, data=True):
        arc_info = {
            'u': u,
            'v': v,
            'is_required': data_dict['is_required'],
            'priority': data_dict['priority'],
            'euclidean_dist': data_dict['d_a_prime'],
            'traversal_time': data_dict['traversal_time'],
            'service_time': data_dict['service_time'],
            'demand': data_dict['demand']
        }
        dataset['arcs'].append(arc_info)
        
    return dataset

if __name__ == "__main__":
    dataset = generate_graph_data(min_arcs=30, max_arcs=200)
    
    # Save dataset to JSON
    output_filename = "generated_dataset.json"
    with open(output_filename, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nSuccessfully generated dataset and saved to {output_filename}")
