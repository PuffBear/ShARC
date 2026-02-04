import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load the data
data_path = 'instances/100/100_39_099.npz'
data = np.load(data_path)

req_data = data['req']
nonreq_data = data['nonreq']
all_arcs = np.vstack([req_data, nonreq_data])

# 2. Reconstruct the Graph Structure
# G is a Directed Multigraph to account for multiple paths between nodes
G = nx.MultiDiGraph()

# Map columns to attributes: 
# 0: u, 1: v, 2: demand, 3: priority, 4: service_time, 5: travel_time
for row in all_arcs:
    u, v = int(row[0]), int(row[1])
    G.add_edge(u, v, 
               demand=row[2], 
               priority=int(row[3]), 
               service_time=row[4], 
               travel_time=row[5])

print(f"Graph reconstructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} arcs.")

# 3. Visualization - Network Topology
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=455, k=0.5) # Consistent layout

# Draw Nodes
nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightskyblue', alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

# Edge Separation for visualization
req_edges = [(int(row[0]), int(row[1])) for row in req_data]
nonreq_edges = [(int(row[0]), int(row[1])) for row in nonreq_data]

# Draw Required Arcs (Tasks) in Red
nx.draw_networkx_edges(G, pos, edgelist=req_edges, 
                       edge_color='crimson', width=2.5, arrowstyle='-|>', 
                       arrowsize=15, label='Required Arcs (Tasks)')

# Draw Traversal Arcs in Gray
nx.draw_networkx_edges(G, pos, edgelist=nonreq_edges, 
                       edge_color='gray', alpha=0.3, style='dashed', 
                       arrowstyle='-|>', arrowsize=10, label='Traversal Arcs')

# Highlight Depot (Node 0) in Gold
if 0 in G:
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color='gold', 
                           node_size=700, edgecolors='black', linewidths=2)

plt.title("H-ShARC Network Topology: Required vs. Traversal Arcs", fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.savefig('network_topology.png')

# 4. Visualization - Data Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Priority Distribution
req_priorities = req_data[:, 3]
unique, counts = np.unique(req_priorities, return_counts=True)
ax1.bar(unique, counts, color='teal', alpha=0.8, edgecolor='black')
ax1.set_title("Distribution of Priority Classes", fontsize=14)
ax1.set_xlabel("Priority Level (1=High, 3=Low)", fontsize=12)
ax1.set_ylabel("Count of Arcs", fontsize=12)
ax1.set_xticks([1, 2, 3])

# Travel Time (Cost) Distribution
all_travel_times = all_arcs[:, 5]
ax2.hist(all_travel_times, bins=25, color='darkorange', alpha=0.7, edgecolor='white')
ax2.set_title("Nominal Travel Time (Cost) Distribution", fontsize=14)
ax2.set_xlabel("Time Units", fontsize=12)
ax2.set_ylabel("Frequency", fontsize=12)

plt.tight_layout()
plt.savefig('data_distribution.png')