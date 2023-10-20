import matplotlib.pyplot as plt
import networkx as nx
import csrgraph as cg
import numpy as np
import random
import string


def get_graph(nodes = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]):

    G = nx.DiGraph()
    east_pairs = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[3], nodes[4]),
                  (nodes[4], nodes[5]), (nodes[6], nodes[7]), (nodes[7], nodes[8])]
    south_pairs = [(nodes[0], nodes[3]), (nodes[3], nodes[6]), (nodes[1], nodes[4]),
                   (nodes[4], nodes[7]), (nodes[2], nodes[5]), (nodes[5], nodes[8])]
    north_pairs = [(i[1], i[0]) for i in south_pairs]
    west_pairs = [(i[1], i[0]) for i in east_pairs]

    for n in nodes:
        G.add_node(n)

    for tple in east_pairs:
        G.add_edge(tple[0], tple[1], direction='E')
    for tple in north_pairs:
        G.add_edge(tple[0], tple[1], direction='N')
    for tple in west_pairs:
        G.add_edge(tple[0], tple[1], direction='W')
    for tple in south_pairs:
        G.add_edge(tple[0], tple[1], direction='S')

    return G

def get_random_walks(G, n_walks=1):
    csr_G = cg.csrgraph(G, threads=12)
    node_names = csr_G.names
    walks = csr_G.random_walks(walklen=50, # length of the walks
                    epochs=1,
                    start_nodes=list(range(0, n_walks)),
                    return_weight=1.,
                    neighbor_weight=1.)
    walks = np.vectorize(lambda x: node_names[x])(walks)
    return walks

def walk_to_string(walk, G):
    walk_string = ""
    for i in range(len(walk)-1):
        node1 = walk[i]
        node2 = walk[i+1]
        direc = G.edges[(node1, node2)]['direction']
        walk_string += str(node1) + " "+ str(direc) + " "
    walk_string += walk[-1]
    return walk_string

def get_walks_as_strings(n_graphs=1000, n_walks=10):
    entities_for_graphs =[random.sample(string.ascii_letters[0:26], 9) for i in range(n_graphs)]

    walks_as_strings = []
    for nodes in entities_for_graphs:
        G = get_graph(nodes=nodes)
        walks = get_random_walks(G, n_walks=n_walks)
        walks_as_strings.extend([walk_to_string(walk, G) for walk in walks])
    return walks_as_strings

def plot_path(input_string):
    directions = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}
    steps = input_string.split(' ')

    # Initialize position and label
    x, y = 0, 0
    label = steps[0]

    # List to store the trajectory (includes positions, labels, and directions)
    trajectory = [(x, y, label)]

    # Update position and label based on each step
    for i in range(1, len(steps), 2):
        movement = steps[i]
        label = steps[i + 1]
        dx, dy = directions[movement]
        x, y = x + dx, y + dy
        trajectory.append((x, y, label))

    # Plot the trajectory
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot each step in the trajectory
    for i in range(len(trajectory) - 1):
        x, y, label = trajectory[i]
        dx = trajectory[i + 1][0] - x
        dy = trajectory[i + 1][1] - y
        ax.scatter(x, y, marker='x', color='red')
        ax.text(x, y, label, fontsize=12, ha='right')

        # Draw the arrows at the middle of each line
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                     arrowprops=dict(arrowstyle="->", color='blue', connectionstyle="arc3,rad=.2"))

    # Add label and scatter for the last position
    x, y, label = trajectory[-1]
    ax.scatter(x, y, marker='x', color='red')
    ax.text(x, y, label, fontsize=12, ha='right')

    # Hide the axes but keep the grid
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks(range(min(x for x, _, _ in trajectory), max(x for x, _, _ in trajectory) + 1), minor=False)
    ax.set_yticks(range(min(y for _, y, _ in trajectory), max(y for _, y, _ in trajectory) + 1), minor=False)
    ax.grid(True)

    plt.show()
