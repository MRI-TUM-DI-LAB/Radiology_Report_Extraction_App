import networkx as nx
from pyvis.network import Network
from itertools import combinations
import pickle

with open("data/RadLex_graph.gpickle", "rb") as f:
    RadLex_graph = pickle.load(f)

def extract_subgraph_with_shortest_paths(codes, G=RadLex_graph):
    """Extract subgraph containing all nodes on shortest paths between any pair of codes."""
    nodes_to_include = set(codes)
    for u, v in combinations(codes, 2):
        if nx.has_path(G, u, v):
            nodes_to_include.update(nx.shortest_path(G, u, v))
        if nx.has_path(G, v, u):
            nodes_to_include.update(nx.shortest_path(G, v, u))
    return G.subgraph(nodes_to_include).copy()

def simplify_graph_for_display(G):
    """Remove self-loops and reciprocal edges from graph for clearer visualization."""
    simplified = nx.DiGraph()
    seen_pairs = set()
    for u, v, d in G.edges(data=True):
        if u == v:
            continue  # remove self-loops
        pair = tuple(sorted((u, v)))
        if pair in seen_pairs:
            continue  # remove reciprocal edges (keep first encountered)
        seen_pairs.add(pair)
        simplified.add_edge(u, v, label=d.get('label'))
    return simplified

def add_shortest_paths_to_root(G, H, codes, root):
    """Add shortest paths from root to each code into the graph H with original edge labels."""
    highlight_edges = set()
    for code in codes:
        if nx.has_path(G, root, code):
            path = nx.shortest_path(G, root, code)
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = G.get_edge_data(u, v)
                # For MultiDiGraph, get first edge's label
                first_key = next(iter(edge_data))
                label = edge_data[first_key].get('label', '')
                H.add_edge(u, v, label=label)
                highlight_edges.add((u, v))
            for node in path:
                H.add_node(node)  # ensure all path nodes are included
    return highlight_edges

def visualize_graph(G, H, codes, root, highlight_edges, output_path):
    """Visualize the graph H with pyvis and save to output_path."""
    net = Network(height="700px", width="100%", directed=True, notebook=False)

    for node in H.nodes():
        if node == root:
            color = 'orange'
            size = 30
        elif node in codes:
            color = 'skyblue'
            size = 25
        else:
            color = 'lightgray'
            size = 10
        label = node
        title = G.nodes[node].get('description', '')  # Tooltip on hover
        net.add_node(node, label=label, title=title, color=color, size=size)

    for u, v, d in H.edges(data=True):
        color = 'black' if (u, v) in highlight_edges else '#cccccc'
        net.add_edge(u, v, label=d.get('label', ''), color=color)

    net.repulsion(node_distance=150, spring_length=200)
    net.show(output_path, notebook=False)

def add_legend_to_html(G, codes, output_path):
    """Add a fixed legend with codes and descriptions to the saved HTML visualization."""
    with open(output_path, 'r', encoding='utf-8') as f:
        html = f.read()

    legend_items = []
    for code in codes:
        desc = G.nodes[code].get('description', '')
        legend_items.append(f"<li><b>{code}</b>: {desc}</li>")

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; background: white; 
                border: 2px solid black; padding: 10px; max-width: 300px; 
                font-family: Arial, sans-serif; font-size: 12px; overflow-y: auto; max-height: 200px; z-index:9999;">
      <h4>Extracted RadLex codes</h4>
      <ul style="padding-left: 1em; margin: 0;">{''.join(legend_items)}</ul>
    </div>
    """

    html = html.replace("</body>", legend_html + "</body>")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

def build_and_visualize_subgraph(codes, G=RadLex_graph, root='RID1', output_path="data/output/graph_sample.html"):
    """Complete pipeline: extract, clean, add shortest paths, visualize and add legend."""
    H_sub = extract_subgraph_with_shortest_paths(codes, G)
    cleaned_H = simplify_graph_for_display(H_sub)
    highlight_edges = add_shortest_paths_to_root(G, cleaned_H, codes, root)
    visualize_graph(G, cleaned_H, codes, root, highlight_edges, output_path)
    add_legend_to_html(G, codes, output_path)
