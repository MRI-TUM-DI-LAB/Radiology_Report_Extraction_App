import os
import pickle
import networkx as nx
from pyvis.network import Network
from itertools import combinations

# Load the pre-built RadLex graph
# HERE = os.path.dirname(__file__)
HERE = os.path.abspath(os.path.dirname(__file__))
GPICKLE_PATH = os.path.join(HERE, ".." ,"data", "RadLex_graph.gpickle")
with open(GPICKLE_PATH, "rb") as f:
    RadLex_graph = pickle.load(f)


def extract_subgraph_with_shortest_paths(codes, G=RadLex_graph):
    """
    Extract subgraph containing all nodes on shortest paths between any pair of codes.
    """
    nodes_to_include = set(codes)
    for u, v in combinations(codes, 2):
        if nx.has_path(G, u, v):
            nodes_to_include.update(nx.shortest_path(G, u, v))
        if nx.has_path(G, v, u):
            nodes_to_include.update(nx.shortest_path(G, v, u))
    return G.subgraph(nodes_to_include).copy()


def simplify_graph_for_display(G):
    """
    Remove self-loops and reciprocal edges from graph for clearer visualization.
    """
    simplified = nx.DiGraph()
    seen_pairs = set()
    for u, v, d in G.edges(data=True):
        if u == v:
            continue  # drop self-loops
        pair = tuple(sorted((u, v)))
        if pair in seen_pairs:
            continue  # drop reciprocal duplicates
        seen_pairs.add(pair)
        simplified.add_edge(u, v, label=d.get("label"))
    return simplified


def add_shortest_paths_to_root(G, H, codes, root):
    """
    Add shortest paths from root to each code into the graph H with original edge labels.
    Returns the set of edges to highlight.
    """
    highlight_edges = set()
    for code in codes:
        if nx.has_path(G, root, code):
            path = nx.shortest_path(G, root, code)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = G.get_edge_data(u, v)
                # For MultiDiGraph, pick the first edge's label
                first_key = next(iter(edge_data))
                label = edge_data[first_key].get("label", "")
                H.add_edge(u, v, label=label)
                highlight_edges.add((u, v))
            # ensure all nodes on the path are present
            for node in path:
                H.add_node(node)
    return highlight_edges


def visualize_graph(G, H, codes, root, highlight_edges, output_path):
    """
    Visualize the graph H with PyVis and save to output_path (an .html file).
    """
    net = Network(height="700px", width="100%", directed=True, notebook=False)

    # add nodes
    for node in H.nodes():
        if node == root:
            color, size = "orange", 30
        elif node in codes:
            color, size = "skyblue", 25
        else:
            color, size = "lightgray", 10
        title = G.nodes[node].get("description", "")
        net.add_node(node, label=node, title=title, color=color, size=size)

    # add edges
    for u, v, d in H.edges(data=True):
        edge_color = "black" if (u, v) in highlight_edges else "#cccccc"
        net.add_edge(u, v, label=d.get("label", ""), color=edge_color)

    net.repulsion(node_distance=150, spring_length=200)
    net.show(output_path, notebook=False)


def add_legend_to_html(G, codes, output_path):
    """
    Inject a fixed legend panel into the saved HTML file,
    showing each extracted code and its description.
    """
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()

    items = []
    for code in codes:
        desc = G.nodes[code].get("description", "")
        items.append(f"<li><b>{code}</b>: {desc}</li>")

    legend = f"""
<div style="position: fixed; bottom: 20px; left: 20px; background: white;
            border: 2px solid black; padding: 10px; max-width: 300px;
            font-family: Arial, sans-serif; font-size: 12px;
            overflow-y: auto; max-height: 200px; z-index:9999;">
  <h4>Extracted RadLex codes</h4>
  <ul style="padding-left: 1em; margin: 0;">{''.join(items)}</ul>
</div>
"""
    html = html.replace("</body>", legend + "</body>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

# -----------------------------------------------------------------------------
def get_subgraph_html(G=None, codes=None, root="RID1"):
    """
    Build & return a full PyVis HTML string (using CDN resources) for the given codes.
    """
    G = G or RadLex_graph
    codes = codes or []

    # 1) extract & simplify subgraph
    H = extract_subgraph_with_shortest_paths(codes, G)
    H = simplify_graph_for_display(H)
    highlight = add_shortest_paths_to_root(G, H, codes, root)

    # 2) build network in CDN mode
    net = Network(
        height="700px",
        width="100%",
        directed=True,
        notebook=True,
        cdn_resources="remote"
    )
    for node in H.nodes():
        if node == root:
            color, size = "orange", 30
        elif node in codes:
            color, size = "skyblue", 25
        else:
            color, size = "lightgray", 10
        title = G.nodes[node].get("description", "")
        net.add_node(node, label=node, title=title, color=color, size=size)
    for u, v, d in H.edges(data=True):
        edge_color = "black" if (u, v) in highlight else "#cccccc"
        net.add_edge(u, v, label=d.get("label", ""), color=edge_color)
    net.repulsion(node_distance=150, spring_length=200)

    # 3) generate HTML in-memory
    html_str = net.generate_html()

    # 4) inject legend as before
    items = [ f"<li><b>{c}</b>: {G.nodes[c].get('description','')}</li>" for c in codes ]
    legend = (
        "<div style='position:fixed;bottom:20px;left:20px;"
        "background:white;border:2px solid black;padding:10px;"
        "max-width:300px;font-size:12px;overflow-y:auto;max-height:200px;z-index:9999;'>"
        "<h4>Extracted RadLex codes</h4><ul>"
        + "".join(items) +
        "</ul></div>"
    )
    html_str = html_str.replace("</body>", legend + "</body>")

    return html_str

def build_and_visualize_subgraph(codes, root="RID1", output_path=None, G=None):
    """
    High-level function: builds the subgraph, visualizes it, and adds the legend.
    - codes: list of RadLex codes to highlight.
    - root: starting code for shortest paths.
    - output_path: file path for the HTML (e.g. "logic/radlex/data/output/graph_foo.html").
    - G: optional custom graph (defaults to the loaded RadLex_graph).
    """
    G = G or RadLex_graph
    # default output path if not provided
    if output_path is None:
        output_dir  = os.path.join(HERE, "data", "output")
        os.makedirs(output_dir, exist_ok=True)       # ensure it exists
        output_path = os.path.join(output_dir, "graph_sample.html")
    # extract and simplify
    H = extract_subgraph_with_shortest_paths(codes, G)
    H = simplify_graph_for_display(H)
    highlight = add_shortest_paths_to_root(G, H, codes, root)
    visualize_graph(G, H, codes, root, highlight, output_path)
    add_legend_to_html(G, codes, output_path)
