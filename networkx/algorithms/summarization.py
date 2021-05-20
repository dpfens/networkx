"""
Graph summarization finds smaller representations of graphs resulting in faster
runtime of algorithms, reduced storage needs, and noise reduction.
Summarization has applications in areas such as visualization, pattern mining,
clustering and community detection, and more.  Core graph summarization
techniques are grouping/aggregation, bit-compression,
simplification/sparsification, and influence based. Graph summarization
algorithms often produce either summary graphs in the form of supergraphs or
sparsified graphs, or a list of independent structures. Supergraphs are the
most common product, which consist of supernodes and original nodes and are
connected by edges and superedges, which represent aggregate edges between
nodes and supernodes.

Grouping/aggregation based techniques compress graphs by representing
close/connected nodes and edges in a graph by a single node/edge in a
supergraph. Nodes can be grouped together into supernodes based on their
structural similarities or proximity within a graph to reduce the total number
of nodes in a graph. Edge-grouping techniques group edges into lossy/lossless
nodes called compressor or virtual nodes to reduce the total number of edges in
a graph. Edge-grouping techniques can be lossless, meaning that they can be
used to re-create the original graph, or techniques can be lossy, requiring
less space to store the summary graph, but at the expense of lower
recontruction accuracy of the original graph.

Bit-compression techniques minimize the amount of information needed to
describe the original graph, while revealing structural patterns in the
original graph.  The two-part minimum description length (MDL) is often used to
represent the model and the original graph in terms of the model.  A key
difference between graph compression and graph summarization is that graph
summarization focuses on finding structural patterns within the original graph,
whereas graph compression focuses on compressions the original graph to be as
small as possible.  **NOTE**: Some bit-compression methods exist solely to
compress a graph without creating a summary graph or finding comprehensible
structural patterns.

Simplification/Sparsification techniques attempt to create a sparse
representation of a graph by removing unimportant nodes and edges from the
graph.  Sparsified graphs differ from supergraphs created by
grouping/aggregation by only containing a subset of the original nodes and
edges of the original graph.

Influence based techniques aim to find a high-level description of influence
propagation in a large graph.  These methods are scarce and have been mostly
applied to social graphs.

*dedensification* is a grouping/aggregation based technique to compress the
neighborhoods around high-degree nodes in unweighted graphs by adding
compressor nodes that summarize multiple edges of the same type to
high-degree nodes (nodes with a degree greater than a given threshold).
Dedensification was developed for the purpose of increasing performance of
query processing around high-degree nodes in graph databases and enables direct
operations on the compressed graph.  The structural patterns surrounding
high-degree nodes in the original is preserved while using fewer edges and
adding a small number of compressor nodes.  The degree of nodes present in the
original graph is also preserved. The current implementation of dedensification
supports graphs with one edge type.

For more information on graph summarization, see `Graph Summarization Methods
and Applications: A Survey <https://dl.acm.org/doi/abs/10.1145/3186727>`_
"""
import networkx as nx
from collections import Counter, defaultdict


__all__ = ["dedensify", "snap_aggregation"]


def dedensify(G, threshold, prefix=None, copy=True):
    """Compresses neighborhoods around high-degree nodes

    Reduces the number of edges to high-degree nodes by adding compressor nodes
    that summarize multiple edges of the same type to high-degree nodes (nodes
    with a degree greater than a given threshold).  Dedensification also has
    the added benefit of reducing the number of edges around high-degree nodes.
    The implementation currently supports graphs with a single edge type.

    Parameters
    ----------
    G: graph
       A networkx graph
    threshold: int
       Minimum degree threshold of a node to be considered a high degree node.
       The threshold must be greater than or equal to 2.
    prefix: str or None, optional (default: None)
       An optional prefix for denoting compressor nodes
    copy: bool, optional (default: True)
       Indicates if dedensification should be done inplace

    Returns
    -------
    dedensified networkx graph : (graph, set)
        2-tuple of the dedensified graph and set of compressor nodes

    Notes
    -----
    According to the algorithm in [1]_, removes edges in a graph by
    compressing/decompressing the neighborhoods around high degree nodes by
    adding compressor nodes that summarize multiple edges of the same type
    to high-degree nodes.  Dedensification will only add a compressor node when
    doing so will reduce the total number of edges in the given graph. This
    implementation currently supports graphs with a single edge type.

    Examples
    --------
    Dedensification will only add compressor nodes when doing so would result
    in fewer edges::

        >>> original_graph = nx.DiGraph()
        >>> original_graph.add_nodes_from(
        ...     ["1", "2", "3", "4", "5", "6", "A", "B", "C"]
        ... )
        >>> original_graph.add_edges_from(
        ...     [
        ...         ("1", "C"), ("1", "B"),
        ...         ("2", "C"), ("2", "B"), ("2", "A"),
        ...         ("3", "B"), ("3", "A"), ("3", "6"),
        ...         ("4", "C"), ("4", "B"), ("4", "A"),
        ...         ("5", "B"), ("5", "A"),
        ...         ("6", "5"),
        ...         ("A", "6")
        ...     ]
        ... )
        >>> c_graph, c_nodes = nx.dedensify(original_graph, threshold=2)
        >>> original_graph.number_of_edges()
        15
        >>> c_graph.number_of_edges()
        14

    A dedensified, directed graph can be "densified" to reconstruct the
    original graph::

        >>> original_graph = nx.DiGraph()
        >>> original_graph.add_nodes_from(
        ...     ["1", "2", "3", "4", "5", "6", "A", "B", "C"]
        ... )
        >>> original_graph.add_edges_from(
        ...     [
        ...         ("1", "C"), ("1", "B"),
        ...         ("2", "C"), ("2", "B"), ("2", "A"),
        ...         ("3", "B"), ("3", "A"), ("3", "6"),
        ...         ("4", "C"), ("4", "B"), ("4", "A"),
        ...         ("5", "B"), ("5", "A"),
        ...         ("6", "5"),
        ...         ("A", "6")
        ...     ]
        ... )
        >>> c_graph, c_nodes = nx.dedensify(original_graph, threshold=2)
        >>> # re-densifies the compressed graph into the original graph
        >>> for c_node in c_nodes:
        ...     all_neighbors = set(nx.all_neighbors(c_graph, c_node))
        ...     out_neighbors = set(c_graph.neighbors(c_node))
        ...     for out_neighbor in out_neighbors:
        ...         c_graph.remove_edge(c_node, out_neighbor)
        ...     in_neighbors = all_neighbors - out_neighbors
        ...     for in_neighbor in in_neighbors:
        ...         c_graph.remove_edge(in_neighbor, c_node)
        ...         for out_neighbor in out_neighbors:
        ...             c_graph.add_edge(in_neighbor, out_neighbor)
        ...     c_graph.remove_node(c_node)
        ...
        >>> nx.is_isomorphic(original_graph, c_graph)
        True

    References
    ----------
    .. [1] Maccioni, A., & Abadi, D. J. (2016, August).
       Scalable pattern matching over compressed graphs via dedensification.
       In Proceedings of the 22nd ACM SIGKDD International Conference on
       Knowledge Discovery and Data Mining (pp. 1755-1764).
       http://www.cs.umd.edu/~abadi/papers/graph-dedense.pdf
    """
    if threshold < 2:
        raise nx.NetworkXError("The degree threshold must be >= 2")

    degrees = G.in_degree if G.is_directed() else G.degree
    # Group nodes based on degree threshold
    high_degree_nodes = set([n for n, d in degrees if d > threshold])
    low_degree_nodes = G.nodes() - high_degree_nodes

    auxillary = {}
    for node in G:
        high_degree_neighbors = frozenset(high_degree_nodes & set(G[node]))
        if high_degree_neighbors:
            if high_degree_neighbors in auxillary:
                auxillary[high_degree_neighbors].add(node)
            else:
                auxillary[high_degree_neighbors] = {node}

    if copy:
        G = G.copy()

    compressor_nodes = set()
    for index, (high_degree_nodes, low_degree_nodes) in enumerate(auxillary.items()):
        low_degree_node_count = len(low_degree_nodes)
        high_degree_node_count = len(high_degree_nodes)
        old_edges = high_degree_node_count * low_degree_node_count
        new_edges = high_degree_node_count + low_degree_node_count
        if old_edges <= new_edges:
            continue
        compression_node = "".join(str(node) for node in high_degree_nodes)
        if prefix:
            compression_node = str(prefix) + compression_node
        for node in low_degree_nodes:
            for high_node in high_degree_nodes:
                if G.has_edge(node, high_node):
                    G.remove_edge(node, high_node)

            G.add_edge(node, compression_node)
        for node in high_degree_nodes:
            G.add_edge(compression_node, node)
        compressor_nodes.add(compression_node)
    return G, compressor_nodes


def _snap_build_graph(
    G,
    groups,
    node_attributes,
    edge_attributes,
    neighbor_info,
    edge_types,
    prefix,
    supernode_attribute,
    superedge_attribute,
):
    """
    Build the summary graph from the data structures produced in the SNAP aggregation algorithm

    Used in the SNAP aggregation algorithm to build the output summary graph and supernode
    lookup dictionary.  This process uses the original graph and the data structures to
    create the supernodes with the correct node attributes, and the superedges with the correct
    edge attributes

    Parameters
    ----------
    G: networkx.Graph
        the original graph to be summarized
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    node_attributes: iterable
        An iterable of the node attributes considered in the summarization process
    edge_attributes: iterable
        An iterable of the edge attributes considered in the summarization process
    neighbor_info: dict
        A data structure indicating the number of edges a node has with the
        groups in the current summarization of each edge type
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization
    prefix: string
        The prefix to be added to all supernodes
    supernode_attribute: str
        The node attribute for recording the supernode groupings of nodes
    superedge_attribute: str
        The edge attribute for recording the edge types represented by superedges

    Returns
    -------
    summary graph: Networkx graph
    """
    output = G.__class__()
    node_label_lookup = dict()
    for index, group_id in enumerate(groups):
        group_set = groups[group_id]
        supernode = "%s%s" % (prefix, index)
        node_label_lookup[group_id] = supernode
        supernode_attributes = {
            attr: G.nodes[next(iter(group_set))][attr] for attr in node_attributes
        }
        supernode_attributes[supernode_attribute] = group_set
        output.add_node(supernode, **supernode_attributes)

    for group_id in groups:
        group_set = groups[group_id]
        source_supernode = node_label_lookup[group_id]
        for other_group, group_edge_types in neighbor_info[
            next(iter(group_set))
        ].items():
            if group_edge_types:
                target_supernode = node_label_lookup[other_group]
                summary_graph_edge = (source_supernode, target_supernode)

                edge_types = [
                    dict(zip(edge_attributes, edge_type))
                    for edge_type in group_edge_types
                ]

                has_edge = output.has_edge(*summary_graph_edge)
                if output.is_multigraph():
                    if not has_edge:
                        for edge_type in edge_types:
                            output.add_edge(*summary_graph_edge, **edge_type)
                    elif not output.is_directed():
                        existing_edge_data = output.get_edge_data(*summary_graph_edge)
                        for edge_type in edge_types:
                            if edge_type not in existing_edge_data.values():
                                output.add_edge(*summary_graph_edge, **edge_type)
                else:
                    superedge_attributes = {superedge_attribute: edge_types}
                    output.add_edge(*summary_graph_edge, **superedge_attributes)

    return output


def _snap_eligible_group(G, groups, group_lookup, edge_types):
    """
    Determines if a group is eligible to be split.

    A group is eligible to be split if all nodes in the group have edges of the same type(s)
    with the same other groups.

    Parameters
    ----------
    G: graph
        graph to be summarized
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    group_lookup: dict
        dictionary of nodes and their current corresponding group ID
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization

    Returns
    -------
    tuple: group ID to split, and neighbor-groups participation_counts data structure
    """
    neighbor_info = {node: {gid: Counter() for gid in groups} for node in group_lookup}
    for group_id in groups:
        current_group = groups[group_id]

        # build neighbor_info for nodes in group
        for node in current_group:
            neighbor_info[node] = {group_id: Counter() for group_id in groups}
            edges = G.edges(node, keys=True) if G.is_multigraph() else G.edges(node)
            for edge in edges:
                neighbor = edge[1]
                edge_type = edge_types[edge]
                neighbor_group_id = group_lookup[neighbor]
                neighbor_info[node][neighbor_group_id][edge_type] += 1

        # check if group_id is eligible to be split
        group_size = len(current_group)
        for other_group_id in groups:
            edge_counts = Counter()
            for node in current_group:
                edge_counts.update(neighbor_info[node][other_group_id].keys())

            if not all(count == group_size for count in edge_counts.values()):
                # only the neighbor_info of the returned group_id is required for handling group splits
                return group_id, neighbor_info

    # if no eligible groups, complete neighbor_info is calculated
    return None, neighbor_info


def _snap_split(
    groups,
    neighbor_info,
    group_lookup,
    group_id,
):
    """
    Splits a group based on edge types and updates the groups accordingly

    Splits the group with the given group_id based on the edge types
    of the nodes so that each new grouping will all have the same
    edges with other nodes.

    Parameters
    ----------
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    neighbor_info: dict
        A data structure indicating the number of edges a node has with the
        groups in the current summarization of each edge type
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization
    group_lookup: dict
        dictionary of nodes and their current corresponding group ID
    group_id: object
        ID of group to be split

    Returns
    -------
    dict
        The updated groups based on the split
    """
    new_group_mappings = defaultdict(set)
    for node in groups[group_id]:
        signature = tuple(
            frozenset(edge_types) for edge_types in neighbor_info[node].values()
        )
        new_group_mappings[signature].add(node)

    # leave the biggest new_group as the original group
    new_groups = sorted(new_group_mappings.values(), key=len)
    for new_group in new_groups[:-1]:
        # Assign unused integer as the new_group_id
        # ids are tuples, so will not interact with the original group_ids
        new_group_id = len(groups)
        groups[new_group_id] = new_group
        groups[group_id] -= new_group
        for node in new_group:
            group_lookup[node] = new_group_id

    return groups


def snap_aggregation(
    G,
    node_attributes,
    edge_attributes=(),
    prefix="Supernode-",
    supernode_attribute="group",
    superedge_attribute="types",
):
    """Creates a summary graph based on attributes and connectivity.

    This function uses the Summarization by Grouping Nodes on Attributes
    and Pairwise edges (SNAP) algorithm for summarizing a given
    graph by grouping nodes by node attributes and their edge attributes
    into supernodes in a summary graph.  This name SNAP should not be
    confused with the Stanford Network Analysis Project (SNAP).

    Here is a high-level view of how this algorithm works:

    1) Group nodes by node attribute values.

    2) Iteratively split groups until all nodes in each group have edges
    to nodes in the same groups. That is, until all the groups are homogeneous
    in their member nodes' edges to other groups.  For example,
    if all the nodes in group A only have edge to nodes in group B, then the
    group is homogeneous and does not need to be split. If all nodes in group B
    have edges with nodes in groups {A, C}, but some also have edges with other
    nodes in B, then group B is not homogeneous and needs to be split into
    groups have edges with {A, C} and a group of nodes having
    edges with {A, B, C}.  This way, viewers of the summary graph can
    assume that all nodes in the group have the exact same node attributes and
    the exact same edges.

    3) Build the output summary graph, where the groups are represented by
    super-nodes. Edges represent the edges shared between all the nodes in each
    respective groups.

    A SNAP summary graph can be used to visualize graphs that are too large to display
    or visually analyze, or to efficiently identify sets of similar nodes with similar connectivity
    patterns to other sets of similar nodes based on specified node and/or edge attributes in a graph.

    Parameters
    ----------
    G: graph
        Networkx Graph to be summarized
    edge_attributes: iterable, optional
        An iterable of the edge attributes considered in the summarization process.  If provided, unique
        combinations of the attribute values found in the graph are used to
        determine the edge types in the graph.  If not provided, all edges
        are considered to be of the same type.
    prefix: str
        The prefix used to denote supernodes in the summary graph. Defaults to 'Supernode-'.
    supernode_attribute: str
        The node attribute for recording the supernode groupings of nodes. Defaults to 'group'.
    superedge_attribute: str
        The edge attribute for recording the edge types of multiple edges. Defaults to 'types'.

    Returns
    -------
    networkx.Graph: summary graph

    Examples
    --------
    SNAP aggregation takes a graph and summarizes it in the context of user-provided
    node and edge attributes such that a viewer can more easily extract and
    analyze the information represented by the graph

    >>> nodes = {
    ...     "A": dict(color="Red"),
    ...     "B": dict(color="Red"),
    ...     "C": dict(color="Red"),
    ...     "D": dict(color="Red"),
    ...     "E": dict(color="Blue"),
    ...     "F": dict(color="Blue"),
    ... }
    >>> edges = [
    ...     ("A", "E", "Strong"),
    ...     ("B", "F", "Strong"),
    ...     ("C", "E", "Weak"),
    ...     ("D", "F", "Weak"),
    ... ]
    >>> G = nx.Graph()
    >>> for node in nodes:
    ...     attributes = nodes[node]
    ...     G.add_node(node, **attributes)
    ...
    >>> for source, target, type in edges:
    ...     G.add_edge(source, target, type=type)
    ...
    >>> node_attributes = ('color', )
    >>> edge_attributes = ('type', )
    >>> summary_graph = nx.snap_aggregation(G, node_attributes=node_attributes, edge_attributes=edge_attributes)

    Notes
    -----
    The summary graph produced is called a maximum Attribute-edge
    compatible (AR-compatible) grouping.  According to [1]_, an
    AR-compatible grouping means that all nodes in each group have the same
    exact node attribute values and the same exact edges and
    edge types to one or more nodes in the same groups.  The maximal
    AR-compatible grouping is the grouping with the minimal cardinality.

    The AR-compatible grouping is the most detailed grouping provided by
    any of the SNAP algorithms.

    References
    ----------
    .. [1] Y. Tian, R. A. Hankins, and J. M. Patel. Efficient aggregation
       for graph summarization. In Proc. 2008 ACM-SIGMOD Int. Conf.
       Management of Data (SIGMOD’08), pages 567–580, Vancouver, Canada,
       June 2008.
    """
    edge_types = {
        edge: tuple(attrs.get(attr) for attr in edge_attributes)
        for edge, attrs in G.edges.items()
    }
    if not G.is_directed():
        if G.is_multigraph():
            # list is needed to avoid mutating while iterating
            edges = [((v, u, k), etype) for (u, v, k), etype in edge_types.items()]
        else:
            # list is needed to avoid mutating while iterating
            edges = [((v, u), etype) for (u, v), etype in edge_types.items()]
        edge_types.update(edges)

    group_lookup = {
        node: tuple(attrs[attr] for attr in node_attributes)
        for node, attrs in G.nodes.items()
    }
    groups = defaultdict(set)
    for node, node_type in group_lookup.items():
        groups[node_type].add(node)

    eligible_group_id, neighbor_info = _snap_eligible_group(
        G, groups, group_lookup, edge_types
    )
    while eligible_group_id:
        groups = _snap_split(groups, neighbor_info, group_lookup, eligible_group_id)
        eligible_group_id, neighbor_info = _snap_eligible_group(
            G, groups, group_lookup, edge_types
        )
    return _snap_build_graph(
        G,
        groups,
        node_attributes,
        edge_attributes,
        neighbor_info,
        edge_types,
        prefix,
        supernode_attribute,
        superedge_attribute,
    )


def ksnap_neighbor_info(
    G,
    groups,
    edge_attributes,
    edge_types,
    group_lookup,
    neighbor_info=None,
    updated_groups=None,
):
    """
    Determines if a group is eligible to be split.
    A group is eligible to be split if all nodes in the group have edges of the same type(s)
    with the same other groups.
    Parameters
    ----------
    G: graph
        graph to be summarized
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    group_lookup: dict
        dictionary of nodes and their current corresponding group ID
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization
    Returns
    -------
    tuple: group ID to split, and neighbor-groups participation_counts data structure
    """
    participation_counts = {gid: {ogid: Counter() for ogid in groups} for gid in groups}
    if not neighbor_info:
        neighbor_info = {
            node: {gid: Counter() for gid in groups} for node in group_lookup
        }

    if updated_groups:
        groups_to_update = updated_groups
    else:
        groups_to_update = groups

    for group_id in groups_to_update:
        current_group = groups[group_id]

        # build neighbor_info for nodes in group
        for node in current_group:
            neighbor_info[node] = {group_id: Counter() for group_id in groups}
            edges = G.edges(node, keys=True) if G.is_multigraph() else G.edges(node)
            for edge in edges:
                neighbor = edge[1]
                edge_type = edge_types[edge]
                neighbor_group_id = group_lookup[neighbor]
                neighbor_info[node][neighbor_group_id][edge_type] += 1

        # check if group_id is eligible to be split
        for other_group_id in groups_to_update:
            edge_counts = Counter()
            for node in current_group:
                edge_counts.update(neighbor_info[node][other_group_id].keys())
                participation_counts[group_id][other_group_id].update(
                    neighbor_info[node][other_group_id].keys()
                )

    # if no eligible groups, complete neighbor_info is calculated
    return neighbor_info, participation_counts


def ksnap_participation_ratio(
    groups, participation_counts, group_i, group_j, edge_type
):
    """
    The proportion of nodes in both groups that have edges with the other
    group of a with a given edge type
    Parameters
    ----------
    groups: iterable
        Group membership of nodes
    participation_counts: dict
        indicates the number of nodes in a group that have edges connecting
        to another group with a given edge type
    group_i: int
    group_j: int
    edge_type: int
    Returns
    -------
    float
    """
    numerator = (
        participation_counts[group_i][group_j][edge_type]
        + participation_counts[group_j][group_i][edge_type]
    )
    denominator = len(groups[group_i]) + len(groups[group_j])
    return numerator / float(denominator)


def ksnap_update_groups(
    groups,
    edge_attributes,
    old_group_id,
    new_group,
    group_lookup,
):
    """
    Updates data structures in SNAP aggregation
    Used in the SNAP aggregation algorithm to update the group id, group sets, and
    group_lookup after a group has been split.
    Parameters
    ----------
    groups: iterable
        Group membership of nodes
    edge_attributes: iterable
        The edge attributes that can be found in edges in the graph, and
        are to be recognized in the summarization
    old_group_id: int
        the ID of the group that was last split
    new_groups: iterable
        List of sets of new groups of nodes
    group_lookup: dict
        Dictionary containing the group assignment of each node in the graph
    Returns
    -------
    2-tuple:
        updated group_ids
        updated_groups
    """
    old_group_count = len(groups)

    groups[old_group_id] -= set(new_group)
    new_group_id = len(groups)
    groups[new_group_id] = new_group

    new_total_group_count = old_group_count + 1
    updated_groups = set([old_group_id]) | set(
        range(old_group_count, new_total_group_count)
    )
    for group_id in updated_groups:
        for node in groups[group_id]:
            group_lookup[node] = group_id
    return groups, new_group_id


def ksnaptd_aggregation(
    G,
    k,
    node_attributes,
    edge_attributes=(),
    prefix="Supernode-",
    supernode_attribute="group",
    superedge_attribute="types",
):
    """
    Executes the KSNAP Top-down summarization algorithm.  Stops when k
    supernodes exist in the summary graph.

    Parameters
    ----------
    G: graph
        The networkx.Graph to summarize
    k: int
        The number of nodes to produce in the summary graph
    edge_attributes: iterable, optional
        The edge attributes that can be found in edges in the graph, and
        are to be recognized in the summarization.  If provided, unique
        combinations of the attribute values found in the graph are used to
        determine the edge types in the graph.  If not provided, all edges
        are considered to be of the same type
    prefix: str
        The prefix used to denote supernodes in the summary graph

    Returns
    -------
    tuple:
        networkx.Graph: The summary graph with k supernodes
        supernodes (dict<string>:<set>): Mappings of the supernodes to
            the original graph nodes
    Raises
    ------
    ValueError: k must be an integer >= 2

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> summarizer = KSNAPTD.from_graph(G, node_attributes=('club', ))
    >>> smaller_summary_graph, larger_supernodes = summarizer.summarize(G, k=5)
    >>> larger_summary_graph, smaller_supernodes = summarizer.summarize(G, k=8)

    Notes
    -----
    The Top-Down approach is generally more efficient and produces
    significantly better quality summaries than the bottom-up approach for
    small values of k.  According to [1]_, when a small k is used, fewer
    split decisions need to be made to obtain the desired k from the
    initial A-compatible grouping.  Split decisions are made based on the
    grouping produced by the previous split decision.  When many splits
    need to occur to reach the target k groups, split decision errors can
    accrue to signficantly impact the summary quality.  Minimizing the
    number of split decisions is generally desireable, as it tends to
    increase the summary quality.
    In practice, users are more likely to choose small values of k to
    generate summaries, making the top-down approach used more often.
    The Top-down approach first groups nodes by their node attributes.
    Then, groups are split iteratively until k groups exist.  On each
    iteration, the split will be chosen on the current grouping to both
    result in fewer overall discrepancies in edges in the grouping
    and separate the fewest nodes from the existing group.

    References
    ----------
    .. [1] Y. Tian, R. A. Hankins, and J. M. Patel. Efficient aggregation
       for graph summarization. In Proc. 2008 ACM-SIGMOD Int. Conf.
       Management of Data (SIGMOD’08), pages 567–580, Vancouver, Canada,
       June 2008.
    """
    if not edge_attributes:
        edge_attributes = set()
    edge_types = {
        edge: tuple(attrs.get(attr) for attr in edge_attributes)
        for edge, attrs in G.edges.items()
    }
    if not G.is_directed():
        if G.is_multigraph():
            # list is needed to avoid mutating while iterating
            edges = [((v, u, k), etype) for (u, v, k), etype in edge_types.items()]
        else:
            # list is needed to avoid mutating while iterating
            edges = [((v, u), etype) for (u, v), etype in edge_types.items()]
        edge_types.update(edges)

    group_lookup = {
        node: tuple(attrs[attr] for attr in node_attributes)
        for node, attrs in G.nodes.items()
    }
    groups = defaultdict(set)
    for node, node_type in group_lookup.items():
        groups[node_type].add(node)

    neighbor_info, participation_counts = ksnap_neighbor_info(
        G,
        groups,
        edge_attributes,
        edge_types,
        group_lookup,
    )

    while len(groups) < k:
        group_i, group_t, edge_type = ksnaptd_identify_split(
            groups, participation_counts, edge_types
        )
        new_group = set(
            node for node in groups[group_i] if neighbor_info[node][group_t][edge_type]
        )
        groups, new_group_id = ksnap_update_groups(
            groups,
            edge_attributes,
            group_i,
            new_group,
            group_lookup,
        )

        updated_groups = set([group_i, group_t, new_group_id])
        for node in groups[group_i]:
            updated_groups |= set(neighbor_info[node].keys())
        for node in groups[group_t]:
            updated_groups |= set(neighbor_info[node].keys())

        neighbor_info, participation_counts = ksnap_neighbor_info(
            G,
            groups,
            edge_attributes,
            edge_types,
            group_lookup=group_lookup,
            neighbor_info=neighbor_info,
            updated_groups=updated_groups,
        )
    return snap_build_graph(
        G,
        groups,
        node_attributes,
        edge_attributes,
        neighbor_info,
        edge_types,
        prefix,
        supernode_attribute,
        superedge_attribute,
    )


def ksnaptd_identify_split(groups, participation_counts, edge_types):
    """
    Identifies and returns the group ID to be split

    Parameters
    ----------
    groups: iterable
        Group membership of nodes
    participation_counts: dict
        indicates the number of nodes in a group that have edges connecting
        to another group with a given edge type
    edge_types: dict
        A edge key mapping the edge edge values to it's
        index/ID
    Returns
    -------
    tuple:
        3-tuple of the group to split, and the neighbor group and
        edge index by which to split the group
    """
    max_group = max_neighbor_group = max_edge_type = 0
    max_ar_delta = float("-inf")
    for group_id in groups:
        for other_group_id in groups:
            for edge_type in participation_counts[group_id][other_group_id]:
                participation_ratio = ksnap_participation_ratio(
                    groups, participation_counts, group_id, other_group_id, edge_type
                )
                ar_delta = participation_counts[group_id][other_group_id][edge_type]

                if participation_ratio > 0.5:
                    ar_delta = len(groups[group_id]) - ar_delta

                if ar_delta > max_ar_delta:
                    max_ar_delta = ar_delta
                    max_group = group_id
                    max_neighbor_group = other_group_id
                    max_edge_type = edge_type

    return max_group, max_neighbor_group, max_edge_type


def ksnapbu_aggregation(
    G,
    k,
    node_attributes,
    edge_attributes=(),
    prefix="Supernode-",
    supernode_attribute="group",
    superedge_attribute="types",
):
    """
    Executes the Bottom-up summarization algorithm.  Stops when k
    supernodes exist in the summary graph.

    Parameters
    ----------
    G: graph
        The networkx.Graph to summarize
    k: int
        The number of nodes to produce in the summary graph
    edge_attributes: iterable, optional
        The edge attributes that can be found in edges in the graph, and
        are to be recognized in the summarization.  If provided, unique
        combinations of the attribute values found in the graph are used to
        determine the edge types in the graph.  If not provided, all edges
        are considered to be of the same type
    prefix: str
        The prefix used to denote supernodes in the summary graph

    Returns
    -------
    tuple:
        networkx.Graph:  The summary graph with k supernodes

    Raises
    ------
    ValueError: K must be an integer 2 <= k <= len(AR-Compatible grouping of G)

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> node_attributes=('club', )
    >>> smaller_summary_graph = ksnapbu_aggregation(G, k=10, node_attributes=node_attributes)
    >>> larger_summary_graph = ksnapbu_aggregation(G, k=14, node_attributes=node_attributes)

    Notes
    -----
    The bottom-up approach starts with the AR-compatible groups, which,
    according to [1]_, gives it an advantage over the top down approach
    when producing summary graphs with larger values of k as fewer merges
    need to be made than if the top-down approach were used.  Since each
    merge decision is made on the current grouping, merge decision errors
    can compound and result in more decisions being based on those errors
    when many merges have to occur to obtain the desired k groups.
    Minimizing the number of merge decisions is generally desireable, as
    it tends to increase the summary quality.
    As low k values tend to be used more often, the bottom-up approach is
    used less often.
    The Bottom-up approach first executes the SNAP algorithm to produce a
    Attribute-edge compatible (AR-compatible) grouping.
    Then, groups are merged iteratively until k groups exist.  On each
    iteration, the 2 groups with the most similar neighbor groups and
    participation ratios within those groups are merged into 1 group.  If
    a group has multiple groups with the same participation ratios,
    the smaller group is preferred.

    References
    ----------
    .. [1] Y. Tian, R. A. Hankins, and J. M. Patel. Efficient aggregation
       for graph summarization. In Proc. 2008 ACM-SIGMOD Int. Conf.
       Management of Data (SIGMOD’08), pages 567–580, Vancouver, Canada,
       June 2008.
    """
    if k < 2 or not isinstance(k, int):
        raise ValueError("k(%r) must be an integer >= 2" % k)

    (
        groups,
        neighbor_info,
        group_lookup,
        edge_types,
    ) = snap_build_data_structures(G, node_attributes, edge_attributes)

    if k > len(groups):
        raise ValueError(
            "k (%r) must be an integer <= the number of group in a maximal AR-compatible grouping (%r)"
            % (k, len(groups))
        )

    neighbor_info, participation_counts = ksnap_neighbor_info(
        G, groups, edge_attributes, edge_types, group_lookup
    )

    node_attribute_groups = dict()
    for group_id in groups:
        node = next(iter(groups[group_id]))
        node_attribute_values = tuple(
            [G.nodes[node][attribute] for attribute in node_attributes]
        )
        node_attribute_groups.setdefault(node_attribute_values, set())
        node_attribute_groups[node_attribute_values].add(group_id)

    is_directed = G.is_directed()

    while len(groups) > k:
        group_i, group_j = ksnapbu_identify_groups(
            G,
            node_attributes,
            groups,
            participation_counts,
            node_attribute_groups,
            is_directed,
        )
        groups, group_lookup = ksnapbu_merge(
            groups, group_i, group_j, neighbor_info, group_lookup
        )

        neighbor_info, participation_counts = ksnap_neighbor_info(
            G, groups, edge_attributes, edge_types, group_lookup
        )

    return snap_build_graph(
        G,
        groups,
        node_attributes,
        edge_attributes,
        neighbor_info,
        edge_types,
        prefix,
        supernode_attribute,
        superedge_attribute,
    )


def ksnapbu_merge(groups, group_i, group_j, neighbor_info, group_lookup):
    """
    Merges group j into group i. updates groups

    Parameters
    ----------
    groups: iterable
        Group membership of nodes
    group_i (int):
        ID of group to be merged with group j
    group_j (int):
        ID of group to be merged with group i
    group_lookup: dict
        dictionary of nodes and their current group ID
    """
    for node in groups[group_j]:
        group_lookup[node] = group_i
        neighbor_info[node][group_i].update(neighbor_info[node][group_j].keys())
        neighbor_info[node].pop(group_j)

    groups[group_i] |= groups[group_j]
    groups.pop(group_j)
    return groups, group_lookup


def ksnapbu_merge_distance(groups, participation_counts, group_i, group_j):
    """
    The accumulated differences in participation ratios between groups i
    and j with other groups.

    Parameters
    ----------
    groups: iterable
        Group membership of nodes in the current graph summary
    participation_counts: dict
        indicates the number of nodes in a group that have edges connecting
        to another group with a given edge type
    group_i (int):
        ID of group
    group_j (int):
        ID of group

    Returns
    -------
    int
    """
    output = 0.0
    for group_id in groups:
        if group_id in (group_i, group_j):
            continue
        for edge_type in participation_counts[group_id][group_i]:
            output += abs(
                ksnap_participation_ratio(
                    groups, participation_counts, group_i, group_id, edge_type
                )
                - ksnap_participation_ratio(
                    groups, participation_counts, group_j, group_id, edge_type
                )
            )
    return output


def ksnapbu_agreements(groups, participation_counts, group_i, group_j):
    """
    Indicates the total number of mutual neighbors with which groups i and
    j both have strong/weak edges.

    Parameters
    ----------
    groups: iterable
        Group membership of nodes in the current graph summary
    participation_counts: dict
        indicates the number of nodes in a group that have edges connecting
        to another group with a given edge type
    group_i (int):
        ID of group
    group_j (int):
        ID of group

    Returns
    -------
    int
    """
    neighbor_groups_i = set()
    neighbor_groups_j = set()
    for group_id in groups:
        if any(participation_counts[group_i][group_id]):
            neighbor_groups_i.add(group_id)
        if any(participation_counts[group_j][group_id]):
            neighbor_groups_j.add(group_id)

    mutual_neighbors = neighbor_groups_i & neighbor_groups_j
    agreements = 0
    for neighbor in mutual_neighbors:
        for edge_type in participation_counts[group_i][group_j]:
            participation_ratio_ikr = (
                ksnap_participation_ratio(
                    groups, participation_counts, group_i, neighbor, edge_type
                )
                <= 0.5
            )
            participation_ratio_jkr = (
                ksnap_participation_ratio(
                    groups, participation_counts, group_j, neighbor, edge_type
                )
                <= 0.5
            )
            if not participation_ratio_ikr ^ participation_ratio_jkr:
                agreements += 1

    return agreements, mutual_neighbors


def ksnapbu_identify_groups(
    G, node_attributes, groups, participation_counts, node_attribute_groups, is_directed
):
    """
    Identifies the optimal groups to merge together based on the groups'
    similarity in participation ratios with other groups

    Parameters
    ----------
    groups: iterable
        Group membership of nodes in the current graph summary
    participation_counts: dict
        indicates the number of nodes in a group that have edges connecting
        to another group with a given edge type
    group_i (int):
        ID of group
    group_j (int):
        ID of group
    is_directed: boolean
        indicates if the graph is directed

    Returns
    -------
    tuple:
        The 2-tuple of groups to be merged together
    """
    potential_groups = dict()
    for node_attribute_group in node_attribute_groups:
        attribute_groups = list(node_attribute_groups[node_attribute_group])
        attribute_group_count = len(attribute_groups)
        for i in range(attribute_group_count):
            group_id = attribute_groups[i]
            group_size = len(groups[group_id])
            if is_directed:
                other_group_iterator = range(attribute_group_count)
            else:
                other_group_iterator = range(i + 1, attribute_group_count)
            for j in other_group_iterator:
                if i == j:
                    continue
                other_group_id = attribute_groups[j]
                args = (group_id, other_group_id)
                merge_dist = ksnapbu_merge_distance(groups, participation_counts, *args)
                agreements, mutual_neighbors = ksnapbu_agreements(
                    groups, participation_counts, group_id, other_group_id
                )
                # invert agreements to ensure groups with more agreements are selected first
                inv_agreements = len(mutual_neighbors) - agreements
                min_group_size = min(group_size, len(groups[other_group_id]))
                potential_groups[group_id, other_group_id, node_attribute_group] = (
                    merge_dist,
                    inv_agreements,
                    min_group_size,
                )
    group_i, group_j, node_attribute_group = min(
        potential_groups, key=potential_groups.get
    )
    node_attribute_groups[node_attribute_group].discard(group_j)
    return group_i, group_j
