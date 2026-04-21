import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .agent import BaseAgent
from .item import ScheduleItem
from .optimization import StudentAllocationProgram

"""Initializations functions"""


def initialize_allocation_matrix(items: list[ScheduleItem], agents: list[BaseAgent]):
    """Initialize allocation matrix.

    Initially, no items are allocated, matrix X is all zeros, except for last column, which displays
    course capacities.

    Args:
        items (list[ScheduleItem]): Items from class BaseItem
        agents (list[BaseAgent]): Agents from class BaseAgent

    Returns:
        X: len(items) x (len(agents)+1) numpy array
    """
    n = len(agents)
    m = len(items)
    X = np.zeros([m, n + 1], dtype=int)
    for i in range(m):
        X[i][n] = items[i].capacity
    return X


def initialize_exchange_graph(items: list[ScheduleItem]):
    """Generate exchange graph.

    There is one node for every item and a sink node 't' representing the pile of unnasigned items.
    Initially, there are no edges between items, and an edge from every item node to node 't'.
    Disclaimer: The previous assumes that every item has capacity > 0

    Args:
        N (int): number of items

    Returns:
        nx.graph: networkx graph object
    """
    exchange_graph = nx.DiGraph()
    exchange_graph.add_node("t")
    for i, item in enumerate(items):
        exchange_graph.add_node(i)
        if item.capacity > 0:
            exchange_graph.add_edge(i, "t", weight=1)
    return exchange_graph


def get_bundle_from_allocation_matrix(
    X: type[np.ndarray], items: list[ScheduleItem], agent_index: int
):
    """Get list of agent's current bundle

    Get list of all items currently owned by a certain agent (bundle), given the current allocation

    Args:
        X (type[np.ndarray]): Allocation matrix
        items (list[ScheduleItem]): List of items from class BaseItem
        agent_index (int): index of the agent for which we want to get the current bundle
    Returns:
        list[ScheduleItem]: List of items from the BaseItem class currently owned by the agent
    """
    bundle0 = []
    items_list = X[:, agent_index]
    for i in range(len(items_list)):
        if int(items_list[i]) == 1:
            bundle0.append(items[i])
    return bundle0


def get_bundle_indexes_from_allocation_matrix(X: type[np.ndarray], agent_index: int):
    """Get list of agent's current bundle's indices.

    Get list of indices of all items currently owned by a certain agent (bundle), given the current allocation

    Args:
        X (type[np.ndarray]): Allocation matrix
        agent_index (int): index of the agent for which we want to get the current bundle
    Returns:
        list[int]: List of indices of the items from the BaseItem class currently owned by the agent
    """
    bundle_indexes = []
    items_list = X[:, agent_index]
    for i in range(len(items_list)):
        if int(items_list[i]) == 1:
            bundle_indexes.append(i)
    return bundle_indexes


def get_gain_function(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    agent_picked: int,
    criteria: str,
    weights: list[float],
):
    """
    Get agent's current gain function value.

    Get gain function for a certain agent from the current allocation, according to the gain function criteria and agent assigned weight.
    This function is used to update the gain function for the agent that just played, to keep track of priority agents.

    Args:
        X (type[np.ndarray]): allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        items (list[ScheduleItem]): Items from class BaseItem
        agent_picked (int): index of the agent that just played
        criteria (str): general yankee swap criteria
        weights (list[float]): list of weights assigned to the agents, if any

    Returns:
        float: updated gain fucntion for the agent that just played
    """
    agent = agents[agent_picked]
    bundle = get_bundle_from_allocation_matrix(X, items, agent_picked)
    val = agent.valuation(bundle)
    if criteria == "LorenzDominance":
        return -val
    w_i = weights[agent_picked]
    if criteria == "WeightedLeximin":
        return -val / w_i
    if criteria == "WeightedNash":
        if val == 0:
            return float("inf")
        else:
            return (1 + 1 / val) ** w_i
    if criteria == "WeightedHarmonic":
        return w_i / (val + 1)


"""Update allocation after finding the shortest path in exchange graph"""


def update_allocation(
    X: type[np.ndarray],
    exchange_graph: type[nx.Graph],
    edge_matrix: list[list],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    path_og: list[int],
    agent_picked: int,
):
    """Udate allocation matrix, edge matrix, and exchange graph.

    Execute the transfer path found, updating the allocation of items and edge matrix accordingly.
    Edge matrix is a list of lists containing the indices of agents responsible for each edge on the exchange graph
    This function is for the edge_matrix version of yankee swap

    Args:
        X (type[np.ndarray]): allocation matrix
        exchange_graph (type[nx.Graph]): exchange graph
        edge_matrix (list[list]): edge matrix
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        path_og (list[int]): shortest path, list of items indices
        agent_picked (int): index of the agent currently playing

    Returns:
        X (type[np.ndarray]): updated allocation matrix
        exchange_graph (type[nx.Graph]): updated exchange graph
        edge_matrix (list[list]): updated edge matrix
        agents_involved (list[int]): indices of the agents involved in the transfer path
    """
    path = path_og.copy()
    path = path[1:-1]
    last_item = path[-1]
    agents_involved = [agent_picked]
    X[last_item, len(agents)] -= 1
    while len(path) > 0:
        last_item = path.pop(len(path) - 1)
        if len(path) > 0:
            next_to_last_item = path[-1]
            current_agent = edge_matrix[next_to_last_item][last_item][0]
            agents_involved.append(current_agent)
            X[last_item, current_agent] = 1
            X[next_to_last_item, current_agent] = 0
            for item_index in range(len(items)):
                if current_agent in edge_matrix[next_to_last_item][item_index]:
                    edge_matrix[next_to_last_item][item_index].remove(current_agent)
                    if len(
                        edge_matrix[next_to_last_item][item_index]
                    ) == 0 and exchange_graph.has_edge(next_to_last_item, item_index):
                        exchange_graph.remove_edge(next_to_last_item, item_index)
        else:
            X[last_item, agent_picked] = 1
    return X, exchange_graph, edge_matrix, agents_involved


"""Graph functions for the exchange graph"""


def find_shortest_path(G: type[nx.Graph], start: str, end: str, weight=None):
    """Find shortest path on exchange graph.

    Find and return shortest path from start to end nodes on graph G. Return False if there is no path

    Args:
        G (type[nx.Graph]): exchange graph
        start (str): start node
        end (str): target node

    Returns:
        list[int]: list of nodes (item indices) on the shortest path
        of False: if there is no such path
    """
    try:
        if weight is None:
            p = nx.shortest_path(G, source=start, target=end)
        else:
            p = nx.shortest_path(G, source=start, target=end, weight=weight)
        return p
    except:
        return False


def add_agent_to_exchange_graph(
    X: type[np.ndarray],
    exchange_graph: type[nx.Graph],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    agent_picked: int,
    valuations,
    desired_item_indexes,
):
    """Add picked agent to the exchange graph.

    Create node representing the agent currently playing, add edges from the node to items that would increase their utility

    Args:
        X (type[np.ndarray]): allocation matrix
        exchange_graph (type[nx.Graph]): exchange graph
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        agent_picked (int): index of the agent currently playing

    Returns:
        exchange_graph (type[nx.Graph]): Updated exchange graph
    """
    exchange_graph.add_node("s")
    bundle = get_bundle_from_allocation_matrix(X, items, agent_picked)
    agent = agents[agent_picked]
    if desired_item_indexes is None:
        # this is just for backward compatability - its cached so this dont happen
        agent_desired_items = agent.get_desired_items_indexes(items)
    else:
        agent_desired_items = desired_item_indexes[agent_picked]
    for i in agent_desired_items:
        g = items[i]
        if g not in bundle and agent.marginal_contribution(bundle, g) == 1:
            if valuations is None:
                exchange_graph.add_edge("s", i, weight=1)
            else:
                exchange_graph.add_edge(
                    "s", i, weight=(8 - valuations[agent_picked, i]) * len(items)
                )
    return exchange_graph


def update_exchange_graph(
    X: type[np.ndarray],
    exchange_graph: type[nx.Graph],
    edge_matrix: list[list],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    path_og: list[int],
    agents_involved: list[int],
    valuations,
    desired_item_indexes,
):
    """Update the exchange graph and edge matrix after the transfers made.

    Given the updated allocation, path found and list of involved agents in the transfer path, update the exchange graph and edge matrix.
    This function is for the edge_matrix version of yankee swap (general_yankee_swap_E)

    Args:
        X (type[np.ndarray]): allocation matrix
        exchange_graph (type[nx.Graph]): exchange graph
        edge_matrix (list[list]): edge matrix
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        path_og (list[int]): shortest path, list of items indices
        agents_involved (list[int]): list of the indices of the agents invovled in the transfer path

    Returns:
        exchange_graph (type[nx.Graph]): updated exchange graph
        edge_matrix (list[list]): updated edge matrix
    """
    path = path_og.copy()
    path = path[1:-1]
    last_item = path[-1]
    if X[last_item, len(agents)] == 0:
        exchange_graph.remove_edge(last_item, "t")
    for agent_index in agents_involved:
        agent = agents[agent_index]
        agent_bundle = get_bundle_indexes_from_allocation_matrix(X, agent_index)
        agent_bundle_items = get_bundle_from_allocation_matrix(X, items, agent_index)
        if desired_item_indexes is None:
            # this is just for backward compatability - its cached so this dont happen
            agent_desired_items = agent.get_desired_items_indexes(items)
        else:
            agent_desired_items = desired_item_indexes[agent_index]
        for item1_idx in agent_bundle:
            item1 = items[item1_idx]
            for item2_idx in agent_desired_items:
                item2 = items[item2_idx]
                if item1_idx != item2_idx:
                    if agent_index in edge_matrix[item1_idx][item2_idx]:
                        if not agent.exchange_contribution(
                            agent_bundle_items, item1, item2
                        ):
                            edge_matrix[item1_idx][item2_idx].remove(agent_index)
                            if len(
                                edge_matrix[item1_idx][item2_idx]
                            ) == 0 and exchange_graph.has_edge(item1_idx, item2_idx):
                                exchange_graph.remove_edge(item1_idx, item2_idx)
                    else:
                        if agent.exchange_contribution(
                            agent_bundle_items, item1, item2
                        ):
                            if valuations is None or (
                                valuations[agent_index, item1_idx]
                                == valuations[agent_index, item2_idx]
                            ):
                                edge_matrix[item1_idx][item2_idx].append(agent_index)
                                exchange_graph.add_edge(item1_idx, item2_idx, weight=1)
    return exchange_graph, edge_matrix


"""Allocation algorithms"""


def integer_linear_program(
    agents: list[BaseAgent], items: list[ScheduleItem], valuations=None
):
    if valuations is not None:
        valuations = valuations.flatten()
    orig_students = [student.student for student in agents]
    program = StudentAllocationProgram(orig_students, items).compile()
    opt_alloc = program.formulateUSW(valuations=valuations).solve()
    return opt_alloc.reshape(len(agents), len(items)).transpose()


def serial_dictatorship(
    agents: list[BaseAgent], items: list[ScheduleItem], valuations=None
):
    """SPIRE allocation algorithm.

    In each round, give the playing agent all items they can add to their bundle that give them positive utility

    Args:
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem

    Returns:
         X (type[np.ndarray]): allocation matrix
    """
    if valuations is None:
        X = initialize_allocation_matrix(items, agents)
        agent_index = 0
        for agent_index, agent in enumerate(agents):
            bundle = []
            desired_items = agent.get_desired_items_indexes(items)
            for item in desired_items:
                if X[item, len(agents)] > 0:
                    current_val = agent.valuation(bundle)
                    new_bundle = bundle.copy()
                    new_bundle.append(items[item])
                    new_valuation = agent.valuation(new_bundle)
                    if new_valuation > current_val:
                        X[item, agent_index] = 1
                        X[item, len(agents)] -= 1
                        bundle = new_bundle.copy()
        return X
    else:
        # Initialize allocation
        n = len(agents)
        m = len(items)
        X = np.zeros([m, n], dtype=int)
        # Make deep copy of the schedule to alter capacities
        # schedule_copy = copy.deepcopy(items)
        schedule_copy = [copy.copy(item) for item in items]

        for student_idx in range(len(agents)):

            small_ilp_students = agents[student_idx : student_idx + 1]

            c_small_ilp = np.array([valuations[student_idx]])

            orig_students = [student.student for student in small_ilp_students]

            program = StudentAllocationProgram(orig_students, schedule_copy).compile()
            opt_alloc = program.formulateUSW(valuations=c_small_ilp.flatten()).solve()

            X[:, student_idx] = opt_alloc

            for i, take in enumerate(opt_alloc):
                if take == 1:
                    schedule_copy[i].capacity -= 1
        return X


def round_robin(agents: list[BaseAgent], items: list[ScheduleItem], valuations=None):
    """Round Robin allocation algorithm.

    In each round, give the playing agent one item they can add to their bundle that give them positive utility, if any

    Args:
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem

    Returns:
         X (type[np.ndarray]): allocation matrix
    """
    n = len(agents)
    players = list(range(n))
    X = initialize_allocation_matrix(items, agents)
    gain_vector = np.zeros([n])
    while len(players) > 0:
        player = np.argmax(gain_vector)
        val = 0
        current_item = []
        agent = agents[player]
        desired_items = agent.get_desired_items_indexes(items)
        bundle = get_bundle_from_allocation_matrix(X, items, player)
        for item in desired_items:
            if X[item, len(agents)] > 0:
                if valuations is None:
                    current_val = agent.marginal_contribution(bundle, items[item])
                else:
                    current_val = (
                        agent.marginal_contribution(bundle, items[item])
                        * valuations[player, item]
                    )
                if current_val > val:
                    current_item.clear()
                    current_item.append(item)
                    val = current_val
        if len(current_item) > 0:
            X[current_item[0], player] = 1
            X[current_item[0], len(agents)] -= 1
            gain_vector[player] = get_gain_function(
                X, agents, items, player, criteria="LorenzDominance", weights=[]
            )
        else:
            players.remove(player)
            gain_vector[player] = float("-inf")
    return X


def yankee_swap(
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    criteria: str = "LorenzDominance",
    weights: list = [],
    plot_exchange_graph: bool = False,
    valuations=None,
):
    """General Yankee swap allocation algorithm, edge matrix version.
    ("smart" bookkeeping to speed things up)

    Args:
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        criteria (str, optional): gain function criteria. Defaults to "LorenzDominance". See get_gain_function to see other alternatives
        weights (list[float]): list of agents assigned weights
        plot_exchange_graph (bool, optional): Defaults to False. Change to True to display exchange graph plot after every modification to it.
        valuations : algorithm runs considering binary valuations by default, but it is allowed to receive non-binary matrix with valuations.

    Returns:
        X (type[np.ndarray]): allocation matrix
    """

    n = len(agents)
    m = len(items)
    players = list(range(n))
    X = initialize_allocation_matrix(items, agents)
    exchange_graph = initialize_exchange_graph(items)
    edge_matrix = [[[] for i in range(m)] for j in range(m)]
    gain_vector = np.zeros([n])
    desired_item_indexes = [agent.get_desired_items_indexes(items) for agent in agents]
    count = 0
    while len(players) > 0:
        # print("Iteration: %d" % count, end="\r")
        count += 1
        agent_picked = np.argmax(gain_vector)
        exchange_graph = add_agent_to_exchange_graph(
            X,
            exchange_graph,
            agents,
            items,
            agent_picked,
            valuations,
            desired_item_indexes,
        )
        if plot_exchange_graph:
            nx.draw(exchange_graph, with_labels=True)
            plt.show()

        path = find_shortest_path(
            exchange_graph, "s", "t", weight=None if valuations is None else "weight"
        )
        exchange_graph.remove_node("s")

        if path == False:
            players.remove(agent_picked)
            gain_vector[agent_picked] = float("-inf")
        else:
            X, exchange_graph, edge_matrix, agents_involved = update_allocation(
                X, exchange_graph, edge_matrix, agents, items, path, agent_picked
            )
            exchange_graph, edge_matrix = update_exchange_graph(
                X,
                exchange_graph,
                edge_matrix,
                agents,
                items,
                path,
                agents_involved,
                valuations,
                desired_item_indexes,
            )
            gain_vector[agent_picked] = get_gain_function(
                X, agents, items, agent_picked, criteria, weights
            )
            if plot_exchange_graph:
                nx.draw(exchange_graph, with_labels=True)
                plt.show()
    return X
