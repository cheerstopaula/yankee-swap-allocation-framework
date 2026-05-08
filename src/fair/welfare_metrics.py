import copy
import numpy as np
from .agent import BaseAgent
from .allocation import get_bundle_from_allocation_matrix
from .item import ScheduleItem


def utilitarian_welfare(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations=None,
):
    """Compute utilitarian social welfare (USW)

    Calculates the average of utilities across all agents.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        float: USW / len(agents)
    """
    if valuations is None:
        util = 0
        for agent_index, agent in enumerate(agents):
            bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
            val = agent.valuation(bundle)
            util += val
        return util / (len(agents))
    else:
        current_utilities = np.diag(np.dot(valuations, X))
        return sum(current_utilities) / len(agents)


def nash_welfare(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations=None,
):
    """Compute Nash social welfare (NSW)

    Calculates the number of agents with 0 utility and product of utilities across all
    agents with utility > 0.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        int: number of agents with utility 0
        float: n-root of NSW
    """
    util = 0
    num_zeros = 0
    if valuations is not None:
        current_utilities = np.diag(np.dot(valuations, X))
    for agent_index, agent in enumerate(agents):
        if valuations is None:
            bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
            val = agent.valuation(bundle)
        else:
            val = current_utilities[agent_index]
        if val == 0:
            num_zeros += 1
        else:
            util += np.log(val)
    return num_zeros, np.exp(util / (len(agents) - num_zeros))


def first_preference_count(
    X: np.ndarray,
    valuations: np.ndarray,
) -> int:
    """
    Returns number of agents who received at least one
    item with maximal valuation.
    """

    num_agents = valuations.shape[0]

    # Highest valuation per agent
    max_vals = valuations.max(axis=1)  # shape (num_agents,)

    count = 0

    for i in range(num_agents):
        # Items assigned to agent i
        assigned_items = X[:, i] == 1

        if not np.any(assigned_items):
            continue

        # Values of assigned items
        assigned_values = valuations[i, assigned_items]

        # Did they receive a max-valued item?
        if np.any(assigned_values == max_vals[i]):
            count += 1

    return count


def leximin(X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]):
    """Compute Leximin vector, i.e. vector with agents utilities, sorted in decreasing order

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        list[int]: utilities for all agents
    """
    valuations = []
    for agent_index, agent in enumerate(agents):
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        valuations.append(val)
    valuations.sort()
    valuations.reverse()
    return valuations
