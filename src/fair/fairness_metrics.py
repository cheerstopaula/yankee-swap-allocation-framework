import numpy as np
import scipy
import copy

from .agent import BaseAgent, LegacyStudent
from .allocation import get_bundle_from_allocation_matrix, yankee_swap
from .item import ScheduleItem, sub_schedule
from .constraint import CourseTimeConstraint, MutualExclusivityConstraint
from .optimization import StudentAllocationProgram
from .simulation import SubStudent


# Precompute agents valuations for all bundles for the binary case
def precompute_bundles_valuations(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Precompute all agents bundles and all agent valuations for said bundles.
    This is a step necessary to run all envy metrics.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        bundles (list(list[ScheduleItem])): ordered list of agnets bundles
        valuations (type[np.ndarray]): len(agents) x len(agents) matrix, element i,j is agent's i valuation of agent's j bundle under X
    """
    bundles = [
        get_bundle_from_allocation_matrix(X, items, i) for i in range(len(agents))
    ]
    valuations = np.zeros((len(agents), len(agents)))
    for i, agent in enumerate(agents):
        for j, bundle in enumerate(bundles):
            valuations[i, j] = agent.valuation(bundle)
    return bundles, valuations


# envy metrics for the binary case
def EF_violations(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations: type[np.ndarray] | None = None,
):
    """Compute envy-free violations.

    Compare every agent to all other agents, fill EF_matrix where EF_matrix[i,j]=1 if agent of index i
    envies agent of index j, 0 otherwise.

    Returns the number of EF violations, and the number of envious agents.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem
        valuations (type[np.ndarray]): Valuations of all agents for all bundles under X

    Returns:
        int: number of EF_violations
        int: number of envious agents
    """

    num_agents = len(agents)
    EF_matrix = np.zeros((num_agents, num_agents))

    if valuations is None:
        _, valuations = precompute_bundles_valuations(X, agents, items)

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if valuations[i, i] < valuations[i, j]:
                EF_matrix[i, j] = 1
            if valuations[j, j] < valuations[j, i]:
                EF_matrix[j, i] = 1
    return np.sum(EF_matrix > 0), np.sum(np.any(EF_matrix > 0, axis=1))


def EF1_violations(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    bundles: list[list[ScheduleItem]] | None = None,
    valuations: type[np.ndarray] | None = None,
):
    """Compute envy-free up to one item (EF-1) violations.

    Compare every agent to all other agents, fill EF1_matrix where EF1_matrix[i,j]=1 if agent of index i
    envies agent of index j in the EF1 sense.

    Returns the number of EF-1 violations, and the number of envious agents in the EF1 sense.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem
        bundles (list(list[ScheduleItem])): List of all agents bundles
        valuations (type[np.ndarray]): Valuations of all agents for all bundles under X

    Returns:
        int: number of EF-1 violations
        int: number of envious agents in the EF-1 sense
    """

    if valuations is None:
        bundles, valuations = precompute_bundles_valuations(X, agents, items)

    num_agents = len(agents)
    EF1_matrix = np.zeros((num_agents, num_agents))

    def there_is_item(i, j):
        for item in range(len(bundles[j])):
            new_bundle = bundles[j].copy()
            new_bundle.pop(item)
            if agents[i].valuation(new_bundle) <= valuations[i][i]:
                return True
        return False

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if valuations[i, i] < valuations[i, j]:
                if not there_is_item(i, j):
                    EF1_matrix[i, j] = 1
            if valuations[j, j] < valuations[j, i]:
                if not there_is_item(j, i):
                    EF1_matrix[j, i] = 1
    return np.sum(EF1_matrix > 0), np.sum(np.any(EF1_matrix > 0, axis=1))


def EFX_violations(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    bundles: list[list[ScheduleItem]] | None = None,
    valuations: type[np.ndarray] | None = None,
):
    """Compute envy-free up to any item (EF-X) violations.

    Compare every agent to all other agents, fill EF1_matrix where EFX_matrix[i,j]=1 if agent of index i
    envies agent of index j in the EF1 sense.

    Returns the number of EF-X violations, and the number of envious agents in the EF-X sense.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem
        bundles (list(list[ScheduleItem])): List of all agents bundles
        valuations (type[np.ndarray]): Valuations of all agents for all bundles under X

    Returns:
        int: number of EF-X violations
        int: number of envious agents in the EF-X sense
    """

    if valuations is None:
        bundles, valuations = precompute_bundles_valuations(X, agents, items)

    num_agents = len(agents)
    EFX_matrix = np.zeros((num_agents, num_agents))

    def for_every_item(i, j):
        for item in range(len(bundles[j])):
            new_bundle = bundles[j].copy()
            new_bundle.pop(item)
            if agents[i].valuation(new_bundle) > valuations[i][i]:
                return False
        return True

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if valuations[i, i] < valuations[i, j]:
                if not for_every_item(i, j):
                    EFX_matrix[i, j] = 1
            if valuations[j, j] < valuations[j, i]:
                if not for_every_item(j, i):
                    EFX_matrix[j, i] = 1
    return np.sum(EFX_matrix > 0), np.sum(np.any(EFX_matrix > 0, axis=1))


# envy metrics for the non-binary case


def _compute_best_response(
    agent_idx: int,
    bundle_vector: np.ndarray,
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations: np.ndarray,
    memo: dict,
) -> float:
    """Compute agent's best achievable value from a given bundle vector, with memoization.

    Args:
        agent_idx: index of the agent
        bundle_vector: binary array of length len(items), 1 where an item is available
        agents: full agent list
        items: full item list
        valuations: agent x item valuation matrix
        memo: shared cache keyed by (agent_idx, bundle_signature)

    Returns:
        float: best value agent_idx can extract from the bundle
    """
    key = (agent_idx, tuple(bundle_vector.astype(int)))
    if key in memo:
        return memo[key]

    c_small_ilp = valuations[agent_idx]
    schedule_copy = [copy.copy(item) for item in items]
    for item_idx, item in enumerate(schedule_copy):
        item.capacity = int(bundle_vector[item_idx] == 1)

    program = StudentAllocationProgram(
        [agents[agent_idx].student], schedule_copy
    ).compile()
    opt_alloc = program.formulateUSW(valuations=c_small_ilp).solve()
    value = c_small_ilp @ opt_alloc
    memo[key] = value
    return value


def EF_violations_responses(
    X: np.ndarray,
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations: np.ndarray,
    student_status_map: dict = None,
    memo: dict = None,
):
    """Compute envy-free violations using best-response values (non-binary case).

    For each pair (i, j), agent i's value for j's bundle is their best achievable utility
    under j's items, computed via ILP and memoized across calls.

    Args:
        X (np.ndarray): Allocation matrix (items x agents)
        agents (list[BaseAgent]): list of agents
        items (list[ScheduleItem]): list of items
        valuations (np.ndarray): agent x item valuation matrix
        student_status_map (dict, optional): maps agent -> status int for stratified counts
        memo (dict, optional): shared ILP cache; pass in to reuse across calls

    Returns:
        Without student_status_map: (total_envy, EF_matrix, memo)
        With student_status_map:    (total_envy, status_envy, downward_envy, EF_matrix, memo)
    """
    if memo is None:
        memo = {}

    num_agents = len(agents)
    potential_utilities = valuations @ X
    current_utilities = np.diag(potential_utilities)

    EF_matrix = np.zeros((num_agents, num_agents))
    np.fill_diagonal(EF_matrix, current_utilities)

    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            if current_utilities[i] >= potential_utilities[i, j]:
                EF_matrix[i, j] = potential_utilities[i, j]
            else:
                EF_matrix[i, j] = _compute_best_response(
                    i, X[:, j], agents, items, valuations, memo
                )

    envy_mask = EF_matrix > current_utilities[:, None]
    np.fill_diagonal(envy_mask, False)
    total_envy = int(np.sum(envy_mask))
    num_envious = int(np.sum(np.any(envy_mask, axis=1)))

    if student_status_map is None:
        return total_envy, num_envious, EF_matrix, memo

    statuses = np.array([student_status_map[a] for a in agents])
    si = statuses[:, None]
    sj = statuses[None, :]
    status_envy = int(np.sum(envy_mask & (si == sj)))
    downward_envy = int(np.sum(envy_mask & (si > sj)))

    return total_envy, num_envious, status_envy, downward_envy, EF_matrix, memo


def EF1_violations_responses(
    X: np.ndarray,
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations: np.ndarray,
    bundles: list[list[ScheduleItem]] | None = None,
    student_status_map: dict = None,
    memo: dict = None,
    EF_matrix: np.ndarray | None = None,
):
    """Compute EF1 violations using best-response values (non-binary case).

    Agent i EF1-envies agent j if, after removing any single item from j's bundle,
    i's best response value still exceeds i's current utility.

    Reuses the EF_matrix and memo from EF_violations_responses to avoid redundant ILP calls.

    Args:
        X (np.ndarray): Allocation matrix (items x agents)
        agents (list[BaseAgent]): list of agents
        items (list[ScheduleItem]): list of items
        valuations (np.ndarray): agent x item valuation matrix
        bundles (list[list[ScheduleItem]], optional): precomputed bundles; derived from X if None
        student_status_map (dict, optional): maps agent -> status int for stratified counts
        memo (dict, optional): shared ILP cache; populated and returned for further reuse
        EF_matrix (np.ndarray, optional): precomputed EF_matrix; skips EF_violations_responses if provided

    Returns:
        Without student_status_map: (total_ef1, num_ef1_envious, EF1_matrix, memo)
        With student_status_map:    (total_ef1, num_ef1_envious, status_ef1, downward_ef1, EF1_matrix, memo)
    """
    if memo is None:
        memo = {}

    if bundles is None:
        bundles = [
            get_bundle_from_allocation_matrix(X, items, i) for i in range(len(agents))
        ]

    if EF_matrix is None:
        ef_result = EF_violations_responses(
            X, agents, items, valuations, student_status_map, memo
        )
        EF_matrix = ef_result[-2]
        memo = ef_result[-1]

    num_agents = len(agents)
    current_utilities = np.diag(valuations @ X)
    item_to_idx = {item: idx for idx, item in enumerate(items)}

    envy_mask = EF_matrix > current_utilities[:, None]
    np.fill_diagonal(envy_mask, False)

    EF1_matrix = np.zeros((num_agents, num_agents))

    for i, j in zip(*np.where(envy_mask)):
        # Try dropping each item from j's bundle; if any removal eliminates i's envy, EF1 holds
        for item in bundles[j]:
            g_idx = item_to_idx.get(item)
            if g_idx is None:
                continue
            bundle_vector = X[:, j].copy()
            bundle_vector[g_idx] = 0
            if (
                _compute_best_response(
                    i, bundle_vector, agents, items, valuations, memo
                )
                <= current_utilities[i]
            ):
                break
        else:
            EF1_matrix[i, j] = 1

    total_ef1 = int(np.sum(EF1_matrix > 0))
    num_ef1_envious = int(np.sum(np.any(EF1_matrix > 0, axis=1)))

    if student_status_map is None:
        return total_ef1, num_ef1_envious, EF1_matrix, memo

    statuses = np.array([student_status_map[a] for a in agents])
    si = statuses[:, None]
    sj = statuses[None, :]
    ef1_mask = EF1_matrix > 0
    status_ef1 = int(np.sum(ef1_mask & (si == sj)))
    downward_ef1 = int(np.sum(ef1_mask & (si > sj)))

    return total_ef1, num_ef1_envious, status_ef1, downward_ef1, EF1_matrix, memo


# FUNCTIONS TO COMPUTE PAIRWISE MAXIMIN SHARE
# For the binary case


def yankee_swap_sub_problem(
    agent: type[BaseAgent],
    new_schedule: list[ScheduleItem],
):
    """Given an agent and information of a reduced schedule (new_schedule, course_strings, course), compute their MMS for the reduced problem,
    considering 2 identical agents competing for the items in the reduced schedule.
    We do this by computing a leximin allocation through yankee swap.

    Args:
        agent (type[BaseAgent]): Agent from the class BaseAgent
        new_schedule (list[ScheduleItem]): Items from class BaseItem, new reduced schedule
        course_strings (list[str]): List of course strings of the new schedule
        course (type[Course]): Course instance of the new schedule

    Returns:
        int: Agent's MMS for the subproblem
    """
    course, slot, weekday, section = new_schedule[0].features

    course_time_constr = CourseTimeConstraint.from_items(new_schedule, slot, weekday)
    course_sect_constr = MutualExclusivityConstraint.from_items(new_schedule, course)
    preferred = agent.preferred_courses
    new_student = SubStudent(
        agent.student.quantities,
        [
            [item for item in topic if item in new_schedule]
            for topic in agent.student.preferred_topics
        ],
        [item for item in preferred if item in new_schedule],
        agent.student.total_courses,
        course,
        section,
        [course_time_constr, course_sect_constr],
        new_schedule,
    )

    legacy_student = LegacyStudent(new_student, new_student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    sub_student = legacy_student

    X_sub = yankee_swap([sub_student, sub_student], new_schedule)

    bundle_1 = get_bundle_from_allocation_matrix(X_sub, new_schedule, 0)
    bundle_2 = get_bundle_from_allocation_matrix(X_sub, new_schedule, 1)

    return min([sub_student.valuation(bundle_1), sub_student.valuation(bundle_2)])


def pairwise_maximin_share(
    agent1: type[BaseAgent],
    agent2: type[BaseAgent],
    current_bundle_1: list[ScheduleItem],
    current_bundle_2: list[ScheduleItem],
):
    """Given two agents and their current bundles, compute their Pairwise Maximin Share (PMMS)

    Args:
        agent1 (type[BaseAgent]): First agent
        agent2 (type[BaseAgent]): Second agent
        current_bundle_1 (list[ScheduleItem]): first agent's current bundle
        current_bundle_2 (list[ScheduleItem]): second agent's current bundle

    Returns:
        PMMS[BaseAgent] (type[int]): for agents 1 and 2, return their PMMS for the subproblem
    """

    PMMS = {}

    # bundle_1 = copy.deepcopy([sched for sched in current_bundle_1])
    bundle_1 = [copy.copy(sched) for sched in current_bundle_1]
    for sched in bundle_1:
        sched.capacity = 1
    # bundle_2 = copy.deepcopy([sched for sched in current_bundle_2])
    bundle_2 = [copy.copy(sched) for sched in current_bundle_2]
    for sched in bundle_2:
        sched.capacity = 1

    new_schedule = sub_schedule([bundle_1, bundle_2])

    PMMS[agent1] = yankee_swap_sub_problem(agent1, new_schedule)

    return PMMS


def PMMS_violations(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    bundles: list[list[ScheduleItem]] | None = None,
    valuations: type[np.ndarray] | None = None,
):
    """Compute number of violations of the Pairwise Maximin Share (PMMS) for an allocation X

    Compare every agent to all agents of higher index and determine whether they receive their PMMS.
    Runs intermediate functions to compute PMMS and returns tuple with the number of comparison which did not comply with the PMMS,
    and the number of agents that for at least one comparison, did not receive their PMMS.

     Args:
         X (type[np.ndarray]): Allocation matrix
         agents (list[BaseAgent]): Agents from class BaseAgent
         schedule (list[ScheduleItem]): Items from class BaseItem
         bundles (list(list[ScheduleItem])): List of all agents bundles
         valuations (type[np.ndarray]): Valuations of all agents for all bundles under X

     Returns:
         int: Number of PMMS violations
         int: Number of agents who did not receive their PMMS in every comparison
    """
    if valuations is None:
        bundles, valuations = precompute_bundles_valuations(X, agents, items)

    PMMS_matrix = np.zeros((len(agents), len(agents)))
    for i, student_1 in enumerate(agents):
        bundle_1 = bundles[i]

        for j in range(i + 1, len(agents)):
            student_2 = agents[j]
            bundle_2 = bundles[j]

            if valuations[i, i] < valuations[i, j] - 1:
                PMMS = pairwise_maximin_share(student_1, student_2, bundle_1, bundle_2)
                PMMS_matrix[i, j] = valuations[i, i] - PMMS[student_1]

            if valuations[j, j] < valuations[j, i] - 1:
                PMMS = pairwise_maximin_share(student_2, student_1, bundle_2, bundle_1)
                PMMS_matrix[j, i] = valuations[j, j] - PMMS[student_2]

    return np.sum(PMMS_matrix < 0), np.sum(np.any(PMMS_matrix < 0, axis=1))


# for the non-banary case
def _compute_pmms(
    agent_idx: int,
    combined_vector: np.ndarray,
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations: np.ndarray,
    memo: dict,
) -> float:
    """Compute the exact PMMS for agent_idx over a combined item pool via a maximin MILP.

    Formulation:
        maximize  t
        subject to  feasibility constraints for student copy 1 (time, mutual exclusivity)
                    feasibility constraints for student copy 2
                    capacity constraints: each item goes to at most one copy
                    val @ x_1 >= t          (copy 1 utility lower bound)
                    val @ x_2 >= t          (copy 2 utility lower bound)
                    x_1, x_2 in {0,1},  t >= 0

    StudentAllocationProgram([student, student], sub_schedule) already encodes the
    first three constraint groups. We augment it with the two utility rows and the
    continuous variable t, then hand off to scipy.optimize.milp.

    Args:
        agent_idx: index of the agent
        combined_vector: binary array of length len(items), 1 for items in the joint pool
        agents: full agent list
        items: full item list
        valuations: agent x item valuation matrix
        memo: shared cache; keyed by ("pmms", agent_idx, bundle_signature)

    Returns:
        float: exact PMMS value (maximum guaranteed utility from the best partition)
    """
    key = ("pmms", agent_idx, tuple(combined_vector.astype(int)))
    if key in memo:
        return memo[key]

    val = valuations[agent_idx]

    # Sub-schedule: available items have capacity 1, all others 0
    schedule_copy = [copy.copy(item) for item in items]
    for g, item in enumerate(schedule_copy):
        item.capacity = int(combined_vector[g] == 1)

    # Compile the 2-student program — handles feasibility + capacity-1 partition constraint
    student = agents[agent_idx].student
    program = StudentAllocationProgram([student, student], schedule_copy).compile()

    n = program.A.shape[1]  # total allocation variables = 2 * extent
    extent = n // 2  # variables per student block (identical students → equal extents)

    # Augment A with a column for t (no existing constraint involves t)
    A_aug = scipy.sparse.hstack(
        [program.A, scipy.sparse.csr_matrix((program.A.shape[0], 1))]
    )

    # Two new rows:  -val @ x_k + t <= 0  (i.e. utility_k >= t) for k in {1, 2}
    # Student k's variable for item g sits at column: k*extent + item.index
    row1 = np.zeros(n + 1)
    row2 = np.zeros(n + 1)
    for g, item in enumerate(items):
        row1[item.index] = -val[g]  # student 1 block
        row2[extent + item.index] = -val[g]  # student 2 block
    row1[n] = 1.0  # t
    row2[n] = 1.0  # t

    A_full = scipy.sparse.vstack(
        [
            A_aug,
            scipy.sparse.csr_matrix(row1.reshape(1, -1)),
            scipy.sparse.csr_matrix(row2.reshape(1, -1)),
        ]
    )
    b_full = scipy.sparse.vstack(
        [
            program.b,
            scipy.sparse.csr_matrix([[0.0]]),
            scipy.sparse.csr_matrix([[0.0]]),
        ]
    )

    # Objective: minimize -t  (i.e. maximise t)
    c_obj = np.zeros(n + 1)
    c_obj[n] = -1.0

    # Bounds: binary for x vars, [0, raw_sum] for t
    ub_t = float(val @ combined_vector)
    bounds = scipy.optimize.Bounds(
        lb=np.zeros(n + 1),
        ub=np.concatenate([np.ones(n), [ub_t]]),
    )

    # Integrality: 1 (integer) for allocation variables, 0 (continuous) for t
    integrality = np.ones(n + 1)
    integrality[n] = 0

    constraint = scipy.optimize.LinearConstraint(
        A_full, ub=b_full.toarray().reshape(-1)
    )
    res = scipy.optimize.milp(
        c=c_obj, integrality=integrality, bounds=bounds, constraints=constraint
    )

    value = float(res.x[n]) if res.success else 0.0
    memo[key] = value
    return value


def PMMS_violations_responses(
    X: np.ndarray,
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations: np.ndarray,
    bundles: list[list[ScheduleItem]] | None = None,
    EF_matrix: np.ndarray | None = None,
    memo: dict = None,
):
    """Compute exact PMMS violations using best-response values (non-binary case).

    For each ordered pair (a, b), checks whether agent a receives at least their
    pairwise maximin share over the combined bundle bundle_a ∪ bundle_b.

    Two fast filters avoid the MILP for most pairs:

    1. EF skip — if EF_matrix[a,b] <= current_utilities[a], the trivial partition
       (bundle_a | bundle_b) already gives agent a their best guaranteed value,
       so no PMMS violation is possible. This is the dominant skip in practice.

    2. Raw-value upper bound — computes sum(valuations[a, S_ab]) / 2. For additive
       values this is a valid upper bound on PMMS; if it does not exceed current
       utility, no violation is possible (no ILP needed).

    For pairs that pass both filters, the exact PMMS is computed via _compute_pmms,
    a maximin MILP that finds the best 2-partition of S_ab respecting the student's
    feasibility constraints (time conflicts, mutual exclusivity).

    Args:
        X (np.ndarray): Allocation matrix (items x agents)
        agents (list[BaseAgent]): list of agents
        items (list[ScheduleItem]): list of items
        valuations (np.ndarray): agent x item valuation matrix
        bundles (list[list[ScheduleItem]], optional): precomputed bundles; derived if None
        EF_matrix (np.ndarray, optional): from EF_violations_responses; computed if None
        memo (dict, optional): shared ILP cache; populated and returned for further reuse

    Returns:
        (total_pmms_violations, num_pmms_violating_agents, memo)
    """
    if memo is None:
        memo = {}

    if bundles is None:
        bundles = [
            get_bundle_from_allocation_matrix(X, items, i) for i in range(len(agents))
        ]

    if EF_matrix is None:
        _, _, EF_matrix, memo = EF_violations_responses(
            X, agents, items, valuations, memo=memo
        )

    num_agents = len(agents)
    current_utilities = np.diag(valuations @ X)

    PMMS_matrix = np.zeros((num_agents, num_agents))

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            for a, b in ((i, j), (j, i)):
                # Level 1: no envy -> trivial partition is already best guaranteed value
                if EF_matrix[a, b] <= current_utilities[a]:
                    continue

                # Level 2: raw additive upper bound on PMMS
                combined = np.clip(X[:, a] + X[:, b], 0, 1)
                if (valuations[a] @ combined) / 2 <= current_utilities[a]:
                    continue

                # Level 3: exact maximin MILP
                pmms = _compute_pmms(a, combined, agents, items, valuations, memo)

                if current_utilities[a] < pmms:
                    PMMS_matrix[a, b] = current_utilities[a] - pmms

    total_violations = int(np.sum(PMMS_matrix < 0))
    num_violating_agents = int(np.sum(np.any(PMMS_matrix < 0, axis=1)))

    return total_violations, num_violating_agents, memo
