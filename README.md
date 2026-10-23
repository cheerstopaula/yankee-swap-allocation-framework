# Fair Allocation

`fair` is a Python framework for allocating indivisible items to agents with constraints, developed for course allocation: students (agents) receive seats in courses (items) subject to schedule conflicts and enrollment capacities.

It includes:

* **Agents, items, and constraints** (`fair.agent`, `fair.item`, `fair.constraint`, `fair.valuation`): students whose valuations are defined by linear constraints (schedule conflicts, enrollment capacities) over course ratings.
* **Allocation mechanisms** (`fair.allocation`):
    * `yankee_swap`: Yankee Swap, generalized beyond binary valuations to students' course ratings, operating directly on courses with multiple seats;
    * `round_robin`: Round Robin (the draft mechanism);
    * `serial_dictatorship`: Serial Dictatorship, where students choose in priority order;
    * `integer_linear_program`: an integer linear program that maximizes utilitarian welfare.

  All four take a list of agents, a list of items, and optionally a valuation matrix (`valuations=`), and return an allocation matrix.
* **Justice criteria**, which evaluate the quality of an allocation in terms of welfare and fairness:
    * welfare metrics (`fair.welfare_metrics`): utilitarian welfare, Nash welfare;
    * fairness metrics (`fair.fairness_metrics`): envy-freeness (EF), EF-1, EFX, and pairwise maximin share (PMMS) violations.
* **Statistical tools** (`fair.stats`): the multivariate Beta models used to generate synthetic students from survey responses.

The software easily extends to additional allocation mechanisms and justice criteria: all mechanisms share common inputs (agents, items, and optional valuations) and an allocation matrix out as output, and all justice criteria evaluate an allocation matrix.

## Preferences

Students' feasible bundles are defined by linear constraints: schedule conflicts (`CourseTimeConstraint`), at most one section of each course (`MutualExclusivityConstraint`), the courses they are interested in (`PreferenceConstraint`), and their maximum course load. Within these constraints, the allocation mechanisms and the metrics support two kinds of preferences:

* **Binary (default):** every course a student is interested in, and can add to their bundle without violating a constraint, is worth 1. This is the setting of the original Yankee Swap algorithm, in which its theoretical guarantees hold.
* **Non-binary:** pass a valuation matrix (`valuations=`, one row per student and one column per course) to value each course by the student's rating. Utilities are additive over the courses in a feasible bundle (i.e., additive preferences under conflict and capacity constraints). The Yankee Swap implementation assumes ratings from 0 to 7; other scales require adjusting (particularly, the constant on edge weights in function `add_agent_to_exchange_graph` in `fair/allocation.py`).

## Simulated data

For testing and experimentation without real data, the framework can generate random students: `fair.simulation` creates students with randomly drawn preferences and course loads over a course schedule. `resources/fall2023schedule.csv` provides an example schedule, and `scripts/run_allocation_algorithms.py` runs all four mechanisms on randomly generated students over that schedule.

To see the framework applied to real student preferences, visit the [course-allocation-data](https://github.com/cheerstopaula/course-allocation-data) repository, which contains course-preference data collected from Computer Science students at the University of Massachusetts Amherst for the Fall 2024 semester, and the experiments that use this framework.

## Installation

```bash
pip install -e .
```

## Development

### VSCode

Format and docstring settings are committed to the repository in `.vscode/settings.json`. The following extensions should also be installed.

* Pylance
* autoDocstring - Python
* Black Formatter
* isort

## Testing

```bash
pytest
```

## Citing

If you use this software, please cite the paper that introduces the dataset and the software:

> George Bissias, Cyrus Cousins, Paula Navarrete Diaz, and Yair Zick. *Deploying Fair and Efficient Course Allocation Mechanisms*. arXiv:2502.10592, 2025.

```bibtex
@misc{bissias2025deploying,
  title         = {Deploying Fair and Efficient Course Allocation Mechanisms},
  author        = {Bissias, George and Cousins, Cyrus and Navarrete Diaz, Paula and Zick, Yair},
  year          = {2025},
  eprint        = {2502.10592},
  archivePrefix = {arXiv},
  primaryClass  = {cs.GT},
  url           = {https://arxiv.org/abs/2502.10592}
}
```

This version of the paper predates the extension to non-binary preferences; we will update this reference when the new version is available.
