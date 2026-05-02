from typing import List

from fair.constraint import PreferenceConstraint
from fair.feature import Course

from .item import BaseItem, ScheduleItem
from .valuation import RankValuation, UniqueItemsValuation


def exchange_contribution(
    valuation: RankValuation,
    bundle: set,
    og_item: BaseItem,
    new_item: BaseItem,
):
    """Check for improvement in utility

    Determine whether the agent can exchange original_item for new_item and keep the same utility


    Args:
        valuation (BaseValuation): Valuation object to be used for comparison
        bundle (List[BaseItem]): Original set of items
        og_item (BaseItem): Item to be removed
        new_item (BaseItem): Item to be added

    Returns:
        bool: True if utility can be improved; False otherwise
    """

    if og_item == new_item:
        return False

    if og_item not in bundle or new_item in bundle:
        return False

    base_val = valuation.value(list(bundle))

    bundle_list = list(bundle)
    bundle_list.remove(og_item)
    bundle_list.append(new_item)

    return valuation.value(bundle_list) == base_val

def marginal_contribution(
    valuation: RankValuation, bundle: set, item: BaseItem
):
    """Marginal change in utility

    Compute the marginal utility the agent gets form adding a particular item to a particular bundle of items


    Args:
        valuation (BaseValuation): Valuation object to be used for computing utility
        bundle (List[BaseItem]): Initial set of items
        item (BaseItem): Item to be added

    Returns:
        Any: Change in value
    """
    if item in bundle:
        return 0

    base_value = valuation.value(list(bundle))
    new_bundle = list(bundle)
    new_bundle.append(item)

    return valuation.value(new_bundle) - base_value


class BaseAgent:
    """A wrapper class for apply a valuation to bundles of items"""

    def __init__(self, valuation: RankValuation):
        """
        Args:
            valuation (BaseValuation): Valuation object to apply to bundles
        """
        self.valuation = valuation

    def value(self, bundle: List[BaseItem]):
        """Apply valuation to bundle

        Args:
            bundle (List[BaseItem]): Items to evaluate

        Returns:
            Any: Value of bundle
        """
        return self.valuation.value(bundle)


class Student(BaseAgent):
    """A student agent"""

    def __init__(self, valuation: RankValuation):
        super().__init__(valuation)


class LegacyStudent:
    """A student compatible with https://github.com/cheerstopaula/Allocation"""

    def __init__(
        self,
        student: BaseAgent,
        preferred_courses: list[ScheduleItem],
        course: Course,
    ):
        """
        Args:
            student (BaseAgent): Student to delegate value queries to
            preferred_courses (list[str]): All courses preferred by student
            course (Course): Course feature associated with preferred_courses

        Raises:
            AttributeError: student must include valuation as member
        """
        if not hasattr(student, "valuation"):
            raise AttributeError("student delegate must have valuation member")

        student.valuation = UniqueItemsValuation(student.valuation)
        self.student = student
        self.preferred_courses = preferred_courses
        self.course = course

    def valuation(self, bundle: List[BaseItem]):
        """Delegate to value function
        Args:
            bundle (List[BaseItem]): Items to evaluate
        """
        return self.student.value(bundle)

    def marginal_contribution(self, bundle: List[BaseItem], item: BaseItem):
        """Delegate to marginal_contribution function

        Args:
            bundle (List[BaseItem]): Initial set of items
            item (BaseItem): Item to be added
        """
        return marginal_contribution(self.student.valuation, bundle, item)

    def exchange_contribution(
        self, bundle: List[BaseItem], og_item: BaseItem, new_item: BaseItem
    ):
        """Delegate to exchange_contribution function

        Args:
            bundle (List[BaseItem]): Initial set of items
            og_item (BaseItem): Item to be removed
            new_item (BaseItem): Item to be added
        """
        return exchange_contribution(self.student.valuation, bundle, og_item, new_item)

    def get_desired_items_indexes(self, items: List[BaseItem]):
        """Return subset of indices from items that are preferred by the student

        Args:
            items (List[BaseItem]): Candidate items list

        Returns:
            List[int]: Indices of desired items in list
        """
        return [item.index for item in items if item in self.preferred_courses]
