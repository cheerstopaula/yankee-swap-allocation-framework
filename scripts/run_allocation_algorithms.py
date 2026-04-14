import os
from collections import defaultdict
import numpy as np
import pandas as pd
import time

from fair.agent import LegacyStudent
from fair.allocation import (
    yankee_swap,
    round_robin,
    serial_dictatorship,
    integer_linear_program,
)
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.envy import (
    EF_violations,
    EF1_violations,
    EFX_violations,
)
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.metrics import (
    nash_welfare,
    utilitarian_welfare,
    precompute_bundles_valuations,
    PMMS_violations,
)
from fair.simulation import RenaissanceMan

NUM_STUDENTS = 200
MAX_COURSES_PER_TOPIC = 5
LOWER_MAX_COURSES_TOTAL = 2
UPPER_MAX_COURSES_TOTAL = 5
SEED = 1

EXCEL_SCHEDULE_PATH = os.path.join(
    os.path.dirname(__file__), "../resources/fall2023schedule.csv"
)
CAPACITY_SCALE_FACTOR = 0.1

SPARSE = False
FIND_OPTIMAL = True

# load schedule as DataFrame
with open(EXCEL_SCHEDULE_PATH, "rb") as fd:
    df = pd.read_csv(fd)

# construct features from DataFrame
course = Course(df["Catalog"].astype(str).unique().tolist())

time_ranges = df["Mtg Time"].dropna().unique()
slot = Slot.from_time_ranges(time_ranges, "15T")
weekday = Weekday()

section = Section(df["Section"].dropna().unique().tolist())
features = [course, slot, weekday, section]

# construct schedule
schedule = []
topic_map = defaultdict(list)
for idx, (_, row) in enumerate(df.iterrows()):
    crs = str(row["Catalog"])
    slt = slots_for_time_range(row["Mtg Time"], slot.times)
    sec = row["Section"]
    capacity = row["CICScapacity"]
    dys = tuple([day.strip() for day in row["zc.days"].split(" ")])
    item = ScheduleItem(
        features,
        [crs, slt, dys, sec],
        index=idx,
        capacity=round(capacity * CAPACITY_SCALE_FACTOR),
    )
    schedule.append(item)
    topic_map[row["Categories"]].append(item)

topics = [topic for topic in topic_map.values()]

# global constraints
course_time_constr = CourseTimeConstraint.from_items(schedule, slot, weekday, SPARSE)
course_sect_constr = MutualExclusivityConstraint.from_items(schedule, course, SPARSE)


# randomly generate students
students = []
for i in range(NUM_STUDENTS):
    student = RenaissanceMan(
        topics,
        [min(len(topic), MAX_COURSES_PER_TOPIC) for topic in topics],
        LOWER_MAX_COURSES_TOTAL,
        UPPER_MAX_COURSES_TOTAL,
        course,
        section,
        [course_time_constr, course_sect_constr],
        schedule,
        seed=SEED * NUM_STUDENTS + i,
        sparse=SPARSE,
    )
    legacy_student = LegacyStudent(student, student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    students.append(legacy_student)


def run_allocation_compute_metrics(algorithm, compute_envy_share_metrics=False):
    start = time.time()
    X = algorithm(students, schedule)
    runtime = time.time() - start
    print("Running time: ", runtime)
    print("USW: ", utilitarian_welfare(X, students, schedule))
    print("NSW: ", nash_welfare(X, students, schedule)[1])
    print("Empty Bundles: ", nash_welfare(X, students, schedule)[0])
    # compute envy and share based metrics
    if compute_envy_share_metrics:
        bundles, valuations = precompute_bundles_valuations(X, students, schedule)
        print(
            "EF violations (count, agents): ",
            EF_violations(X, students, schedule, valuations),
        )
        print(
            "EF-1 violations (count, agents): ",
            EF1_violations(X, students, schedule, bundles, valuations),
        )
        print(
            "PMMS violations (count, agents): ",
            PMMS_violations(X, students, schedule, bundles, valuations),
        )


# run allocation algorithms and compute metrics, binary case
print("=========INTEGER LINEAR PROGRAM=========")
run_allocation_compute_metrics(integer_linear_program)

print("==========SERIAL DICTATORSHIP==========")
run_allocation_compute_metrics(serial_dictatorship)

print("==============ROUND ROBIN==============")
run_allocation_compute_metrics(round_robin)

print("==============YANKEE SWAP==============")
run_allocation_compute_metrics(yankee_swap)

