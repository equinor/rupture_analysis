import importlib
import logging
import math

from analysis.config import LOGGER

logger = logging.getLogger(LOGGER)

ALGORITHM_PELT = {
    "module": "ruptures",
    "class": "Pelt",
    # cost function using mean
    "params": {"model": "l2", "min_size": 3, "jump": 5},
    "predict_params": {"pen": 3},
}


def get_changepoints(algorithm, data):
    algo_class = getattr(
        importlib.import_module(algorithm["module"]), algorithm["class"]
    )
    algo = algo_class(**algorithm["params"]).fit(data)
    my_bkps = algo.predict(**algorithm["predict_params"])
    return my_bkps


def remove_small_splits(remain_percentage, splits):
    # Max points to remove
    length_limit = math.floor((1 - remain_percentage) * splits[-1])

    # Get split length, sort by length
    intervals = list(zip(splits, splits[1:]))
    lengths = list(map(lambda x: x[1] - x[0], intervals))
    interval_length = list(zip(intervals, lengths))
    interval_length_sorted = sorted(interval_length, key=lambda x: x[1])
    final_interval_length = []
    for interval_length in interval_length_sorted:
        if interval_length[1] <= length_limit:
            length_limit -= interval_length[1]
            logger.debug(f"Removed {interval_length}")
        else:
            final_interval_length.append(interval_length)
    return final_interval_length


def select_periods(df, periods):
    indexes = []
    for period in periods:
        indexes.extend(df.loc[period[0] : period[1]].index.to_list())

    new_signal = df.loc[indexes]
    return new_signal.sort_index()


def normalize_changepoints(break_points, split_index):
    if break_points:
        if break_points[0] != 0:
            break_points.insert(0, 0)
        if break_points[-1] < split_index:
            break_points.append(split_index)
    else:
        logger.warning("No breakpoints found, taking all distribution")
        break_points = [0, split_index]
    return break_points
