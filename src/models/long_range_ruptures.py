"""
Does run a long range changepoint analysis on time series data
"""

import json
import logging
from datetime import datetime, timedelta

from slugify import slugify
import matplotlib
import matplotlib.pyplot as plt


from analysis.config import PLOTTING, LOGGER
from analysis.model_selection import get_fit_distribution
from analysis.tag_data_extractor import get_data

matplotlib.use("GTK3Agg")

logger = logging.getLogger(LOGGER)
logger.setLevel(logging.DEBUG)


def get_random_tags(n):
    return []


if __name__ == "__main__":
    #   6m => 15.5 M datapoints (1s)
    #   60 seconds "random pick" resample => 0.25 M

    # In the test: 1.5K datapoints = 1 week => 5sec runtime
    time_end = datetime(2019, 1, 18, 14, 56, 48)
    time_start = datetime(2019, 1, 18, 14, 56, 48) - timedelta(days=90)
    logging.debug(f"Times: {time_start} {time_end}")

    # Get tags, select 30 random series
    for tag in get_random_tags(30):
        df = get_data(tag, time_start=time_start, time_end=time_end)
        distribution_info = get_fit_distribution(df, tag)
        # All done!
        with open(f"analysis/models/{slugify(tag)}_parameters.json", "w") as fdesc:
            json.dump(distribution_info, fdesc)
    if PLOTTING:
        plt.close("all")
