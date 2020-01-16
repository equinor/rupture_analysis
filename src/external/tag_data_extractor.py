import logging
import matplotlib.pyplot as plt
from datetime import datetime
from config import PLOTTING, LOGGER

from slugify import slugify

logger = logging.getLogger(LOGGER)


def query_resampled(
    conn,
    measurement,
    tag_id: str,
    time_start: datetime,
    time_end: datetime,
    grouping_minutes: int = 4,
):
    INFLUX_STRFTIME = "%Y-%m-%dT%H:%M:%SZ"
    grouping_str = f"{grouping_minutes}m"
    str_query = """
            SELECT MEAN("value")
            FROM "data"."{}" 
            WHERE ("tag" = '{}') 
                AND (time >= '{}') AND (time <= '{}')  
            group by time({}) fill(null)""".format(
        measurement,
        tag_id,
        time_start.strftime(INFLUX_STRFTIME),
        time_end.strftime(INFLUX_STRFTIME),
        grouping_str,
    )
    result = conn.query(str_query)
    return result


class TSConnectionFactory(object):
    pass


def get_data(tag, time_start, time_end):
    conn_factory = TSConnectionFactory()
    conn_dataframe = conn_factory.get_client_dataframe()
    grouping_minutes = 40

    logger.debug(f"Processing tag: {tag}")
    res = query_resampled(
        conn_dataframe, "plant", tag, time_start, time_end, grouping_minutes
    )
    try:
        df = res["plant"]
    except KeyError:
        res = query_resampled(
            conn_dataframe, "ioc_calc", tag, time_start, time_end, grouping_minutes
        )
        if not res:
            logger.error(f"Cannot get data for: {tag}")

        df = res["ioc_calc"]
    df.dropna(inplace=True)
    logger.debug(df.describe())
    if PLOTTING:
        # Save plot of the data
        plt.subplots()
        df["mean"].plot()
        plt.title(f"{tag} Real values")
        plt.savefig(f"analysis/images/{slugify(tag)}_raw.png", format="png")

        # Get freq distribution
        plt.subplots()
        df["mean"].reset_index(drop=True).hist(bins=50)
        plt.title(f"{tag} Frequency distribution")
        plt.savefig(f"analysis/images/{slugify(tag)}_hist.png", format="png")
        plt.show()

    return df
