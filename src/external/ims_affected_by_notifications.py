import os
from datetime import timedelta
from collections import defaultdict
import pyodbc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging

from config import LOGGER
from external.connection import MSSQLConnectionFactory
from models.model_selection import get_fit_distribution
from external.tag_data_extractor import get_data
from typing import List, Dict

logger = logging.getLogger(LOGGER)
logger.setLevel(logging.DEBUG)


def get_ims_notifications(conn: pyodbc.Connection) -> List[dict]:
    sql = """
        SELECT n.CreationDate, n.RequiredEndDate, n.NotificationNumber, 
            i.ID as IMSTag_ID, i.tag, mel.*
        FROM TechnicalInfoTag as t,
             TechnicalInfoTag_IMSTag as ti,
             Notification as n,
             IMSTag i,
             ModelElement me left join aas.vModelElementLimitPivot mel 
                on me.ID = mel.ModelElement_ID
        where 
            t.ID = ti.TechnicalInfoTag_ID
            and ti.IMSTag_ID = i.ID
            and t.ID = n.TechnicalInfoTag_ID
            and i.ID = me.IMSTag_ID
            and (n.FailureImpact IN ('D','S') or NOTIFICATIONTYPE IN ('M2','M3','M4'))
        order by n.NotificationNumber desc 
    """

    cursor = conn.cursor()
    cursor.execute(sql)
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in cursor.fetchall()]
    cursor.close()
    return data


def plot_old_limits(notification):
    high_list = ["Software High", "High", "High High"]
    low_list = ["Software Low", "Low", "Low Low"]
    for l in high_list:
        if notification[l]:
            plt.axhline(y=notification[l], color=(1, 0, 0), dashes=[6, 2], label=l)
    for l in low_list:
        if notification[l]:
            plt.axhline(y=notification[l], color=(0, 1, 0), dashes=[6, 2], label=l)


if __name__ == "__main__":
    conn_str = os.getenv("MSSQL_CONN_STRING") or exit("Cannot get MSSQL con string")
    cf = MSSQLConnectionFactory()
    mssql_conn = cf.get_client(conn_str)
    ims_notifications = get_ims_notifications(mssql_conn)

    plt.rcParams["figure.figsize"] = (10.0, 8.0)

    grouped_notifications: Dict = defaultdict(list)
    for notification in ims_notifications:
        grouped_notifications[notification["NotificationNumber"]].append(notification)

    for notification_id, notifications in grouped_notifications.items():
        notifications_number = len(notifications)
        pdf = PdfPages(f"analysis/models/{notification_id}.pdf")
        tag_list: List = []
        for index, notification in enumerate(notifications):
            fig = plt.figure(figsize=(10, 6))  # inches
            tag = notification["tag"]
            if tag not in tag_list:
                tag_list.append(tag)
            else:
                logger.info("Tag already analyzed")
                continue
            # Find new IMS limits
            prior_incident_time_start = notification["CreationDate"] - timedelta(
                days=91
            )
            prior_incident_time_end = notification["CreationDate"] - timedelta(days=1)

            in_incident_time_start = notification["CreationDate"] - timedelta(days=30)

            if notification["RequiredEndDate"]:
                in_incident_time_end = notification["RequiredEndDate"] + timedelta(
                    days=7
                )
            else:
                in_incident_time_end = notification["CreationDate"] + timedelta(days=30)

            df_prior_incident_data = get_data(
                tag,
                time_start=prior_incident_time_start,
                time_end=prior_incident_time_end,
            )
            distribution_info = get_fit_distribution(df_prior_incident_data, tag)

            # Get data for IMSs in the down period
            df_during_incident_data = get_data(
                tag, time_start=in_incident_time_start, time_end=in_incident_time_end
            )
            # Plot data with new limits
            logger.debug(notification)
            df_during_incident_data["mean"].plot()
            plt.title(f"{tag} {distribution_info['distribution']}")
            for interval in distribution_info["confidence"]:
                for index, y in enumerate(interval):
                    color = (1, 0, 0) if index == 1 else (0, 1, 0)
                    plt.axhline(y=y, color=color)
            # Check if we can plot old limits
            plot_old_limits(notification)
            # Check if we can plot the break points
            if notification["CreationDate"]:
                plt.axvline(
                    x=notification["CreationDate"],
                    color=(0, 0, 0),
                    label="Creation Date",
                    dashes=[6, 2],
                )
            if notification["RequiredEndDate"]:
                plt.axvline(
                    x=notification["RequiredEndDate"],
                    color=(0, 0, 0),
                    label="Required EndDate",
                    dashes=[6, 2],
                )

            plt.legend(loc="best").get_frame().set_alpha(0.5)
            pdf.savefig()
            plt.close()
        pdf.close()
