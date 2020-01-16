import logging

from scipy.stats import skewnorm, fisk, norm
from slugify import slugify
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn import preprocessing
import numpy as np
import ruptures as rpt
import time

from changepoint.changepoint import (
    get_changepoints,
    ALGORITHM_PELT,
    normalize_changepoints,
    remove_small_splits,
    select_periods,
)
from config import PLOTTING, CONFIDENCE, LOGGER

logger = logging.getLogger(LOGGER)


def fit_distribution(signal, selected_periods_timestamps, tag):
    # Fit multigaussian
    try:
        bayes_multigaussian = is_distribution_bayesian_multigaussian(signal)
    except Exception as e:
        logging.exception(e)
        bayes_multigaussian = False

    if bayes_multigaussian:
        # Try to differenciate a deprecated working mode from a bimodal
        deprecated_periods = has_deprecated_working_mode(
            signal, selected_periods_timestamps
        )
        if deprecated_periods:  # Historic data has deprecated working mode
            # Remove old data
            filtered_signal = remove_deprecated_working_mode(
                signal, selected_periods_timestamps
            )
            return fit_skewnormal(filtered_signal, tag)
        else:
            # multigaussian fit
            return bayes_multigaussian
    else:
        return fit_skewnormal(signal, tag)


def remove_deprecated_working_mode(signal, selected_periods_timestamps):
    return signal


def fit_skewnormal(signal, tag):
    mu, loc, std = skewnorm.fit(signal["mean"].values)
    confidence_interval = skewnorm.interval(CONFIDENCE, mu, loc=loc, scale=std)
    if PLOTTING:
        # Plot the histogram.
        plt.subplots()
        plt.hist(signal["mean"].values, bins=25, density=True, alpha=0.6, color="g")

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = skewnorm.pdf(x, mu, loc, std)
        plt.plot(x, p, "k", linewidth=1)
        title = "Fit results skewnormal: a=%2f loc = %.2f,  std = %.2f" % (mu, loc, std)
        plt.title(title)
        plt.axvline(x=confidence_interval[0])
        plt.axvline(x=confidence_interval[1])

        # Plot the confidence interval
        plt.savefig(
            f"analysis/images/{slugify(tag)}_fit_skew_histogram.png", format="png"
        )

    return {
        "distribution": "skewnormal",
        "params": [{"a": mu, "loc": loc, "std": std}],
        "confidence": [confidence_interval],
    }


def is_distribution_bayesian_multigaussian(data):
    model = BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.3,
        n_components=3,
        reg_covar=0,
        init_params="random",
        max_iter=1000,
        mean_precision_prior=0.8,
        random_state=2,
    )
    model.fit(data)
    logger.info(f"Weights: {model.weights_}")
    sorted_weights = sorted(model.weights_, reverse=True)
    logger.info(f"Means: {model.means_}")
    volume_top_two = sorted_weights[0] + sorted_weights[1]

    std_dev_check = model.covariances_.flatten() / model.means_.flatten()
    logger.info(f"Std Dev check: {std_dev_check}")

    if volume_top_two > 0.9 and (sorted_weights[0] / volume_top_two) < 0.9:
        biggest_element_index = list(model.weights_).index(sorted_weights[0])
        second_element_index = list(model.weights_).index(sorted_weights[1])
        mean_ratio = (
            model.means_[biggest_element_index] / model.means_[second_element_index]
        )

        if (
            (mean_ratio > 1.1 or mean_ratio < 0.9)
            and std_dev_check[biggest_element_index] < 0.2
            and std_dev_check[second_element_index] < 0.2
        ):  # Mean ratio different enough...
            logging.info(f"Clasiffied as multigaussian")
            result = {"distribution": "multigaussian", "params": [], "confidence": []}
            for index in [biggest_element_index, second_element_index]:
                variance = model.covariances_[index][0][0]
                mean = model.means_[index][0]
                confidence_interval = norm.interval(
                    CONFIDENCE, loc=mean, scale=variance
                )
                result["params"].append({"mu": mean, "std": variance})
                result["confidence"].append(confidence_interval)
            return result


def has_deprecated_working_mode(signal, selected_periods_timestamps):

    return []


def fit_normal(signal, tag):
    mu, std = norm.fit(signal["mean"].values)
    confidence_interval = norm.interval(CONFIDENCE, loc=mu, scale=std)
    if PLOTTING:
        # Plot the histogram.
        plt.subplots()
        plt.hist(signal["mean"].values, bins=25, density=True, alpha=0.6, color="g")
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, "k", linewidth=2)
        title = "Fit results normal: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)
        plt.axvline(x=confidence_interval[0])
        plt.axvline(x=confidence_interval[1])

        # Plot the confidence interval
        plt.savefig(f"analysis/images/{slugify(tag)}_fit_histogram.png", format="png")

    return {
        "distribution": "normal",
        "params": [{"mu": mu, "std": std}],
        "confidence": [confidence_interval],
    }


def fit_fisk(signal, tag):
    mu, loc, std = fisk.fit(signal["mean"].values)
    confidence_interval = fisk.interval(CONFIDENCE, mu, loc=loc, scale=std)
    if PLOTTING:
        # Plot the histogram.
        plt.subplots()
        plt.hist(signal["mean"].values, bins=25, density=True, alpha=0.6, color="g")
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = fisk.pdf(x, mu, loc, std)
        plt.plot(x, p, "k", linewidth=1)
        title = "Fit results fisk: a=%2f loc = %.2f,  std = %.2f" % (mu, loc, std)
        plt.title(title)
        plt.axvline(x=confidence_interval[0])
        plt.axvline(x=confidence_interval[1])

        # Plot the confidence interval
        plt.savefig(
            f"analysis/images/{slugify(tag)}_fit_fisk_histogram.png", format="png"
        )

    return {
        "distribution": "fisk",
        "params": [{"a": mu, "loc": loc, "std": std}],
        "confidence": [confidence_interval],
    }


def get_fit_distribution(df, tag):
    ruptures_split = 2
    # Max Min transform? Robust transform?
    data = preprocessing.MinMaxScaler().fit_transform(
        np.array(df["mean"].values).reshape(-1, 1)
    )
    if PLOTTING:
        fig, ax = plt.subplots()
        ax.plot(data)
        plt.title(f"{tag} MinMax transform")
        plt.savefig(f"analysis/images/{slugify(tag)}_min_max.png", format="png")

    # Find break points
    splits = np.array_split(data, ruptures_split)
    break_points = []
    t_start = time.time()
    split_index = 0
    for index, split in enumerate(splits):
        new_break_points = get_changepoints(ALGORITHM_PELT, split)
        if len(new_break_points) > 1:
            new_break_points = (np.array(new_break_points) + split_index).tolist()
            break_points.extend(new_break_points)
        split_index += len(split)

    break_points = normalize_changepoints(break_points, split_index)

    logger.debug(f"It has taken: {time.time() - t_start}")
    logger.info(f"Breakpoints: {break_points}")
    if PLOTTING:
        # Save plot the split
        fig, (ax,) = rpt.show.display(data, break_points, [], figsize=(10, 3))
        plt.savefig(f"analysis/images/{slugify(tag)}_break_points.png", format="png")

    selected_periods = remove_small_splits(0.9, break_points)
    logger.debug(f"Remaining periods: {selected_periods}")

    # Converting dots to timestamps
    selected_periods_timestamps = list(
        map(
            lambda x: (
                df.sort_index().index[x[0][0]],
                df.sort_index().index[x[0][1] - 1],
            ),
            selected_periods,
        )
    )
    logger.debug(f"{selected_periods_timestamps}")

    # Assemble new signal
    new_signal = select_periods(df, selected_periods_timestamps)
    logger.debug(new_signal.describe())
    if PLOTTING:
        plt.subplots()
        new_signal["mean"].plot()
        plt.title(f"{tag} Filtered signal")
        plt.savefig(f"analysis/images/{slugify(tag)}_filtered.png", format="png")

    # Fit distribution
    return fit_distribution(new_signal, selected_periods_timestamps, tag)
