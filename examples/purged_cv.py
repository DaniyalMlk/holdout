"""Why purging matters: a model that learns nothing looks skilful under shuffled k-fold.

The label is the next 20 days' return and the only feature is a slow moving
average of the price, so neighbouring observations have nearly the same
feature *and* overlapping labels. A nearest-neighbour model scored with
shuffled k-fold finds a training neighbour whose label shares most of the
test label's days, and appears to forecast. Purged k-fold removes those
neighbours and the skill disappears — as it must, because the prices are a
random walk.

Run with ``python examples/purged_cv.py``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from holdout import LeakageError, Split, leakage_audit, purged_kfold

HORIZON = 20


def data(n: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n + HORIZON + 60)
    price = np.cumsum(returns)
    kernel = np.ones(60) / 60
    # The feature on day t uses prices up to t; the label is the next HORIZON returns.
    feature: NDArray[np.float64] = np.convolve(price, kernel, mode="valid")[:n].astype(np.float64)
    label = np.array([returns[t + 60 : t + 60 + HORIZON].sum() for t in range(n)], dtype=np.float64)
    return feature, label


def nearest_neighbour_ic(
    feature: NDArray[np.float64], label: NDArray[np.float64], splits: list[Split]
) -> float:
    """Correlation between a 1-nearest-neighbour forecast and the realised label."""
    predictions = np.empty_like(label)
    for split in splits:
        train_x = feature[split.train]
        for i in split.test:
            nearest = split.train[int(np.argmin(np.abs(train_x - feature[i])))]
            predictions[i] = label[nearest]
    return float(np.corrcoef(predictions, label)[0, 1])


def shuffled_kfold(n: int, folds: int, seed: int) -> list[Split]:
    order = np.random.default_rng(seed).permutation(n)
    groups = np.array_split(order, folds)
    everything = np.arange(n, dtype=np.int64)
    return [
        Split(train=np.setdiff1d(everything, g), test=np.sort(g).astype(np.int64)) for g in groups
    ]


def main() -> dict[str, float]:
    n = 2000
    feature, label = data(n, seed=0)
    start = np.arange(n, dtype=np.float64)
    end = start + HORIZON  # each label spans the next HORIZON days

    shuffled = shuffled_kfold(n, 5, seed=1)
    purged = purged_kfold(5, start=start, end=end, embargo=HORIZON)

    try:
        leakage_audit(shuffled, start, end)
        leaked = False
    except LeakageError as exc:
        leaked = True
        print(f"shuffled k-fold fails the audit: {exc}")
    leakage_audit(purged, start, end)
    print("purged k-fold passes the audit")

    ic_shuffled = nearest_neighbour_ic(feature, label, shuffled)
    ic_purged = nearest_neighbour_ic(feature, label, purged)
    print(f"forecast/outcome correlation, shuffled k-fold: {ic_shuffled:+.3f}")
    print(f"forecast/outcome correlation, purged k-fold:   {ic_purged:+.3f}")
    return {"leaked": float(leaked), "shuffled": ic_shuffled, "purged": ic_purged}


if __name__ == "__main__":
    main()
