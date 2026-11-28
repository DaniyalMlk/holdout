"""The README's known-gaps list defends itself.

A list of shortcomings is the first thing in a document to go stale, because
closing a gap is satisfying and deleting the line that described it is not. So
each line carries a predicate, and the predicate holds only while the gap is
real: implement the feature and its line fails, forcing the line out.

Two rules about what belongs on the list:

**Absences, not refusals.** Most of what this library declines to do it
declines deliberately -- a single Sharpe ratio admits only single-step
haircuts, ``RANGE`` refuses duplicated columns -- and those are documented as
behaviour where they belong. A list of deliberate refusals relabelled as gaps
would read as candour and be worth less than no list at all.

**Call something.** A predicate that searches for a name cannot see the feature
arriving under a different one. Each predicate below either calls an entry
point and observes what happens, or asserts the exact shape of a public
signature, so that a new parameter under any name breaks it.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from holdout import (
    assess_drawdown,
    conditional_interval,
    deflated_sharpe_ratio,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    selection_coverage,
    selection_inference,
)

README = Path(__file__).resolve().parent.parent / "README.md"

#: The heading the list lives under, and the heading that ends it.
SECTION = "## Known gaps"
NEXT_SECTION = "## Layout"


def parameters(function: Callable[..., object]) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


# --- one predicate per line of the list ----------------------------------- #


def the_selection_interval_assumes_independence() -> None:
    """Correlation is something coverage is measured under, and not an input."""
    assert "correlation" in parameters(selection_coverage)
    assert parameters(selection_inference) == ("estimates", "deviations", "level")
    assert parameters(conditional_interval) == ("estimate", "deviation", "threshold", "level")
    with pytest.raises(TypeError, match="correlation"):
        selection_inference([0.1, 0.2], [0.05, 0.05], correlation=0.5)  # type: ignore[call-arg]
    selection_inference([0.1, 0.2], [0.05, 0.05])


def the_sharpe_tests_take_no_serial_correlation() -> None:
    """Exactly these arguments, so an autocorrelation input under any name fails."""
    assert parameters(probabilistic_sharpe_ratio) == (
        "sharpe",
        "n",
        "benchmark",
        "skewness",
        "kurtosis",
    )
    assert parameters(deflated_sharpe_ratio) == (
        "sharpe",
        "n",
        "n_trials",
        "trials_variance",
        "skewness",
        "kurtosis",
        "trials_mean",
        "method",
    )
    assert parameters(minimum_track_record_length) == (
        "sharpe",
        "benchmark",
        "confidence",
        "skewness",
        "kurtosis",
    )


def the_drawdown_law_is_brownian() -> None:
    """Reorder a record, or fatten its tails, and the law it is judged by holds still.

    The three records share a mean and a standard deviation and nothing else:
    one is normal, one is the same returns shuffled, one is Student t with three
    degrees of freedom rescaled to match. Their observed drawdowns differ, which
    is the check that the inputs really are different; the law's expected and
    median maximum drawdown and its limit do not, to rounding.
    """
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0004, 0.01, 1000)
    shuffled = rng.permutation(normal)
    tails = rng.standard_t(3, 1000)
    tails = (tails - tails.mean()) / tails.std(ddof=1) * normal.std(ddof=1) + normal.mean()

    reference = assess_drawdown(normal)
    for other in (shuffled, tails):
        assessed = assess_drawdown(other)
        assert assessed.observed.depth != pytest.approx(reference.observed.depth, rel=1e-3)
        for field in ("expected", "median", "limit"):
            assert getattr(assessed, field) == pytest.approx(
                getattr(reference, field), rel=1e-12, abs=0.0
            ), field


GAPS: tuple[tuple[str, Callable[[], None]], ...] = (
    (
        "The selection interval assumes independent candidates.",
        the_selection_interval_assumes_independence,
    ),
    (
        "The Sharpe tests take no serial correlation.",
        the_sharpe_tests_take_no_serial_correlation,
    ),
    (
        "There is one drawdown law, and it is Brownian.",
        the_drawdown_law_is_brownian,
    ),
)


def gaps_section() -> str:
    """The text from the list's heading up to the heading that follows it."""
    text = README.read_text(encoding="utf-8")
    start = text.index(SECTION)
    end = text.index(NEXT_SECTION, start + len(SECTION))
    return text[start:end]


def rendered_bullets() -> list[str]:
    """The bold lead of each bullet under the heading, in page order."""
    return re.findall(r"^- \*\*(.+?)\*\*", gaps_section(), re.MULTILINE)


@pytest.mark.parametrize("line, predicate", GAPS, ids=[line for line, _ in GAPS])
def test_the_gap_is_still_real(line: str, predicate: Callable[[], None]) -> None:
    predicate()


@pytest.mark.parametrize("line", [line for line, _ in GAPS])
def test_the_line_is_still_on_the_page(line: str) -> None:
    assert line in rendered_bullets(), f"not a bullet under {SECTION!r}: {line!r}"


def test_the_page_lists_nothing_that_has_no_predicate() -> None:
    """Same lines, same order: a bullet added to the page needs a predicate too."""
    assert rendered_bullets() == [line for line, _ in GAPS]


def test_every_predicate_is_distinct() -> None:
    predicates = [predicate for _, predicate in GAPS]
    assert len({p.__name__ for p in predicates}) == len(predicates)


def test_the_section_ends_at_the_heading_that_follows_it() -> None:
    """The list is bounded by the next ``## `` heading, whatever it is called.

    ``NEXT_SECTION not in gaps_section()`` looks like it checks this and cannot
    fail, because the slice is cut at that heading. What can go wrong is a new
    section inserted between the two, whose bullets would then be read as gaps.
    """
    headings = re.findall(r"^## .+$", README.read_text(encoding="utf-8"), re.MULTILINE)
    assert SECTION in headings
    following = headings[headings.index(SECTION) + 1]
    assert following == NEXT_SECTION, (
        f"the list is bounded by {NEXT_SECTION!r} and is followed by {following!r}"
    )
