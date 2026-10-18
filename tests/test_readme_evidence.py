"""The figures on the front page are the figures the code produces.

The README opens with a table of results checked against things outside this
repository. A table like that is worth exactly as much as the guarantee that it
still matches, and a number copied into a document once stops being true
quietly: the library changes, the suite stays green because nothing in it reads
the document, and the front page goes on claiming a precision the code no
longer has.

So the claim is pinned here, in both directions. Every row the README's table
holds has to be a row ``examples/evidence.py`` renders, which fails on the
commit that moves a figure; and every row label has to still be in the
document, which catches the other way a claim disappears — somebody deletes it.

This test is also what makes "runs in CI" true. CI runs the suite, so the
evidence script runs here rather than in a workflow step of its own.
"""

from __future__ import annotations

import re
import runpy
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
EVIDENCE = ROOT / "examples" / "evidence.py"

#: Row labels of the README's evidence table, as the script renders them.
#: Written out rather than parsed from the README so that deleting a row from
#: the table does not silently delete the check that defends it.
ROWS = (
    "Bailey and López de Prado (2014), their worked example end to end",
    "Expected maximum of n normals against quadrature, n = 2 to 10^6",
    "Skewness and kurtosis against scipy.stats, both bias conventions",
    "Lo's eta(q) against its AR(1) closed form, eight cases",
    "CPCV path count against k/N x C(N, k), every N up to 12",
    "Log normal upper tail against scipy.stats.norm.logsf, z = 0.01 to 400",
    "50 pure-noise strategies, best selected, 200 runs",
)

#: Results the README quotes, exactly as the script renders them.
RESULTS = (
    "SR0 = 0.1132 and DSR = 0.9004, as published",
    "agree to 1e-10",
    "agree to 1e-12",
    "exact",
    "agree to 1e-14 relative, past where the tail itself underflows",
    "PSR(0) > 95% in 90% of them, DSR > 95% in none",
)


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    """Run the evidence script once and share its figures across the tests."""
    with redirect_stdout(StringIO()):
        namespace = runpy.run_path(str(EVIDENCE))
        result = namespace["main"]()
    assert isinstance(result, dict)
    return result


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize("row", ROWS)
def test_the_evidence_table_still_has_the_row(row: str, readme: str) -> None:
    assert row in readme, f"the evidence table no longer has: {row}"


@pytest.mark.parametrize("result", RESULTS)
def test_a_quoted_result_is_one_the_script_renders(
    result: str, evidence: dict[str, Any], readme: str
) -> None:
    rendered = evidence["rendered"]
    assert isinstance(rendered, dict)
    assert any(result in text for text in rendered.values()), (
        f"the README quotes {result!r}, which examples/evidence.py no longer renders"
    )
    assert result in readme


def test_every_row_the_script_renders_is_on_the_front_page(
    evidence: dict[str, Any], readme: str
) -> None:
    """A check added to the script cannot go unquoted.

    The other direction of the pin. Without it, a new row lands in the script,
    nothing on the front page mentions it, and the evidence quietly stops being
    what the README says it is.
    """
    rendered = evidence["rendered"]
    assert isinstance(rendered, dict)
    assert set(rendered) == set(ROWS), "the script's checks and the pinned rows disagree"
    for label, result in rendered.items():
        assert f"| {label} | {result} |" in readme, f"not on the front page: {label}"


def test_every_figure_is_inside_the_bound_the_readme_states(
    evidence: dict[str, Any],
) -> None:
    """The script already enforces this, so this asserts that it did.

    ``main()`` raises ``SystemExit`` on a breached bound, which means a figure
    moving fails this module at fixture time rather than here. The point of
    asserting it again is that a future edit which turns the gate into a
    printout -- dropping the ``raise`` -- would otherwise go unnoticed.
    """
    checks = evidence["checks"]
    assert isinstance(checks, dict)
    assert set(checks) == set(ROWS)
    bounds = {
        "Bailey and López de Prado (2014), their worked example end to end": 0.0,
        "Expected maximum of n normals against quadrature, n = 2 to 10^6": 1e-10,
        "Skewness and kurtosis against scipy.stats, both bias conventions": 1e-12,
        "Lo's eta(q) against its AR(1) closed form, eight cases": 1e-15,
        "CPCV path count against k/N x C(N, k), every N up to 12": 0.0,
        "Log normal upper tail against scipy.stats.norm.logsf, z = 0.01 to 400": 1e-14,
        "50 pure-noise strategies, best selected, 200 runs": 0.0,
    }
    for label, bound in bounds.items():
        assert checks[label] <= bound, f"{label} measured {checks[label]}, bound {bound}"


def test_the_noise_sweep_fools_the_naive_ratio_and_not_the_deflated_one(
    evidence: dict[str, Any], readme: str
) -> None:
    """The row the library exists for, read off the measurement.

    Both halves are asserted because either one alone would be a weaker claim
    than the README makes. A deflated ratio fooled in no run is only
    interesting next to a probabilistic one that is fooled most of the time --
    otherwise the data might simply not have looked skilful to begin with.
    """
    runs = evidence["runs"]
    psr_hits = evidence["psr_hits"]
    assert isinstance(runs, int)
    assert isinstance(psr_hits, int)
    assert evidence["dsr_hits"] == 0
    assert psr_hits / runs > 0.8
    assert f"in {psr_hits} runs out of" in readme, (
        f"the prose no longer states the measured count of {psr_hits}"
    )


def test_the_suite_size_on_the_front_page_is_not_wildly_stale() -> None:
    """A loose bound, deliberately.

    Pinning the exact count would fail on every commit that adds a test, which
    trains people to update the number without reading it. A quarter either way
    still catches the README claiming hundreds when there are thousands, which
    is the error that matters.
    """
    claimed = re.search(r"([\d,]+) tests", README.read_text(encoding="utf-8"))
    assert claimed is not None, "the README no longer states a suite size"
    stated = int(claimed.group(1).replace(",", ""))
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-o", "addopts="],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    found = re.search(r"(\d+) tests collected", collected.stdout)
    assert found is not None, collected.stdout[-500:]
    actual = int(found.group(1))
    assert 0.75 * actual <= stated <= 1.25 * actual, (
        f"the README says {stated} tests and there are {actual}"
    )
