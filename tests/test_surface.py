"""The package's public surface is what its modules declare.

``src/holdout/__init__.py`` re-exports the package's public names and it is a
hand-maintained list. A hand-maintained list is a list with something missing
from it, and what was missing included the way into the library:

    >>> import holdout          # the README's own opening line
    >>> holdout.read_returns_csv
    AttributeError: module 'holdout' has no attribute 'read_returns_csv'

``__init__.py`` had no ``from .io import`` line at all, and none for
``synthetic`` either: two whole modules, three declared names each. ``io`` is
how a caller gets their own returns in, so the first thing anybody with a CSV
reaches for was not there. Fourteen more names were missing one at a time from
modules otherwise wired in — ``series`` declared eight and the root carried
two, ``sharpe`` declared nine and carried six — among them ``FloatArray`` and
``IndexArray``, the array aliases the package's own signatures are written in.

Nothing noticed because nothing looked, in either of the two places that could
have. ``tests/test_io.py`` opens with ``from holdout.io import ...``: a
submodule import, which is the right style for a test and exactly the style
that cannot see a missing re-export — the module is exercised in full and the
surface it is absent from is never touched. And the README, whose thirteen
other examples all write ``from holdout import ...``, never shows the CSV
reader, so the one entry point a new user needs was in neither the house style
nor the documentation.

So the surface is recomputed here from the modules rather than compared against
a second list that would need the same remembering.

**A module's own ``__all__`` is the authority.** Deciding here which names are
public puts the judgement somewhere other than beside the code it is about, and
makes the gate wrong whenever a module changes its mind. What is asserted is a
disagreement *inside* the package: a module declaring a name public while the
root does not carry it.

**Containment, not equality.** Two names reach the root without appearing in
any module's ``__all__``. Asserting the equality would mean ruling on those
here, which is the judgement this file refuses to make; and thirteen modules
still define a public name they have not declared, which belongs in the modules
rather than in a gate. Both are recorded in ``ROADMAP.md`` as what this phase
leaves open.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections import Counter
from types import ModuleType

import pytest

import holdout

#: Neither is part of the importable surface: ``cli`` is an entry point whose
#: names are wired to ``argparse``, and ``__main__`` exists so that
#: ``python -m holdout`` works.
NOT_SURFACE = frozenset({"cli", "__main__"})

#: Modules that declare no ``__all__``. Pinned rather than discovered, so that
#: a new module arriving without one is a decision somebody makes here instead
#: of a silent exemption from every check in this file.
NO_DUNDER_ALL: frozenset[str] = frozenset()


def submodules() -> dict[str, ModuleType]:
    """Every importable submodule of the package, bar the entry points."""
    return {
        info.name: importlib.import_module(f"holdout.{info.name}")
        for info in pkgutil.iter_modules(holdout.__path__)
        if info.name not in NOT_SURFACE
    }


def declared() -> dict[str, frozenset[str]]:
    """Each module's own declaration of what it makes public."""
    return {
        name: frozenset(module.__all__)
        for name, module in submodules().items()
        if getattr(module, "__all__", None) is not None
    }


@pytest.fixture(scope="module")
def root() -> frozenset[str]:
    return frozenset(holdout.__all__)


def test_every_declared_name_reaches_the_root(root: frozenset[str]) -> None:
    """A name a module declares public is importable from the package.

    This is the check the defect needed. Twenty names failed it.
    """
    missing: dict[str, list[str]] = {}
    for module, names in declared().items():
        absent = sorted(names - root)
        if absent:
            missing[module] = absent
    assert not missing, f"declared public but not re-exported from holdout/__init__.py: {missing}"


def test_no_module_is_absent_from_the_root_entirely(root: frozenset[str]) -> None:
    """A module with no name at the root is reported as the module.

    The check above catches it too, as six loose names across two modules,
    which reads as six small omissions rather than the two whole modules it
    was. ``io`` and ``synthetic`` had no import line in ``__init__.py`` at all,
    and that is a different mistake from a name somebody forgot.
    """
    absent = sorted(module for module, names in declared().items() if names and not (names & root))
    assert not absent, (
        "these modules declare public names and none of them is at the root, so "
        f"the module was never wired into holdout/__init__.py: {absent}"
    )


def test_the_way_into_the_library_is_at_the_root(root: frozenset[str]) -> None:
    """The CSV entry point, named, because it is what a new user reaches for.

    Containment covers these three as part of twenty, which does not say what
    broke. A reader who follows the README's ``import holdout`` and then cannot
    load a file has hit something worse than an incomplete namespace, so it
    fails here with a message that says so.
    """
    for name in ("read_returns_csv", "write_returns_csv", "ReturnTable"):
        assert name in root, (
            f"holdout.{name} is not importable from the package root. This is how "
            "a caller gets their own returns in; the README's first line is "
            "`import holdout`, so it has to be reachable from there."
        )


def test_the_aliases_the_signatures_are_written_in_are_at_the_root(
    root: frozenset[str],
) -> None:
    """A caller who can call a function can annotate against its types.

    ``FloatArray`` and ``IndexArray`` are the array aliases this package's own
    signatures use. Losing them means the functions are reachable and the types
    they are written in are not, which is the same failure as the entry point
    above and is worth its own name.
    """
    for alias in ("FloatArray", "IndexArray"):
        assert alias in root, (
            f"holdout.{alias} is not at the root, so a caller cannot annotate "
            "against a type this package's own signatures are written in"
        )


def test_every_exported_name_is_actually_there(root: frozenset[str]) -> None:
    """Nothing in ``__all__`` is a name the package does not have.

    ``__all__`` is a list of strings and nothing checks it against the module,
    so a rename that updates the import and not the list leaves an entry that
    only fails for somebody doing ``from holdout import *``.
    """
    phantom = sorted(name for name in root if not hasattr(holdout, name))
    assert not phantom, f"in holdout.__all__ but not attributes of it: {phantom}"


def test_the_root_exports_each_name_once() -> None:
    """``__all__`` does not repeat itself."""
    repeated = {name: count for name, count in Counter(holdout.__all__).items() if count > 1}
    assert not repeated, f"repeated in holdout.__all__: {repeated}"


def test_every_module_declares_something() -> None:
    """No module is exempt from the checks above by declaring nothing.

    Every other assertion here is silent about a module without an ``__all__``,
    so such a module is exempt from all of them. There is currently none, and
    the pinned set is empty: a module arriving without one fails here, which is
    the point at which somebody decides whether that was wanted.
    """
    silent = sorted(
        name for name, module in submodules().items() if getattr(module, "__all__", None) is None
    )
    assert silent == sorted(NO_DUNDER_ALL), (
        f"modules carrying no __all__, and so exempt from every check here: "
        f"{silent}; pinned: {sorted(NO_DUNDER_ALL)}"
    )


def test_the_probe_reads_a_surface_worth_checking(root: frozenset[str]) -> None:
    """The checks above are not passing because they found nothing.

    Every assertion in this file is satisfied by an empty set, so a probe that
    stops importing modules or stops finding ``__all__`` turns the whole file
    green. The sizes are asserted with loose bounds — the claim is that a
    surface was read at all — and ``io`` is named, because it is the module the
    defect centred on and the one a future rewrite is most likely to drop
    again.
    """
    modules = declared()
    assert len(modules) >= 18, f"only {len(modules)} modules declared an __all__"
    assert len(root) >= 120, f"the root surface came back as {len(root)} names"
    assert "io" in modules, "the io module was not read"
    assert {"read_returns_csv", "write_returns_csv", "ReturnTable"} <= modules["io"]
