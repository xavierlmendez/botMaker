"""Recording an A* landmark search on two cells and rendering each as a walkthrough page.

Three cells, chosen so that the set says something a single one cannot. The RBF chain 8x8 fixture at
k = 3 is the cell the unit tests certify, small enough that every frame can be read and the whole
recording committed as a fixture. SPECTF at n = 14, k = 2 is the harness's own smallest real cell —
the same recipe, seed for seed, that `tests/ml/test_nystrom_uci_harness.py` runs — so the page shows
the search working on data rather than on a hand-built kernel. The third is the RBF chain again
under an expansion cap: the same kernel and the same budget of landmarks, stopped before it can
prove anything, so the page shows what a run without a certificate looks like — an incumbent, a
pruned child, and an ending that reports a gap instead of an optimum.

This file is a composition root: it is where an engine and the recorder that watches it meet (D-32).
Nothing in `math` knows it is being recorded, and nothing in `visualization` knows the cells exist.

    uv run python examples/astar_landmark_walkthrough.py --output-dir build/walkthroughs
    uv run python examples/astar_landmark_walkthrough.py --output-dir build --cell rbf_chain_8x8_k3

Each cell writes `<cell>.json` (the recording) and `<cell>.html` (the page). The RBF cell's
recording is committed as `tests/visualization/fixtures/astar_landmark_rbf_chain_8x8_k3.json` and
regenerated deliberately, like a baseline snapshot.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.algorithms.anytime_a_star_search import AnytimeAStarSearch
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
from mllib.ml.projects.nystrom_uci_data import (
    UciDatasetSpec,
    build_rbf_kernel,
    downsample_rows,
    load_feature_matrix,
    standardize_columns,
)
from mllib.visualization.html_renderer import write_walkthrough
from mllib.visualization.recorders.astar_landmark import AStarRecorder
from mllib.visualization.recording import Recording
from mllib.visualization.views import view_for

# examples/ -> repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
KERNEL_FIXTURES = REPOSITORY_ROOT / "tests" / "math" / "fixtures"

# The harness's SPECTF cell, verbatim: the same file, the same 14 sampled rows at sample_seed 5, the
# same standardisation and the same median-heuristic bandwidth. Changing any of them would make the
# page a picture of a different search from the one the harness reports on.
SPECTF = UciDatasetSpec(name="SPECTF", file_name="SPECTF.test", label_position="first")
SPECTF_ROWS = 14
SPECTF_SAMPLE_SEED = 5

# The cap is the point of this cell, not a budget anybody would choose. Seven is the smallest one
# that shows what the cell exists to show: the first complete selection is priced on the fifth
# expansion, so a run stopped before the seventh has no incumbent line and no pruned child on any
# frame it recorded, and the page would be an exact run that stopped early.
CAPPED_EXPANSIONS = 7

CELLS = ("rbf_chain_8x8_k3", "spectf_n14_k2", "rbf_chain_8x8_k3_capped")


def rbf_chain_kernel() -> np.ndarray:
    """The committed 8x8 RBF chain the search unit tests use, read as it is stored."""
    return np.loadtxt(KERNEL_FIXTURES / "rbf_chain_8x8.csv", delimiter=",")


def spectf_kernel() -> np.ndarray:
    """SPECTF at n = 14, built the way `run_nystrom_on_uci_dataset` builds it (order included)."""
    features = load_feature_matrix(SPECTF)
    features = downsample_rows(features, max_rows=SPECTF_ROWS, seed=SPECTF_SAMPLE_SEED)
    features = standardize_columns(features)
    kernel, _ = build_rbf_kernel(features)
    return kernel


def kernel_for(cell: str) -> tuple[np.ndarray, int]:
    """The kernel and landmark budget one cell name stands for."""
    if cell in ("rbf_chain_8x8_k3", "rbf_chain_8x8_k3_capped"):
        return rbf_chain_kernel(), 3
    if cell == "spectf_n14_k2":
        return spectf_kernel(), 2
    raise ValueError(f"Unknown cell {cell!r}; the example runs {', '.join(CELLS)}.")


def engine_for(cell: str) -> tuple[type, dict[str, object]]:
    """Which engine a cell is run with, and the knobs that make it that cell.

    The capped cell is the only one that is not exact A*: it is the same problem under
    ``AnytimeAStarSearch``, which is the pruned engine with a stop on it, so the recording carries
    an incumbent on every frame and a result with a gap rather than a certificate.
    """
    if cell == "rbf_chain_8x8_k3_capped":
        return AnytimeAStarSearch, {"max_expansions": CAPPED_EXPANSIONS}
    return AStarSearch, {}


def record_cell(cell: str) -> Recording:
    """Run one cell's engine with a recorder attached and assemble the recording.

    ``engine_configuration`` is put on the result before it is described: the engines leave it
    unset and a caller fills it in from the engine it ran (the harness does the same), so the page's
    footer states the settings the search actually ran under rather than nothing at all.
    """
    kernel, landmark_count = kernel_for(cell)
    engine, knobs = engine_for(cell)
    problem = NystromLandmarkProblem(kernel, landmark_count)
    recorder = AStarRecorder()
    search = engine(problem, NystromCssCostFunction(problem), recorder=recorder, **knobs)
    result = search.run()

    return Recording.from_recorder(
        recorder,
        problem={
            "kind": "astar_landmark",
            "cell": cell,
            "n": int(kernel.shape[0]),
            "landmark_count": landmark_count,
        },
        configuration=dict(search.configuration),
        result=replace(result, engine_configuration=search.configuration),
    )


def write_cell(cell: str, output_dir: Path) -> tuple[Path, Path]:
    """Record one cell and write both artefacts; returns the recording's path and the page's."""
    recording = record_cell(cell)
    recording_path = recording.save(Path(output_dir) / f"{cell}.json")
    page_path = write_walkthrough(recording, view_for(recording), Path(output_dir) / f"{cell}.html")
    return recording_path, page_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="where both files are written"
    )
    parser.add_argument(
        "--cell",
        action="append",
        choices=CELLS,
        help="a cell to record; repeatable, and every cell by default",
    )
    arguments = parser.parse_args(argv)

    for cell in arguments.cell or list(CELLS):
        recording_path, page_path = write_cell(cell, arguments.output_dir)
        print(f"{cell}: {recording_path}  {page_path}")


if __name__ == "__main__":
    main()
