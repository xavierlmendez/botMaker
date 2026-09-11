"""Record a two-hot span rcut run and write both the recording and the walkthrough page.

Four graphs, each with the collision weight the harness sweep found worth watching and the
spectral initialisation they are all run from: the Guattery-Miller roach at λ = 10, whose antenna
cut is the one spectral bisection famously misses, Zachary's karate club at λ = 1, the
eighty-node cockroach of He, Gu & Zhang 2012 at λ = 10, which is the same roach at k = 20 and is
K = 3 because removing its ladder leaves the two antennae disconnected, and two triangles joined by
a weak bridge at λ = 0.1, the six-node instance whose optimum and datum are both exactly 1/15 so
that a missed page is the optimizer's doing and nothing else's.

`two_triangles` is also the first graph whose entry carries a learning rate and a step count of its
own (0.01 and 5000, against the shared 0.05 and 300): Σλ = 0.064 there, three orders below the
roach's, so the same step size walks straight past the optimum. A graph's own values apply only
when the CLI left `--learning-rate` and `--steps` at their defaults, so naming either on the command
line still means what it said, and the three older graphs — whose entries carry neither — run today's
schedule exactly.

This is a composition root and nothing else. It builds the incidence matrix, the initial V and the
recorder, runs `fit_two_hot_span` with the recorder attached, and writes `<graph>.json` beside
`<graph>.html`. Every number in those documents is the recorder's; every number the *run* reports
comes back through the exact numpy `pinv` projector of `two_hot_span_problem` (D-31). The training
loss on each frame is the ridge training loss and is never E.

`--frame-every` defaults to 5 *here*, in the example, while the recorder's own default stays 1. A
full frame carries V, so karate at every step of 300 is a 34-by-78 matrix three hundred times over —
a twelve-megabyte document, and a page embeds its whole recording. That is not a page anyone opens.
Every fifth step is sixty pictures, which is more than a reader steps through anyway, and the loss
curve is unaffected because a light frame is recorded at every step regardless. `--frame-every 1` is
still there for the run where V's every step is the thing being studied.

The recording also carries a `layout` inside `problem`, because a view has to draw the pair graph
somewhere and node positions are a property of the instance rather than of the run: a ladder for the
roach — the two paths at y = 1 and y = 0, position along the path as x, which is the picture the
graph is actually named after — and a seeded spring layout for anything else, so that two
regenerations of the same recording place the same node in the same spot.

The experiment knobs of slices 2.4 and 2.5 — `--init`, `--collision-weight`, `--adjacency-weight`,
`--adjacency-form`, `--diversity-weight` and `--schedule` — are here so that a configuration the
harness sweep found can be *walked through* rather than only tabulated; `--output-name` puts that
run's `.json`/`.html` beside the default run's instead of over it. All six default to the run this
example already recorded, so a bare invocation writes exactly the document it wrote before. The
reference configuration, the one that reaches the roach optimum, is

    uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir out/ \
        --graph roach_g5 --init random --collision-weight 10 --adjacency-form edge_product \
        --adjacency-weight 0.3 --diversity-weight 10 --output-name roach_g5_diversity

    uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir out/
    uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir out/ \
        --graph roach_g5 --frame-every 1
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import networkx as nx

from mllib.math.graph.two_hot_span_problem import (
    incidence_matrix,
    karate_graph,
    node_count,
    roach_graph,
    spectral_floor,
    spectral_spanning_set,
    two_triangles_graph,
)
from mllib.visualization.html_renderer import write_walkthrough
from mllib.visualization.recorders.two_hot_span import TwoHotSpanRecorder
from mllib.visualization.recording import Recording
from mllib.visualization.views.two_hot_span import two_hot_span_view

DEFAULT_STEP_COUNT = 300
# The example's default, not the recorder's (which stays 1): see the module docstring on page size.
DEFAULT_FRAME_EVERY = 5
DEFAULT_SEED = 0
DEFAULT_LEARNING_RATE = 0.05

# The experiment knobs, at the values that leave the run the one slice 2.2 shipped. They are the
# config's own defaults, restated here so the CLI can default to them without importing torch.
DEFAULT_INIT = "spectral"
DEFAULT_SCHEDULE = "constant"
DEFAULT_ADJACENCY_WEIGHT = 0.0
DEFAULT_ADJACENCY_FORM = "laplacian"
DEFAULT_DIVERSITY_WEIGHT = 0.0
INITS = ("spectral", "random")

ROACH_RUNG_COUNT = 5
ROACH_G20_RUNG_COUNT = 20

# The λ each graph is recorded at, and the cluster count it is cut into. All run from the spectral
# initialisation: it starts at E* = Σλ exactly, so the walkthrough opens on the relaxation's own
# optimum and what it shows is what the collision reward does to it. An entry may also carry
# `learning_rate` and `step_count`, which apply only when the CLI left its own at the default.
GRAPHS: dict[str, dict[str, object]] = {
    "roach_g5": {"cluster_count": 2, "collision_weight": 10.0},
    "karate": {"cluster_count": 2, "collision_weight": 1.0},
    "roach_g20": {"cluster_count": 3, "collision_weight": 10.0},
    "two_triangles": {
        "cluster_count": 2,
        "collision_weight": 0.1,
        "learning_rate": 0.01,
        "step_count": 5000,
    },
}


def build_graph(name: str) -> nx.Graph:
    """The graph a name stands for, built the same way the harness builds it."""
    if name == "roach_g5":
        return roach_graph(ROACH_RUNG_COUNT)
    if name == "roach_g20":
        return roach_graph(ROACH_G20_RUNG_COUNT)
    if name == "karate":
        return karate_graph()
    if name == "two_triangles":
        return two_triangles_graph()
    raise ValueError(f"unknown graph {name!r}; known: {sorted(GRAPHS)}")


TWO_TRIANGLES_POSITIONS: list[list[float]] = [
    # Left triangle 0-1-2: two vertices stacked at x = 0 and the bridge end at x = 1, so that the
    # bridge (2, 3) runs horizontally at y = 0 between the two triangles and the picture says which
    # edge is the cheap one. The right triangle is the mirror image about x = 1.5.
    [0.0, 0.5],
    [0.0, -0.5],
    [1.0, 0.0],
    [2.0, 0.0],
    [3.0, 0.5],
    [3.0, -0.5],
]


def layout_positions(name: str, graph: nx.Graph) -> list[list[float]]:
    """Node positions in sorted node order: a ladder for any roach, a seeded spring otherwise.

    The ladder reads k off the graph — a roach has n = 4k vertices — rather than off a constant, so
    the same layout draws the k = 5 roach and the k = 20 one of He, Gu & Zhang 2012.

    `two_triangles` gets its own six positions rather than a spring layout, for the same reason the
    roach gets the ladder: the thing a reader has to see is the one weak edge, and a spring layout
    that has to guess is free to draw it anywhere. Written out because six nodes is fewer than a
    formula.
    """
    nodes = sorted(graph.nodes())
    if name == "two_triangles":
        return [list(position) for position in TWO_TRIANGLES_POSITIONS]
    if name.startswith("roach"):
        # Top path 0..2k-1 at y = 1, bottom path 2k..4k-1 at y = 0, x the position along the path.
        path_length = len(nodes) // 2
        positions = []
        for node in nodes:
            index = int(node)
            if index < path_length:
                positions.append([float(index), 1.0])
            else:
                positions.append([float(index - path_length), 0.0])
        return positions
    spring = nx.spring_layout(graph, seed=0)
    return [[float(spring[node][0]), float(spring[node][1])] for node in nodes]


def layout_edges(graph: nx.Graph) -> list[list[int]]:
    """The graph's edges in index space as sorted ``[i, j]`` pairs with ``i < j``, ordered."""
    index_of = {node: index for index, node in enumerate(sorted(graph.nodes()))}
    edges = []
    for source, target in graph.edges():
        first, second = index_of[source], index_of[target]
        edges.append([min(first, second), max(first, second)])
    edges.sort()
    return edges


def initial_spanning_set(graph: nx.Graph, cluster_count: int, init: str):
    """The starting V for an init name; ``None`` means the optimizer's own seeded randn."""
    if init == "spectral":
        return spectral_spanning_set(graph, cluster_count)
    if init == "random":
        return None
    raise ValueError(f"unknown init {init!r}; known: {list(INITS)}")


def experiment_keys(config) -> dict[str, object]:
    """The knobs of slices 2.4 and 2.5 that are *on*, for the recording's ``problem``.

    Only the non-default ones, so a default run's document is byte for byte the document the
    example wrote before these flags existed and a reader of a page never sees a knob at zero.
    """
    keys: dict[str, object] = {}
    if config.learning_rate_schedule != DEFAULT_SCHEDULE:
        keys["learning_rate_schedule"] = config.learning_rate_schedule
    if config.adjacency_weight != DEFAULT_ADJACENCY_WEIGHT:
        keys["adjacency_weight"] = float(config.adjacency_weight)
        keys["adjacency_form"] = config.adjacency_form
    if config.diversity_weight != DEFAULT_DIVERSITY_WEIGHT:
        keys["diversity_weight"] = float(config.diversity_weight)
    return keys


def record_graph(
    name: str,
    output_dir: Path,
    *,
    step_count: int = DEFAULT_STEP_COUNT,
    frame_every: int = DEFAULT_FRAME_EVERY,
    seed: int = DEFAULT_SEED,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    init: str = DEFAULT_INIT,
    collision_weight: float | None = None,
    learning_rate_schedule: str = DEFAULT_SCHEDULE,
    adjacency_weight: float = DEFAULT_ADJACENCY_WEIGHT,
    adjacency_form: str = DEFAULT_ADJACENCY_FORM,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
    output_name: str | None = None,
) -> tuple[Path, Path]:
    """Run one graph with a recorder attached; write its recording and its page, in that order.

    Returns both paths, the recording's first. The view is built from the recording rather than
    chosen through ``view_for``, because this example knows exactly which problem it just recorded
    and naming it here is what makes a missing registration fail at the example rather than at the
    CLI, which is the path that must survive on a recording alone.

    ``collision_weight`` of ``None`` means the graph's own λ, and ``output_name`` of ``None`` the
    graph's own name. ``step_count`` and ``learning_rate`` left at the module defaults mean the
    same: a graph entry that carries its own overrides them, and one that does not leaves them
    alone. They are ordinary values rather than ``None`` sentinels because both the CLI and the
    tests pass them positionally-by-keyword at those defaults today, and a graph with no entry of
    its own must keep running exactly the schedule it ran before.

    The experiment knobs reach ``problem`` only when they are off their defaults, so the default
    run's document is the one it always was and an experiment's document says on its face what was
    switched on.
    """
    # torch is an optional group (D-31), so the import is local to the function that needs it.
    from mllib.math.algorithms.two_hot_span_optimizer import TwoHotSpanConfig, fit_two_hot_span

    settings = GRAPHS[name]
    cluster_count = int(settings["cluster_count"])  # type: ignore[arg-type]
    if collision_weight is None:
        collision_weight = float(settings["collision_weight"])  # type: ignore[arg-type]
    collision_weight = float(collision_weight)
    if step_count == DEFAULT_STEP_COUNT and "step_count" in settings:
        step_count = int(settings["step_count"])  # type: ignore[arg-type]
    if learning_rate == DEFAULT_LEARNING_RATE and "learning_rate" in settings:
        learning_rate = float(settings["learning_rate"])  # type: ignore[arg-type]

    graph = build_graph(name)
    count = node_count(graph)
    X = incidence_matrix(graph)
    floor = spectral_floor(graph, cluster_count)
    initial = initial_spanning_set(graph, cluster_count, init)

    config = TwoHotSpanConfig(
        step_count=step_count,
        learning_rate=learning_rate,
        collision_weight=collision_weight,
        seed=seed,
        learning_rate_schedule=learning_rate_schedule,
        adjacency_weight=adjacency_weight,
        adjacency_form=adjacency_form,
        diversity_weight=diversity_weight,
    )
    recorder = TwoHotSpanRecorder(
        X,
        graph,
        cluster_count,
        step_count,
        frame_every=frame_every,
        spectral_floor=floor,
    )
    run = fit_two_hot_span(X, cluster_count, config, initial, recorder=recorder)

    problem: dict[str, object] = {
        "kind": "two_hot_span",
        "graph": name,
        "n": count,
        "cluster_count": cluster_count,
        "collision_weight": collision_weight,
        "init": init,
        "spectral_floor": float(floor),
    }
    problem.update(experiment_keys(config))
    problem["layout"] = {
        "positions": layout_positions(name, graph),
        "edges": layout_edges(graph),
    }

    recording = Recording.from_recorder(
        recorder,
        problem=problem,
        configuration=dataclasses.asdict(config),
        result=run,
    )
    stem = output_name or name
    recording_path = recording.save(Path(output_dir) / f"{stem}.json")
    page_path = write_walkthrough(
        recording, two_hot_span_view(recording), Path(output_dir) / f"{stem}.html"
    )
    return recording_path, page_path


def summarise(name: str, paths: tuple[Path, Path], run_fields: dict[str, object]) -> str:
    """One line per recorded graph, for a reader watching the example run."""
    return (
        f"{name}: E* {float(run_fields['relaxed_objective']):.6f}  "  # type: ignore[arg-type]
        f"Ê {float(run_fields['rounded_cut']):.6f}  "  # type: ignore[arg-type]
        f"Σλ {float(run_fields['spectral_floor']):.6f}  "  # type: ignore[arg-type]
        f"{run_fields['component_count']} components  ->  {paths[0]}  {paths[1]}"
    )


def main(argv: list[str] | None = None) -> None:
    """Record every requested graph and print where each recording went."""
    # Same reason as `record_graph`'s import: the allowed schedule and form names live beside the
    # config that validates against them, in the torch module (D-31), and only the CLI needs them.
    from mllib.math.algorithms.two_hot_span_optimizer import (
        ADJACENCY_FORMS,
        LEARNING_RATE_SCHEDULES,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--graph", action="append", choices=sorted(GRAPHS), default=None, help="repeatable"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEP_COUNT,
        help="left at the default, a graph's own step count wins",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="left at the default, a graph's own learning rate wins",
    )
    parser.add_argument(
        "--frame-every",
        type=int,
        default=DEFAULT_FRAME_EVERY,
        help="steps between full frames; 1 records V at every step and makes a very large page",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # The experiment knobs of slices 2.4 and 2.5, defaulting to the run this example already
    # recorded: naming none of them writes the document it wrote before they existed.
    parser.add_argument("--init", choices=INITS, default=DEFAULT_INIT)
    parser.add_argument(
        "--collision-weight", type=float, default=None, help="λ; default: the graph's own"
    )
    parser.add_argument("--schedule", choices=LEARNING_RATE_SCHEDULES, default=DEFAULT_SCHEDULE)
    parser.add_argument("--adjacency-weight", type=float, default=DEFAULT_ADJACENCY_WEIGHT)
    parser.add_argument("--adjacency-form", choices=ADJACENCY_FORMS, default=DEFAULT_ADJACENCY_FORM)
    parser.add_argument("--diversity-weight", type=float, default=DEFAULT_DIVERSITY_WEIGHT)
    parser.add_argument(
        "--output-name",
        default=None,
        help="stem of the .json/.html written; default: the graph name",
    )
    arguments = parser.parse_args(argv)

    names = arguments.graph if arguments.graph else sorted(GRAPHS)
    if arguments.output_name is not None and len(names) > 1:
        parser.error("--output-name names one document, so it takes exactly one --graph")
    for name in names:
        paths = record_graph(
            name,
            arguments.output_dir,
            step_count=arguments.steps,
            frame_every=arguments.frame_every,
            seed=arguments.seed,
            learning_rate=arguments.learning_rate,
            init=arguments.init,
            collision_weight=arguments.collision_weight,
            learning_rate_schedule=arguments.schedule,
            adjacency_weight=arguments.adjacency_weight,
            adjacency_form=arguments.adjacency_form,
            diversity_weight=arguments.diversity_weight,
            output_name=arguments.output_name,
        )
        recording = Recording.load(paths[0])
        print(summarise(name, paths, recording.result))


if __name__ == "__main__":
    main()
