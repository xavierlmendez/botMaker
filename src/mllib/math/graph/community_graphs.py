"""Rung 3 of the two-hot span stress ladder: real graphs whose communities are published.

Rungs 1 and 2 are synthetic — the roach, planted partitions, two-moons neighbourhood graphs — and
their planted labels are true by construction. Rung 3 asks the harder question: on a graph nobody
built for us, does the rounded ratio cut find the partition the field agrees is there? Three graphs
are small enough to certify against and standard enough that their numbers are quotable:

  ``polbooks``        Krebs' 105 political books co-purchased on Amazon, hand-labelled liberal /
                      neutral / conservative [L6]. n = 105, m = 441, K = 3, connected.
  ``football``        Girvan & Newman's 115 Division I-A teams of the 2000 season, labelled by
                      athletic conference [L7]. n = 115, m = 613, K = 12, connected. Caveat, known
                      and deliberately *not* corrected here: the five independent teams — teams in
                      no conference that season — are mislabelled in the original file. The file is
                      loaded as distributed, so that a number quoted from it is comparable with
                      every other paper's number from the same file; the price is that a handful of
                      planted labels are wrong and no agreement measure on this instance can reach
                      1.0.
  ``email_eu_core``   SNAP's email network of a European research institution, labelled by the
                      sender's department [L8]. 1005 nodes and 16064 undirected edges over 20
                      components; the largest holds 986 of them.

The files are fetched, not committed. They are all well under the 1 MB ceiling of D-19, so the
ceiling is not the reason: the reason is that they are third-party data with their own provenance,
and the repository records where they came from and what they hashed to rather than a copy of them.
That is the same convention the sensor-gap experiment uses. The default directory sits outside the
work tree; the sha256 of every file is pinned below and checked on every load, so a silently
changed upstream file is an error and never a quietly different result.

The email graph is cut down to its largest connected component before it becomes a
``GraphInstance``, and that is not tidying. ``spectral_spanning_set`` raises on a disconnected
graph on purpose: a graph with c components has a c-dimensional Laplacian null space, so at K < c
one of the r largest eigenvectors is still a null-space vector and the columns silently stop being
zero-sum. Every engine here starts from that spectral initialisation, so a disconnected instance
has no run at all. The other 19 components are isolated vertices — all 16064 edges are inside the
largest one — so nothing but 19 unreachable nodes is lost, and the loss is recorded in
``graph.graph["largest_component"]`` so a report can say what it left out.

Pre-fetch once, from the repository root::

    uv run python -m mllib.math.graph.community_graphs

Re-running verifies the hashes and downloads nothing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from mllib.math.graph.two_hot_span_problem import GraphInstance

#: Outside the work tree, created on demand. Nothing here is ever committed.
DEFAULT_DATA_DIR = Path.home() / "Desktop" / "BotMaker" / "two-hot-stress-data"

#: How many bytes to hash at a time; the files are tiny but the loop is the same one everywhere.
_HASH_CHUNK = 1 << 20

#: Sent on every download; see ``_download``.
_USER_AGENT = "mllib-community-graphs/1.0"


@dataclass(frozen=True, slots=True)
class Source:
    """One remote file, pinned. ``filename`` is what it is called inside the data directory."""

    url: str
    sha256: str
    filename: str


SOURCES: dict[str, tuple[Source, ...]] = {
    "polbooks": (
        Source(
            url="https://websites.umich.edu/~mejn/netdata/polbooks.zip",
            sha256="b8e37351ae9ae8ee39f8b75ed52170d2435f290855605680c9b4f4c8b46b3c37",
            filename="polbooks.zip",
        ),
    ),
    "football": (
        Source(
            url="https://websites.umich.edu/~mejn/netdata/football.zip",
            sha256="147081ba047c919782896b26852f812e65880a9e52f69070069f950eea90c396",
            filename="football.zip",
        ),
    ),
    "email_eu_core": (
        Source(
            url="https://snap.stanford.edu/data/email-Eu-core.txt.gz",
            sha256="4b47acdb80197b085fe63c819c357ae488131ee904ed93d1b219a68b0f9e245f",
            filename="email-Eu-core.txt.gz",
        ),
        Source(
            url="https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz",
            sha256="e5abe5b4581a480032a63adcf2576c161785f45692642c6ebb0b1276f0f33669",
            filename="email-Eu-core-department-labels.txt.gz",
        ),
    ),
}

#: The rung-3 instances, in ascending order of size.
RUNG3_NAMES = ("polbooks", "football", "email_eu_core")


def sha256_file(path: Path) -> str:
    """The hex digest of a file, read in chunks so the loop does not care how big it is."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    """Fetch to a neighbouring ``.part`` file first, so an interrupted run leaves nothing to trust.

    An explicit User-Agent is required, not polite: the Michigan host answers urllib's default
    agent string with 403.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        while chunk := response.read(_HASH_CHUNK):
            handle.write(chunk)
    partial.replace(target)


def fetch(name: str, data_dir: Path | str | None = None) -> list[Path]:
    """Download the sources of ``name`` if absent, verify every one, and return their paths.

    A file that is already present is hashed but never re-downloaded — the verification is the
    cheap half and the only half that has to happen every time. A hash mismatch deletes the file
    (a half-written or replaced-upstream file must not survive to be trusted by the next run) and
    raises ``ValueError`` naming it, so the fix is to re-run rather than to hunt for stale bytes.
    """
    if name not in SOURCES:
        raise ValueError(f"unknown community graph {name!r}; known: {sorted(SOURCES)}")
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for source in SOURCES[name]:
        target = directory / source.filename
        if not target.exists():
            _download(source.url, target)
        digest = sha256_file(target)
        if digest != source.sha256:
            target.unlink(missing_ok=True)
            raise ValueError(
                f"sha256 mismatch for {source.filename}: expected {source.sha256}, got {digest}; "
                "the bad download has been deleted"
            )
        paths.append(target)
    return paths


def _labels_by_first_appearance(values: Sequence[object]) -> tuple[np.ndarray, int]:
    """Map published community values to ints 0..K-1 in order of first appearance.

    First appearance over *sorted* nodes, not over file order: the file order of a GML is an
    implementation detail of whoever wrote it, while sorted node order is the order every other
    matrix in this package is built in, so the labels line up with the rows of ``X`` by
    construction rather than by luck.
    """
    index_of: dict[object, int] = {}
    for value in values:
        if value not in index_of:
            index_of[value] = len(index_of)
    labels = np.array([index_of[value] for value in values], dtype=int)
    return labels, len(index_of)


def _simple_undirected(graph: nx.Graph) -> nx.Graph:
    """An unweighted simple undirected copy: no direction, no self-loops, no parallel edges."""
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    simple.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    return simple


def _relabel_sorted(graph: nx.Graph) -> nx.Graph:
    """Nodes → 0..n-1 in sorted original order, the convention of ``two_hot_span_problem``."""
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def _provenance(sources: Sequence[Source]) -> list[dict[str, str]]:
    return [{"url": s.url, "sha256": s.sha256, "filename": s.filename} for s in sources]


def read_gml_zip(path: Path | str) -> nx.Graph:
    """Parse the single ``.gml`` member of one of Newman's zips into a simple undirected graph.

    ``label="id"`` is not optional here. Newman's files carry a human ``label`` per node — a book
    title, a team name — and ``nx.read_gml`` would use it as the node key, which sorts
    alphabetically and would make the node order an artefact of English spelling. The integer
    ``id`` is the identifier the published community labels are indexed by.
    """
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".gml")]
        if len(members) != 1:
            raise ValueError(
                f"expected exactly one .gml member in {Path(path).name}, got {members}"
            )
        with archive.open(members[0]) as handle:
            graph = nx.read_gml(handle, label="id")
    return _simple_undirected(nx.Graph(graph))


def build_gml_instance(
    name: str, gml_zip_path: Path | str, sources: Sequence[Source] = ()
) -> GraphInstance:
    """A ``GraphInstance`` from a Newman-style zip whose node attribute ``value`` is the community."""
    graph = read_gml_zip(gml_zip_path)
    nodes = sorted(graph.nodes())
    if any("value" not in graph.nodes[node] for node in nodes):
        raise ValueError(f"{name}: every node must carry the community attribute 'value'")
    labels, cluster_count = _labels_by_first_appearance(
        [graph.nodes[node]["value"] for node in nodes]
    )
    graph = _relabel_sorted(graph)
    graph.graph = {"source": _provenance(sources), "community_key": "value"}
    return GraphInstance(name, graph, cluster_count, labels)


def read_email_edge_list(edges_path: Path | str, labels_path: Path | str) -> tuple[nx.Graph, dict]:
    """Read the gzipped SNAP pair files: a symmetrised simple graph, and department per node.

    The edge file is directed (``u`` emailed ``v``) and holds both directions of most exchanges
    plus a few self-mails. Communities are undirected here — the ratio cut has no notion of
    direction — so the file is read straight into ``nx.Graph``, which collapses ``u v`` and ``v u``
    into one edge, and self-loops are dropped: a self-loop adds an equal amount to a vertex's
    degree and to its adjacency row, so it cancels out of the Laplacian and only ever distorts the
    edge count.
    """
    graph = nx.Graph()
    with gzip.open(edges_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            source, target = (int(field) for field in line.split())
            graph.add_node(source)
            graph.add_node(target)
            if source != target:
                graph.add_edge(source, target)
    department_of: dict[int, int] = {}
    with gzip.open(labels_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            node, department = (int(field) for field in line.split())
            department_of[node] = department
    return graph, department_of


def build_email_instance(
    name: str,
    edges_path: Path | str,
    labels_path: Path | str,
    sources: Sequence[Source] = (),
) -> GraphInstance:
    """A ``GraphInstance`` on the *largest connected component* of the SNAP email graph.

    The component is taken because the engine cannot run without it: ``spectral_spanning_set``
    raises on a disconnected graph, since the c-dimensional null space of a c-component Laplacian
    leaks a non-zero-sum column into the spanning set. What was dropped is recorded on the graph
    under ``largest_component`` — counts before and after, and how many departments survived — so
    that a report quotes the reduced instance without hiding the reduction.
    """
    graph, department_of = read_email_edge_list(edges_path, labels_path)
    node_count_before = graph.number_of_nodes()
    edge_count_before = graph.number_of_edges()
    labels_before = len({department_of[node] for node in graph.nodes() if node in department_of})

    component = max(nx.connected_components(graph), key=len)
    graph = _simple_undirected(graph.subgraph(component).copy())
    nodes = sorted(graph.nodes())
    missing = [node for node in nodes if node not in department_of]
    if missing:
        raise ValueError(
            f"{name}: {len(missing)} nodes have no department label, e.g. {missing[:5]}"
        )
    labels, cluster_count = _labels_by_first_appearance([department_of[node] for node in nodes])
    graph = _relabel_sorted(graph)
    graph.graph = {
        "source": _provenance(sources),
        "community_key": "department",
        "largest_component": {
            "node_count_before": node_count_before,
            "node_count_after": graph.number_of_nodes(),
            "edge_count_before": edge_count_before,
            "labels_before": labels_before,
            "labels_after": cluster_count,
        },
    }
    return GraphInstance(name, graph, cluster_count, labels)


def load_community_graph(name: str, data_dir: Path | str | None = None) -> GraphInstance:
    """Fetch (if needed), verify and parse one rung-3 graph into a ``GraphInstance``."""
    if name not in SOURCES:
        raise ValueError(f"unknown community graph {name!r}; known: {sorted(SOURCES)}")
    paths = fetch(name, data_dir)
    sources = SOURCES[name]
    if name == "email_eu_core":
        return build_email_instance(name, paths[0], paths[1], sources)
    return build_gml_instance(name, paths[0], sources)


def load_rung3_graphs(data_dir: Path | str | None = None) -> tuple[GraphInstance, ...]:
    """All three rung-3 instances, smallest first."""
    return tuple(load_community_graph(name, data_dir) for name in RUNG3_NAMES)


def _describe(instance: GraphInstance, sources: Sequence[Source]) -> str:
    graph = instance.graph
    lines = [
        f"{instance.name}: n = {graph.number_of_nodes()}  m = {graph.number_of_edges()}  "
        f"K = {instance.cluster_count}  connected = {nx.is_connected(graph)}"
    ]
    component: Mapping | None = graph.graph.get("largest_component")
    if component is not None:
        lines.append(
            f"  largest component: {component['node_count_before']} -> "
            f"{component['node_count_after']} nodes, {component['edge_count_before']} edges before, "
            f"{component['labels_before']} -> {component['labels_after']} labels"
        )
    for source in sources:
        lines.append(f"  {source.filename}  sha256 {source.sha256}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Pre-fetch every rung-3 graph and print what it is. Run once per clone; offline afterwards."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"directory outside git for the raw files (default: {DEFAULT_DATA_DIR})",
    )
    arguments = parser.parse_args(argv)
    for name in RUNG3_NAMES:
        instance = load_community_graph(name, arguments.data_dir)
        print(_describe(instance, SOURCES[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
