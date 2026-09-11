"""Rung-3 community graph loaders: parsing, label mapping, component surgery, and the hash pin.

Every test here is offline. The parsers are exercised on tiny fixtures written into ``tmp_path``
and shaped like the real files — a zip holding one ``.gml``, a gzipped directed edge list with a
self-loop and a reverse duplicate — and handed to the *same* functions the real files go through,
so the fixture tests cover the production path rather than a parallel one. The download path is
covered by monkeypatching ``urllib.request.urlopen``: once to serve bytes, once to fail if it is
called at all, which is how "a verified file is never re-fetched" is asserted rather than assumed.

The three tests that touch the real graphs assert their published n, m and K, and skip unless the
files are already sitting in the default data directory. They never download: the suite stays
deterministic and offline, and the pre-fetch is a deliberate one-off
(``uv run python -m mllib.math.graph.community_graphs``).
"""

import gzip
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from mllib.math.graph.community_graphs import (
    DEFAULT_DATA_DIR,
    RUNG3_NAMES,
    SOURCES,
    Source,
    build_email_instance,
    build_gml_instance,
    fetch,
    load_community_graph,
    read_email_edge_list,
    read_gml_zip,
    sha256_file,
)

# Node ids 0..5 carry labels that sort into exactly the reverse of the id order, so a parser that
# keyed nodes by ``label`` instead of ``id`` would produce a different edge set and be caught.
# ``value`` is the community; the edge (5, 5) is a self-loop that must not survive.
FIXTURE_GML = """graph [
  node [ id 0 label "zeta" value "n" ]
  node [ id 1 label "yankee" value "n" ]
  node [ id 2 label "xray" value "n" ]
  node [ id 3 label "whiskey" value "l" ]
  node [ id 4 label "bravo" value "l" ]
  node [ id 5 label "alpha" value "l" ]
  edge [ source 0 target 1 ]
  edge [ source 1 target 2 ]
  edge [ source 2 target 0 ]
  edge [ source 3 target 4 ]
  edge [ source 4 target 5 ]
  edge [ source 2 target 3 ]
  edge [ source 5 target 5 ]
]
"""

# Directed, the way SNAP ships it: (1, 0) repeats (0, 1) in reverse, (6, 6) is a self-mail that
# leaves node 6 isolated, and the graph falls into three components of sizes 4, 2 and 1.
FIXTURE_EDGES = "0 1\n1 0\n1 2\n2 3\n3 0\n4 5\n5 4\n6 6\n"

# Departments are numbered so that first-appearance order over sorted nodes (7, then 3) differs
# from their numeric order.
FIXTURE_LABELS = "0 7\n1 7\n2 3\n3 7\n4 5\n5 5\n6 9\n"


@pytest.fixture
def gml_zip(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("tiny.gml", FIXTURE_GML)
    return path


@pytest.fixture
def email_files(tmp_path: Path) -> tuple[Path, Path]:
    edges = tmp_path / "tiny-edges.txt.gz"
    labels = tmp_path / "tiny-labels.txt.gz"
    edges.write_bytes(gzip.compress(FIXTURE_EDGES.encode()))
    labels.write_bytes(gzip.compress(FIXTURE_LABELS.encode()))
    return edges, labels


def _urlopen_serving(payload: bytes):
    def opener(request, *args, **kwargs):
        return io.BytesIO(payload)

    return opener


def _urlopen_forbidden(request, *args, **kwargs):
    raise AssertionError("urlopen must not be called for an already verified file")


# --- the GML path ----------------------------------------------------------------------------


def test_gml_zip_is_keyed_by_integer_id_not_by_the_human_label(gml_zip: Path) -> None:
    graph = read_gml_zip(gml_zip)
    assert set(graph.nodes()) == set(range(6))
    assert set(map(frozenset, graph.edges())) == {
        frozenset({0, 1}),
        frozenset({1, 2}),
        frozenset({0, 2}),
        frozenset({3, 4}),
        frozenset({4, 5}),
        frozenset({2, 3}),
    }


def test_gml_self_loop_is_dropped(gml_zip: Path) -> None:
    graph = read_gml_zip(gml_zip)
    assert list(nx.selfloop_edges(graph)) == []
    assert graph.number_of_edges() == 6


def test_gml_communities_map_to_zero_based_ints_in_first_appearance_order(gml_zip: Path) -> None:
    instance = build_gml_instance("tiny", gml_zip)
    assert instance.cluster_count == 2
    # "n" is met first over sorted nodes, so it is 0 even though "l" sorts before it.
    assert instance.planted_labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_gml_instance_nodes_are_zero_to_n_minus_one_and_connected(gml_zip: Path) -> None:
    instance = build_gml_instance("tiny", gml_zip)
    assert sorted(instance.graph.nodes()) == list(range(6))
    assert nx.is_connected(instance.graph)


def test_gml_instance_records_its_provenance(gml_zip: Path) -> None:
    source = Source(url="https://example.invalid/tiny.zip", sha256="0" * 64, filename="tiny.zip")
    instance = build_gml_instance("tiny", gml_zip, (source,))
    assert instance.graph.graph["community_key"] == "value"
    assert instance.graph.graph["source"] == [
        {"url": source.url, "sha256": source.sha256, "filename": source.filename}
    ]


def test_a_gml_zip_without_exactly_one_gml_member_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "no graph here")
    with pytest.raises(ValueError, match=r"exactly one \.gml member"):
        read_gml_zip(path)


# --- the edge-list path ----------------------------------------------------------------------


def test_email_edge_list_drops_self_loops_and_collapses_reverse_duplicates(
    email_files: tuple[Path, Path],
) -> None:
    graph, department_of = read_email_edge_list(*email_files)
    assert graph.number_of_nodes() == 7
    assert graph.number_of_edges() == 5
    assert list(nx.selfloop_edges(graph)) == []
    assert graph.degree(6) == 0
    assert department_of == {0: 7, 1: 7, 2: 3, 3: 7, 4: 5, 5: 5, 6: 9}


def test_email_instance_keeps_only_the_largest_component(email_files: tuple[Path, Path]) -> None:
    instance = build_email_instance("tiny", *email_files)
    assert sorted(instance.graph.nodes()) == [0, 1, 2, 3]
    assert instance.graph.number_of_edges() == 4
    assert nx.is_connected(instance.graph)


def test_email_instance_records_what_the_component_cut_away(
    email_files: tuple[Path, Path],
) -> None:
    instance = build_email_instance("tiny", *email_files)
    assert instance.graph.graph["largest_component"] == {
        "node_count_before": 7,
        "node_count_after": 4,
        "edge_count_before": 5,
        "labels_before": 4,
        "labels_after": 2,
    }
    assert instance.graph.graph["community_key"] == "department"


def test_email_departments_map_to_zero_based_ints_in_first_appearance_order(
    email_files: tuple[Path, Path],
) -> None:
    instance = build_email_instance("tiny", *email_files)
    # Departments over sorted component nodes are 7, 7, 3, 7: department 7 is met first.
    assert instance.planted_labels.tolist() == [0, 0, 1, 0]
    assert instance.cluster_count == 2


def test_a_node_without_a_department_is_rejected(tmp_path: Path) -> None:
    edges = tmp_path / "edges.txt.gz"
    labels = tmp_path / "labels.txt.gz"
    edges.write_bytes(gzip.compress(b"0 1\n1 2\n"))
    labels.write_bytes(gzip.compress(b"0 1\n1 1\n"))
    with pytest.raises(ValueError, match="no department label"):
        build_email_instance("tiny", edges, labels)


# --- fetching and the hash pin -----------------------------------------------------------------


@pytest.fixture
def pinned_payload(monkeypatch: pytest.MonkeyPatch) -> bytes:
    payload = b"pinned bytes\n"
    monkeypatch.setitem(
        SOURCES,
        "tiny",
        (
            Source(
                url="https://example.invalid/tiny.txt",
                sha256=hashlib.sha256(payload).hexdigest(),
                filename="tiny.txt",
            ),
        ),
    )
    return payload


def test_fetch_downloads_and_verifies_an_absent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_payload: bytes
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_serving(pinned_payload))
    (path,) = fetch("tiny", tmp_path)
    assert path == tmp_path / "tiny.txt"
    assert path.read_bytes() == pinned_payload
    assert sha256_file(path) == SOURCES["tiny"][0].sha256


def test_a_verified_file_is_never_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_payload: bytes
) -> None:
    (tmp_path / "tiny.txt").write_bytes(pinned_payload)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_forbidden)
    assert fetch("tiny", tmp_path) == [tmp_path / "tiny.txt"]


def test_a_sha256_mismatch_raises_naming_the_file_and_deletes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_payload: bytes
) -> None:
    target = tmp_path / "tiny.txt"
    target.write_bytes(b"not the pinned bytes\n")
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_forbidden)
    with pytest.raises(ValueError, match=r"sha256 mismatch for tiny\.txt"):
        fetch("tiny", tmp_path)
    assert not target.exists()


def test_fetch_creates_the_data_directory_on_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_payload: bytes
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_serving(pinned_payload))
    directory = tmp_path / "nested" / "data"
    fetch("tiny", directory)
    assert directory.is_dir()


def test_an_unknown_graph_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown community graph"):
        load_community_graph("karate", tmp_path)


# --- the real graphs, only if they are already on disk -----------------------------------------


def _present(name: str) -> bool:
    return all((DEFAULT_DATA_DIR / source.filename).exists() for source in SOURCES[name])


@pytest.mark.parametrize(
    ("name", "expected_nodes", "expected_edges", "expected_clusters"),
    [
        ("polbooks", 105, 441, 3),
        ("football", 115, 613, 12),
        ("email_eu_core", 986, 16064, 42),
    ],
)
def test_the_published_graph_has_its_published_size_and_community_count(
    name: str, expected_nodes: int, expected_edges: int, expected_clusters: int
) -> None:
    if not _present(name):
        pytest.skip(f"{name} not pre-fetched into {DEFAULT_DATA_DIR}; tests never download")
    instance = load_community_graph(name)
    assert instance.graph.number_of_nodes() == expected_nodes
    assert instance.graph.number_of_edges() == expected_edges
    assert instance.cluster_count == expected_clusters
    assert nx.is_connected(instance.graph)
    assert instance.planted_labels.shape == (expected_nodes,)
    assert sorted(np.unique(instance.planted_labels)) == list(range(expected_clusters))
    assert instance.graph.graph["source"] == [
        {"url": s.url, "sha256": s.sha256, "filename": s.filename} for s in SOURCES[name]
    ]


def test_the_email_graph_records_the_component_it_was_cut_down_to() -> None:
    if not _present("email_eu_core"):
        pytest.skip(f"email_eu_core not pre-fetched into {DEFAULT_DATA_DIR}; tests never download")
    record = load_community_graph("email_eu_core").graph.graph["largest_component"]
    assert record["node_count_before"] == 1005
    assert record["node_count_after"] == 986
    assert record["edge_count_before"] == 16064
    assert record["labels_before"] == 42
    assert record["labels_after"] == 42


def test_rung3_names_are_exactly_the_pinned_sources() -> None:
    assert set(RUNG3_NAMES) == set(SOURCES)
