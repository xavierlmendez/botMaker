"""The page: what it carries, and what it must not depend on.

A walkthrough is only useful if it survives being moved, so the two claims worth testing are that
the run is *in* the file (not referenced from it) and that nothing in the file points outward. The
third is the one that would be a bug rather than a regression: a caption is data, and no caption can
close the script element it rides in, whatever it happens to contain.
"""

import json
import re
from dataclasses import replace

import pytest

from mllib.visualization.html_renderer import render_walkthrough, write_walkthrough
from mllib.visualization.views import view_for
from mllib.visualization.views.astar_landmark import explain

# The controls the template owns; the stepper's JavaScript looks every one of them up by id.
STEPPER_IDS = ("previous", "next", "play", "speed", "slider", "position")

EMBEDDED = re.compile(
    r'<script id="(recording|layout)" type="application/json">(.*?)</script>', re.DOTALL
)


def _page(recording) -> str:
    return render_walkthrough(recording, view_for(recording))


def _embedded(page: str) -> dict[str, object]:
    return {name: json.loads(payload) for name, payload in EMBEDDED.findall(page)}


def test_the_page_carries_the_whole_recording_and_not_a_reference_to_it(fixture_recording):
    page = _page(fixture_recording)

    assert _embedded(page)["recording"] == fixture_recording.to_dict()


def test_the_page_carries_the_views_layout_beside_the_recording(fixture_recording):
    page = _page(fixture_recording)

    assert _embedded(page)["layout"] == view_for(fixture_recording).layout


def test_the_page_holds_every_stepper_control_the_walkthrough_looks_up(fixture_recording):
    page = _page(fixture_recording)

    for control in STEPPER_IDS:
        assert f'id="{control}"' in page, control
    assert 'id="caption"' in page
    assert 'id="raw"' in page
    assert 'id="drawing"' in page


def test_the_page_binds_the_keyboard_and_the_play_timer(fixture_recording):
    page = _page(fixture_recording)

    assert "ArrowLeft" in page
    assert "ArrowRight" in page
    assert 'event.key === " "' in page
    assert "setInterval" in page


def test_a_caption_cannot_close_the_script_element_it_rides_in(fixture_recording):
    frames = [dict(frame) for frame in fixture_recording.frames]
    frames[0]["caption"] = "Expanded </script><script>alert(1)</script> at bound 0.5."
    hostile = replace(fixture_recording, frames=frames)

    page = _page(hostile)

    # The pair is escaped on the way in and reads back as itself on the way out, so the document
    # in the page is still exactly the document on the object.
    assert "<\\/script>" in page
    assert _embedded(page)["recording"] == hostile.to_dict()
    assert "</script><script>alert(1)" not in page
    # Four script elements, no more: the two JSON blocks, the view and the stepper.
    assert page.count("</script>") == 4


def test_a_caption_holding_a_template_token_is_not_expanded_into_the_page(fixture_recording):
    frames = [dict(frame) for frame in fixture_recording.frames]
    frames[0]["caption"] = "Expanded {{VIEW_JS}} at bound 0.5."
    page = _page(replace(fixture_recording, frames=frames))

    assert _embedded(page)["recording"]["frames"][0]["caption"] == frames[0]["caption"]


def test_a_run_that_set_a_goal_aside_keeps_both_of_its_goal_frames(superseded_recording):
    # Under a tie tolerance the pruned engine pops one goal and returns another (D-29). The page
    # must carry both frames: dropping either would leave a walkthrough that either never reaches
    # the state that came back or never shows the one that was set aside.
    embedded = _embedded(_page(superseded_recording))["recording"]
    goal_frames = [frame for frame in embedded["frames"] if frame["goal"]]

    assert len(goal_frames) == 2
    popped, returned = goal_frames
    assert "superseded_goal" not in popped["extras"]
    assert returned["extras"]["superseded_goal"] == popped["expanded_state"]
    assert returned["extras"]["superseded_cost"] == popped["bound"]
    assert returned["bound"] == embedded["result"]["cost"] != popped["bound"]
    assert returned is embedded["frames"][-1]


def test_the_page_holds_every_panel_the_explanation_fills(fixture_recording):
    page = _page(fixture_recording)

    for panel in ("opening", "moments", "quantities", "narration", "legend", "ending"):
        assert f'id="{panel}"' in page, panel


def test_the_explanation_rides_in_the_page_with_a_sentence_per_frame(fixture_recording):
    # The page says nothing the recording did not say: the narration is embedded, not composed in
    # the browser, so this is the one place the two can be compared at all.
    page = _page(fixture_recording)
    embedded = _embedded(page)["layout"]["explain"]

    assert embedded["narration"] == list(explain(fixture_recording).narration)
    assert len(embedded["narration"]) == len(fixture_recording.frames)
    assert len(embedded["opening"]) == 4


def test_the_stepper_jumps_to_either_end_of_the_run(fixture_recording):
    page = _page(fixture_recording)

    assert 'event.key === "Home"' in page
    assert 'event.key === "End"' in page


def test_the_page_reaches_out_to_nothing(fixture_recording):
    page = _page(fixture_recording)

    assert "http://" not in page
    assert "https://" not in page
    assert "fetch(" not in page
    assert 'type="module"' not in page


def test_writing_the_page_returns_the_path_it_wrote(fixture_recording, tmp_path):
    written = write_walkthrough(
        fixture_recording, view_for(fixture_recording), tmp_path / "a" / "b.html"
    )

    assert written.read_text() == _page(fixture_recording)


def test_a_view_is_refused_a_recording_of_a_different_problem(fixture_recording):
    view = replace(view_for(fixture_recording), kind="two_hot_span")

    with pytest.raises(ValueError, match="two_hot_span"):
        render_walkthrough(fixture_recording, view)


def test_a_layout_without_an_explanation_still_renders_the_page_it_rendered_before(
    fixture_recording,
):
    # Phase 1's contract: a view that computes no explanation — an older recording's, or one
    # written before this layer existed — still gets a walkthrough. The panels ride in the
    # template with `hidden` on them and the stepper un-hides them only when `explain` is there,
    # so the fallback is the template's default rather than a second rendering path.
    view = view_for(fixture_recording)
    layout = {key: value for key, value in view.layout.items() if key != "explain"}
    page = render_walkthrough(fixture_recording, replace(view, layout=layout))

    assert page.startswith("<!doctype html>")
    assert "explain" not in _embedded(page)["layout"]
    for panel in ("opening", "moments", "quantities", "legend", "ending"):
        assert re.search(rf'id="{panel}"[^>]*\bhidden\b', page), panel
    assert 'id="narration"' in page
    assert 'id="caption"' in page
