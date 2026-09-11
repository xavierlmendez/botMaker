// The picture of one A* expansion: what was popped, what it left on the frontier, what it priced.
//
// Three panels, because a search has three questions a reader asks at every step. What is waiting
// (the frontier, as bars sorted by bound, cheapest at the top — the order the heap will pop them
// in). What this expansion produced (the priced children, coloured by whether they entered the
// frontier, with the mechanism that dropped the ones that did not). And where the run is going (the
// bound of each expanded state against the expansion index, drawn only as far as the current frame,
// so the curve grows as the reader steps rather than giving the ending away).
//
// A frame with "goal": true is the last one: the goal was popped, not expanded, so it priced no
// children. It gets a banner instead of a children panel.

(function () {
  "use strict";

  var KIND = "astar_landmark";

  var INK = "#16181d";
  var MUTED = "#5d6470";
  var RULE = "#d8dce3";
  var FRONTIER = "#9aa6ba";
  var EXPANDED = "#2f5ea8";
  var PUSHED = "#2f7a4f";
  var PRUNED = "#b4462f";
  var INCUMBENT = "#a2761f";

  // Every drawn thing that stands for a quantity carries the key of the legend entry that explains
  // it. The whole attribute is spelled out rather than assembled from a key, so the drawing's keys
  // and the legend's can be read out of the two files and compared without running either.
  var TAG = {
    frontier: 'data-legend="frontier_bar"',
    expanded: 'data-legend="expanded_bar"',
    fresh: 'data-legend="new_bar"',
    child: 'data-legend="child_bar"',
    pruned: 'data-legend="pruned_bar"',
    curve: 'data-legend="bound_curve"',
    incumbent: 'data-legend="incumbent_line"',
    banner: 'data-legend="goal_banner"'
  };

  // A group carrying its legend key and the one line a reader gets by resting on it. <title> is
  // the SVG tooltip: no script, no positioning, and it survives the page being saved.
  function tagged(attribute, hover, body, draw) {
    return "<g " + attribute + ">" + draw.tag("title", {}, draw.escape(hover)) + body + "</g>";
  }

  function stateLabel(state) {
    return "(" + state.join(", ") + ")";
  }

  function stateKey(state) {
    return state.join(",");
  }

  // Bars are compared to each other, so the scale spans every bound in the whole run rather than
  // the ones in this frame: a bar that grows between frames then means the bound grew.
  function fraction(value, layout) {
    var span = layout.bound_max - layout.bound_min;
    return span > 0 ? (value - layout.bound_min) / span : 1;
  }

  function bar(box, row, name, value, colour, layout, draw, note) {
    var y = box.y + row * box.row_height;
    var left = box.x + box.label_width;
    var track = box.width - box.label_width - 76;
    var width = Math.max(2, 6 + fraction(value, layout) * track);
    return (
      draw.label(box.x, y + box.bar_height - 2, name, { fill: INK, "font-size": 11 }) +
      // The track is the run's whole bound range, so a short bar reads as a cheap state rather
      // than as a bar that failed to draw.
      draw.tag("rect", {
        x: left,
        y: y,
        width: track + 6,
        height: box.bar_height,
        rx: 2,
        fill: RULE,
        "fill-opacity": 0.45
      }) +
      draw.tag("rect", {
        x: left,
        y: y,
        width: width,
        height: box.bar_height,
        rx: 2,
        fill: colour,
        "fill-opacity": 0.85
      }) +
      draw.label(left + width + 6, y + box.bar_height - 2, draw.round(value) + (note || ""), {
        fill: MUTED,
        "font-size": 11
      })
    );
  }

  function panelFrontier(frame, layout, draw) {
    var box = layout.frontier;
    var pushedNow = {};
    frame.children.forEach(function (child) {
      if (child[2]) {
        pushedNow[stateKey(child[0])] = true;
      }
    });

    var parts = [
      draw.label(box.x, box.y - 12, "Frontier after this expansion (" + frame.frontier_size + ")", {
        fill: MUTED,
        "font-size": 11,
        "letter-spacing": "0.05em"
      }),
      tagged(
        TAG.expanded,
        "expanded state " +
          stateLabel(frame.expanded_state) +
          ": bound " +
          draw.round(frame.bound) +
          ", depth " +
          frame.expanded_state.length,
        bar(
          box,
          0,
          stateLabel(frame.expanded_state),
          frame.bound,
          EXPANDED,
          layout,
          draw,
          frame.goal
            ? frame.extras.superseded_goal
              ? "  incumbent, returned"
              : "  goal, popped"
            : "  expanded"
        ),
        draw
      )
    ];

    var shown = Math.min(frame.frontier.length, box.max_rows - 1);
    for (var i = 0; i < shown; i++) {
      var entry = frame.frontier[i];
      var justPushed = pushedNow[stateKey(entry[1])];
      parts.push(
        tagged(
          justPushed ? TAG.fresh : TAG.frontier,
          "state " +
            stateLabel(entry[1]) +
            ": bound " +
            draw.round(entry[0]) +
            ", depth " +
            entry[1].length,
          bar(
            box,
            i + 1,
            stateLabel(entry[1]),
            entry[0],
            justPushed ? PUSHED : FRONTIER,
            layout,
            draw,
            justPushed ? "  new" : ""
          ),
          draw
        )
      );
    }
    if (frame.frontier.length > shown) {
      parts.push(
        draw.label(
          box.x,
          box.y + (shown + 1) * box.row_height + box.bar_height - 2,
          "+ " + (frame.frontier.length - shown) + " more waiting, all at a larger bound",
          { fill: MUTED, "font-size": 11 }
        )
      );
    }
    return parts.join("");
  }

  function panelChildren(frame, layout, draw) {
    var box = layout.children;
    var reasons = {};
    (frame.extras.pruned || []).forEach(function (entry) {
      reasons[stateKey(entry[0])] = entry[2];
    });

    var pushed = frame.children.filter(function (child) {
      return child[2];
    }).length;
    var parts = [
      draw.label(
        box.x,
        box.y - 12,
        "Priced children (" + pushed + " pushed, " + (frame.children.length - pushed) + " pruned)",
        { fill: MUTED, "font-size": 11, "letter-spacing": "0.05em" }
      )
    ];

    if (frame.children.length === 0) {
      parts.push(
        draw.label(
          box.x,
          box.y + box.bar_height - 2,
          frame.goal ? "None: a goal is popped, never expanded." : "None: no successor was left.",
          { fill: MUTED, "font-size": 11 }
        )
      );
      return parts.join("");
    }

    var shown = Math.min(frame.children.length, box.max_rows);
    for (var i = 0; i < shown; i++) {
      var child = frame.children[i];
      var reason = reasons[stateKey(child[0])];
      parts.push(
        tagged(
          child[2] ? TAG.child : TAG.pruned,
          "child " +
            stateLabel(child[0]) +
            ": bound " +
            draw.round(child[1]) +
            (child[2] ? ", pushed" : ", pruned (" + (reason || "pruned") + ")"),
          bar(
            box,
            i,
            stateLabel(child[0]),
            child[1],
            child[2] ? PUSHED : PRUNED,
            layout,
            draw,
            child[2] ? "" : "  " + (reason || "pruned")
          ),
          draw
        )
      );
    }
    if (frame.children.length > shown) {
      parts.push(
        draw.label(
          box.x,
          box.y + shown * box.row_height + box.bar_height - 2,
          "+ " + (frame.children.length - shown) + " more priced",
          { fill: MUTED, "font-size": 11 }
        )
      );
    }
    return parts.join("");
  }

  function curvePoint(box, layout, expansions, value) {
    var span = Math.max(1, layout.expansion_max - 1);
    return {
      x: box.x + ((expansions - 1) / span) * box.width,
      y: box.y + box.height - fraction(value, layout) * box.height
    };
  }

  function panelCurve(frame, recording, layout, draw) {
    var box = layout.curve;
    var parts = [
      draw.label(box.x, box.y - 14, "Bound of the expanded state, by expansion", {
        fill: MUTED,
        "font-size": 11,
        "letter-spacing": "0.05em"
      }),
      draw.tag("rect", {
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        fill: "none",
        stroke: RULE
      }),
      draw.label(box.x - 6, box.y + 10, draw.round(layout.bound_max), {
        fill: MUTED,
        "font-size": 10,
        "text-anchor": "end"
      }),
      draw.label(box.x - 6, box.y + box.height, draw.round(layout.bound_min), {
        fill: MUTED,
        "font-size": 10,
        "text-anchor": "end"
      }),
      draw.label(box.x + box.width, box.y + box.height + 14, "expansion " + layout.expansion_max, {
        fill: MUTED,
        "font-size": 10,
        "text-anchor": "end"
      })
    ];

    var points = [];
    for (var i = 0; i <= frame.index; i++) {
      var each = recording.frames[i];
      var point = curvePoint(box, layout, each.expansions, each.bound);
      points.push(point.x.toFixed(2) + "," + point.y.toFixed(2));
    }
    var here = curvePoint(box, layout, frame.expansions, frame.bound);
    parts.push(
      tagged(
        TAG.curve,
        "expansion " + frame.expansions + ": bound " + draw.round(frame.bound),
        draw.tag("polyline", {
          points: points.join(" "),
          fill: "none",
          stroke: EXPANDED,
          "stroke-width": 1.6
        }) + draw.tag("circle", { cx: here.x, cy: here.y, r: 3.5, fill: EXPANDED }),
        draw
      )
    );

    if (frame.extras.incumbent !== undefined && frame.extras.incumbent !== null) {
      var line = curvePoint(box, layout, 1, frame.extras.incumbent);
      parts.push(
        tagged(
          TAG.incumbent,
          "incumbent " + draw.round(frame.extras.incumbent),
          draw.tag("line", {
            x1: box.x,
            y1: line.y,
            x2: box.x + box.width,
            y2: line.y,
            stroke: INCUMBENT,
            "stroke-width": 1.2,
            "stroke-dasharray": "5 4"
          }) +
            draw.label(
              box.x + 6,
              line.y - 4,
              "incumbent " +
                (frame.extras.incumbent_state
                  ? stateLabel(frame.extras.incumbent_state) + " "
                  : "") +
                draw.round(frame.extras.incumbent),
              { fill: INCUMBENT, "font-size": 10 }
            ),
          draw
        )
      );
    }
    return parts.join("");
  }

  // What a goal frame says. The cost is always the frame's own bound — at a goal state the bound is
  // the objective (D-23) — and never the result's cost, because a run under a tie tolerance can pop
  // one goal and return another: on that run the first goal frame's state cost what its bound says,
  // not what came back. "The search is done" is likewise reserved for the last frame, since a
  // superseded goal has a frame after it.
  function goalText(frame, recording, draw) {
    var last = frame.index === recording.frames.length - 1;
    var closing = last ? " — the search is done." : ".";
    if (frame.extras.superseded_goal) {
      return (
        "Returned the held incumbent " +
        stateLabel(frame.expanded_state) +
        " at cost " +
        draw.round(frame.bound) +
        "; the popped goal " +
        stateLabel(frame.extras.superseded_goal) +
        " at cost " +
        draw.round(frame.extras.superseded_cost) +
        " was set aside" +
        closing
      );
    }
    return (
      "Goal " +
      stateLabel(frame.expanded_state) +
      " popped at cost " +
      draw.round(frame.bound) +
      " after " +
      frame.expansions +
      " expansions" +
      closing
    );
  }

  function goalBanner(frame, recording, layout, draw) {
    if (!frame.goal) {
      return "";
    }
    var text = goalText(frame, recording, draw);
    return tagged(
      TAG.banner,
      text,
      draw.tag("rect", {
        x: layout.banner.x,
        y: layout.banner.y,
        width: layout.banner.width,
        height: layout.banner.height,
        rx: 4,
        fill: PUSHED,
        "fill-opacity": 0.12,
        stroke: PUSHED
      }) +
        draw.label(layout.banner.x + 12, layout.banner.y + 20, text, {
          fill: PUSHED,
          "font-size": 13,
          "font-weight": "600"
        }),
      draw
    );
  }

  window.walkthroughViews = window.walkthroughViews || {};
  window.walkthroughViews[KIND] = function (frame, recording, svg, layout, draw) {
    var parts = [
      draw.label(layout.frontier.x, 22, layout.subtitle, { fill: INK, "font-size": 12 }),
      goalBanner(frame, recording, layout, draw),
      panelFrontier(frame, layout, draw),
      panelChildren(frame, layout, draw),
      panelCurve(frame, recording, layout, draw)
    ];
    svg.innerHTML = parts.join("");
  };
})();
