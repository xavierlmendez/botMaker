// The picture of one optimizer step: what V looks like, and what it would round to if we stopped.
//
// Four panels, because the run has two halves and each of them has a "now" and a "so far". The pair
// graph is the answer as a reader would state it — which nodes ended up together — drawn on the
// instance's own coordinates so the roach looks like a ladder and a node stays where it was on the
// previous frame. The heatmap is the thing being optimised, V itself, one row per node and one
// column per spanning vector, diverging around zero because the sign of an entry is what decides
// which end of a pair a node lands on. Below them the two curves: the ridge training loss, which
// the optimizer actually descends, and Ê, the cut the rounding would buy, against the spectral
// floor Σλ that no cut can go below.
//
// A light frame has no derived numbers — the recorder only stored its training loss — so the two
// upper panels keep the last full frame's picture, greyed, and say so. Redrawing them empty would
// make the graph flicker between the frames that have one and the frames that do not.

(function () {
  "use strict";

  var KIND = "two_hot_span";

  var INK = "#16181d";
  var MUTED = "#5d6470";
  var RULE = "#d8dce3";
  var LOSS = "#2f5ea8";
  var CUT = "#2f7a4f";
  var FLOOR = "#a2761f";
  var POSITIVE = "#b4462f";
  var NEGATIVE = "#2f5ea8";

  // A small categorical palette, cycled: the component labels are arbitrary integers and a run can
  // produce more of them than any palette has colours. Cycling is honest here because the legend
  // states the count — the colours separate neighbours, they do not name them.
  var PALETTE = [
    "#2f5ea8",
    "#b4462f",
    "#2f7a4f",
    "#a2761f",
    "#6b4c9a",
    "#0f7b8a",
    "#a83f6e",
    "#5d6470"
  ];

  function colourOf(label) {
    return PALETTE[((label % PALETTE.length) + PALETTE.length) % PALETTE.length];
  }

  // Every drawn thing that stands for a quantity carries the key of the legend entry that explains
  // it. The whole attribute is spelled out rather than assembled from a key, so the drawing's keys
  // and the legend's can be read out of the two files and compared without running either.
  var TAG = {
    edge: 'data-legend="graph_edge"',
    pairEdge: 'data-legend="pair_edge"',
    pairChord: 'data-legend="pair_chord"',
    vertex: 'data-legend="vertex"',
    cell: 'data-legend="heat_cell"',
    loss: 'data-legend="loss_curve"',
    cut: 'data-legend="cut_curve"',
    floor: 'data-legend="floor_line"',
    banner: 'data-legend="end_banner"'
  };

  // A group carrying its legend key and the one line a reader gets by resting on it. <title> is
  // the SVG tooltip: no script, no positioning, and it survives the page being saved.
  function tagged(attribute, hover, body, draw) {
    return "<g " + attribute + ">" + draw.tag("title", {}, draw.escape(hover)) + body + "</g>";
  }

  // The instance's own edges, as a lookup, so a rounded pair can be drawn as what it is: a pair
  // the graph already joins, or a chord across it that the relaxation was free to choose.
  function edgeIndex(layout) {
    var joined = {};
    layout.edges.forEach(function (edge) {
      joined[Math.min(edge[0], edge[1]) + "," + Math.max(edge[0], edge[1])] = true;
    });
    return joined;
  }

  function isEdge(joined, pair) {
    return joined[Math.min(pair[0], pair[1]) + "," + Math.max(pair[0], pair[1])] === true;
  }

  // The last frame at or before this one that carried V and the derived numbers, or null if the
  // run has not produced one yet.
  function lastFull(recording, index) {
    for (var i = index; i >= 0; i--) {
      if (recording.frames[i].full) {
        return recording.frames[i];
      }
    }
    return null;
  }

  function panelFrame(box, heading, draw) {
    return (
      draw.label(box.x, box.y - 12, heading, {
        fill: MUTED,
        "font-size": 11,
        "letter-spacing": "0.05em"
      }) +
      draw.tag("rect", {
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        fill: "none",
        stroke: RULE
      })
    );
  }

  // ------------------------------------------------------------------------------------------
  // The pair graph.
  // ------------------------------------------------------------------------------------------

  function pairGraph(source, layout, draw) {
    var positions = layout.positions;
    var joined = edgeIndex(layout);
    var pairs = source.rounded_pairs || [];
    // A column index is only nameable when every column rounded to a pair: a column that rounds
    // to zero leaves no pair, and the frame does not say which column each surviving pair came
    // from, so a shorter list is drawn without column numbers rather than with guessed ones.
    var columnsKnown = pairs.length === layout.column_count;
    var parts = [];

    // The instance's own edges first and faint: they are the background the rounded pairs are read
    // against, and a rounded pair need not be one of them.
    layout.edges.forEach(function (edge) {
      var from = positions[edge[0]];
      var to = positions[edge[1]];
      if (!from || !to) {
        return;
      }
      parts.push(
        tagged(
          TAG.edge,
          "edge (" + edge[0] + ", " + edge[1] + ")",
          draw.tag("line", {
            x1: from[0],
            y1: from[1],
            x2: to[0],
            y2: to[1],
            stroke: RULE,
            "stroke-width": 1
          }),
          draw
        )
      );
    });

    pairs.forEach(function (pair, column) {
      var from = positions[pair[0]];
      var to = positions[pair[1]];
      if (!from || !to) {
        return;
      }
      var onEdge = isEdge(joined, pair);
      parts.push(
        tagged(
          onEdge ? TAG.pairEdge : TAG.pairChord,
          "pair (" +
            pair[0] +
            ", " +
            pair[1] +
            ")" +
            (columnsKnown ? ", column " + column : "") +
            (onEdge ? "" : " — not an edge of the graph"),
          draw.tag("line", {
            x1: from[0],
            y1: from[1],
            x2: to[0],
            y2: to[1],
            stroke: INK,
            "stroke-width": 2.2,
            "stroke-opacity": 0.75,
            // A pair the graph does not join is a pair the reader must not read as an edge.
            "stroke-dasharray": onEdge ? null : "6 3"
          }),
          draw
        )
      );
    });

    var labels = source.labels || [];
    positions.forEach(function (position, node) {
      var label = labels[node] === undefined ? 0 : labels[node];
      parts.push(
        tagged(
          TAG.vertex,
          "vertex " + node + ": component " + label,
          draw.tag("circle", {
            cx: position[0],
            cy: position[1],
            r: 6,
            fill: colourOf(label),
            stroke: "#ffffff",
            "stroke-width": 1.2
          }),
          draw
        )
      );
    });
    return parts.join("");
  }

  function pairGraphLegend(source, layout, draw) {
    var box = layout.graph;
    var count = source.component_count;
    var plural = count === 1 ? "" : "s";
    // Two short lines rather than one long one: a single line of this is wider than the panel and
    // runs under the heatmap's own footer, which is how two captions become one unreadable row.
    var parts = [
      draw.label(
        box.x,
        box.y + box.height + 16,
        count + " component" + plural + " of up to " + layout.component_max + " in this run",
        { fill: MUTED, "font-size": 11 }
      ),
      draw.label(
        box.x,
        box.y + box.height + 30,
        (source.rounded_pairs || []).length +
          " rounded pairs: solid on an edge, dashed off the graph",
        { fill: MUTED, "font-size": 11 }
      )
    ];
    var shown = Math.min(count, PALETTE.length);
    for (var i = 0; i < shown; i++) {
      parts.push(
        draw.tag("rect", {
          x: box.x + i * 16,
          y: box.y + box.height + 38,
          width: 11,
          height: 11,
          rx: 2,
          fill: colourOf(i)
        })
      );
    }
    return parts.join("");
  }

  // ------------------------------------------------------------------------------------------
  // V as a heatmap: n rows, r columns, diverging around zero on the run's largest magnitude.
  // ------------------------------------------------------------------------------------------

  function heatmap(source, layout, draw) {
    var box = layout.heatmap;
    var rows = source.spanning_set;
    var columns = layout.column_count || 1;
    var cell = Math.min(box.width / columns, box.height / Math.max(rows.length, 1));
    var left = box.x + (box.width - cell * columns) / 2;
    var top = box.y + (box.height - cell * rows.length) / 2;
    var parts = [];

    for (var row = 0; row < rows.length; row++) {
      for (var column = 0; column < rows[row].length; column++) {
        var value = rows[row][column];
        var weight = Math.min(1, Math.abs(value) / layout.value_max);
        parts.push(
          tagged(
            TAG.cell,
            "V[" + row + "," + column + "] = " + draw.round(value),
            draw.tag("rect", {
              x: left + column * cell,
              y: top + row * cell,
              width: cell,
              height: cell,
              fill: value >= 0 ? POSITIVE : NEGATIVE,
              // Zero is the page's own white, so the sign of an entry is its hue and the size of
              // it is how far the cell has travelled away from the background.
              "fill-opacity": weight
            }),
            draw
          )
        );
      }
    }
    // Split for the same reason the pair graph's footer is: one line of it overruns the panel.
    parts.push(
      draw.label(box.x, box.y + box.height + 16, rows.length + " nodes x " + columns + " columns", {
        fill: MUTED,
        "font-size": 11
      }),
      draw.label(
        box.x,
        box.y + box.height + 30,
        "red positive, blue negative, scaled to |V| " +
          draw.round(layout.value_max, 3) +
          " (the run's largest)",
        { fill: MUTED, "font-size": 11 }
      )
    );
    return parts.join("");
  }

  // ------------------------------------------------------------------------------------------
  // The two curves, drawn only as far as the current frame.
  // ------------------------------------------------------------------------------------------

  function pointOf(box, layout, step, value, low, high) {
    var span = high - low;
    var across = layout.step_count > 1 ? step / (layout.step_count - 1) : 0;
    var up = span > 0 ? (value - low) / span : 0.5;
    return { x: box.x + across * box.width, y: box.y + box.height - up * box.height };
  }

  function axisLabels(box, low, high, layout, draw) {
    return (
      draw.label(box.x - 6, box.y + 10, draw.round(high, 3), {
        fill: MUTED,
        "font-size": 10,
        "text-anchor": "end"
      }) +
      draw.label(box.x - 6, box.y + box.height, draw.round(low, 3), {
        fill: MUTED,
        "font-size": 10,
        "text-anchor": "end"
      }) +
      draw.label(box.x + box.width, box.y + box.height + 14, "step " + (layout.step_count - 1), {
        fill: MUTED,
        "font-size": 10,
        "text-anchor": "end"
      })
    );
  }

  function curve(box, points, colour, draw) {
    if (points.length === 0) {
      return "";
    }
    var text = points.map(function (point) {
      return point.x.toFixed(2) + "," + point.y.toFixed(2);
    });
    var last = points[points.length - 1];
    return (
      draw.tag("polyline", {
        points: text.join(" "),
        fill: "none",
        stroke: colour,
        "stroke-width": 1.6
      }) + draw.tag("circle", { cx: last.x, cy: last.y, r: 3.5, fill: colour })
    );
  }

  function lossCurve(frame, recording, layout, draw) {
    var box = layout.loss_curve;
    var points = [];
    for (var i = 0; i <= frame.index; i++) {
      var each = recording.frames[i];
      // The end frame restates the last step's training loss, so plotting it would put a second
      // point on top of the last one and make the curve look as if the run took a step it did not.
      if (each.end) {
        continue;
      }
      points.push(
        pointOf(box, layout, each.step, each.training_loss, layout.loss_min, layout.loss_max)
      );
    }
    return (
      panelFrame(box, "training loss (ridge), every step", draw) +
      axisLabels(box, layout.loss_min, layout.loss_max, layout, draw) +
      tagged(
        TAG.loss,
        "step " + frame.step + ": training loss " + draw.round(frame.training_loss),
        curve(box, points, LOSS, draw),
        draw
      )
    );
  }

  function cutCurve(frame, recording, layout, draw) {
    var box = layout.cut_curve;
    var points = [];
    var shown = null;
    for (var i = 0; i <= frame.index; i++) {
      var each = recording.frames[i];
      if (each.full) {
        points.push(
          pointOf(box, layout, each.step, each.rounded_cut, layout.cut_min, layout.cut_max)
        );
        shown = each;
      }
    }
    var floor = pointOf(box, layout, 0, layout.spectral_floor, layout.cut_min, layout.cut_max);
    return (
      panelFrame(box, "rounded cut Ê, on full frames", draw) +
      axisLabels(box, layout.cut_min, layout.cut_max, layout, draw) +
      tagged(
        TAG.floor,
        "spectral floor Σλ " + draw.round(layout.spectral_floor),
        draw.tag("line", {
          x1: box.x,
          y1: floor.y,
          x2: box.x + box.width,
          y2: floor.y,
          stroke: FLOOR,
          "stroke-width": 1.2,
          "stroke-dasharray": "5 4"
        }) +
          draw.label(box.x + 6, floor.y - 4, "Σλ " + draw.round(layout.spectral_floor, 4), {
            fill: FLOOR,
            "font-size": 10
          }),
        draw
      ) +
      tagged(
        TAG.cut,
        shown === null
          ? "no full frame yet"
          : "step " + shown.step + ": Ê " + draw.round(shown.rounded_cut),
        curve(box, points, CUT, draw),
        draw
      )
    );
  }

  // ------------------------------------------------------------------------------------------
  // The run's last frame.
  // ------------------------------------------------------------------------------------------

  function endBanner(frame, layout, draw) {
    if (!frame.end) {
      return "";
    }
    var plural = frame.component_count === 1 ? "" : "s";
    var text =
      "Run ended: E* " +
      draw.round(frame.relaxed_objective) +
      ", Ê " +
      draw.round(frame.rounded_cut) +
      ", Ê - Σλ " +
      draw.round(frame.rounded_cut_minus_floor) +
      ", " +
      frame.component_count +
      " component" +
      plural +
      ".";
    return tagged(
      TAG.banner,
      text,
      draw.tag("rect", {
        x: layout.banner.x,
        y: layout.banner.y,
        width: layout.banner.width,
        height: layout.banner.height,
        rx: 4,
        fill: CUT,
        "fill-opacity": 0.12,
        stroke: CUT
      }) +
        draw.label(layout.banner.x + 12, layout.banner.y + 20, text, {
          fill: CUT,
          "font-size": 13,
          "font-weight": "600"
        }),
      draw
    );
  }

  window.walkthroughViews = window.walkthroughViews || {};
  window.walkthroughViews[KIND] = function (frame, recording, svg, layout, draw) {
    var source = frame.full ? frame : lastFull(recording, frame.index);
    var upper;

    if (source === null) {
      upper = draw.label(
        layout.graph.x,
        layout.graph.y + 20,
        "light frame: training loss only — no full frame has been recorded yet.",
        { fill: MUTED, "font-size": 12 }
      );
    } else if (frame.full) {
      upper =
        pairGraph(source, layout, draw) +
        pairGraphLegend(source, layout, draw) +
        heatmap(source, layout, draw);
    } else {
      // The picture is carried, so it is greyed: what is on screen is the last full frame's
      // answer, not this one's, and this one has nothing to say beyond its training loss.
      upper =
        draw.tag(
          "g",
          { opacity: 0.35 },
          pairGraph(source, layout, draw) + pairGraphLegend(source, layout, draw)
        ) +
        draw.label(
          layout.heatmap.x,
          layout.heatmap.y + 20,
          "light frame: training loss only (picture carried from step " + source.step + ")",
          { fill: MUTED, "font-size": 12 }
        );
    }

    var parts = [
      draw.label(layout.graph.x, 22, layout.subtitle, { fill: INK, "font-size": 12 }),
      endBanner(frame, layout, draw),
      panelFrame(layout.graph, "Pair graph on the instance's own positions", draw),
      panelFrame(layout.heatmap, "V — one row per node, one column per spanning vector", draw),
      upper,
      lossCurve(frame, recording, layout, draw),
      cutCurve(frame, recording, layout, draw)
    ];
    svg.innerHTML = parts.join("");
  };
})();
