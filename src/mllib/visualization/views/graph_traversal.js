// The picture of one visit of a graph traversal: where the walk has been, where it is, what waits.
//
// One graph, drawn in the same place on every frame, with four states a node can be in. Visited
// (filled: it is in the traversal order). Current (filled and ringed: the node this frame is
// about). Pending (outlined: it is in the queue or on the stack, waiting). Untouched (faint).
// The edges the walk has already crossed — both ends visited — are drawn heavier than the rest,
// so the shape of the walk accumulates on the page instead of having to be remembered.
//
// The reason the same view serves both algorithms is that the difference between them is entirely
// in the pending list, which is drawn in the container's own order: the queue's next node is at the
// head of the strip, the stack's next node is at its tail. Stepping the two pages side by side is
// the clearest statement of what "breadth-first" and "depth-first" actually mean.
//
// The last frame ("end": true) is about no node at all — it says whether the traversal stopped on
// its target or ran out of graph — so it draws the finished walk under a banner.

(function () {
  "use strict";

  var KIND = "graph_traversal";

  var INK = "#16181d";
  var MUTED = "#5d6470";
  var RULE = "#d8dce3";
  var UNTOUCHED = "#eef0f4";
  var VISITED = "#2f5ea8";
  var CURRENT = "#b4462f";
  var PENDING = "#a2761f";
  var FOUND = "#2f7a4f";

  // Every drawn thing that stands for something carries the key of the legend entry that explains
  // it. The whole attribute is spelled out rather than assembled from a key, so the drawing's keys
  // and the legend's can be read out of the two files and compared without running either.
  var TAG = {
    edge: 'data-legend="edge"',
    crossed: 'data-legend="crossed_edge"',
    visited: 'data-legend="node_visited"',
    current: 'data-legend="node_current"',
    pending: 'data-legend="node_pending"',
    untouched: 'data-legend="node_untouched"',
    strip: 'data-legend="pending_strip"',
    banner: 'data-legend="banner"'
  };

  function escapeText(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function tag(name, attributes, body) {
    var markup = "<" + name;
    for (var key in attributes) {
      if (attributes[key] !== null && attributes[key] !== undefined) {
        markup += " " + key + '="' + escapeText(attributes[key]) + '"';
      }
    }
    if (body === null || body === undefined) {
      return markup + " />";
    }
    return markup + ">" + body + "</" + name + ">";
  }

  function label(x, y, value, attributes) {
    var merged = { x: x, y: y };
    for (var key in attributes || {}) {
      merged[key] = attributes[key];
    }
    return tag("text", merged, escapeText(value));
  }

  // A group carrying its legend key and the one line a reader gets by resting on it. <title> is
  // the SVG tooltip: no script, no positioning, and it survives the page being saved.
  function tagged(attribute, hover, body) {
    return "<g " + attribute + ">" + tag("title", {}, escapeText(hover)) + body + "</g>";
  }

  function membership(list) {
    var seen = {};
    for (var i = 0; i < (list || []).length; i += 1) {
      seen[list[i]] = true;
    }
    return seen;
  }

  function placement(layout) {
    var byId = {};
    for (var i = 0; i < layout.nodes.length; i += 1) {
      byId[layout.nodes[i].id] = layout.nodes[i];
    }
    return byId;
  }

  function edges(layout, node, walked) {
    var markup = "";
    for (var i = 0; i < layout.edges.length; i += 1) {
      var u = node[layout.edges[i][0]];
      var v = node[layout.edges[i][1]];
      if (!u || !v) {
        continue;
      }
      var crossed = walked[layout.edges[i][0]] && walked[layout.edges[i][1]];
      var joins = "edge " + layout.edges[i][0] + "–" + layout.edges[i][1];
      markup += tagged(
        crossed ? TAG.crossed : TAG.edge,
        crossed ? joins + ": both ends visited" : joins,
        tag("line", {
          x1: u.x,
          y1: u.y,
          x2: v.x,
          y2: v.y,
          stroke: crossed ? VISITED : RULE,
          "stroke-width": crossed ? 2.5 : 1.2,
          "stroke-opacity": crossed ? 0.75 : 1
        })
      );
    }
    return markup;
  }

  // What a reader gets by resting on a node: which of the four states it is in, and the one
  // number that goes with that state — the visit it was taken at, or the depth it waits at.
  function nodeHover(id, visit, pendingDepth) {
    if (visit) {
      return "node " + id + ": visited at " + visit;
    }
    if (pendingDepth !== undefined) {
      return "node " + id + ": pending at depth " + pendingDepth;
    }
    return "node " + id + ": untouched";
  }

  function nodes(layout, frame, visitOf, pendingDepths) {
    var markup = "";
    for (var i = 0; i < layout.nodes.length; i += 1) {
      var placed = layout.nodes[i];
      var isCurrent = frame.node_id !== null && placed.id === frame.node_id;
      var fill = UNTOUCHED;
      var stroke = RULE;
      var strokeWidth = 1.2;
      var key = TAG.untouched;
      if (isCurrent) {
        fill = CURRENT;
        stroke = INK;
        strokeWidth = 3;
        key = TAG.current;
      } else if (visitOf[placed.id]) {
        fill = VISITED;
        stroke = VISITED;
        key = TAG.visited;
      } else if (pendingDepths[placed.id] !== undefined) {
        fill = "#ffffff";
        stroke = PENDING;
        strokeWidth = 2.5;
        key = TAG.pending;
      }
      markup += tagged(
        key,
        nodeHover(placed.id, visitOf[placed.id], pendingDepths[placed.id]),
        tag("circle", {
          cx: placed.x,
          cy: placed.y,
          r: layout.node_radius,
          fill: fill,
          stroke: stroke,
          "stroke-width": strokeWidth
        }) +
          label(placed.x, placed.y + 4, placed.id, {
            "text-anchor": "middle",
            "font-size": 11,
            fill: fill === UNTOUCHED || fill === "#ffffff" ? MUTED : "#ffffff"
          })
      );
    }
    return markup;
  }

  function pendingStrip(layout, frame) {
    var pending = frame.pending || [];
    if (frame.end) {
      return "";
    }
    var heading =
      layout.algorithm === "dfs"
        ? "Stack (bottom first; the last one pops next)"
        : "Queue (front first; the first one comes out next)";
    var markup = label(16, layout.height - 44, heading, { "font-size": 11, fill: MUTED });
    if (!pending.length) {
      return markup + label(16, layout.height - 26, "empty", { "font-size": 12, fill: MUTED });
    }
    var x = 16;
    for (var i = 0; i < pending.length && x < layout.width - 60; i += 1) {
      var text = pending[i][0] + " @" + pending[i][1];
      var width = 14 + text.length * 6.6;
      markup += tagged(
        TAG.strip,
        "node " +
          pending[i][0] +
          ": pending at depth " +
          pending[i][1] +
          ", " +
          (i + 1) +
          " of " +
          pending.length +
          " in the " +
          (layout.algorithm === "dfs" ? "stack" : "queue"),
        tag("rect", {
          x: x,
          y: layout.height - 40,
          width: width,
          height: 20,
          rx: 4,
          fill: "#ffffff",
          stroke: PENDING,
          "stroke-width": 1.4
        }) +
          label(x + width / 2, layout.height - 26, text, {
            "text-anchor": "middle",
            "font-size": 11,
            fill: INK
          })
      );
      x += width + 6;
    }
    if (x >= layout.width - 60) {
      markup += label(x, layout.height - 26, "…", { "font-size": 12, fill: MUTED });
    }
    return markup;
  }

  function banner(layout, frame) {
    var box = layout.banner;
    var text;
    var colour;
    if (frame.end) {
      text = frame.found
        ? "Ended: target found after " + frame.traversal_order.length + " visits."
        : "Ended: the traversal ran out of nodes after " +
          frame.traversal_order.length +
          " visits; target not found.";
      colour = frame.found ? FOUND : CURRENT;
    } else {
      text =
        "Visit " +
        frame.traversal_order.length +
        " — node " +
        frame.node_id +
        " at depth " +
        frame.depth;
      colour = MUTED;
    }
    return tagged(
      TAG.banner,
      text,
      tag("rect", {
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        rx: 5,
        fill: frame.end ? "#f3f6fa" : "#ffffff",
        stroke: frame.end ? colour : RULE,
        "stroke-width": frame.end ? 2 : 1
      }) +
        label(box.x + 10, box.y + 20, text, { "font-size": 13, fill: frame.end ? colour : INK })
    );
  }

  window.walkthroughViews = window.walkthroughViews || {};
  window.walkthroughViews[KIND] = function (frame, recording, svg, layout) {
    var placed = placement(layout);
    var walked = membership(frame.traversal_order);
    // The visit a node was taken at, counting from one, so its hover can name it.
    var visitOf = {};
    for (var v = 0; v < (frame.traversal_order || []).length; v += 1) {
      visitOf[frame.traversal_order[v]] = v + 1;
    }
    var pendingDepths = {};
    for (var i = 0; i < (frame.pending || []).length; i += 1) {
      pendingDepths[frame.pending[i][0]] = frame.pending[i][1];
    }

    svg.innerHTML =
      label(16, 20, layout.subtitle, { "font-size": 12, fill: MUTED }) +
      banner(layout, frame) +
      edges(layout, placed, walked) +
      nodes(layout, frame, visitOf, pendingDepths) +
      pendingStrip(layout, frame) +
      label(
        layout.legend.x,
        layout.legend.y,
        "filled: visited · ringed: current · outlined: pending",
        { "font-size": 11, fill: MUTED }
      );
  };
})();
