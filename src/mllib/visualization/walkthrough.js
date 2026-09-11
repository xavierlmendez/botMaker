// The stepper: everything a walkthrough does that does not depend on which problem was recorded.
//
// It reads two JSON blocks the renderer embedded (the recording and the view's layout), holds one
// number — which frame is showing — and re-renders on every change to it. The picture itself is
// none of its business: it calls the view function the page registered under the recording's kind
// and hands it the frame, the whole recording, the <svg> element and a few markup helpers.
//
// Deliberately plain: an IIFE, no modules, no fetch, no dependencies. The page has to work when it
// is opened from file://, where a module script and a fetch both fail, and the whole point of the
// artefact is that a reader can save it, mail it, and open it in five years.
//
// It also fills the explanation panels, if the layout carries one under "explain". Not a sentence
// of it is composed here: Python derived every line from the recording, and this only decides
// which of them is on screen. A layout without an "explain" leaves every one of those panels
// hidden, so an older recording still renders as the page it was rendered as before.

(function () {
  "use strict";

  var recording = JSON.parse(document.getElementById("recording").textContent);
  var layout = JSON.parse(document.getElementById("layout").textContent);
  var frames = recording.frames;
  var kind = recording.problem.kind;

  var svg = document.getElementById("drawing");
  var caption = document.getElementById("caption");
  var raw = document.getElementById("raw");
  var slider = document.getElementById("slider");
  var position = document.getElementById("position");
  var play = document.getElementById("play");
  var speed = document.getElementById("speed");

  var explanation = layout.explain || null;
  var openingPanel = document.getElementById("opening");
  var momentStrip = document.getElementById("moments");
  var quantityList = document.getElementById("quantities");
  var narration = document.getElementById("narration");
  var legendPanel = document.getElementById("legend");
  var endingPanel = document.getElementById("ending");
  var momentButtons = [];
  var quantityCells = [];

  // Markup helpers handed to the view. The view builds SVG as a string and assigns it to
  // svg.innerHTML, which parses in the SVG namespace: that avoids createElementNS, and with it the
  // one namespace URL that would otherwise be the only external-looking reference on the page.
  var draw = {
    escape: function (value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    },
    tag: function (name, attributes, body) {
      var markup = "<" + name;
      for (var key in attributes) {
        if (attributes[key] !== null && attributes[key] !== undefined) {
          markup += " " + key + '="' + draw.escape(attributes[key]) + '"';
        }
      }
      if (body === null || body === undefined) {
        return markup + " />";
      }
      return markup + ">" + body + "</" + name + ">";
    },
    label: function (x, y, value, attributes) {
      var merged = { x: x, y: y };
      for (var key in attributes || {}) {
        merged[key] = attributes[key];
      }
      return draw.tag("text", merged, draw.escape(value));
    },
    round: function (value, decimals) {
      return Number(value).toFixed(decimals === undefined ? 4 : decimals);
    }
  };

  // A glossary term is looked up by its lowercase spelling, so a sentence that starts one with a
  // capital and another mid-clause finds the same entry. "2-hot vector" answers to "2-hot" too,
  // which is how the word is actually written in a sentence.
  function glossaryIndex(glossary) {
    var bySpelling = {};
    Object.keys(glossary || {}).forEach(function (term) {
      bySpelling[term.toLowerCase()] = glossary[term];
      if (term === "2-hot vector") {
        bySpelling["2-hot"] = glossary[term];
      }
    });
    return bySpelling;
  }

  // Longest first, so "goal state" is marked as one term rather than as "state" with a word in
  // front of it. The prefix character is captured instead of using a look-behind, which browsers
  // took until recently to agree on and this page cannot afford to need.
  function termPattern(bySpelling) {
    var spellings = Object.keys(bySpelling).sort(function (a, b) {
      return b.length - a.length;
    });
    if (spellings.length === 0) {
      return null;
    }
    var alternation = spellings
      .map(function (spelling) {
        return spelling.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      })
      .join("|");
    return new RegExp("(^|[^A-Za-z0-9_])(" + alternation + ")(?![A-Za-z0-9_])", "gi");
  }

  function markTerms(text, bySpelling, pattern) {
    var safe = draw.escape(text);
    if (pattern === null) {
      return safe;
    }
    // Escaping first and matching second is what keeps the markup honest: by the time a term is
    // wrapped there is no other tag in the string for the wrapper to land inside of.
    pattern.lastIndex = 0;
    return safe.replace(pattern, function (whole, prefix, term) {
      var definition = bySpelling[term.toLowerCase()];
      if (definition === undefined) {
        return whole;
      }
      return prefix + '<abbr title="' + draw.escape(definition) + '">' + term + "</abbr>";
    });
  }

  var terms = glossaryIndex(explanation ? explanation.glossary : null);
  var termsPattern = termPattern(terms);

  function marked(text) {
    return markTerms(text, terms, termsPattern);
  }

  function fillLines(parent, lines) {
    lines.forEach(function (line) {
      var item = document.createElement("li");
      item.innerHTML = marked(line);
      parent.appendChild(item);
    });
  }

  // Roughly the width of a marker's label. Whether two markers collide is a question about
  // pixels, not about frames: the same pair of frames is comfortably apart on a wide page and on
  // top of each other on a phone, so the rule is measured against the track as it is drawn.
  var MOMENT_CROWDING_PX = 56;

  // Assign each marker a row. Markers are in frame order, so crowding is always with the last
  // marker left on row 1; a crowded one drops to row 2 and the next comes back up, which spreads
  // a run of near neighbours over two rows rather than hiding every one behind the last.
  function layoutMoments() {
    var width = momentStrip.clientWidth;
    var lastRowOnePosition = null;
    var staggered = false;
    momentButtons.forEach(function (button) {
      var position = Number(button.getAttribute("data-fraction")) * width;
      var crowded =
        lastRowOnePosition !== null && position - lastRowOnePosition < MOMENT_CROWDING_PX;
      if (crowded) {
        button.classList.add("row-2");
        staggered = true;
      } else {
        button.classList.remove("row-2");
        lastRowOnePosition = position;
      }
    });
    if (staggered) {
      momentStrip.classList.add("two-rows");
    } else {
      momentStrip.classList.remove("two-rows");
    }
  }

  function addMoment(moment) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "moment";
    button.setAttribute("data-frame", String(moment.frame_index));
    button.title = moment.reason;
    button.textContent = moment.label;
    // Positioned by frame fraction against the slider's own width; a run of one frame has no
    // fraction to take, so its single marker sits at the left end. The two ends are anchored by
    // their own edge rather than centred, or half of a marker at 0 % or 100 % would be drawn
    // outside the strip and clipped by the stepper.
    var span = frames.length - 1;
    var fraction = span > 0 ? moment.frame_index / span : 0;
    button.style.left = fraction * 100 + "%";
    // Kept on the element so the row assignment can be redone against a track that has changed
    // width without recomputing which frame this was.
    button.setAttribute("data-fraction", String(fraction));
    if (fraction === 0) {
      button.classList.add("at-start");
    } else if (fraction === 1) {
      button.classList.add("at-end");
    }
    button.addEventListener("click", function () {
      stop();
      show(moment.frame_index);
    });
    momentStrip.appendChild(button);
    momentButtons.push(button);
  }

  function addQuantity(quantity) {
    var pair = document.createElement("div");
    var term = document.createElement("dt");
    var value = document.createElement("dd");
    // The strip names the number, the hover names the glossary term it belongs to. Two numbers
    // can share a term — a frontier has a minimum and a size — and a strip printing the term
    // twice would label neither.
    term.textContent = quantity.label || quantity.term;
    term.title = quantity.term + " — " + quantity.definition;
    pair.appendChild(term);
    pair.appendChild(value);
    quantityList.appendChild(pair);
    quantityCells.push({ values: quantity.values, cell: value });
  }

  function addLegendEntry(entry) {
    var item = document.createElement("li");
    if (entry.swatch) {
      var swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = entry.swatch;
      item.appendChild(swatch);
    }
    var name = document.createElement("b");
    name.textContent = entry.name;
    item.appendChild(name);
    item.appendChild(document.createTextNode(" — " + entry.meaning));
    legendPanel.querySelector("ul").appendChild(item);
  }

  function fillExplanation() {
    if (explanation === null) {
      return;
    }
    fillLines(openingPanel.querySelector("ol"), explanation.opening);
    openingPanel.hidden = false;
    explanation.moments.forEach(addMoment);
    momentStrip.hidden = explanation.moments.length === 0;
    layoutMoments();
    // The rows are a function of the track's width, so they are reassigned when it changes.
    window.addEventListener("resize", layoutMoments);
    explanation.quantities.forEach(addQuantity);
    quantityList.hidden = explanation.quantities.length === 0;
    explanation.legend.forEach(addLegendEntry);
    legendPanel.hidden = explanation.legend.length === 0;
    fillLines(endingPanel.querySelector("ul"), explanation.ending);
    endingPanel.hidden = false;
  }

  function showExplanation() {
    if (explanation === null) {
      return;
    }
    var reason = null;
    momentButtons.forEach(function (button) {
      var here = Number(button.getAttribute("data-frame")) === index;
      if (here) {
        reason = button.title;
        button.classList.add("current");
      } else {
        button.classList.remove("current");
      }
    });
    var line = marked(explanation.narration[index] || "");
    if (reason !== null) {
      line += " <strong>" + marked(reason) + "</strong>";
    }
    narration.innerHTML = line;
    quantityCells.forEach(function (quantity) {
      var value = quantity.values[index];
      quantity.cell.textContent =
        value === null || value === undefined ? "—" : String(value);
    });
    if (index === frames.length - 1) {
      endingPanel.classList.add("reached");
    } else {
      endingPanel.classList.remove("reached");
    }
  }

  var index = 0;
  var timer = null;

  function show(next) {
    index = Math.max(0, Math.min(frames.length - 1, next));
    var frame = frames[index];
    var view = window.walkthroughViews[kind];
    if (view) {
      view(frame, recording, svg, layout, draw);
    } else {
      svg.innerHTML = draw.label(16, 32, "No view is registered for " + kind + ".");
    }
    caption.textContent = frame.caption;
    raw.textContent = JSON.stringify(frame, null, 2);
    slider.value = index;
    position.textContent = "frame " + (index + 1) + " of " + frames.length;
    document.getElementById("previous").disabled = index === 0;
    document.getElementById("next").disabled = index === frames.length - 1;
    showExplanation();
  }

  function step(delta) {
    if (index + delta < 0 || index + delta > frames.length - 1) {
      stop();
      return;
    }
    show(index + delta);
  }

  function stop() {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
    play.textContent = "Play";
  }

  function start() {
    stop();
    if (index === frames.length - 1) {
      show(0);
    }
    timer = window.setInterval(function () {
      step(1);
    }, Number(speed.value));
    play.textContent = "Pause";
  }

  function toggle() {
    if (timer === null) {
      start();
    } else {
      stop();
    }
  }

  function describe(fields) {
    var parts = [];
    Object.keys(fields).sort().forEach(function (key) {
      var value = fields[key];
      parts.push(key + " = " + (typeof value === "object" ? JSON.stringify(value) : String(value)));
    });
    return parts.join("  ·  ");
  }

  document.getElementById("previous").addEventListener("click", function () {
    stop();
    step(-1);
  });
  document.getElementById("next").addEventListener("click", function () {
    stop();
    step(1);
  });
  slider.addEventListener("input", function () {
    stop();
    show(Number(slider.value));
  });
  play.addEventListener("click", toggle);
  speed.addEventListener("change", function () {
    if (timer !== null) {
      start();
    }
  });

  document.addEventListener("keydown", function (event) {
    if (event.target && event.target.tagName === "SELECT") {
      return;
    }
    if (event.key === "ArrowLeft") {
      stop();
      step(-1);
    } else if (event.key === "ArrowRight") {
      stop();
      step(1);
    } else if (event.key === " " || event.key === "Spacebar") {
      event.preventDefault();
      toggle();
    } else if (event.key === "Home") {
      stop();
      show(0);
    } else if (event.key === "End") {
      stop();
      show(frames.length - 1);
    } else {
      return;
    }
    event.stopPropagation();
  });

  document.getElementById("problem-fields").textContent = describe(recording.problem);
  document.getElementById("configuration-fields").textContent = describe(recording.configuration);
  document.getElementById("result-fields").textContent = describe(recording.result);
  slider.max = frames.length - 1;
  fillExplanation();
  show(0);
})();
