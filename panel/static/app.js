/* slerm panel -- vanilla JS, no build step. */
(function () {
  "use strict";

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    attrs = attrs || {};
    Object.keys(attrs).forEach(function (k) {
      if (k === "class") node.className = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k === "text") node.textContent = attrs[k];
      else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) {
      if (c === null || c === undefined) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // Make the live node `a` look like the freshly built `b` without replacing
  // what has not changed, so a poll moves bars and numbers in place (and CSS
  // transitions run) instead of swapping the whole block in. Buttons are
  // always swapped: their click handlers close over the poll that built them.
  function morph(a, b) {
    if (a.nodeType !== b.nodeType || a.nodeName !== b.nodeName || a.nodeName === "BUTTON") {
      a.replaceWith(b); return;
    }
    if (a.nodeType === 3) { if (a.data !== b.data) a.data = b.data; return; }
    if (a.nodeType !== 1) { a.replaceWith(b); return; }
    Array.prototype.slice.call(a.attributes).forEach(function (at) {
      if (!b.hasAttribute(at.name)) a.removeAttribute(at.name);
    });
    Array.prototype.forEach.call(b.attributes, function (at) {
      if (a.getAttribute(at.name) !== at.value) a.setAttribute(at.name, at.value);
    });
    morphInto(a, Array.prototype.slice.call(b.childNodes));
  }

  function morphInto(host, kids) {
    var old = Array.prototype.slice.call(host.childNodes);
    if (old.length !== kids.length) {
      clear(host);
      kids.forEach(function (k) { host.appendChild(k); });
      return;
    }
    kids.forEach(function (k, i) { morph(old[i], k); });
  }

  function bytes(n) {
    if (n === null || n === undefined) return "—";
    var units = ["B", "KB", "MB", "GB", "TB"], i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return (i === 0 ? n : n.toFixed(n < 10 ? 1 : 0)) + " " + units[i];
  }

  function dur(sec) {
    if (sec === null || sec === undefined) return "—";
    sec = Math.max(0, Math.floor(sec));
    var d = Math.floor(sec / 86400), h = Math.floor(sec % 86400 / 3600),
        m = Math.floor(sec % 3600 / 60), s = sec % 60;
    if (d) return d + "d " + h + "h";
    if (h) return h + "h " + m + "m";
    if (m) return m + "m " + s + "s";
    return s + "s";
  }

  function when(ts) {
    if (!ts) return "—";
    var d = new Date(ts * 1000);
    return d.toLocaleString(undefined, { month: "short", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  }

  // Scan times come off the wire as UTC epoch seconds and are always rendered
  // in the panel's configured zone (not the viewer's), so the label matches the
  // timestamp matplotlib burns into each PNG.
  function scanTime(ts, withDate) {
    if (ts === null || ts === undefined) return "—";
    var opts = { timeZone: state.tz, hour: "2-digit", minute: "2-digit",
                 second: "2-digit", hour12: false, timeZoneName: "short" };
    if (withDate !== false) {
      opts.year = "numeric"; opts.month = "2-digit"; opts.day = "2-digit";
    }
    try {
      return new Date(ts * 1000).toLocaleString("en-CA", opts);
    } catch (e) {
      return new Date(ts * 1000).toISOString();
    }
  }

  function ago(ts) {
    if (!ts) return "—";
    return dur(Date.now() / 1000 - ts) + " ago";
  }

  function api(path, opts) {
    opts = opts || {};
    if (opts.body && typeof opts.body !== "string") {
      opts.body = JSON.stringify(opts.body);
      opts.headers = { "Content-Type": "application/json" };
    }
    return fetch(path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (!r.ok) throw new Error(data.error || (r.status + " " + r.statusText));
        return data;
      });
    });
  }

  function bar(pct, cls) {
    var fill = el("i", { class: cls || "" });
    fill.style.width = Math.max(0, Math.min(100, pct || 0)) + "%";
    return el("div", { class: "bar" }, [fill]);
  }

  function barRow(label, right, pct, cls) {
    return el("div", { class: "bar-row" }, [
      el("div", { class: "bar-label" }, [
        el("span", { text: label }), el("span", { text: right })
      ]),
      bar(pct, cls)
    ]);
  }

  function loadClass(pct) {
    if (pct === null || pct === undefined) return "";
    if (pct >= 90) return "bad";
    if (pct >= 70) return "warn";
    return "ok";
  }

  function table(headers, rows, emptyText) {
    var t = el("table");
    var thead = el("thead", {}, [el("tr", {}, headers.map(function (h) {
      return el("th", { text: h });
    }))]);
    t.appendChild(thead);
    var tbody = el("tbody");
    if (!rows.length) {
      tbody.appendChild(el("tr", {}, [
        el("td", { colspan: headers.length, class: "empty",
                   text: emptyText || "nothing here" })
      ]));
    }
    rows.forEach(function (cells) {
      tbody.appendChild(el("tr", {}, cells.map(function (c) {
        return (c && c.nodeType) ? el("td", {}, [c]) : el("td", { text: c === null || c === undefined ? "—" : String(c) });
      })));
    });
    t.appendChild(tbody);
    return t;
  }

  function statePill(state) {
    var s = String(state || "").toUpperCase(), cls = "idle";
    if (["IDLE", "COMPLETED"].indexOf(s) >= 0) cls = "ok";
    else if (["ALLOCATED", "MIXED", "RUNNING", "COMPLETING", "SUBMITTED", "LISTING", "SUBMITTING"].indexOf(s) >= 0) cls = "busy";
    else if (["PENDING", "CONFIGURING", "SUSPENDED"].indexOf(s) >= 0) cls = "warn";
    // CANCELLED stays neutral: the orchestrator cancels router/worker jobs as
    // routine, and a column of red for that trained the eye to ignore red.
    else if (["DOWN", "DRAIN", "DRAINED", "DRAINING", "FAIL", "FAILED", "NOT_RESPONDING", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "ERROR"].indexOf(s) >= 0) cls = "bad";
    return el("span", { class: "pill " + cls, text: s || "?" });
  }

  function notice(target, message, kind) {
    clear(target);
    if (!message) return;
    target.appendChild(el("div", { class: "notice " + (kind || "err"), text: message }));
  }

  // ------------------------------------------------------------------ state
  var state = {
    tab: "cluster",
    projects: [],
    views: {},
    historyDays: 3,
    busy: false,
    tz: "UTC",
    poll: { tick: 1000, cluster: 1000, history: 20000, project: 1000,
            runProgress: 2000, frames: 5000, gif: 8000, schedules: 3000,
            wazuh: 2000, ai: 5000, aiLive: 1500 },
    aiWindow: "24h",
    aiLive: {},       // job id -> the latest /api/ai/live entry for that server
    nodeRows: {},     // node name -> its row element (reconciled, never rebuilt)
    phaseSeen: {},    // node name -> {phase, since} as first seen by this page
    probing: {},      // node name -> when its probe button was pressed
    last: {}
  };

  // Each target is due on its own interval off a shared 1s tick, so the live
  // views stay current without dragging sacct and the frame index along with
  // them. Returns true (and stamps) only when the interval has elapsed.
  function due(key, interval) {
    var now = Date.now();
    if (state.last[key] && now - state.last[key] < interval) return false;
    state.last[key] = now;
    return true;
  }

  function dueReset() { state.last = {}; }

  // ---------------------------------------------------------------- cluster
  // The cluster tab re-renders every second, so everything here reconciles in
  // place: the stat strip only flashes a figure that changed, and node rows are
  // keyed by name so an open row stays open and nothing replays its entrance.

  function esc(s) {
    return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function flashIfChanged(node, html) {
    if (node.innerHTML === html) return;
    var had = node.innerHTML !== "";
    node.innerHTML = html;
    if (had) { node.classList.remove("flash"); void node.offsetWidth; node.classList.add("flash"); }
  }

  // One stat in a strip of big numbers. Built once per label, then updated.
  function statTile(host, key, label, valueHtml, sub) {
    var t = host.querySelector('[data-stat="' + key + '"]');
    if (!t) {
      t = el("div", { class: "tile", "data-stat": key }, [
        el("div", { class: "value" }), el("div", { class: "label" }), el("div", { class: "sub" })]);
      host.appendChild(t);
    }
    flashIfChanged(t.querySelector(".value"), valueHtml);
    t.querySelector(".label").textContent = label;
    t.querySelector(".sub").textContent = sub || "";
  }

  // "slerm-node[01-02,05]" -> ["slerm-node01", "slerm-node02", "slerm-node05"]
  function expandHosts(text) {
    var out = [];
    String(text || "").replace(/([^,\[]+)(?:\[([^\]]+)\])?/g, function (_, pre, range) {
      if (!range) { if (pre) out.push(pre); return; }
      range.split(",").forEach(function (r) {
        var ab = r.split("-"), w = ab[0].length, a = parseInt(ab[0], 10), b = parseInt(ab[1] || ab[0], 10);
        for (var i = a; i <= b; i++) out.push(pre + String(i).padStart(w, "0"));
      });
    });
    return out;
  }

  var RUNNING_PHASES = { busy: 1, idle: 1, outside: 1 };
  var PHASE_VIEW = {
    busy:         { dot: "busy pulse", word: "busy", c: "ph-busy" },
    idle:         { dot: "ok", word: "idle", c: "ph-idle" },
    outside:      { dot: "violet pulse slow", word: "in use by you", c: "ph-out" },
    sleeping:     { dot: "off", word: "asleep", c: "ph-off" },
    waking:       { dot: "warn pulse", word: "waking", c: "ph-trans", step: 0 },
    booting:      { dot: "warn pulse", word: "booting", c: "ph-trans", step: 1 },
    initialising: { dot: "warn pulse", word: "joining Slurm", c: "ph-trans", step: 2 },
    stopping:     { dot: "warn", word: "shutting down", c: "ph-trans" },
    down:         { dot: "bad", word: "down", c: "ph-bad" }
  };
  var CHEV = '<svg class="chev" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5"/></svg>';

  // Without the orchestrator there is no power lifecycle, only Slurm's view.
  function nodePhase(n) {
    var pw = n.power;
    if (pw && pw.phase) {
      if (pw.outside && RUNNING_PHASES[pw.phase]) return "outside";
      return pw.phase;
    }
    if (!n.healthy) return "down";
    return n.alloc_cpus ? "busy" : "idle";
  }

  function nodeRole(n) {
    var gpus = n.gres.filter(function (g) { return g.type === "gpu"; });
    var cores = n.cpus + " cores";
    if (n.name === "slerm-node05") return "deadlock · " + cores;
    if (gpus.length) return "GPU · " + gpus.map(function (g) { return g.model || "gpu"; }).join("/");
    return "CPU · " + cores;
  }

  function meterHTML(label, right, pct, cls, key) {
    return '<div class="bar-row"' + (key ? ' data-k="' + key + '"' : "") + '><div class="bar-label"><span>' + esc(label) +
      '</span><span>' + esc(right) + '</span></div><div class="bar"><i class="' + (cls || "") +
      '" style="width:' + Math.max(0, Math.min(100, pct || 0)) + '%"></i></div></div>';
  }

  function stepsHTML(labels, cur, down) {
    return '<div class="steps' + (down ? " down" : "") + '">' + labels.map(function (l, i) {
      return '<div class="step' + (i < cur ? " done" : i === cur ? " cur" : "") + '"><div class="b"></div>' + l + "</div>";
    }).join("") + "</div>";
  }

  function nodeWork(n, jobs, orch) {
    if (!jobs.length) return '<div class="job free">no jobs</div><div class="model" style="color:var(--muted)">free for batch</div>';
    var j = jobs[0], more = jobs.length > 1 ? " +" + (jobs.length - 1) : "";
    var sub = j.partition;
    var router = orch && orch.router;
    if (j.name === "llm-router") {
      sub = router && router.node === n.name && router.loaded && router.loaded.length
        ? router.loaded.join(", ") : "router · no model loaded";
    } else if (/^llm-rpc/.test(j.name)) {
      sub = "rpc backend" + (router && router.node ? " for " + router.node.replace("slerm-", "") : "");
    } else if (j.name === "llm-stage") {
      var st = ((orch && orch.staged) || []).filter(function (s) {
        return s.nodes && s.nodes[n.name] && s.nodes[n.name].state !== "done";
      }).map(function (s) { return s.model; });
      sub = "staging " + (st.length ? st.join(", ") : "models");
    }
    return '<div class="job"><span class="num">' + esc(j.job_id) + "</span>" + esc(j.name) + esc(more) +
      '</div><div class="model">' + esc(sub) + "</div>";
  }

  function nodeLife(n, phase) {
    var pw = n.power || {}, v = PHASE_VIEW[phase], detail = pw.detail || "";
    if (phase === "sleeping") {
      if (!pw.can_wake && detail && detail !== "powered off") {
        return '<div class="life-txt blocked">Can’t wake: ' + esc(detail) + '</div><span class="el" data-age></span>';
      }
      return '<div class="life-txt">Powered off <span class="dim">· wakes on the next request that needs it</span></div><span class="el" data-age></span>';
    }
    if (phase === "stopping") {
      return stepsHTML(["Drain", "Shut down", "Off"], 1, true) + '<div class="life-txt"><span class="dim">' + esc(detail) + '</span></div><span class="el" data-age></span>';
    }
    if (phase === "down") {
      return '<div class="life-txt bad">Slurm ' + esc((n.states || [n.state]).join("+").toLowerCase()) +
        (n.reason ? " · " + esc(n.reason) : "") + (n.reason_by ? ' <span class="dim">· set by ' + esc(n.reason_by) + "</span>" : "") + "</div>";
    }
    return stepsHTML(["Power on", "Boot", "Join Slurm", "Ready"], v.step) +
      '<div class="life-txt"><span class="dim">' + esc(detail) + '</span></div><span class="el" data-age></span>';
  }

  function nodeMain(n, phase, jobs, orch) {
    var v = PHASE_VIEW[phase] || PHASE_VIEW.idle;
    var html = '<div class="who"><span class="dot ' + v.dot + '"></span><div><div class="name">' + esc(n.name) +
      '</div><div class="sub">' + esc(nodeRole(n)) + ' · <b class="' + v.c + '">' + v.word + "</b></div></div></div>";
    if (!RUNNING_PHASES[phase]) return html + '<div class="life">' + nodeLife(n, phase) + "</div>" + CHEV;
    var work = nodeWork(n, jobs, orch);
    if (phase === "outside") work += '<div class="outside-note">' + esc(n.power.outside) + " <span>· slerm capped, won’t power off</span></div>";
    var hasGpu = n.gres.some(function (g) { return g.type === "gpu"; });
    return html + '<div class="work">' + work + "</div>" +
      '<div data-k="cpu"></div><div data-k="mem"></div>' +
      '<div class="gpu-cell" data-k="gpu">' + (hasGpu ? "" : '<span class="gpu-none">no GPU</span>') + "</div>" + CHEV;
  }

  // Values that move every poll: patched without rebuilding, so bars ease.
  function patchMeter(host, label, right, pct, cls) {
    if (!host) return;
    if (!host.firstChild) { host.innerHTML = meterHTML(label, right, pct, cls); return; }
    host.querySelector(".bar-label span:last-child").textContent = right;
    var i = host.querySelector(".bar > i");
    i.className = cls || ""; i.style.width = Math.max(0, Math.min(100, pct || 0)) + "%";
  }

  function patchGpu(row, n) {
    var cell = row.querySelector('[data-k="gpu"]');
    if (!cell || !n.gres.some(function (g) { return g.type === "gpu"; })) return;
    var live = n.gpu_live, gpus = (live && live.gpus) || [];
    if (!gpus.length) {
      if (cell.dataset.mode !== "none") {
        cell.dataset.mode = "none";
        cell.innerHTML = '<div class="bar-label"><span>' + esc(nodeRole(n).replace("GPU · ", "")) +
          '</span><span>—</span></div><div class="chips"><span>no live feed · probe in details</span></div>';
      }
      return;
    }
    // Every card the feed lists (node04 has two P100s; this showed only the first until 2026-10-21), then the
    // node's own counters when the feed carries them (gpu-publish.py host_stats).
    var mode = "live" + gpus.length;
    if (cell.dataset.mode !== mode) {
      cell.dataset.mode = mode;
      cell.innerHTML = gpus.map(function () {
        return '<div class="bar-row"><div class="bar-label"><span class="gpu-name"></span><span data-t></span></div>' +
          '<div class="bar"><i></i></div></div><div class="chips"><span data-v></span><span data-c></span><span data-w></span></div>';
      }).join("") + '<div class="chips" data-h></div>';
    }
    var rows = cell.querySelectorAll(".bar-row"), chips = cell.querySelectorAll(".chips:not([data-h])");
    gpus.forEach(function (g, k) {
      var r = rows[k], c = chips[k];
      r.querySelector(".gpu-name").textContent = (g.name || "GPU").replace(/^NVIDIA (GeForce )?/, "") +
        (gpus.length > 1 ? " #" + k : "");
      r.querySelector("[data-t]").textContent = (g.util_pct !== null ? g.util_pct : "–") + "% util";
      var i = r.querySelector(".bar > i");
      i.style.width = Math.max(g.util_pct || 0, 2) + "%";
      i.className = (g.util_pct || 0) > 5 ? "live" : "";
      c.querySelector("[data-v]").textContent = g.mem_total_mb
        ? (g.mem_used_mb / 1024).toFixed(1) + "/" + (g.mem_total_mb / 1024).toFixed(0) + " GB" : "";
      c.querySelector("[data-c]").textContent = g.temp_c !== null ? g.temp_c + "°C" : "";
      c.querySelector("[data-w]").textContent = g.power_w ? Math.round(g.power_w) + " W" : "";
    });
    var h = live.host_stats, hc = cell.querySelector("[data-h]");
    hc.textContent = h && h.cpu_pct !== undefined
      ? "cpu " + Math.round(h.cpu_pct) + "% busy · disk " + h.disk_read_mb_s.toFixed(1) + " MB/s read · net " +
        h.net_rx_mb_s.toFixed(1) + " MB/s in"
      : "";
  }

  function nodeDrawer(n, probes) {
    var kv = function (k, v) { return '<div class="kv"><div class="k">' + k + '</div><div class="v">' + esc(v) + "</div></div>"; };
    var pw = n.power || {};
    var html = kv("Partitions", n.partitions.join(", ")) +
      kv("Slurm", (n.states || [n.state]).join("+").toLowerCase() + " · slurmd " + (n.version || "?")) +
      kv("Layout", n.sockets + " socket · " + n.cores + " cores · " + n.threads + " thread") +
      kv("Uptime", n.uptime_s && nodePhase(n) !== "sleeping" ? dur(n.uptime_s) : "—") +
      kv("CPU load", n.cpu_load !== null && n.cpu_load !== undefined ? n.cpu_load + " (" + (n.load_pct || 0) + "%)" : "—") +
      kv("Memory allocated", (n.alloc_memory_mb / 1024).toFixed(1) + " of " + (n.real_memory_mb / 1024).toFixed(1) + " GB") +
      kv("Power", pw.detail || "—") +
      (n.reason ? kv("Reason", n.reason + (n.reason_by ? " (" + n.reason_by + ")" : "")) : "");
    if (n.gres.some(function (g) { return g.type === "gpu"; })) {
      var probe = probes[n.name], line = "";
      var live = n.gpu_live;
      if (live && live.gpus && live.gpus.length) {
        line = live.gpus.map(function (g) {
          return "<b>" + esc(g.name) + "</b> · vram " + g.mem_used_mb + "/" + g.mem_total_mb + " MB · fan " +
            (g.fan_pct !== null && g.fan_pct !== undefined ? g.fan_pct + "%" : "—");
        }).join("<br>") + " · live " + live.age_s + "s ago, job " + esc(live.job_id || "?");
      } else if (probe) {
        line = probe.ok && probe.gpus && probe.gpus.length
          ? probe.gpus.map(function (g) {
              return "<b>" + esc(g.name) + "</b> · " + (g.temp_c !== null ? g.temp_c + "°C" : "—") + " · util " +
                (g.util_pct !== null ? g.util_pct + "%" : "—") + " · vram " + g.mem_used_mb + "/" + g.mem_total_mb + " MB";
            }).join("<br>") + " · probed " + ago(probe.ts)
          : '<span class="bad-text">probe failed: ' + esc(probe.error || "unknown") + "</span> · " + ago(probe.ts);
      } else {
        line = "No live feed: a running llama.cpp server publishes GPU load for free; a probe spends a 1-minute job slot.";
      }
      html += '<div class="gres-detail">' + line + '</div><div><button class="tiny" data-probe="' + esc(n.name) + '">' +
        (state.probing[n.name] && Date.now() - state.probing[n.name] < 6000 ? "Probing…" : "Probe GPU") + "</button></div>";
    }
    return html;
  }

  function nodeRow(name) {
    var row = state.nodeRows[name];
    if (row) return row;
    row = el("div", { class: "nrow" });
    row.innerHTML = '<div class="nrow-main" tabindex="0" role="button" aria-expanded="false"></div>' +
      '<div class="drawer"><div><div class="drawer-in"></div></div></div>';
    var main = row.firstChild;
    var toggle = function () {
      row.classList.toggle("open");
      main.setAttribute("aria-expanded", row.classList.contains("open"));
    };
    main.addEventListener("click", toggle);
    main.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); toggle(); }
    });
    row.querySelector(".drawer-in").addEventListener("click", function (ev) {
      var b = ev.target.closest("[data-probe]");
      if (!b) return;
      state.probing[b.dataset.probe] = Date.now();
      b.disabled = true; b.textContent = "Probing…";
      api("/api/cluster/gpu-probe", { method: "POST", body: { nodes: [b.dataset.probe] } }).catch(function () {});
      setTimeout(refresh, 4000);
    });
    state.nodeRows[name] = row;
    return row;
  }

  function renderNodes(data) {
    var grid = document.getElementById("node-grid");
    var orch = data.orch || null;
    var jobsOn = {};
    (data.queue || []).forEach(function (j) {
      if (j.state !== "RUNNING") return;
      expandHosts(j.nodes).forEach(function (h) { (jobsOn[h] = jobsOn[h] || []).push(j); });
    });
    var seen = {};
    (data.nodes || []).forEach(function (n, idx) {
      seen[n.name] = 1;
      var row = nodeRow(n.name), phase = nodePhase(n), jobs = jobsOn[n.name] || [];
      if (grid.children[idx] !== row) grid.insertBefore(row, grid.children[idx] || null);
      var prev = state.phaseSeen[n.name];
      if (!prev || prev.phase !== phase) state.phaseSeen[n.name] = { phase: phase, since: Date.now() };
      var sig = [phase, (n.power || {}).detail, (n.power || {}).can_wake, (n.power || {}).outside, n.reason,
                 jobs.map(function (j) { return j.job_id; }).join(","),
                 nodeWork(n, jobs, orch)].join("|");
      var main = row.firstChild;
      if (row.dataset.sig !== sig) {
        main.innerHTML = nodeMain(n, phase, jobs, orch);
        row.dataset.sig = sig;
        row.classList.toggle("asleep", phase === "sleeping");
      }
      if (RUNNING_PHASES[phase]) {
        var alloc = n.alloc_cpus;
        patchMeter(main.querySelector('[data-k="cpu"]'), "CPU", alloc + "/" + n.cpus, n.cpu_pct, "dim");
        var memPct = n.mem_used_pct !== null && n.mem_used_pct !== undefined ? n.mem_used_pct : n.mem_pct;
        patchMeter(main.querySelector('[data-k="mem"]'), "Memory",
          Math.round(memPct) + "% of " + (n.real_memory_mb / 1024).toFixed(1) + " GB", memPct,
          memPct >= 90 ? "bad" : memPct >= 75 ? "warn" : "");
        patchGpu(row, n);
      }
      var age = main.querySelector("[data-age]");
      if (age) {
        var s = (Date.now() - state.phaseSeen[n.name].since) / 1000;
        age.textContent = (phase === "sleeping" ? "asleep " : "") + dur(s);
        age.title = "time in this state since the panel first saw it";
      }
      row.querySelector(".drawer-in").innerHTML = nodeDrawer(n, data.probes || {});
    });
    Object.keys(state.nodeRows).forEach(function (name) {
      if (!seen[name]) { state.nodeRows[name].remove(); delete state.nodeRows[name]; }
    });
    var partitions = (data.partitions || []).map(function (p) { return p.name; }).join(" · ");
    document.getElementById("nodes-meta").textContent = partitions;
  }

  function fmtMin(s) { return s < 60 ? Math.round(s) + "s" : s <= 5400 ? Math.floor(s / 60) + "m" : Math.floor(s / 3600) + "h " + Math.floor(s % 3600 / 60) + "m"; }

  // Header chip + idle clock, both from the orchestrator's published state.
  function renderOrch(orch) {
    var gw = document.getElementById("gw"), clock = document.getElementById("idle-clock");
    if (!orch) { gw.hidden = true; if (clock) clock.hidden = true; return; }
    gw.hidden = false;
    var queued = Object.keys(orch.queue || {});
    var busy = (orch.inflight || 0) > 0 || queued.length > 0;
    document.getElementById("gw-dot").className = "dot pulse " + (busy ? "busy" : "ok slow");
    var lk = orch.lock;
    document.getElementById("gw-txt").textContent = queued.length ? "bringing up " + queued.join(", ")
      : busy ? "gateway busy" : lk ? "locked to " + lk.model : "gateway idle " + fmtMin(orch.idle_s || 0);
    document.getElementById("gw-sub").textContent = (orch.inflight || 0) + " in flight";
    gw.title = orch.gateway || "";
    if (!clock) return;
    var pol = orch.policy || {}, rel = pol.idle_release_s || 1200, off = pol.idle_poweroff_s || 3600;
    var idle = busy ? 0 : (orch.idle_s || 0);
    clock.hidden = false;
    document.getElementById("ic-idle").textContent = fmtMin(idle);
    document.getElementById("ic-fill").style.width = Math.min(100, 100 * idle / off) + "%";
    document.getElementById("ic-m1").style.left = (100 * rel / off) + "%";
    document.getElementById("ic-m1-t").textContent = fmtMin(rel) + " · unload, shrink to GPUs";
    document.getElementById("ic-m2-t").textContent = fmtMin(off) + " · power off";
    document.getElementById("ic-next").textContent = busy ? "serving"
      : lk ? "locked · " + ((orch.lock_nodes || []).map(function (n) { return n.replace(/^slerm-/, ""); })
                                                     .join(", ") || "its nodes") +
             ((orch.lock_nodes || []).length === 1 ? " stays on" : " stay on")
      : idle < rel ? "unload in " + fmtMin(rel - idle)
      : idle < off ? "power off in " + fmtMin(off - idle) : "powered down";
  }

  function renderCluster(data) {
    var s = data.summary;
    var nodes = data.nodes || [];
    var q = data.queue || [];
    var running = q.filter(function (j) { return j.state === "RUNNING"; }).length;
    var pending = q.filter(function (j) { return j.state === "PENDING"; }).length;

    var phases = nodes.map(nodePhase);
    var up = phases.filter(function (p) { return RUNNING_PHASES[p]; }).length;
    var counts = {};
    phases.forEach(function (p) { if (!RUNNING_PHASES[p]) counts[PHASE_VIEW[p].word] = (counts[PHASE_VIEW[p].word] || 0) + 1; });
    var notUp = Object.keys(counts).map(function (k) { return counts[k] + " " + k; }).join(" · ");

    var vramUsed = 0, vramTotal = 0;
    nodes.forEach(function (n) {
      ((n.gpu_live && n.gpu_live.gpus) || []).forEach(function (g) { vramUsed += g.mem_used_mb || 0; vramTotal += g.mem_total_mb || 0; });
    });

    var tiles = document.getElementById("cluster-tiles");
    statTile(tiles, "nodes", notUp ? "nodes up · " + notUp : "nodes up",
      up + "<small>/" + nodes.length + "</small>", "");
    statTile(tiles, "cpus", "CPUs allocated", s.cpus_alloc + "<small>/" + s.cpus_total + "</small>",
      (s.cpus_total ? Math.round(100 * s.cpus_alloc / s.cpus_total) : 0) + "% busy");
    statTile(tiles, "jobs", "jobs running", String(running), pending + " pending");
    if (vramTotal) {
      statTile(tiles, "vram", "VRAM in use", (vramUsed / 1024).toFixed(1) + "<small>/" + Math.round(vramTotal / 1024) + " GB</small>",
        "live from the server jobs");
    } else {
      statTile(tiles, "vram", "GPUs allocated", s.gpus_used + "<small>/" + s.gpus_total + "</small>", "no live GPU feed");
    }
    statTile(tiles, "sched", "scheduler cycle",
      data.diag && data.diag.sched_mean_cycle_us !== undefined ? data.diag.sched_mean_cycle_us + "<small>µs</small>" : "—",
      data.diag && data.diag.jobs_submitted !== undefined ? data.diag.jobs_submitted + " submitted" : "sdiag");

    renderNodes(data);
    renderOrch(data.orch);

    // head node
    var head = data.head || {};
    document.getElementById("head-name").textContent =
      (head.hostname || "").split(".")[0] + (head.uptime_s ? " · up " + dur(head.uptime_s) : "");
    var ht = document.getElementById("head-tiles");
    var meters = [];
    if (head.load) {
      var lpct = head.cpus ? 100 * head.load[0] / head.cpus : 0;
      meters.push(["load", "Load (" + head.cpus + " CPUs)", head.load.join(" · "), lpct, lpct >= 90 ? "bad" : lpct >= 70 ? "warn" : "dim"]);
    }
    if (head.mem) {
      meters.push(["mem", "Memory", head.mem.used_mb + " MB of " + (head.mem.total_mb / 1024).toFixed(1) + " GB",
        head.mem.used_pct, head.mem.used_pct >= 90 ? "bad" : head.mem.used_pct >= 75 ? "warn" : ""]);
      if (head.mem.swap_total_mb) {
        var su = head.mem.swap_total_mb - head.mem.swap_free_mb;
        meters.push(["swap", "Swap", su + " MB of " + (head.mem.swap_total_mb / 1024).toFixed(1) + " GB",
          100 * su / head.mem.swap_total_mb, "dim"]);
      }
    }
    // /data, /home and /apps are one NFS export: show it once.
    var mounts = {};
    (head.mounts || []).forEach(function (m) {
      var k = m.total_gb + "|" + m.used_gb;
      (mounts[k] = mounts[k] || { m: m, paths: [] }).paths.push(m.path);
    });
    Object.keys(mounts).forEach(function (k, i) {
      var m = mounts[k].m;
      meters.push(["mnt" + i, mounts[k].paths.join(" · "), Math.round(m.used_gb) + " of " + Math.round(m.total_gb) + " GB",
        m.used_pct, m.used_pct >= 90 ? "bad" : m.used_pct >= 80 ? "warn" : ""]);
    });
    meters.forEach(function (m) {
      var host = ht.querySelector('[data-k="' + m[0] + '"]');
      if (!host) { host = el("div", { "data-k": m[0] }); ht.appendChild(host); }
      patchMeter(host, m[1], m[2], m[3], m[4]);
      host.querySelector(".bar-label span").textContent = m[1];
    });

    // queue
    document.getElementById("queue-meta").textContent = running + " running · " + pending + " pending";
    var qt = document.getElementById("queue-table");
    var qrows = q.map(function (j) {
      var name = j.array_task_string
        ? (j.array_job_id + "_[" + j.array_task_string + "]")
        : String(j.array_task_id !== null && j.array_task_id !== undefined
            ? j.array_job_id + "_" + j.array_task_id : j.job_id);
      return [name, j.name, j.partition, statePill(j.state),
              j.nodes || j.reason || "—",
              j.start_time ? dur(Date.now() / 1000 - j.start_time) : "—",
              el("button", { class: "tiny danger", text: "Cancel", title: "scancel " + (j.array_job_id || j.job_id),
                onclick: function () { cancelJob(j.array_job_id || j.job_id); } })];
    });
    var qnew = table(["Job", "Name", "Partition", "State", "Nodes / reason", "Elapsed", ""],
      qrows, "Queue is empty");
    qnew.id = "queue-table";
    qt.replaceWith(qnew);
  }
  function cancelJob(id) {
    api("/api/jobs/" + id + "/cancel", { method: "POST" })
      .then(refresh)
      .catch(function (e) { alert("cancel failed: " + e.message); });
  }

  function refreshHistory() {
    api("/api/cluster/history?days=" + state.historyDays).then(function (data) {
      var rows = data.jobs.map(function (j) {
        var name = j.array_task_id !== null && j.array_task_id !== undefined
          ? j.array_job_id + "_" + j.array_task_id : String(j.job_id);
        return [name, j.name, j.partition, statePill(j.state), j.nodes || "—",
                j.elapsed !== null ? dur(j.elapsed) : "—", when(j.start)];
      });
      var old = document.getElementById("history-table");
      var t = table(["Job", "Name", "Partition", "State", "Nodes", "Ran for", "Started"],
        rows, "No jobs in this window");
      t.id = "history-table";
      old.replaceWith(t);
    }).catch(function () {});
  }

  // --------------------------------------------------------------- projects
  // The project view is one tab with inner panes: a Run pane plus one pane per
  // product, sharing a single dataset selector and a single time cursor. The
  // cursor is a frame *name*, not an index, so panes stay aligned even when a
  // product has been rendered for only some of the scans -- which is what makes
  // flipping between reflectivity and rotation a comparison rather than a jump.
  //
  // Wazuh is a deliberately separate project view. It has no frame products,
  // and trying to squeeze it through the run/product/cursor UI would make every
  // one of those assumptions leak. Its own reconciler follows the same rule as
  // the runs view below: keep list nodes and open detail/evidence panes in place
  // while the report lifecycle is being polled.

  function localDateTimeValue(date) {
    var shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
    return shifted.toISOString().slice(0, 16);
  }

  function wazuhTime(value) {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "number") {
      return scanTime(value > 100000000000 ? value / 1000 : value);
    }
    var numeric = Number(value);
    if (String(value).trim() !== "" && isFinite(numeric)) {
      return scanTime(numeric > 100000000000 ? numeric / 1000 : numeric);
    }
    var parsed = new Date(value);
    if (isNaN(parsed.getTime())) return String(value);
    return scanTime(parsed.getTime() / 1000);
  }

  function reportState(value) {
    return String(value || "pending").toLowerCase().replace(/_/g, "-");
  }

  function wazuhStatePill(value) {
    var raw = reportState(value), cls = "idle";
    if (["ready", "complete", "completed", "done"].indexOf(raw) >= 0) cls = "ok";
    else if (["queued", "fetching", "uploading", "importing", "normalising",
              "normalizing", "listing", "submitting", "submitted", "running",
              "reducing", "analysing", "analyzing"].indexOf(raw) >= 0) cls = "busy";
    else if (["pending", "waiting"].indexOf(raw) >= 0) cls = "warn";
    else if (["failed", "error", "cancelled"].indexOf(raw) >= 0) cls = "bad";
    return el("span", { class: "pill " + cls, text: raw || "pending" });
  }

  function isLiveReport(row) {
    return ["ready", "complete", "completed", "done", "failed", "error",
            "cancelled"].indexOf(reportState(row && row.state)) < 0;
  }

  function normaliseProfiles(data) {
    var input = (data && data.profiles) || [];
    if (!Array.isArray(input)) {
      input = Object.keys(input).map(function (id) {
        var value = input[id];
        if (typeof value === "string") return { id: id, state: value };
        var copy = {};
        Object.keys(value || {}).forEach(function (k) { copy[k] = value[k]; });
        copy.id = copy.id || id;
        return copy;
      });
    }
    return input.map(function (item) {
      if (typeof item === "string") return { id: item, profile: item };
      var health = item.health && typeof item.health === "object" ? item.health : {};
      var out = {};
      Object.keys(health).forEach(function (k) { out[k] = health[k]; });
      Object.keys(item || {}).forEach(function (k) {
        if (k !== "health") out[k] = item[k];
      });
      out.id = out.id || out.profile || out.name;
      return out;
    }).filter(function (p) { return !!p.id; }).sort(function (a, b) {
      var order = { nbdevlab: 0, remote: 1 };
      var ai = order[a.id] !== undefined ? order[a.id] : 10;
      var bi = order[b.id] !== undefined ? order[b.id] : 10;
      return ai === bi ? String(a.id).localeCompare(String(b.id)) : ai - bi;
    });
  }

  function buildWazuhView(p) {
    var view = el("div", { class: "view", id: "view-" + p.id });
    var ctx = { kind: "wazuh", project: p, pane: null, panes: {}, buttons: {},
                profiles: {}, loadingProfiles: false, profileLoaded: false };
    state.views[p.id] = ctx;

    ctx.notice = el("div");
    ctx.summary = el("div", { class: "wazuh-overview" }, [
      el("span", { class: "muted mono", text: "loading indexer profiles…" })
    ]);
    view.appendChild(el("section", { class: "card" }, [
      el("h2", {}, [document.createTextNode(p.label),
        el("span", { class: "h2-actions muted mono", text: "deterministic cluster analysis" })]),
      el("div", { class: "muted wazuh-intro", text: p.description ||
        "Pull Wazuh alerts into an immutable snapshot, then score them against the window's own baseline." }),
      ctx.notice, ctx.summary
    ]));

    ctx.subnav = el("nav", { class: "tabs subtabs wazuh-subtabs" });
    ctx.paneHost = el("div", { class: "wazuh-pane-host" });
    view.appendChild(ctx.subnav);
    view.appendChild(ctx.paneHost);

    ctx.panes.import = buildWazuhImportPane(ctx);
    ctx.paneHost.appendChild(ctx.panes.import.box);
    ctx.buttons.import = el("button", { text: "Import", "data-pane": "import",
      onclick: function () { selectWazuhPane(ctx, "import", true); } });
    ctx.subnav.appendChild(ctx.buttons.import);
    selectWazuhPane(ctx, "import", false);
    return view;
  }

  function buildWazuhProfilePane(ctx, id) {
    var st = { ctx: ctx, id: id, kind: "profile", reportNodes: {}, reports: [],
               selectedReport: null, detail: null, detailLoading: false,
               reportsLoading: false, indicesLoading: false };
    st.box = el("div", { class: "wazuh-profile-pane", "data-profile": id });
    st.healthPill = el("span", {});
    st.healthMeta = el("div", { class: "wazuh-health-meta muted mono",
      text: "checking indexer…" });
    st.healthNotice = el("div");
    st.fetchNotice = el("div");

    var now = new Date(), start = new Date(now.getTime() - 24 * 3600 * 1000);
    st.startInput = el("input", { type: "datetime-local", name: "start", step: "1",
      value: localDateTimeValue(start), required: "required" });
    st.endInput = el("input", { type: "datetime-local", name: "end", step: "1",
      value: localDateTimeValue(now), required: "required" });
    st.indexInput = el("input", { name: "pattern", value: "wazuh-alerts-4.x-*",
      list: "wazuh-indices-" + id.replace(/[^A-Za-z0-9_-]/g, "-"),
      placeholder: "wazuh-alerts-4.x-*" });
    st.indexList = el("datalist", { id: st.indexInput.getAttribute("list") });
    st.indexHelp = el("div", { class: "help", text: "loading available indices…" });
    st.queryInput = el("input", { name: "query", placeholder: "rule.id:92657" });
    st.fetchButton = el("button", { class: "primary", type: "submit",
      text: "Fetch & analyse", disabled: "disabled" });
    st.fetchForm = el("form", {}, [
      el("div", { class: "form-grid wazuh-fetch-grid" }, [
        el("div", { class: "field" }, [el("label", { text: "Start" }), st.startInput,
          el("div", { class: "help", text: "browser-local time; sent to the indexer as UTC" })]),
        el("div", { class: "field" }, [el("label", { text: "End" }), st.endInput,
          el("div", { class: "help", text: "inclusive end of the snapshot window" })]),
        el("div", { class: "field wazuh-index-field" }, [
          el("label", { text: "Index or pattern" }), st.indexInput, st.indexList, st.indexHelp]),
        el("div", { class: "field" }, [el("label", { text: "Optional query filter" }),
          st.queryInput, el("div", { class: "help", text: "OpenSearch query_string syntax" })])
      ]),
      el("div", { class: "wazuh-submit-row" }, [st.fetchButton,
        el("span", { class: "muted", text: "The fetch is saved first; cluster analysis can finish after the indexer disconnects." })])
    ]);
    st.fetchForm.addEventListener("submit", function (ev) {
      ev.preventDefault();
      submitWazuhFetch(st);
    });

    st.reportsBox = el("div", { class: "wazuh-report-list" });
    st.detailHost = el("div", { class: "wazuh-detail-host" });
    st.box.appendChild(el("section", { class: "card wazuh-health-card" }, [
      el("h2", {}, [document.createTextNode("Indexer · " + id),
        el("span", { class: "h2-actions" }, [st.healthPill])]),
      st.healthMeta, st.healthNotice
    ]));
    st.box.appendChild(el("section", { class: "card" }, [
      el("h2", { text: "Fetch a snapshot" }), st.fetchNotice, st.fetchForm
    ]));
    st.box.appendChild(el("section", { class: "card" }, [
      el("h2", {}, [document.createTextNode("Snapshots & reports"),
        el("span", { class: "h2-actions muted mono", text: "newest first" })]),
      st.reportsBox
    ]));
    st.box.appendChild(st.detailHost);
    return st;
  }

  function buildWazuhImportPane(ctx) {
    var st = { ctx: ctx, id: "import", kind: "import", reportNodes: {}, reports: [],
               selectedReport: null, detail: null, detailLoading: false,
               reportsLoading: false, uploading: false };
    st.box = el("div", { class: "wazuh-profile-pane", "data-profile": "import" });
    st.notice = el("div");
    st.fileInput = el("input", { type: "file", name: "file",
      accept: ".csv,.json,.jsonl,.gz,application/json,text/csv",
      class: "wazuh-file-input" });
    st.dropTitle = el("strong", { text: "Drop a Wazuh export here" });
    st.dropMeta = el("span", { text: "CSV, JSON, JSONL, or gzip · up to 512 MB" });
    st.dropZone = el("div", { class: "wazuh-drop-zone", tabindex: "0", role: "button" }, [
      el("span", { class: "wazuh-drop-icon", text: "⇧" }), st.dropTitle, st.dropMeta,
      el("button", { type: "button", class: "tiny", text: "Choose file" })
    ]);
    st.progressLabel = el("div", { class: "bar-label" }, [
      el("span", { text: "Upload" }), el("span", { text: "waiting" })
    ]);
    st.progressBar = bar(0, "ok");
    st.progress = el("div", { class: "wazuh-upload-progress", style: "display:none" }, [
      st.progressLabel, st.progressBar
    ]);
    function choose(ev) {
      if (ev && ev.target && ev.target.tagName === "INPUT") return;
      st.fileInput.click();
    }
    st.dropZone.addEventListener("click", choose);
    st.dropZone.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); st.fileInput.click(); }
    });
    ["dragenter", "dragover"].forEach(function (name) {
      st.dropZone.addEventListener(name, function (ev) {
        ev.preventDefault();
        if (!st.uploading) st.dropZone.classList.add("dragging");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      st.dropZone.addEventListener(name, function (ev) {
        ev.preventDefault();
        st.dropZone.classList.remove("dragging");
      });
    });
    st.dropZone.addEventListener("drop", function (ev) {
      var file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      if (file) uploadWazuhFile(st, file);
    });
    st.fileInput.addEventListener("change", function () {
      if (st.fileInput.files && st.fileInput.files[0]) uploadWazuhFile(st, st.fileInput.files[0]);
      st.fileInput.value = "";
    });

    st.reportsBox = el("div", { class: "wazuh-report-list" });
    st.detailHost = el("div", { class: "wazuh-detail-host" });
    st.box.appendChild(el("section", { class: "card" }, [
      el("h2", { text: "Import an offline export" }),
      el("div", { class: "muted wazuh-import-help", text:
        "Imports use the same normalisation, global baseline, detectors, and Slurm analysis as a live indexer fetch." }),
      st.notice, st.fileInput, st.dropZone, st.progress
    ]));
    st.box.appendChild(el("section", { class: "card" }, [
      el("h2", {}, [document.createTextNode("Imported reports"),
        el("span", { class: "h2-actions muted mono", text: "newest first" })]),
      st.reportsBox
    ]));
    st.box.appendChild(st.detailHost);
    return st;
  }

  function selectWazuhPane(ctx, id, userSelected) {
    if (!ctx.panes[id]) return;
    ctx.pane = id;
    if (userSelected) ctx.userSelectedPane = true;
    Object.keys(ctx.panes).forEach(function (key) {
      ctx.panes[key].box.style.display = key === id ? "" : "none";
    });
    Object.keys(ctx.buttons).forEach(function (key) {
      ctx.buttons[key].classList.toggle("active", key === id);
    });
    dueReset();
    refreshWazuh(ctx, true);
  }

  function reconcileWazuhProfiles(ctx, data) {
    var profiles = normaliseProfiles(data), seen = {};
    profiles.forEach(function (profile) {
      var id = profile.id;
      seen[id] = true;
      ctx.profiles[id] = profile;
      if (!ctx.panes[id]) {
        ctx.panes[id] = buildWazuhProfilePane(ctx, id);
        ctx.paneHost.insertBefore(ctx.panes[id].box, ctx.panes.import.box);
        ctx.buttons[id] = el("button", { text: id, "data-pane": id,
          onclick: function () { selectWazuhPane(ctx, id, true); } });
      }
      renderWazuhHealth(ctx.panes[id], profile);
    });

    // Preserve nodes for configured profiles, but remove a profile that has
    // actually disappeared from the credentials file.
    Object.keys(ctx.profiles).forEach(function (id) {
      if (seen[id]) return;
      delete ctx.profiles[id];
      if (ctx.panes[id]) ctx.panes[id].box.remove();
      if (ctx.buttons[id]) ctx.buttons[id].remove();
      delete ctx.panes[id];
      delete ctx.buttons[id];
      if (ctx.pane === id) selectWazuhPane(ctx, "import", false);
    });

    profiles.forEach(function (profile, index) {
      var button = ctx.buttons[profile.id];
      var at = ctx.subnav.children[index];
      if (at !== button) ctx.subnav.insertBefore(button, at || ctx.buttons.import);
    });
    ctx.subnav.appendChild(ctx.buttons.import);

    var up = profiles.filter(function (p) { return !!p.ok; }).length;
    clear(ctx.summary);
    ctx.summary.appendChild(el("span", { class: "mono", text:
      profiles.length ? up + " / " + profiles.length + " indexers reachable" : "No live indexer profiles configured" }));
    ctx.summary.appendChild(el("span", { class: "muted", text:
      "Stored snapshots and imported reports remain usable when an indexer is offline." }));

    if (!ctx.profileLoaded && profiles.length && !ctx.userSelectedPane) {
      ctx.profileLoaded = true;
      selectWazuhPane(ctx, profiles[0].id, false);
    } else {
      ctx.profileLoaded = true;
    }
  }

  function renderWazuhHealth(st, health) {
    var stateName = String(health.state || (health.ok ? "ok" : "unknown")).toLowerCase();
    var label = health.ok ? "reachable · auth ok" : stateName;
    var kind = health.ok ? "ok" : "err";
    if (!health.ok && stateName === "unreachable" && st.id === "remote") label = "tunnel down";
    else if (!health.ok && stateName === "unreachable") label = "indexer unreachable";
    else if (!health.ok && ["unauthorised", "unauthorized"].indexOf(stateName) >= 0) {
      label = "credentials rejected";
    } else if (!health.ok && stateName === "unconfigured") label = "not configured";

    clear(st.healthPill);
    st.healthPill.appendChild(el("span", { class: "wazuh-health-chip " + kind,
      text: label }));
    st.ctx.buttons[st.id].className = (st.ctx.pane === st.id ? "active " : "") +
      "wazuh-profile-tab health-" + (health.ok ? "ok" : "bad");
    st.fetchButton.disabled = !health.ok;

    var details = [];
    if (health.url) details.push(health.url);
    if (health.user) details.push("user " + health.user);
    if (health.cluster) details.push("cluster " + health.cluster);
    if (health.version) details.push("OpenSearch " + health.version);
    if (health.elapsed !== undefined) details.push(Math.round(health.elapsed * 1000) + " ms");
    st.healthMeta.textContent = details.join(" · ") || "No indexer connection details available.";

    if (!health.ok && stateName === "unreachable" && st.id === "remote") {
      notice(st.healthNotice,
        "Remote tunnel down — alert availability is unknown. This is not a quiet or zero-alert result." +
        (health.error ? " " + health.error : ""), "err");
    } else if (!health.ok) {
      notice(st.healthNotice, (health.error || "This indexer cannot be queried right now.") +
        " Stored reports below are still available.", "err");
    } else {
      notice(st.healthNotice, "");
    }
  }

  function submitWazuhFetch(st) {
    var start = new Date(st.startInput.value), end = new Date(st.endInput.value);
    if (isNaN(start.getTime()) || isNaN(end.getTime()) || start >= end) {
      notice(st.fetchNotice, "Choose a valid window whose end is after its start.", "err");
      return;
    }
    st.fetchButton.disabled = true;
    notice(st.fetchNotice, "Starting the indexer fetch…", "ok");
    api("/api/wazuh/" + encodeURIComponent(st.id) + "/fetch", {
      method: "POST", body: {
        start: start.toISOString(), end: end.toISOString(),
        pattern: st.indexInput.value.trim() || "wazuh-alerts-4.x-*",
        query: st.queryInput.value.trim() || null
      }
    }).then(function (res) {
      var label = res.fetch_id !== undefined ? "fetch #" + res.fetch_id : "fetch";
      notice(st.fetchNotice, label + " started. Alerts will be snapshotted before cluster analysis begins.", "ok");
      dueReset();
      refreshWazuh(st.ctx, true);
    }).catch(function (e) {
      notice(st.fetchNotice, e.message, "err");
    }).then(function () {
      var health = st.ctx.profiles[st.id];
      st.fetchButton.disabled = !(health && health.ok);
    });
  }

  function uploadWazuhFile(st, file) {
    if (st.uploading) return;
    if (!file) return;
    st.uploading = true;
    st.dropZone.classList.add("uploading");
    st.progress.style.display = "";
    st.progressLabel.lastChild.textContent = "0% · " + bytes(file.size);
    var fill = st.progressBar.querySelector("i");
    fill.style.width = "0%";
    notice(st.notice, "Uploading " + file.name + "…", "ok");

    var xhr = new XMLHttpRequest(), form = new FormData();
    form.append("file", file, file.name);
    xhr.open("POST", "/api/wazuh/import");
    xhr.upload.onprogress = function (ev) {
      if (!ev.lengthComputable) {
        st.progressLabel.lastChild.textContent = bytes(ev.loaded) + " uploaded";
        return;
      }
      var pct = Math.round(100 * ev.loaded / ev.total);
      fill.style.width = pct + "%";
      st.progressLabel.lastChild.textContent = pct + "% · " + bytes(ev.loaded) + " / " + bytes(ev.total);
    };
    xhr.onerror = function () { finishWazuhUpload(st, null, "Upload failed: network error"); };
    xhr.onload = function () {
      var data = {};
      try { data = JSON.parse(xhr.responseText || "{}"); } catch (e) { data = {}; }
      if (xhr.status < 200 || xhr.status >= 300) {
        finishWazuhUpload(st, null, data.error || (xhr.status + " " + xhr.statusText));
        return;
      }
      finishWazuhUpload(st, data, null);
    };
    xhr.send(form);
  }

  function finishWazuhUpload(st, data, error) {
    st.uploading = false;
    st.dropZone.classList.remove("uploading");
    if (error) {
      notice(st.notice, error, "err");
      st.progressLabel.lastChild.textContent = "failed";
      return;
    }
    var fill = st.progressBar.querySelector("i");
    fill.style.width = "100%";
    st.progressLabel.lastChild.textContent = "uploaded · normalising snapshot";
    var row = (data && data.report) || {};
    var id = data && (data.report_id !== undefined ? data.report_id : row.id);
    var truncated = !!(row.truncated || (data && data.truncated));
    notice(st.notice, truncated
      ? "Upload accepted, but it contains exactly 10,000 CSV rows: the dashboard truncated the export. Treat the analysed window as incomplete."
      : "Upload accepted" + (id !== undefined ? " as report #" + id : "") +
        ". Normalisation and cluster analysis are running.", truncated ? "err" : "ok");
    dueReset();
    refreshWazuh(st.ctx, true);
    if (id !== undefined) selectWazuhReport(st, id);
  }

  function refreshWazuh(ctx, force) {
    var interval = state.poll.wazuh || 2000;
    if (due("wazuh:profiles", force ? 0 : Math.max(interval, 5000))) {
      refreshWazuhProfiles(ctx);
    }
    var st = ctx.panes[ctx.pane];
    if (!st) return;
    if (st.kind === "profile" && due("wazuh:indices:" + st.id,
        force ? 0 : Math.max(30000, interval * 10))) {
      refreshWazuhIndices(st);
    }
    if (due("wazuh:reports:" + st.id, force ? 0 : interval)) {
      refreshWazuhReports(st, force);
    }
    if (st.selectedReport !== null && st.selectedReport !== undefined &&
        (!st.detail || !st.detail.data) &&
        due("wazuh:detail:" + st.selectedReport, force ? 0 : interval)) {
      loadWazuhReportDetail(st, st.selectedReport, false);
    }
  }

  function refreshWazuhProfiles(ctx) {
    if (ctx.loadingProfiles) return;
    ctx.loadingProfiles = true;
    api("/api/wazuh/profiles").then(function (data) {
      reconcileWazuhProfiles(ctx, data);
      notice(ctx.notice, "");
      stamp();
    }).catch(function (e) {
      notice(ctx.notice, "Could not refresh Wazuh profile health: " + e.message, "err");
      stamp("Wazuh health error");
    }).then(function () { ctx.loadingProfiles = false; });
  }

  function refreshWazuhIndices(st) {
    var health = st.ctx.profiles[st.id];
    if (!health || !health.ok || st.indicesLoading) {
      if (health && !health.ok) st.indexHelp.textContent = "index list unavailable while this profile is offline";
      return;
    }
    st.indicesLoading = true;
    api("/api/wazuh/" + encodeURIComponent(st.id) + "/indices")
      .then(function (data) {
        var indices = (data && data.indices) || [];
        clear(st.indexList);
        indices.forEach(function (row) {
          var name = typeof row === "string" ? row : row.index;
          if (name) st.indexList.appendChild(el("option", { value: name }));
        });
        var docs = indices.reduce(function (n, row) {
          return n + Number((row && row.docs) || 0); }, 0);
        st.indexHelp.textContent = indices.length
          ? indices.length + " matching index" + (indices.length === 1 ? "" : "es") +
            " · " + docs.toLocaleString() + " documents; a wildcard is allowed"
          : "No matching indices returned. Health is good; verify the pattern and collection window.";
      }).catch(function (e) {
        st.indexHelp.textContent = "Could not list indices: " + e.message;
      }).then(function () { st.indicesLoading = false; });
  }

  function refreshWazuhReports(st, force) {
    if (st.reportsLoading) return;
    st.reportsLoading = true;
    var path = "/api/wazuh/reports";
    if (st.kind === "profile") path += "?profile=" + encodeURIComponent(st.id);
    else path += "?source=import";
    api(path).then(function (data) {
      var rows = (data && data.reports) || [];
      rows.sort(function (a, b) { return Number(b.id || 0) - Number(a.id || 0); });
      st.reports = rows;
      renderWazuhReports(st, rows);
      var selected = rows.filter(function (row) {
        return String(row.id) === String(st.selectedReport); })[0];
      if (selected && (!st.detail || !st.detail.data || isLiveReport(selected))) {
        var oldUpdated = st.detail && st.detail.report && st.detail.report.updated;
        if (force || !st.detail || oldUpdated !== selected.updated) {
          loadWazuhReportDetail(st, selected.id, false);
        }
      }
      stamp();
    }).catch(function (e) {
      if (!Object.keys(st.reportNodes).length) {
        clear(st.reportsBox);
        st.reportsBox.appendChild(el("div", { class: "notice err",
          text: "Could not load reports: " + e.message }));
      }
    }).then(function () { st.reportsLoading = false; });
  }

  function renderWazuhReports(st, rows) {
    if (!rows.length) {
      if (!st.reportsEmpty) {
        clear(st.reportsBox);
        st.reportNodes = {};
        st.reportsEmpty = el("div", { class: "empty", text: st.kind === "import"
          ? "no imported reports yet — drop an export above"
          : "no snapshots for this profile yet" });
        st.reportsBox.appendChild(st.reportsEmpty);
      }
      return;
    }
    if (st.reportsEmpty) { st.reportsEmpty.remove(); st.reportsEmpty = null; }

    var seen = {};
    rows.forEach(function (row, index) {
      var id = String(row.id);
      seen[id] = true;
      var node = st.reportNodes[id];
      if (!node) {
        node = createWazuhReportNode(st, row);
        st.reportNodes[id] = node;
      }
      updateWazuhReportNode(st, node, row);
      var at = st.reportsBox.children[index];
      if (at !== node.box) st.reportsBox.insertBefore(node.box, at || null);
    });
    Object.keys(st.reportNodes).forEach(function (id) {
      if (seen[id]) return;
      st.reportNodes[id].box.remove();
      delete st.reportNodes[id];
    });
  }

  function createWazuhReportNode(st, row) {
    var node = {};
    node.idEl = el("b", { class: "wazuh-report-id" });
    node.pill = el("span");
    node.title = el("span", { class: "wazuh-report-title" });
    node.age = el("span", { class: "muted mono" });
    node.open = el("button", { class: "tiny", text: "open",
      onclick: function () { selectWazuhReport(st, node.row.id); } });
    node.meta = el("div", { class: "wazuh-report-meta" });
    node.message = el("div", { class: "msg" });
    node.progress = el("div", { class: "progress-wrap wazuh-report-progress" });
    node.actions = el("div", { class: "wazuh-report-actions" });
    node.box = el("div", { class: "wazuh-report" }, [
      el("div", { class: "wazuh-report-head" }, [
        node.idEl, node.pill, node.title, el("span", { class: "grow" }),
        node.age, node.actions, node.open
      ]), node.meta, node.message, node.progress
    ]);
    return node;
  }

  function updateWazuhReportNode(st, node, row) {
    node.row = row;
    node.idEl.textContent = "#" + row.id;
    clear(node.pill);
    node.pill.appendChild(wazuhStatePill(row.state));
    node.title.textContent = row.filename || (row.source === "import" ? "imported export" :
      ((row.profile || st.id) + " snapshot"));
    var ts = row.updated || row.created;
    node.age.textContent = ts ? ago(ts) : "";
    node.box.classList.toggle("selected", String(row.id) === String(st.selectedReport));

    clear(node.meta);
    var facts = [];
    if (row.fetch_id !== null && row.fetch_id !== undefined) facts.push("fetch #" + row.fetch_id);
    if (row.fetch && row.fetch.state) facts.push("snapshot " + row.fetch.state);
    if (row.run_id !== null && row.run_id !== undefined) facts.push("run #" + row.run_id);
    if (row.alerts !== null && row.alerts !== undefined) facts.push(Number(row.alerts).toLocaleString() + " alerts");
    if (row.bytes !== null && row.bytes !== undefined) facts.push(bytes(Number(row.bytes)));
    if (typeof row.findings === "number") facts.push(row.findings + " findings");
    if (row.actionable !== null && row.actionable !== undefined) facts.push(row.actionable + " actionable");
    if (row.created) facts.push("created " + wazuhTime(row.created));
    facts.forEach(function (fact) { node.meta.appendChild(el("span", { text: fact })); });

    var message = row.message || (row.fetch && row.fetch.message) || "";
    node.message.textContent = message;
    node.message.style.display = message ? "" : "none";
    clear(node.progress);
    if (isLiveReport(row)) {
      node.progress.appendChild(el("div", { class: "bar indeterminate" }, [el("i", { class: "ok" })]));
    }
    // A fetch can run for a long time against a wide window, so it needs a way
    // out that is not "restart the panel". Only offered while the snapshot is
    // still being pulled -- once the array is running, cancelling is the run's
    // job, not the fetch's.
    clear(node.actions);
    var fetchState = row.fetch && row.fetch.state;
    if (row.fetch_id && (fetchState === "fetching" || fetchState === "queued")) {
      node.actions.appendChild(el("button", {
        class: "tiny danger", text: "Cancel fetch",
        onclick: function (ev) {
          ev.stopPropagation();
          ev.target.disabled = true;
          ev.target.textContent = "cancelling…";
          api("/api/wazuh/fetches/" + row.fetch_id + "/cancel", { method: "POST", body: {} })
            .catch(function (e) { notice(node.message, e.message); });
        }
      }));
    } else if (fetchState === "cancelling") {
      node.actions.appendChild(el("span", { class: "muted", text: "cancelling…" }));
    }
  }

  function selectWazuhReport(st, id) {
    if (String(st.selectedReport) !== String(id)) st.detailSignature = null;
    st.selectedReport = id;
    Object.keys(st.reportNodes).forEach(function (key) {
      st.reportNodes[key].box.classList.toggle("selected", key === String(id));
    });
    if (!st.detail || String(st.detail.report && st.detail.report.id) !== String(id)) {
      clear(st.detailHost);
      st.detailHost.appendChild(el("section", { class: "card" }, [
        el("div", { class: "empty", text: "loading report #" + id + "…" })
      ]));
    }
    loadWazuhReportDetail(st, id, true);
  }

  function closeWazuhReport(st) {
    st.selectedReport = null;
    st.detail = null;
    st.detailSeq = (st.detailSeq || 0) + 1;
    clear(st.detailHost);
    Object.keys(st.reportNodes).forEach(function (key) {
      st.reportNodes[key].box.classList.remove("selected");
    });
  }

  function loadWazuhReportDetail(st, id, userAction) {
    if (st.detailLoading && !userAction) return;
    st.detailLoading = true;
    st.detailSeq = (st.detailSeq || 0) + 1;
    var seq = st.detailSeq;
    api("/api/wazuh/reports/" + encodeURIComponent(id)).then(function (payload) {
      if (seq !== st.detailSeq || String(id) !== String(st.selectedReport)) return;
      var row = payload.report || (payload.data ? {} : payload);
      var data = payload.data !== undefined ? payload.data :
        (payload.findings || payload.alerts !== undefined ? payload : null);
      st.detail = { report: row || {}, data: data };
      // Rebuilding the detail wholesale on every poll destroys the findings
      // section a second after it is built -- taking the filters, any expanded
      // evidence and the live AI-triage panel with it. This is the same trap
      // the runs list hit. Only re-render when something
      // that is actually rendered has changed; triage updates arrive through
      // renderWazuhFindingRows(), which patches the list in place.
      var signature = JSON.stringify([
        row && row.id, row && row.state, row && row.message,
        row && row.alerts, row && row.findings, row && row.actionable,
        row && row.truncated, row && row.run_id,
        data ? (data.generated || data.finding_count) : null
      ]);
      if (userAction || signature !== st.detailSignature) {
        st.detailSignature = signature;
        renderWazuhReportDetail(st);
      }
    }).catch(function (e) {
      if (seq !== st.detailSeq || String(id) !== String(st.selectedReport)) return;
      clear(st.detailHost);
      st.detailHost.appendChild(el("section", { class: "card" }, [
        el("h2", {}, [document.createTextNode("Report #" + id),
          el("span", { class: "h2-actions" }, [el("button", { class: "tiny", text: "close",
            onclick: function () { closeWazuhReport(st); } })])]),
        el("div", { class: "notice err", text: "Could not load report: " + e.message })
      ]));
    }).then(function () { if (seq === st.detailSeq) st.detailLoading = false; });
  }

  function statusTile(label, value, sub, cls) {
    return el("div", { class: "tile" + (cls ? " " + cls : "") }, [
      el("div", { class: "label", text: label }),
      el("div", { class: "value", text: value === null || value === undefined ? "—" : String(value) }),
      el("div", { class: "sub", text: sub || "" })
    ]);
  }

  function renderWazuhReportDetail(st) {
    var row = st.detail.report || {}, report = st.detail.data;
    clear(st.detailHost);
    var headActions = el("span", { class: "h2-actions" }, [
      wazuhStatePill(row.state || (report ? "ready" : "pending")),
      el("button", { class: "tiny", text: "close",
        onclick: function () { closeWazuhReport(st); } })
    ]);
    var children = [el("h2", {}, [document.createTextNode("Report #" +
      (row.id !== undefined ? row.id : st.selectedReport)), headActions])];
    if (row.message) children.push(el("div", { class: "wazuh-report-message muted mono", text: row.message }));

    var truncated = !!(row.truncated || (report && report.truncated) ||
      (report && report.meta && report.meta.truncated));
    if (truncated) {
      children.push(el("div", { class: "notice err wazuh-truncated", text:
        "Incomplete export: exactly 10,000 CSV rows were imported, which is the dashboard cap. Findings describe only the truncated slice, not the requested window." }));
    }
    if (!report) {
      children.push(el("div", { class: "wazuh-pending-detail" }, [
        el("div", { class: "bar indeterminate" }, [el("i", { class: "ok" })]),
        el("div", { class: "muted mono", text:
          isLiveReport(row) ? "The snapshot or cluster analysis is still running. This view will update automatically."
                            : "No completed reducer report is available." })
      ]));
      st.detailHost.appendChild(el("section", { class: "card" }, children));
      return;
    }

    var meta = report.meta || {};
    var alerts = report.alerts !== undefined ? report.alerts : row.alerts;
    var findings = Array.isArray(report.findings) ? report.findings : [];
    var actionable = report.actionable_count !== undefined ? report.actionable_count :
      findings.filter(function (f) { return !f.routine; }).length;
    var assumedZone = meta.assumed_zone || meta.assumed_tz || null;
    var windowSub = report.t_min !== null && report.t_min !== undefined &&
      report.t_max !== null && report.t_max !== undefined
      ? wazuhTime(report.t_min) + " → " + wazuhTime(report.t_max) : "timestamp range unavailable";
    var tiles = el("div", { class: "tiles wazuh-status-tiles" }, [
      statusTile("Parsed alerts", Number(alerts || 0).toLocaleString(),
        row.bytes ? bytes(Number(row.bytes)) + " snapshot" : "normalised JSONL"),
      statusTile("Unparseable", Number(report.unparseable || 0).toLocaleString(),
        report.unparseable ? "excluded from time-based detectors" : "all rows parsed",
        report.unparseable ? "warn-tile" : ""),
      statusTile("Window", report.t_min !== null && report.t_min !== undefined &&
        report.t_max !== null && report.t_max !== undefined ? dur(report.t_max - report.t_min) : "—", windowSub),
      statusTile("Agents", Number(report.agent_count || 0).toLocaleString(), "distinct agents"),
      statusTile("Rules", Number(report.rule_count || 0).toLocaleString(), "distinct rules"),
      statusTile("Findings", actionable + " actionable", findings.length + " total · routine hidden by default"),
      statusTile("Assumed zone", assumedZone || "not needed", assumedZone
        ? "used for timestamps without an offset" : "no ingest-time zone assumption recorded")
    ]);
    children.push(tiles);
    if (report.generated) children.push(el("div", { class: "wazuh-generated muted mono",
      text: "generated " + wazuhTime(report.generated) +
        ((meta.shards || meta.expected_shards) ? " · " +
          (meta.shards || meta.expected_shards) + " merged shards" : "") +
        (report.bucket_minutes ? " · " + report.bucket_minutes + " minute buckets" : "") }));
    st.detailHost.appendChild(el("section", { class: "card wazuh-report-detail" }, children));
    st.detailHost.appendChild(buildWazuhTimeline(report));
    st.detailHost.appendChild(buildWazuhFindings(st, report));
  }

  function normaliseTimeline(report) {
    var raw = report.timeline || report.volume_timeline || report.volume_buckets || report.buckets;
    if (!raw) return [];
    var flagged = {};
    (report.flagged_buckets || []).forEach(function (key) { flagged[String(key)] = true; });
    if (!Array.isArray(raw)) {
      raw = Object.keys(raw).map(function (key) {
        return { t: Number(key), count: raw[key], flagged: !!flagged[key] };
      });
    }
    return raw.map(function (item, index) {
      if (typeof item === "number") {
        return { t: report.t_min + index * (report.bucket_minutes || 1) * 60,
                 count: item, flagged: false };
      }
      var t = item.t !== undefined ? item.t :
        (item.ts !== undefined ? item.ts : (item.time !== undefined ? item.time : item.bucket));
      var count = item.count !== undefined ? item.count :
        (item.alerts !== undefined ? item.alerts : (item.value !== undefined ? item.value : item.n));
      return { t: t, count: Number(count || 0),
        flagged: !!(item.flagged || item.alert || item.state === "flagged" || flagged[String(t)]) };
    }).filter(function (point) { return point.t !== undefined && point.t !== null; });
  }

  function buildWazuhTimeline(report) {
    var points = normaliseTimeline(report);
    var body = [];
    if (!points.length) {
      body.push(el("div", { class: "empty", text:
        "Timeline unavailable for this report. Older reducer reports did not persist per-bucket volume." }));
    } else {
      var maximum = points.reduce(function (m, p) { return Math.max(m, p.count); }, 0) || 1;
      var strip = el("div", { class: "wazuh-timeline", role: "img",
        "aria-label": "Alert volume over time" });
      points.forEach(function (point) {
        var height = Math.max(point.count ? 5 : 1, Math.round(100 * point.count / maximum));
        strip.appendChild(el("div", { class: "wazuh-timeline-bucket" +
          (point.flagged ? " flagged" : ""),
          style: "--bucket-height:" + height + "%",
          title: wazuhTime(point.t) + " · " + point.count.toLocaleString() + " alerts" }));
      });
      body.push(strip);
      body.push(el("div", { class: "wazuh-timeline-axis muted mono" }, [
        el("span", { text: wazuhTime(points[0].t) }),
        el("span", { text: "peak " + maximum.toLocaleString() + " alerts / " +
          (report.bucket_minutes || "?") + " min" }),
        el("span", { text: wazuhTime(points[points.length - 1].t) })
      ]));
    }
    return el("section", { class: "card" }, [
      el("h2", {}, [document.createTextNode("Alert volume timeline"),
        el("span", { class: "h2-actions muted mono", text:
          report.bucket_minutes ? report.bucket_minutes + " minute buckets" : "" })])
    ].concat(body));
  }

  function buildWazuhFindings(st, report) {
    var reportId = (report.report && report.report.id) || report.id ||
                   report.report_id || st.selectedReport;
    var findings = (report.findings || []).map(function (finding, index) {
      return { finding: finding, index: index };
    });
    findings.sort(function (a, b) {
      return Number(b.finding.score || 0) - Number(a.finding.score || 0);
    });
    var ui = { st: st, reportId: st.selectedReport, findings: findings,
               expanded: {}, evidence: {}, evidenceLoading: {}, targets: {} };
    st.findingsUi = ui;
    ui.detector = el("select", {}, [el("option", { value: "", text: "all detectors" })]);
    var detectors = {};
    findings.forEach(function (item) { detectors[item.finding.detector || "unknown"] = true; });
    Object.keys(detectors).sort().forEach(function (name) {
      ui.detector.appendChild(el("option", { value: name, text: name }));
    });
    ui.agent = el("input", { type: "search", placeholder: "agent contains…" });
    ui.rule = el("input", { type: "search", placeholder: "rule contains…" });
    ui.routine = el("input", { type: "checkbox" });
    ui.meta = el("span", { class: "muted mono" });
    ui.reportId = reportId;
    // The findings list is rebuilt wholesale every poll, so any <details> the
    // user opened is recreated closed and appears to collapse a second after
    // being clicked. Remember the open state per finding and restore it.
    ui.openTriage = {};
    ui.openBaseline = {};
    ui.list = el("div", { class: "wazuh-findings" });
    [ui.detector, ui.agent, ui.rule].forEach(function (input) {
      input.addEventListener(input.tagName === "SELECT" ? "change" : "input",
        function () { renderWazuhFindingRows(ui); });
    });
    ui.routine.addEventListener("change", function () { renderWazuhFindingRows(ui); });

    var controls = el("div", { class: "wazuh-finding-controls" }, [
      el("div", { class: "field" }, [el("label", { text: "Detector" }), ui.detector]),
      el("div", { class: "field" }, [el("label", { text: "Agent" }), ui.agent]),
      el("div", { class: "field" }, [el("label", { text: "Rule" }), ui.rule]),
      el("label", { class: "wazuh-routine-toggle" }, [ui.routine,
        el("span", { text: "Show routine / demoted" })]), ui.meta
    ]);
    ui.triageBtn = el("button", { class: "tiny", text: "AI triage",
      onclick: function () { runWazuhTriage(ui); } });
    ui.triageStop = el("button", { class: "tiny danger", text: "Stop",
      style: "display:none",
      onclick: function () { stopWazuhTriage(ui); } });
    ui.triageState = el("span", { class: "muted mono" });
    ui.triageNow = el("div", { class: "wazuh-triage-now", style: "display:none" });
    ui.llmPanel = el("div", { class: "wazuh-llm-settings", style: "display:none" });
    ui.llmToggle = el("button", { class: "tiny", text: "Model settings",
      onclick: function () {
        var open = ui.llmPanel.style.display === "none";
        ui.llmPanel.style.display = open ? "" : "none";
        ui.llmToggle.textContent = open ? "Hide settings" : "Model settings";
        if (open) buildLlmSettings(ui);
      } });
    var section = el("section", { class: "card" }, [
      el("h2", {}, [document.createTextNode("Findings"),
        el("span", { class: "h2-actions" }, [
          ui.triageState, ui.triageStop, ui.llmToggle, ui.triageBtn,
          el("span", { class: "muted mono", text: "score ordered" })])]),
      ui.llmPanel, ui.triageNow, controls, ui.list
    ]);
    renderWazuhFindingRows(ui);
    loadWazuhTriage(ui);
    return section;
  }

  function findingMatches(ui, item) {
    var f = item.finding;
    if (!ui.routine.checked && f.routine) return false;
    if (ui.detector.value && String(f.detector || "unknown") !== ui.detector.value) return false;
    var agent = ui.agent.value.trim().toLowerCase();
    var rule = ui.rule.value.trim().toLowerCase();
    var agentHaystack = [f.agent, f.title, f.why].join(" ").toLowerCase();
    var ruleHaystack = [f.rule, f.title, f.why].join(" ").toLowerCase();
    return (!agent || agentHaystack.indexOf(agent) >= 0) &&
      (!rule || ruleHaystack.indexOf(rule) >= 0);
  }

  // Triage is fetched separately and merged at render time, so the report the
  // cluster produced is never mutated by it -- delete the sidecar and the
  // findings are byte-identical to what the detectors computed.
  function loadWazuhTriage(ui) {
    if (!ui.reportId) return;
    api("/api/wazuh/reports/" + ui.reportId + "/triage").then(function (data) {
      if (!data || data.state === "none") { ui.triageState.textContent = ""; return; }
      ui.triage = data;
      if (data.state === "running") {
        ui.triageState.textContent = "triaging " + (data.done || 0) + "/" + (data.total || 0) + "…";
        ui.triageStop.style.display = "";
        ui.triageBtn.disabled = true;
        ui.triageBtn.textContent = "triaging…";
        // name the finding the model is actually working on: at ~90s each,
        // "triaging 4/20" alone leaves you wondering what it is chewing on
        var now = data.current;
        if (now) {
          clear(ui.triageNow);
          ui.triageNow.style.display = "";
          ui.triageNow.appendChild(el("span", { class: "wazuh-pending",
            text: "analysing" }));
          ui.triageNow.appendChild(el("b", { text: " #" + (now.index + 1) + " " +
            (now.title || "") }));
          ui.triageNow.appendChild(el("span", { class: "muted",
            text: "  ·  " + (now.detector || "") }));
        } else {
          ui.triageNow.style.display = "none";
        }
        setTimeout(function () { loadWazuhTriage(ui); }, 3000);
      } else {
        ui.triageStop.style.display = "none";
        ui.triageNow.style.display = "none";
        ui.triageState.textContent =
          data.state === "cancelled" ? "triage stopped at " + (data.done || 0) + "/" + (data.total || 0)
          : data.state === "failed" ? "triage failed"
          : data.state === "interrupted" ? "triage interrupted — re-run"
          : "triaged " + (data.done || 0)
            + (data.errors ? " (" + data.errors + " errors)" : "");
        ui.triageBtn.disabled = false;
        ui.triageBtn.textContent = "Re-run AI triage";
      }
      // Only rebuild the list when the triage data actually moved. Polling at
      // 3s while each finding takes ~90s meant ~30 pointless full rebuilds per
      // finding, each one resetting scroll position and any open panel.
      var sig = JSON.stringify([data.state, data.done, data.errors,
                                data.current && data.current.index]);
      if (sig !== ui.triageSignature) {
        ui.triageSignature = sig;
        renderWazuhFindingRows(ui);
      }
    }).catch(function () { ui.triageState.textContent = ""; });
  }

  function stopWazuhTriage(ui) {
    if (!ui.reportId) return;
    ui.triageStop.disabled = true;
    ui.triageStop.textContent = "stopping…";
    api("/api/wazuh/reports/" + ui.reportId + "/triage/cancel",
        { method: "POST", body: {} })
      .then(function () {
        // the in-flight model call finishes first, so say so rather than
        // looking frozen for up to ~90s
        ui.triageState.textContent = "stopping after the current finding…";
        setTimeout(function () {
          ui.triageStop.disabled = false;
          ui.triageStop.textContent = "Stop";
          loadWazuhTriage(ui);
        }, 2000);
      })
      .catch(function (e) {
        ui.triageStop.disabled = false;
        ui.triageStop.textContent = "Stop";
        ui.triageState.textContent = e.message;
      });
  }

  function runWazuhTriage(ui) {
    if (!ui.reportId) return;
    ui.triageBtn.disabled = true;
    ui.triageBtn.textContent = "triaging…";
    api("/api/wazuh/reports/" + ui.reportId + "/triage",
        { method: "POST", body: { force: true } })
      .then(function () { setTimeout(function () { loadWazuhTriage(ui); }, 1200); })
      .catch(function (e) {
        ui.triageBtn.disabled = false;
        ui.triageBtn.textContent = "AI triage";
        ui.triageState.textContent = e.message;
      });
  }

  // Model picker and generation settings. Values are saved to the panel host,
  // not the browser, because the model runs there and a triage started from
  // one tab must use the same settings as one started from another.
  function buildLlmSettings(ui) {
    clear(ui.llmPanel);
    ui.llmPanel.appendChild(el("div", { class: "muted", text: "loading model settings…" }));
    api("/api/wazuh/llm/settings").then(function (data) {
      clear(ui.llmPanel);
      if (!data || data.error && !data.settings) {
        ui.llmPanel.appendChild(notice("warn", "Cannot read model settings."));
        return;
      }
      var cur = data.settings || {};
      var fields = data.fields || {};
      var inputs = {};

      var models = (data.models || []).slice();
      if (cur.model && models.indexOf(cur.model) < 0) models.unshift(cur.model);
      var modelSel = el("select", {});
      if (!models.length) {
        modelSel.appendChild(el("option", { value: "", text: "(endpoint unreachable)" }));
      }
      models.forEach(function (m) {
        modelSel.appendChild(el("option", { value: m, text: m }));
      });
      modelSel.value = cur.model || "";
      inputs.model = modelSel;

      function num(name, label, step, hint) {
        var f = fields[name] || {};
        var input = el("input", { type: "number", value: String(cur[name]),
          min: f.min === null ? "" : String(f.min),
          max: f.max === null ? "" : String(f.max), step: step });
        inputs[name] = input;
        return el("div", { class: "field" }, [
          el("label", { text: label }), input,
          el("span", { class: "muted mono",
            text: hint || ("" + f.min + "–" + f.max) })]);
      }
      function flag(name, label, hint) {
        var input = el("input", { type: "checkbox" });
        input.checked = !!cur[name];
        inputs[name] = input;
        return el("label", { class: "wazuh-routine-toggle" }, [input,
          el("span", { text: label }),
          el("span", { class: "muted", text: hint ? " — " + hint : "" })]);
      }

      var status = el("span", { class: "muted mono" });
      var save = el("button", { class: "tiny", text: "Save" });
      var reset = el("button", { class: "tiny danger", text: "Reset to defaults" });

      function send(payload, done) {
        status.textContent = "saving…";
        fetch("/api/wazuh/llm/settings", { method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload) })
          .then(function (r) {
            return r.json().then(function (b) { return { ok: r.ok, body: b }; });
          })
          .then(function (res) {
            if (!res.ok) { status.textContent = res.body.error || "rejected"; return; }
            status.textContent = "saved — applies to the next triage run";
            if (done) done();
          })
          .catch(function (e) { status.textContent = String(e); });
      }
      save.addEventListener("click", function () {
        send({ model: inputs.model.value,
               max_tokens: inputs.max_tokens.value,
               timeout: inputs.timeout.value,
               temperature: inputs.temperature.value,
               top_n: inputs.top_n.value,
               json_schema: inputs.json_schema.checked,
               thinking: inputs.thinking.checked,
               web_search: inputs.web_search.checked,
               all_findings: inputs.all_findings.checked,
               auto: inputs.auto.checked });
      });
      reset.addEventListener("click", function () {
        send({}, function () { buildLlmSettings(ui); });
      });

      var where = data.ok ? "reachable" : ("unreachable" + (data.error ? ": " + data.error : ""));
      // In cluster mode the endpoint is resolved from the Slurm queue on every
      // call, so which node answers is a fact about right now, not config.
      if (data.source === "cluster" && data.node) {
        where = (data.ok ? "" : where + " — ") + data.node +
                (data.job_id ? " · job " + data.job_id : "") +
                (data.loaded ? " · " + data.loaded + " resident" : " · nothing resident");
      }
      ui.llmPanel.appendChild(el("div", { class: "wazuh-llm-grid" }, [
        el("div", { class: "field" }, [el("label", { text: "Model" }), modelSel,
          el("span", { class: "muted mono", text: data.url || "" }),
          // A model that is not the resident one costs a full load off NFS
          // before the first answer, and evicts whatever was on the card.
          data.warning ? el("span", { class: "warn-text mono",
                                      text: data.warning }) : null]),
        num("top_n", "Findings to triage", "1", "highest scoring first"),
        num("max_tokens", "Max tokens", "50"),
        num("timeout", "Timeout (s)", "10"),
        num("temperature", "Temperature", "0.1"),
        el("div", { class: "field" }, [
          flag("json_schema", "Constrain output to JSON schema",
               "required for reasoning models"),
          flag("thinking", "Allow the model to think",
               "off asks the template to skip reasoning; some ignore it"),
          flag("all_findings", "Triage every ranked finding",
               "off uses the limit above; routine findings are never triaged"),
          flag("web_search", "Look up rule background on the web",
               "sends rule ids and MITRE names to a search engine, never hosts, users or IPs"),
          flag("auto", "Triage automatically when a report is ready")])
      ]));
      ui.llmPanel.appendChild(el("div", { class: "wazuh-llm-actions" }, [
        save, reset, status,
        el("span", { class: "muted mono", text: "endpoint " + where })]));
      if ((data.overridden || []).length) {
        ui.llmPanel.appendChild(el("div", { class: "muted mono",
          text: "overridden: " + data.overridden.join(", ") }));
      }
    }).catch(function (e) {
      clear(ui.llmPanel);
      ui.llmPanel.appendChild(notice("warn", "Cannot read model settings: " + e));
    });
  }

  function renderWazuhFindingRows(ui) {
    clear(ui.list);
    ui.targets = {};
    var visible = ui.findings.filter(function (item) { return findingMatches(ui, item); });
    var routineTotal = ui.findings.filter(function (item) { return item.finding.routine; }).length;
    ui.meta.textContent = visible.length + " shown · " + ui.findings.length + " total" +
      (!ui.routine.checked && routineTotal ? " · " + routineTotal + " routine hidden" : "");
    if (!visible.length) {
      ui.list.appendChild(el("div", { class: "empty", text:
        ui.findings.length ? "no findings match these filters" : "no findings in this report" }));
      return;
    }
    visible.forEach(function (item) {
      ui.list.appendChild(buildWazuhFinding(ui, item));
    });
  }

  function buildWazuhFinding(ui, item) {
    var f = item.finding, index = item.index, open = !!ui.expanded[index];
    var score = Number(f.score || 0);
    var evidenceButton = el("button", { class: "tiny", type: "button",
      text: open ? "hide evidence" : "show evidence",
      "aria-expanded": open ? "true" : "false" });
    var evidenceBox = el("div", { class: "finding-evidence" });
    evidenceButton.addEventListener("click", function () {
      ui.expanded[index] = !ui.expanded[index];
      renderWazuhFindingRows(ui);
    });
    var tags = [el("span", { class: "pill idle", text: f.detector || "unknown" })];
    if (f.agent) tags.push(el("span", { class: "wazuh-tag", text: "agent " + f.agent }));
    if (f.rule) tags.push(el("span", { class: "wazuh-tag", text: "rule " + f.rule }));
    if (f.routine) tags.push(el("span", { class: "pill warn", text: "routine · demoted" }));
    // Model verdict rides alongside the detector's own tags. It is advisory:
    // it never changes the score, the order, or whether a finding is shown.
    var tri = ui.triage && ui.triage.items && ui.triage.items[String(index)];
    // queued for triage but not answered yet: say so, so the list visibly
    // fills in rather than looking like the model skipped this one
    if (!tri && ui.triage && ui.triage.state === "running" &&
        (ui.triage.targets || []).indexOf(index) >= 0) {
      tags.push(el("span", { class: "pill idle wazuh-pending",
                             text: "AI: analysing…" }));
    }
    if (tri && tri.error) {
      // Carry the reason on the pill. "AI: unavailable" with no detail meant
      // the only way to find out why was to read the sidecar on the box.
      tags.push(el("span", { class: "pill warn", title: String(tri.error),
                             text: "AI: unavailable" }));
    }
    if (tri && !tri.error) {
      var verdict = { likely_false_positive: ["ok", "likely false positive"],
                      needs_review: ["warn", "needs review"],
                      likely_genuine: ["bad", "likely genuine"] }[tri.assessment]
                    || ["idle", tri.assessment];
      tags.push(el("span", { class: "pill " + verdict[0],
        text: "AI: " + verdict[1] + " (" + (tri.confidence || "low") + ")" }));
    }

    var baseline = el("pre", { class: "wazuh-baseline",
      text: JSON.stringify(f.baseline || {}, null, 2) });
    var card = el("article", { class: "wazuh-finding" + (f.routine ? " routine" : ""),
      "data-finding-index": String(index) }, [
      el("div", { class: "wazuh-finding-head" }, [
        el("div", { class: "wazuh-score" + (score >= 70 ? " high" : score >= 45 ? " medium" : ""),
          text: score.toFixed(1) }),
        el("div", { class: "wazuh-finding-title" }, [
          el("h3", { text: f.title || "Untitled finding" }),
          el("div", { class: "wazuh-finding-tags" }, tags)
        ]), evidenceButton
      ]),
      el("p", { class: "wazuh-why", text: f.why || "No detector explanation supplied." }),
      (tri && !tri.error) ? el("details", {
        class: "wazuh-triage",
        open: ui.openTriage[index] ? "" : null,
        ontoggle: function (ev) { ui.openTriage[index] = ev.target.open; }
      }, [
        el("summary", { text: "AI triage — advisory, not used in scoring" }),
        el("p", { class: "wazuh-triage-line" }, [
          el("b", { text: "What this is: " }), document.createTextNode(tri.explanation || "—")]),
        el("p", { class: "wazuh-triage-line" }, [
          el("b", { text: "Likely trigger: " }), document.createTextNode(tri.trigger || "—")]),
        el("p", { class: "wazuh-triage-line" }, [
          el("b", { text: "Assessment: " }), document.createTextNode(tri.reasoning || "—")])
      ]) : null,
      f.routine ? el("div", { class: "wazuh-routine-reason", text:
        "Demoted as routine: " + (f.routine_reason || "routine series") }) : null,
      el("details", {
        class: "wazuh-baseline-wrap",
        open: ui.openBaseline[index] ? "" : null,
        ontoggle: function (ev) { ui.openBaseline[index] = ev.target.open; }
      }, [
        el("summary", { text: "baseline used by this detector" }), baseline
      ]), evidenceBox
    ]);
    if (open) {
      ui.targets[index] = evidenceBox;
      evidenceBox.appendChild(el("pre", { class: "log", text: "loading evidence…" }));
      showWazuhEvidence(ui, item);
    }
    return card;
  }

  function evidenceText(evidence, raw) {
    var blocks = [];
    (evidence || []).forEach(function (sample) {
      if (typeof sample === "string") {
        try { blocks.push(JSON.stringify(JSON.parse(sample), null, 2)); }
        catch (e) { blocks.push(sample); }
      } else {
        blocks.push(JSON.stringify(sample, null, 2));
      }
    });
    (raw || []).forEach(function (sample) { blocks.push(String(sample)); });
    return blocks.join("\n\n") || "No raw sample alerts were retained for this finding.";
  }

  function paintWazuhEvidence(ui, index, evidence, raw) {
    var target = ui.targets[index];
    if (!target || !ui.expanded[index]) return;
    clear(target);
    target.appendChild(el("pre", { class: "log", text: evidenceText(evidence, raw) }));
  }

  function showWazuhEvidence(ui, item) {
    var index = item.index, f = item.finding;
    if (ui.evidence[index]) {
      paintWazuhEvidence(ui, index, ui.evidence[index].evidence, ui.evidence[index].raw);
      return;
    }
    if (Array.isArray(f.evidence) && f.evidence.length) {
      ui.evidence[index] = { evidence: f.evidence, raw: [] };
      paintWazuhEvidence(ui, index, f.evidence, []);
      return;
    }
    if (ui.evidenceLoading[index]) return;
    ui.evidenceLoading[index] = true;
    api("/api/wazuh/reports/" + encodeURIComponent(ui.reportId) +
        "/evidence?finding=" + index).then(function (data) {
      ui.evidence[index] = { evidence: data.evidence || data.samples || [], raw: data.raw || [] };
      paintWazuhEvidence(ui, index, ui.evidence[index].evidence, ui.evidence[index].raw);
    }).catch(function (e) {
      var target = ui.targets[index];
      if (target) {
        clear(target);
        target.appendChild(el("div", { class: "notice err", text: "Could not load evidence: " + e.message }));
      }
    }).then(function () { delete ui.evidenceLoading[index]; });
  }

  function buildProjectView(p) {
    if (p.view === "wazuh") return buildWazuhView(p);

    var view = el("div", { class: "view", id: "view-" + p.id });
    var ctx = { project: p, products: p.products, pane: "run",
                framesByProduct: {}, cursor: null, playing: false,
                playTimer: null, dayHints: {}, panes: {}, gifPolls: {} };
    state.views[p.id] = ctx;

    ctx.tiles = el("div", { class: "tiles" });

    // shared dataset + time cursor
    ctx.groupSelect = el("select", { name: "group" },
      [el("option", { value: "all", text: "all frames" })]);
    ctx.groupSelect.addEventListener("change", function () {
      // hold the selection in state: the DOM value is lost every time the
      // option list is rebuilt, and it is rebuilt whenever a product reports
      // a different set of groups
      ctx.selectedGroup = ctx.groupSelect.value || "all";
      ctx.cursor = null;
      ctx.framesByProduct = {};
      dueReset();
      refreshProject(ctx, true);
    });

    ctx.slider = el("input", { type: "range", min: "0", max: "0", value: "0" });
    ctx.slider.addEventListener("input", function () {
      var frames = referenceFrames(ctx);
      var f = frames[parseInt(ctx.slider.value, 10)];
      if (f) { ctx.cursor = f.name; renderPanes(ctx); }
    });
    ctx.playBtn = el("button", { class: "tiny", text: "▶ play",
      onclick: function () { togglePlay(ctx); } });
    ctx.cursorLabel = el("div", { class: "frame-name", text: "—" });

    ctx.scrubBar = el("div", { class: "scrub" }, [
      ctx.playBtn, ctx.slider, ctx.cursorLabel
    ]);

    view.appendChild(el("section", { class: "card" }, [
      el("h2", {}, [document.createTextNode(p.label),
        el("span", { class: "h2-actions muted mono", text: p.out_dir })]),
      el("div", { class: "muted", text: p.description,
                  style: "font-size:12.5px;margin-bottom:12px" }),
      ctx.tiles,
      el("div", { class: "inline-form", style: "margin-top:12px" }, [
        el("div", { class: "field", style: "min-width:220px" }, [
          el("label", { text: "Dataset" }), ctx.groupSelect,
          el("div", { class: "help", text: "shared by every view below" })
        ])
      ]),
      ctx.scrubBar
    ]));

    // inner nav
    ctx.subnav = el("nav", { class: "tabs subtabs" });
    view.appendChild(ctx.subnav);

    var entries = [{ id: "run", label: "Run" }].concat(
      p.products.map(function (pr) { return { id: pr.id, label: pr.label }; }),
      [{ id: "compare", label: "Compare" }]);
    entries.forEach(function (e) {
      ctx.subnav.appendChild(el("button", { text: e.label, "data-pane": e.id,
        onclick: function () { selectPane(ctx, e.id); } }));
    });

    ctx.paneHost = el("div");
    view.appendChild(ctx.paneHost);

    ctx.panes.run = buildRunPane(ctx, p);
    ctx.paneHost.appendChild(ctx.panes.run);
    p.products.forEach(function (pr) {
      ctx.panes[pr.id] = buildProductPane(ctx, pr);
      ctx.paneHost.appendChild(ctx.panes[pr.id]);
    });
    ctx.panes.compare = buildComparePane(ctx, p);
    ctx.paneHost.appendChild(ctx.panes.compare);

    selectPane(ctx, "run");
    return view;
  }

  function selectPane(ctx, id) {
    ctx.pane = id;
    Object.keys(ctx.panes).forEach(function (k) {
      ctx.panes[k].style.display = (k === id) ? "" : "none";
    });
    Array.prototype.forEach.call(ctx.subnav.querySelectorAll("button"),
      function (b) { b.classList.toggle("active", b.dataset.pane === id); });
    // the time cursor is meaningless on the Run pane
    ctx.scrubBar.style.display = (id === "run") ? "none" : "";
    dueReset();
    refreshProject(ctx, true);
  }

  function buildRunPane(ctx, p) {
    var pane = el("div");
    ctx.runNotice = el("div");

    var fields = p.params.map(function (param) {
      if (param.kind === "day") return dayField(ctx, param);
      return el("div", { class: "field" }, [
        el("label", { text: param.label }),
        el("input", { name: param.name, value: param.default,
                      placeholder: param.placeholder || "" }),
        param.help ? el("div", { class: "help", text: param.help }) : null
      ]);
    });

    var renderable = p.products.filter(function (pr) { return pr.kind === "series"; });
    ctx.productSelect = el("select", { name: "product" },
      renderable.map(function (pr) {
        return el("option", { value: pr.id, text: pr.label });
      }));
    fields.push(el("div", { class: "field" }, [
      el("label", { text: "Product" }), ctx.productSelect,
      el("div", { class: "help", text: "which image series this run renders" })
    ]));
    fields.push(el("div", { class: "field" }, [
      el("label", { text: "Partition" }),
      el("input", { name: "partition", value: p.default_partition }),
      el("div", { class: "help",
        text: "comma-separated; gpu nodes are used for their cpus, no gres" })
    ]));
    ctx.memInput = el("input", { name: "mem", value: p.default_mem || "" });
    ctx.memHelp = el("div", { class: "help",
                              text: "lower = more tasks run at once" });
    // products have very different footprints (dealiasing costs ~2 GB), so the
    // field follows the selection instead of silently keeping a value that
    // would get the tasks OOM-killed
    function syncMem() {
      var chosen = p.products.filter(function (pr) {
        return pr.id === ctx.productSelect.value; })[0];
      var want = (chosen && chosen.default_mem) || p.default_mem || "";
      ctx.memInput.value = want;
      ctx.memHelp.textContent = chosen && chosen.default_mem
        ? "this product needs more than the project default"
        : "lower = more tasks run at once";
    }
    ctx.productSelect.addEventListener("change", syncMem);
    setTimeout(syncMem, 0);
    fields.push(el("div", { class: "field" }, [
      el("label", { text: "Memory / task" }), ctx.memInput, ctx.memHelp
    ]));
    fields.push(el("div", { class: "field" }, [
      el("label", { text: "Array throttle" }),
      el("input", { name: "throttle", value: String(p.default_throttle),
                    type: "number", min: "1" })
    ]));
    fields.push(el("div", { class: "field" }, [
      el("label", { text: "Time limit" }),
      el("input", { name: "timelimit", value: p.default_timelimit })
    ]));
    fields.push(el("div", { class: "field" }, [
      el("label", { text: "Task limit" }),
      el("input", { name: "limit", value: "", type: "number", min: "1",
                    placeholder: "all" }),
      el("div", { class: "help", text: "cap tasks — handy for a test run" })
    ]));

    ctx.runForm = el("form", {}, [
      el("div", { class: "form-grid" }, fields),
      el("div", { style: "margin-top:12px" },
         [el("button", { class: "primary", type: "submit", text: "Run scan" })])
    ]);
    ctx.runForm.addEventListener("submit", function (ev) {
      ev.preventDefault();
      submitRun(ctx);
    });

    ctx.runsBox = el("div");
    pane.appendChild(el("section", { class: "card" },
      [el("h2", { text: "Run a scan" }), ctx.runNotice, ctx.runForm]));
    pane.appendChild(el("section", { class: "card" },
      [el("h2", { text: "Runs" }), ctx.runsBox]));
    return pane;
  }

  function buildProductPane(ctx, product) {
    var pane = el("div");
    var st = { product: product };
    ctx.gifPolls[product.id] = st;

    st.stage = el("div", { class: "stage" },
      [el("div", { class: "empty", text: "nothing rendered yet" })]);
    st.meta = el("div", { class: "gif-meta" });
    st.notice = el("div");
    st.progress = el("div", { class: "progress-wrap" });

    var head = [el("h2", {}, [document.createTextNode(product.label),
      el("span", { class: "h2-actions muted mono", text: product.out_dir })]),
      el("div", { class: "muted", text: product.description,
                  style: "font-size:12.5px;margin-bottom:10px" })];
    if (product.note) {
      head.push(el("div", { class: "notice ok", text: product.note }));
    }
    if (product.assembles) {
      st.assembleNotice = el("div");
      head.push(el("div", { class: "inline-form", style: "margin-bottom:10px" }, [
        el("button", { class: "primary", type: "button",
          text: "Assemble profile for this dataset",
          onclick: function () { assembleProduct(ctx, st); } }),
        el("span", { class: "muted", style: "font-size:12px",
          text: "combines every per-scan profile in the selected dataset" })
      ]));
      head.push(st.assembleNotice);
    }
    head.push(st.stage);

    var section = el("section", { class: "card" }, head);
    pane.appendChild(section);

    if (product.kind === "series") {
      st.controls = el("div", { class: "inline-form" }, [
        el("div", { class: "field" }, [
          el("label", { text: "Width (px)" }),
          el("input", { name: "width", value: "0", type: "number", min: "0" }),
          el("div", { class: "help", text: "0 = native" })
        ]),
        el("div", { class: "field" }, [
          el("label", { text: "Stride" }),
          el("input", { name: "stride", value: "1", type: "number", min: "1" })
        ]),
        el("div", { class: "field" }, [
          el("label", { text: "FPS" }),
          el("input", { name: "fps", value: "8", type: "number", min: "1", max: "50" })
        ]),
        el("button", { class: "primary", type: "button", text: "Build GIF",
          onclick: function () { buildGif(ctx, st, false); } }),
        el("button", { type: "button", text: "Rebuild",
          onclick: function () { buildGif(ctx, st, true); } }),
        el("a", { class: "btn", href: "#", text: "Download",
                  style: "display:none;text-decoration:none" })
      ]);
      st.download = st.controls.querySelector("a");
      st.gifStage = el("div", { class: "stage" },
        [el("div", { class: "empty", text: "not built yet" })]);
      pane.appendChild(el("section", { class: "card" }, [
        el("h2", {}, [document.createTextNode("Animation"),
          el("span", { class: "h2-actions" }, [st.meta])]),
        st.notice, st.controls, st.progress, st.gifStage
      ]));
    }
    return pane;
  }

  function assembleProduct(ctx, st) {
    // prefer the stored choice: the DOM value is empty while the option list
    // is being rebuilt, which silently widened requests back to "all"
    var group = ctx.selectedGroup || ctx.groupSelect.value || "all";
    notice(st.assembleNotice, "assembling for " + group + "…", "ok");
    api("/api/p/" + ctx.project.id + "/prod/" + st.product.id + "/assemble", {
      method: "POST", body: { group: group }
    }).then(function () { pollAssemble(ctx, st, group); })
      .catch(function (e) { notice(st.assembleNotice, e.message, "err"); });
  }

  function pollAssemble(ctx, st, group) {
    api("/api/p/" + ctx.project.id + "/prod/" + st.product.id +
        "/assemble/status?group=" + encodeURIComponent(group))
      .then(function (s) {
        if (s.running) {
          setTimeout(function () { pollAssemble(ctx, st, group); }, 2000);
          return;
        }
        if (s.error) { notice(st.assembleNotice, s.error, "err"); return; }
        notice(st.assembleNotice, "assembled — refreshing", "ok");
        dueReset();
        refreshProject(ctx, true);
      })
      .catch(function () {});
  }

  function buildComparePane(ctx, p) {
    var pane = el("div");
    var series = p.products;
    function picker(defaultId) {
      return el("select", {}, series.map(function (pr) {
        return el("option", { value: pr.id, text: pr.label,
                              selected: pr.id === defaultId ? "selected" : null });
      }));
    }
    ctx.cmpLeft = picker(series[0] && series[0].id);
    ctx.cmpRight = picker(series[1] ? series[1].id : (series[0] || {}).id);
    ctx.cmpLeftStage = el("div", { class: "stage" });
    ctx.cmpRightStage = el("div", { class: "stage" });
    [ctx.cmpLeft, ctx.cmpRight].forEach(function (sel) {
      sel.addEventListener("change", function () { renderPanes(ctx); });
    });

    pane.appendChild(el("section", { class: "card" }, [
      el("h2", { text: "Compare" }),
      el("div", { class: "muted", style: "font-size:12.5px;margin-bottom:10px",
        text: "Both panels follow the shared time cursor above, so you are "
            + "always looking at the same scan two ways." }),
      el("div", { class: "compare-grid" }, [
        el("div", {}, [el("div", { class: "cmp-head" }, [ctx.cmpLeft]),
                       ctx.cmpLeftStage]),
        el("div", {}, [el("div", { class: "cmp-head" }, [ctx.cmpRight]),
                       ctx.cmpRightStage])
      ])
    ]));
    return pane;
  }

  function isoDate(value) {
    var m = /^(\d{4})[\/-](\d{2})[\/-](\d{2})$/.exec(String(value || "").trim());
    return m ? m[1] + "-" + m[2] + "-" + m[3] : "";
  }

  // A day picker plus the zone that day is measured in. Central is the default
  // because that is how you read a storm; UTC is there because it is how the
  // bucket is actually keyed.
  function dayField(ctx, param) {
    var input = el("input", { type: "date", name: param.name,
                              value: isoDate(param.default) });
    var tz = el("select", { name: param.name + "_tz" }, [
      el("option", { value: "central", text: "Central" }),
      el("option", { value: "utc", text: "UTC" })
    ]);
    var hint = el("div", { class: "help", text: param.help || "" });
    ctx.dayHints[param.name] = { input: input, tz: tz, hint: hint };

    function update() { refreshDayHint(ctx, param.name); }
    input.addEventListener("change", update);
    tz.addEventListener("change", update);
    setTimeout(update, 0);

    return el("div", { class: "field" }, [
      el("label", { text: param.label }),
      el("div", { style: "display:flex;gap:6px" }, [
        el("div", { style: "flex:1 1 auto" }, [input]),
        el("div", { style: "flex:0 0 auto" }, [tz])
      ]),
      hint
    ]);
  }

  function refreshDayHint(ctx, name) {
    var f = ctx.dayHints[name];
    if (!f || !f.input.value) return;
    api("/api/p/" + ctx.project.id + "/resolve-day?date=" +
        encodeURIComponent(f.input.value) + "&mode=" + f.tz.value)
      .then(function (info) {
        if (info.mode === "utc") {
          f.hint.textContent = "S3 prefix " + info.prefixes[0] +
            " — taken as a UTC day, exactly as the bucket stores it";
          return;
        }
        var many = info.prefixes.length > 1;
        f.hint.textContent =
          info.window_start + " → " + info.window_end +
          " (" + info.hours + "h) · " +
          (many ? "spans " + info.prefixes.length + " UTC prefixes ("
                  + info.prefixes.join(", ") + "), trimmed to the local day"
                : "UTC prefix " + info.prefixes[0]);
      })
      .catch(function (e) { f.hint.textContent = e.message; });
  }

  function submitRun(ctx) {
    var data = new FormData(ctx.runForm);
    var params = {}, options = {};
    ctx.project.params.forEach(function (p) {
      params[p.name] = data.get(p.name);
      if (p.kind === "day") params[p.name + "_tz"] = data.get(p.name + "_tz");
    });
    options.product = ctx.productSelect ? ctx.productSelect.value : undefined;
    ["partition", "throttle", "timelimit", "limit", "mem"].forEach(function (k) {
      var v = data.get(k);
      if (v !== null && v !== "") options[k] = v;
    });
    notice(ctx.runNotice, "");
    var btn = ctx.runForm.querySelector("button[type=submit]");
    btn.disabled = true;
    api("/api/p/" + ctx.project.id + "/run", { method: "POST",
      body: { params: params, options: options } })
      .then(function (res) {
        notice(ctx.runNotice, "run #" + res.run_id + " started — listing scans…", "ok");
        dueReset();
        refreshProject(ctx);
      })
      .catch(function (e) { notice(ctx.runNotice, e.message, "err"); })
      .then(function () { btn.disabled = false; });
  }

  function refreshProject(ctx, force) {
    var pid = ctx.project.id;
    if (force) dueReset();
    if (due("project:" + pid, force ? 0 : state.poll.project)) {
      refreshProjectStatus(ctx);
    }
    if (due("frames:" + pid, force ? 0 : state.poll.frames)) refreshFrames(ctx);
    var st = ctx.gifPolls[ctx.pane];
    if (st && st.controls && due("gif:" + pid + ":" + ctx.pane,
                                force ? 0 : state.poll.gif)) {
      pollGif(ctx, st, false);
    }
  }

  function refreshProjectStatus(ctx) {
    var pid = ctx.project.id;
    state.busy = true;
    api("/api/p/" + pid + "/status").then(function (data) {
      var st = data.status;
      clear(ctx.tiles);
      var tiles = [["Scans listed", String(st.list_total),
                    st.lists.length + " list file(s)"],
                   ["Job logs", String(st.log_count), "in " + ctx.project.logs_dir]];
      ctx.products.forEach(function (pr) {
        var ps = (st.products || {})[pr.id];
        if (!ps) return;
        tiles.push([pr.label, String(ps.frame_count),
                    ps.frame_count ? bytes(ps.frame_bytes) : "not rendered"]);
      });
      tiles.forEach(function (t) {
        ctx.tiles.appendChild(el("div", { class: "tile" }, [
          el("div", { class: "label", text: t[0] }),
          el("div", { class: "value", text: t[1] }),
          el("div", { class: "sub", text: t[2] })
        ]));
      });
      renderRuns(ctx, data.runs);
      state.busy = false;
      stamp();
    }).catch(function (e) {
      state.busy = false;
      stamp("error: " + e.message);
    });
  }

  // Runs are reconciled in place rather than re-created on every poll: a
  // teardown/rebuild made each row collapse to a stub and re-expand when the
  // async progress fetch landed, which read as a flicker every few seconds.
  // Keeping the nodes also keeps an open log pane open.
  function renderRuns(ctx, runList) {
    ctx.runNodes = ctx.runNodes || {};

    if (!runList.length) {
      if (!ctx.runsEmpty) {
        clear(ctx.runsBox);
        ctx.runNodes = {};
        ctx.runsEmpty = el("div", { class: "empty", text: "no runs yet" });
        ctx.runsBox.appendChild(ctx.runsEmpty);
      }
      return;
    }
    if (ctx.runsEmpty) { ctx.runsEmpty.remove(); ctx.runsEmpty = null; }

    var seen = {};
    runList.forEach(function (run, index) {
      seen[run.id] = true;
      var node = ctx.runNodes[run.id];
      if (!node) {
        node = createRunNode(ctx, run);
        ctx.runNodes[run.id] = node;
      }
      updateRunNode(ctx, node, run);
      var at = ctx.runsBox.children[index];
      if (at !== node.box) ctx.runsBox.insertBefore(node.box, at || null);
    });

    Object.keys(ctx.runNodes).forEach(function (id) {
      if (seen[id]) return;
      ctx.runNodes[id].box.remove();
      delete ctx.runNodes[id];
    });
  }

  function createRunNode(ctx, run) {
    var node = { id: run.id, fetched: false };
    node.idEl = el("b", { text: "#" + run.id });
    node.pill = el("span", {});
    node.paramsEl = el("span", { class: "muted" });
    node.jobEl = el("span", { class: "muted" });
    node.sourceEl = el("span", { class: "muted", text: run.source });
    node.createdEl = el("span", { class: "muted", text: when(run.created) });
    node.logBtn = el("button", { class: "tiny", text: "logs",
      onclick: function () { toggleLog(ctx, node.run, node.box); } });
    node.cancelBtn = el("button", { class: "tiny danger", text: "cancel",
      onclick: function () {
        api("/api/runs/" + run.id + "/cancel", { method: "POST" })
          .then(function () { refreshProject(ctx); })
          .catch(function (e) { alert(e.message); });
      } });
    node.msgEl = el("div", { class: "msg" });
    node.countsEl = el("div", { class: "counts" });
    node.barWrap = el("div", { class: "progress-wrap" });

    node.box = el("div", { class: "run" }, [
      el("div", { class: "run-head" }, [
        node.idEl, node.pill, node.paramsEl, node.jobEl, node.sourceEl,
        el("span", { class: "grow" }), node.createdEl, node.logBtn, node.cancelBtn
      ]),
      node.msgEl, node.countsEl, node.barWrap
    ]);
    return node;
  }

  function updateRunNode(ctx, node, run) {
    node.run = run;
    var live = ["listing", "submitting", "submitted", "running"].indexOf(run.state) >= 0;

    if (node.state !== run.state) {
      node.state = run.state;
      clear(node.pill);
      node.pill.appendChild(statePill(run.state));
    }
    var params = Object.keys(run.params).map(function (k) {
      return k + "=" + run.params[k]; }).join(" ");
    if (node.paramsEl.textContent !== params) node.paramsEl.textContent = params;

    var job = run.job_id ? "job " + run.job_id : "";
    if (node.jobEl.textContent !== job) node.jobEl.textContent = job;
    node.logBtn.style.display = run.job_id ? "" : "none";
    node.cancelBtn.style.display = live ? "" : "none";

    var msg = run.message || "";
    if (node.msgEl.textContent !== msg) node.msgEl.textContent = msg;
    node.msgEl.style.display = msg ? "" : "none";

    // A finished run's task breakdown never changes, so fetch it once; a live
    // one refreshes on its own slower interval, because each fetch is a
    // squeue plus a sacct over every task in the array.
    if (run.job_id && (!node.fetched ||
        (live && due("runProgress:" + run.id, state.poll.runProgress)))) {
      node.fetched = true;
      api("/api/runs/" + run.id).then(function (info) {
        if (info.counts) updateRunCounts(node, info);
      }).catch(function () {});
    } else if (!run.job_id) {
      node.countsEl.style.display = "none";
      node.barWrap.style.display = "none";
    }
  }

  function updateRunCounts(node, info) {
    var c = info.counts;
    var pct = c.total ? 100 * c.done / c.total : 0;
    node.countsEl.style.display = "";
    node.barWrap.style.display = "";
    clear(node.countsEl);
    [
      el("span", { html: "<b>" + c.done + "/" + c.total + "</b> tasks" }),
      el("span", { class: "ok-text", html: "done <b>" + c.completed + "</b>" }),
      el("span", { class: "warn-text", html: "pending <b>" + c.pending + "</b>" }),
      el("span", { html: "running <b>" + c.running + "</b>" }),
      c.failed ? el("span", { class: "bad-text", html: "failed <b>" + c.failed + "</b>" }) : null,
      c.cancelled ? el("span", { html: "cancelled <b>" + c.cancelled + "</b>" }) : null,
      info.frames_new !== null && info.frames_new !== undefined
        ? el("span", { class: "muted", html: "+<b>" + info.frames_new + "</b> frames" })
        : null
    ].filter(Boolean).forEach(function (n) { node.countsEl.appendChild(n); });

    var fill = node.barWrap.querySelector("i");
    if (!fill) {
      node.barWrap.appendChild(bar(pct, c.failed ? "warn" : "ok"));
    } else {
      fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
      fill.className = c.failed ? "warn" : "ok";
    }
  }

  function toggleLog(ctx, run, box) {
    var existing = box.querySelector("pre.log");
    if (existing) { existing.remove(); return; }
    var pre = el("pre", { class: "log", text: "loading…" });
    box.appendChild(pre);
    api("/api/p/" + ctx.project.id + "/logs/" + run.job_id + "?lines=200")
      .then(function (data) {
        pre.textContent = data.missing
          ? "no log file found yet for job " + run.job_id
          : (data.path + "\n\n" + data.text);
      })
      .catch(function (e) { pre.textContent = "log read failed: " + e.message; });
  }

  // ------------------------------------------------------------ frames /gif
  function referenceFrames(ctx) {
    // the cursor is anchored to whichever product has the most frames, so it
    // still spans the full day when a derived product is only part-rendered
    var best = [];
    Object.keys(ctx.framesByProduct).forEach(function (id) {
      var f = ctx.framesByProduct[id] || [];
      if (f.length > best.length) best = f;
    });
    return best;
  }

  function refreshFrames(ctx) {
    var group = ctx.selectedGroup ||
                (ctx.groupSelect ? ctx.groupSelect.value : "all") || "all";
    var wanted = ctx.pane === "compare"
      ? [ctx.cmpLeft.value, ctx.cmpRight.value]
      : (ctx.pane === "run" ? ctx.products.map(function (p) { return p.id; })
                            : [ctx.pane]);
    // always keep the reference product loaded so the cursor has a spine
    if (ctx.products.length && wanted.indexOf(ctx.products[0].id) < 0) {
      wanted = wanted.concat([ctx.products[0].id]);
    }
    var pending = wanted.length;
    wanted.forEach(function (id) {
      api("/api/p/" + ctx.project.id + "/prod/" + id + "/frames?group=" +
          encodeURIComponent(group))
        .then(function (data) {
          ctx.framesByProduct[id] = data.frames;
          syncGroups(ctx, data.groups, id);
        })
        .catch(function () { ctx.framesByProduct[id] = []; })
        .then(function () { if (--pending === 0) syncCursor(ctx); });
    });
  }

  function syncCursor(ctx) {
    var frames = referenceFrames(ctx);
    ctx.slider.max = String(Math.max(0, frames.length - 1));
    if (!frames.length) { ctx.cursorLabel.textContent = "no frames"; renderPanes(ctx); return; }
    var idx = cursorIndex(ctx, frames);
    if (idx < 0) { idx = frames.length - 1; ctx.cursor = frames[idx].name; }
    ctx.slider.value = String(idx);
    var f = frames[idx];
    ctx.cursorLabel.textContent = (idx + 1) + " / " + frames.length +
      (f.t ? "  ·  " + scanTime(f.t) : "") + "  ·  " + f.name;
    renderPanes(ctx);
  }

  function cursorIndex(ctx, frames) {
    if (!ctx.cursor) return -1;
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].name === ctx.cursor) return i;
    }
    return -1;
  }

  function frameURL(ctx, productId, name) {
    return "/api/p/" + ctx.project.id + "/prod/" + productId + "/frame/" +
      encodeURIComponent(name);
  }

  function paintStage(ctx, stage, productId) {
    var frames = ctx.framesByProduct[productId] || [];
    var name = ctx.cursor;
    var has = name && frames.some(function (f) { return f.name === name; });
    if (!has) {
      // aligned by name, so we can say precisely why a pane is empty
      clear(stage);
      stage.appendChild(el("div", { class: "empty",
        text: frames.length
          ? "not rendered for this scan — run this product to fill it in"
          : "nothing rendered for this product yet" }));
      return;
    }
    var img = stage.querySelector("img");
    if (!img) { clear(stage); img = el("img", {}); stage.appendChild(img); }
    var url = frameURL(ctx, productId, name);
    if (img.getAttribute("src") !== url) img.src = url;
  }

  function renderPanes(ctx) {
    if (ctx.pane === "compare") {
      paintStage(ctx, ctx.cmpLeftStage, ctx.cmpLeft.value);
      paintStage(ctx, ctx.cmpRightStage, ctx.cmpRight.value);
      return;
    }
    var st = ctx.gifPolls[ctx.pane];
    if (st) paintStage(ctx, st.stage, ctx.pane);
  }

  function togglePlay(ctx) {
    if (ctx.playing) {
      clearInterval(ctx.playTimer);
      ctx.playing = false;
      ctx.playBtn.textContent = "▶ play";
      return;
    }
    var frames = referenceFrames(ctx);
    if (!frames.length) return;
    ctx.playing = true;
    ctx.playBtn.textContent = "❚❚ pause";
    ctx.playTimer = setInterval(function () {
      var f = referenceFrames(ctx);
      if (!f.length) return;
      var next = (cursorIndex(ctx, f) + 1) % f.length;
      ctx.cursor = f[next].name;
      ctx.slider.value = String(next);
      var fr = f[next];
      ctx.cursorLabel.textContent = (next + 1) + " / " + f.length +
        (fr.t ? "  ·  " + scanTime(fr.t) : "") + "  ·  " + fr.name;
      renderPanes(ctx);
    }, 125);
  }

  function syncGroups(ctx, groups, productId) {
    if (!groups) return;
    var sel = ctx.groupSelect;

    // syncGroups runs ONCE PER PRODUCT, and products do not agree: each has
    // rendered a different subset of days, so one returns [X,Y] and another
    // returns [X]. Letting whichever responded last define the option list
    // made the selector thrash every poll and drop back to "all frames".
    // Union them by key so the dataset list is stable regardless of which
    // product answered, and which are only part-rendered.
    ctx.groupsByProduct = ctx.groupsByProduct || {};
    if (productId) ctx.groupsByProduct[productId] = groups;
    var byKey = {};
    Object.keys(ctx.groupsByProduct).forEach(function (pid) {
      (ctx.groupsByProduct[pid] || []).forEach(function (g) {
        var seen = byKey[g.key];
        // the same day holds different frame counts per product; show the
        // fullest one so the number never shrinks as panes load
        if (!seen || g.count > seen.count) byKey[g.key] = g;
      });
    });
    var merged = Object.keys(byKey).sort().map(function (k) { return byKey[k]; });
    if (!merged.length) return;

    var signature = merged.map(function (g) { return g.key; }).join("|");
    var total = merged.reduce(function (n, g) { return n + g.count; }, 0);

    if (sel.dataset.signature === signature) {
      // same datasets, refreshed counts: retitle in place, never rebuild
      var opts = sel.options;
      if (opts.length) opts[0].textContent = "all frames (" + total + ")";
      merged.forEach(function (g, i) {
        var opt = opts[i + 1];
        if (opt) opt.textContent = g.label + " UTC (" + g.count + ")";
      });
      ctx.groups = merged;
      return;
    }
    if (document.activeElement === sel) return;   // never rebuild under an open dropdown

    sel.dataset.signature = signature;
    clear(sel);
    sel.appendChild(el("option", { value: "all", text: "all frames (" + total + ")" }));
    merged.forEach(function (g) {
      // the key is a UTC day because that is how the NOAA bucket is laid out,
      // while the frames inside are labelled in local time -- say so.
      sel.appendChild(el("option", { value: g.key,
        text: g.label + " UTC (" + g.count + ")" }));
    });
    // restore from state, not from the DOM. If the chosen day is not in the
    // list yet, fall back to "all" for display but KEEP the choice, so it
    // comes back when that day's frames appear.
    var want = ctx.selectedGroup || "all";
    sel.value = want;
    if (!sel.value) sel.value = "all";
    ctx.groups = merged;
  }

  function gifQuery(ctx, st) {
    var f = st.controls;
    return "width=" + (f.querySelector("[name=width]").value || "0") +
      "&stride=" + (f.querySelector("[name=stride]").value || "1") +
      "&fps=" + (f.querySelector("[name=fps]").value || "8") +
      "&group=" + encodeURIComponent(ctx.groupSelect.value || "all");
  }

  function buildGif(ctx, st, force) {
    notice(st.notice, "");
    var f = st.controls;
    api("/api/p/" + ctx.project.id + "/prod/" + st.product.id + "/gif/build", {
      method: "POST",
      body: { width: f.querySelector("[name=width]").value || "0",
              stride: f.querySelector("[name=stride]").value || "1",
              fps: f.querySelector("[name=fps]").value || "8",
              group: ctx.groupSelect.value || "all", force: !!force }
    }).then(function () { pollGif(ctx, st, true); })
      .catch(function (e) { notice(st.notice, e.message, "err"); });
  }

  function pollGif(ctx, st, aggressive) {
    if (!st.controls) return;
    api("/api/p/" + ctx.project.id + "/prod/" + st.product.id +
        "/gif/status?" + gifQuery(ctx, st))
      .then(function (info) { renderGif(ctx, st, info, aggressive); })
      .catch(function () {});
  }

  function renderGif(ctx, st, info, aggressive) {
    st.meta.textContent = info.frame_count + " of " + info.total_frames +
      " frames" + (info.ready ? " · " + bytes(info.size) + " · built " +
                                ago(info.built_at) : "");
    clear(st.progress);
    var build = info.build;
    if (build && build.state === "running") {
      var pct = build.total ? 100 * build.done / build.total : 0;
      st.progress.appendChild(el("div", { class: "bar-label" }, [
        el("span", { text: build.stage === "palette" ? "building palette" : "encoding" }),
        el("span", { text: build.done + " / " + build.total })
      ]));
      st.progress.appendChild(bar(pct, "ok"));
      if (st.timer) clearTimeout(st.timer);
      st.timer = setTimeout(function () { pollGif(ctx, st, true); }, 1500);
      return;
    }
    if (build && build.state === "error") notice(st.notice, "build failed: " + build.error, "err");

    var base = "/api/p/" + ctx.project.id + "/prod/" + st.product.id + "/gif?";
    if (!info.ready) {
      st.download.style.display = "none";
      clear(st.gifStage);
      st.gifStage.appendChild(el("div", { class: "empty",
        text: info.frame_count ? "not built for these settings yet — hit Build GIF"
                               : "no frames to animate yet" }));
      return;
    }
    st.download.style.display = "";
    st.download.href = base + gifQuery(ctx, st) + "&download=1";
    var url = base + gifQuery(ctx, st) + "&k=" + info.key;
    var img = st.gifStage.querySelector("img");
    if (img && img.getAttribute("data-key") === info.key) return;
    clear(st.gifStage);
    if (info.size > 8 * 1024 * 1024 && !aggressive) {
      st.gifStage.appendChild(el("button", { class: "primary",
        text: "Load animation (" + bytes(info.size) + ", " + info.frame_count + " frames)",
        onclick: function () {
          clear(st.gifStage);
          var i = el("img", { "data-key": info.key });
          i.src = url;
          st.gifStage.appendChild(i);
        } }));
    } else {
      var i2 = el("img", { "data-key": info.key });
      i2.src = url;
      st.gifStage.appendChild(i2);
    }
  }

  // -------------------------------------------------------------- schedules
  function renderScheduleParams() {
    var pid = document.getElementById("sched-project").value;
    var project = state.projects.filter(function (p) { return p.id === pid; })[0];
    var box = document.getElementById("sched-params");
    clear(box);
    if (!project) return;
    project.params.forEach(function (param) {
      var value = param.default;
      if (param.kind === "day") value = "{{date:yesterday}}";
      // schedules stay free text so the {{date:...}} tokens still work
      box.appendChild(el("div", { class: "field" }, [
        el("label", { text: param.label }),
        el("input", { name: "p_" + param.name, value: value }),
        el("div", { class: "help", text: param.kind === "day"
          ? "{{date:yesterday}}, {{date:today}} or {{date:today-3}}, resolved "
            + "on the panel's clock when the schedule fires"
          : (param.help || "") })
      ]));
      if (param.kind === "day") {
        box.appendChild(el("div", { class: "field" }, [
          el("label", { text: "Day measured in" }),
          el("select", { name: "p_" + param.name + "_tz" }, [
            el("option", { value: "central", text: "Central" }),
            el("option", { value: "utc", text: "UTC" })
          ])
        ]));
      }
    });
  }

  function refreshSchedules() {
    api("/api/schedules").then(function (data) {
      document.getElementById("sched-status").textContent =
        data.scheduler.running
          ? "ticking every " + data.scheduler.tick_seconds + "s"
          : "scheduler thread is NOT running";
      var rows = data.schedules.map(function (s) {
        return [
          s.label || ("#" + s.id),
          s.project,
          s.description,
          Object.keys(s.params).map(function (k) { return k + "=" + s.params[k]; }).join(" "),
          s.enabled ? when(s.next_fire) : "—",
          s.last_fired ? ago(s.last_fired) : "never",
          s.last_error ? el("span", { class: "bad-text", text: s.last_error })
                       : (s.last_run_id ? "run #" + s.last_run_id : "—"),
          el("span", {}, [
            el("button", { class: "tiny", text: s.enabled ? "disable" : "enable",
              onclick: function () {
                api("/api/schedules/" + s.id, { method: "PATCH",
                  body: { enabled: !s.enabled } }).then(refreshSchedules);
              } }),
            document.createTextNode(" "),
            el("button", { class: "tiny", text: "run now",
              onclick: function () {
                api("/api/schedules/" + s.id + "/run", { method: "POST" })
                  .then(refreshSchedules)
                  .catch(function (e) { alert(e.message); });
              } }),
            document.createTextNode(" "),
            el("button", { class: "tiny danger", text: "delete",
              onclick: function () {
                if (!confirm("delete this schedule?")) return;
                api("/api/schedules/" + s.id, { method: "DELETE" }).then(refreshSchedules);
              } })
          ])
        ];
      });
      var t = table(["label", "project", "repeat", "params", "next fire", "last fired", "last result", ""],
        rows, "no schedules yet");
      t.id = "sched-table";
      document.getElementById("sched-table").replaceWith(t);
    }).catch(function () {});
  }

  function initSchedules() {
    var sel = document.getElementById("sched-project");
    state.projects.forEach(function (p) {
      sel.appendChild(el("option", { value: p.id, text: p.label }));
    });
    sel.addEventListener("change", renderScheduleParams);
    renderScheduleParams();

    var kind = document.getElementById("sched-kind");
    kind.addEventListener("change", function () {
      var expr = document.getElementById("sched-expr");
      var help = document.getElementById("sched-expr-help");
      if (kind.value === "interval") {
        expr.value = "3600";
        help.textContent = "seconds between runs (minimum 60)";
      } else {
        expr.value = "0 6 * * *";
        help.textContent = "min hour dom mon dow — 0 6 * * * is 06:00 daily";
      }
    });

    document.getElementById("sched-form").addEventListener("submit", function (ev) {
      ev.preventDefault();
      var form = ev.target;
      var data = new FormData(form);
      var pid = data.get("project");
      var project = state.projects.filter(function (p) { return p.id === pid; })[0];
      var params = {};
      project.params.forEach(function (p) {
        params[p.name] = data.get("p_" + p.name);
        if (p.kind === "day") params[p.name + "_tz"] = data.get("p_" + p.name + "_tz");
      });
      notice(document.getElementById("sched-notice"), "");
      api("/api/schedules", { method: "POST", body: {
        project: pid, label: data.get("label"), kind: data.get("kind"),
        expr: data.get("expr"), params: params, options: {}
      } }).then(function () {
        notice(document.getElementById("sched-notice"), "schedule added", "ok");
        refreshSchedules();
      }).catch(function (e) {
        notice(document.getElementById("sched-notice"), e.message, "err");
      });
    });
  }

  // ================================================================ AI
  // How the cluster's inference servers are being used, by every client of
  // them -- a chat UI, a script, someone with curl. Rendered from one
  // /api/ai/overview payload on every poll, so each container is cleared or
  // replaced before it is filled.

  // ------------------------------------------------------------ AI motion
  // Rendering here is destructive: every container is cleared and refilled on
  // the 5s poll, so an animation attached to an element replays forever unless
  // something decides when it is allowed to run. This is that something.
  //
  // `fresh` means "this render is an arrival" -- you opened the tab, or changed
  // the window -- and is the only time entrance animations play. The maps hold
  // what the previous render showed, so a value can be bumped and a table row
  // flashed when they are genuinely new rather than on every tick.

  var aiMotion = {
    fresh: true,
    live: false,
    tiles: {},        // tile label -> the value string last painted
    calls: null,      // request key -> 1, from the previous render
    events: null,     // null means "nothing rendered yet", see aiSeen()
    servers: null     // server name -> was it live, from the previous render
  };

  function aiFresh() {
    aiMotion.fresh = true;
    aiMotion.tiles = {};
    aiMotion.calls = null;
    aiMotion.events = null;
    aiMotion.servers = null;
  }

  function aiStagger(host) {
    // The entrance is a wave, not a flash. Capped so a forty-row container
    // does not spend two seconds arriving.
    Array.prototype.forEach.call(host.children, function (child, i) {
      child.style.setProperty("--i", Math.min(i, 14));
    });
  }

  function aiReveal(host) {
    // toggle, not add: the class has to come back off or the next poll's
    // children inherit a finished animation and paint at opacity 0.
    host.classList.toggle("ai-reveal", aiMotion.fresh);
    // Staggered every time, not just on arrival: --i also offsets the
    // travelling hairline on the tiles, and without it they beat in unison.
    aiStagger(host);
  }

  function aiBump(node, key, value) {
    var prev = aiMotion.tiles[key];
    aiMotion.tiles[key] = value;
    // The node is rebuilt every poll, so the class never needs removing.
    if (prev !== undefined && prev !== value) node.classList.add("bump");
  }

  function aiSeen(bucket, keys) {
    // One flag per row: true when that row was not in the previous render.
    // The first render seeds the set and flags nothing -- seeing every
    // historical row flash on arrival tells you precisely nothing.
    var prev = aiMotion[bucket], next = {}, flags = [];
    keys.forEach(function (k) { next[k] = 1; });
    keys.forEach(function (k) { flags.push(prev !== null && !prev[k]); });
    aiMotion[bucket] = next;
    return flags;
  }

  // What counts as inferencing. requests_processing is the only direct answer
  // the servers give, but it exists only while a model is resident -- with
  // --models-max 1 the router spends real time holding nothing at all. A fresh
  // GPU sample over the idle floor is the fallback, and while nothing is
  // loaded it is the only signal there is. A stale sample is not a signal:
  // gpu-publish.py stops writing when the job dies and nothing deletes the
  // file, so an old reading would pin the card "live" forever.
  var AI_LIVE_UTIL = 25;
  var AI_GPU_MAX_AGE = 30;

  function aiGpuFeeds(s) {
    // A router's own node holds one card; the rest of its compute is other
    // nodes lent over ggml-rpc, each publishing its own feed. Reading only the
    // local card is how the pooled 980 Ti went missing -- and a router can be
    // flat out with its local card idle, so "live" has to look at all of them.
    var feeds = [];
    if (s.gpu) feeds.push({ node: s.node, feed: s.gpu });
    (s.pool || []).forEach(function (w) {
      if (w.gpu) feeds.push({ node: w.node, feed: w.gpu });
    });
    return feeds;
  }

  function aiFeedFresh(feed) {
    var age = feed.age_s;
    return !(age === undefined || age === null || age > AI_GPU_MAX_AGE);
  }

  function aiServerLive(s) {
    var m = s.metrics;
    if (m && (m.requests_processing || 0) > 0) return true;
    // Strata fills its expert cache on the GPUs while it starts: busy, not serving.
    if (s.engine === "strata" && !s.loaded) return false;
    return aiGpuFeeds(s).some(function (f) {
      return aiFeedFresh(f.feed) && (f.feed.gpus || []).some(function (g) {
        return (g.util_pct || 0) >= AI_LIVE_UTIL;
      });
    });
  }

  function aiGpuBusiest(s) {
    // The rate a model reports belongs to the model instance, so a card with
    // nothing resident has none. GPU utilisation is what is still measured,
    // and it can be any card in the pool that is doing the work.
    var top = 0;
    aiGpuFeeds(s).forEach(function (f) {
      (f.feed.gpus || []).forEach(function (g) {
        if ((g.util_pct || 0) > top) top = g.util_pct;
      });
    });
    return top;
  }

  function aiNum(n) {
    if (n === null || n === undefined) return "—";
    return Number(n).toLocaleString();
  }

  function aiToks(n) {
    // Token counts run to millions; a raw integer stops being readable well
    // before that, so anything over 10k is abbreviated and the exact figure
    // goes in the title attribute.
    if (n === null || n === undefined) return "—";
    if (n >= 1e6) return (n / 1e6).toFixed(n < 1e7 ? 2 : 1) + "M";
    if (n >= 10000) return (n / 1000).toFixed(n < 1e5 ? 1 : 0) + "k";
    return Number(n).toLocaleString();
  }

  function aiRate(n) {
    return (n === null || n === undefined) ? "—" : Number(n).toFixed(1);
  }

  function aiCell(text, title) {
    return el("span", { class: "mono", text: text, title: title || null });
  }

  function aiTiles(data) {
    var host = document.getElementById("ai-tiles");
    clear(host);
    var t = data.totals || {};
    [
      ["Tokens served", aiToks(t.total_tokens),
       aiNum(t.prompt_tokens) + " in · " + aiNum(t.generated_tokens) + " out"],
      ["Calls", aiNum(t.calls),
       t.models + (t.models === 1 ? " model" : " models") +
       (t.truncated ? " · " + aiNum(t.truncated) + " hit the context limit" : "")],
      ["Generated", aiToks(t.generated_tokens),
       "avg " + aiNum(t.avg_generated_tokens) + " per call"],
      ["Generation rate", aiRate(t.generated_tok_s) + " tok/s",
       aiRate(t.prompt_tok_s) + " tok/s reading the prompt"],
      ["GPU time", dur(t.busy_s), "avg " + aiRate(t.avg_s) + "s per call"],
      ["Longest context", aiToks(t.longest_context), "tokens in one call"]
    ].forEach(function (tile) {
      // A figure that moved is worth pointing at; the same figure read off
      // another poll is not, so the bump is keyed on the label and fires only
      // on an actual change.
      var value = el("div", { class: "value", text: tile[1] });
      aiBump(value, tile[0], tile[1]);
      host.appendChild(el("div", { class: "tile", title: tile[3] || null }, [
        el("div", { class: "label", text: tile[0] }),
        value,
        el("div", { class: "sub", text: tile[2] || "" })
      ]));
    });
    aiReveal(host);

    // The panel only counts calls made after it started watching, so an empty
    // tab has to say that rather than look broken.
    var note = document.getElementById("ai-collecting");
    var s = data.sampler || {};
    var parts = [];
    parts.push(data.collecting_since
      ? "watching since " + when(data.collecting_since)
      : "not watching any server yet");
    if (s.last_poll) parts.push("sampled " + ago(s.last_poll));
    if (s.interval_s) parts.push("every " + s.interval_s + "s");
    if (s.error) parts.push("error: " + s.error);
    note.textContent = parts.join(" · ");
  }

  function aiKv(rows) {
    var grid = el("div", { class: "ai-kv" });
    rows.forEach(function (row) {
      if (row[1] === null || row[1] === undefined) return;
      grid.appendChild(el("span", { class: "k", text: row[0] }));
      grid.appendChild(row[1] instanceof Node ? el("span", {}, [row[1]])
                                              : el("span", { text: String(row[1]) }));
    });
    return grid;
  }

  function aiStrataCard(s, live, changed) {
    // The STRATA-* models' server (jobs/strata.sh, started by slerm-orch in
    // place of the router). Same shape as the router card; its start-up is
    // drawn by aiProgressBuild from the job's log, and it has no unload: the
    // orchestrator stops the whole job at the first idle tier.
    var m = s.metrics, st = s.start || {};
    var rate = m && (m.generated_tok_s_recent || m.generated_tok_s);
    var starting = !s.loaded && !s.error;
    var card = el("div", { class: "node ai-backend " + (s.ok ? "ok" : "bad") +
                                  (live ? " live" : "") +
                                  (changed ? " just-changed" : "") });
    card.appendChild(el("h3", {}, [
      live ? el("span", { class: "ai-spin" }) : null,
      document.createTextNode("strata"),
      live ? el("span", { class: "pill busy", text: "generating" })
           : starting ? el("span", { class: "pill warn",
                                     text: st.phase === "restarting" ? "restarting" : "starting" })
           : statePill(s.ok ? "running" : "down")
    ]));
    card.appendChild(el("div", { class: "meta",
      text: [s.node, s.partition, s.job_id ? "job " + s.job_id : "",
             s.started ? "up " + dur(Date.now() / 1000 - s.started) : ""
            ].filter(Boolean).join(" · ") }));
    var progress = el("div", { class: "ai-progress", "data-job": String(s.job_id) });
    aiProgressFill(progress, state.aiLive[s.job_id]);
    card.appendChild(progress);
    if (live) {
      var queued = (m && m.requests_deferred) || 0;
      card.appendChild(el("div", { class: "ai-now" }, [
        el("span", { text: "1 request running" }),
        queued ? el("span", { class: "sep", text: "·" }) : null,
        queued ? el("em", { text: queued + " queued" }) : null,
        rate ? el("span", { class: "sep", text: "·" }) : null,
        rate ? el("span", { text: aiRate(rate) + " tok/s" }) : null
      ]));
    }
    var rows = [
      ["engine", "Strata"],
      ["model", s.model || "?"],
      ["state", s.loaded ? "serving" : (st.text || "starting")]
    ];
    if (!s.loaded && st.expected_s) rows.push(["last start", dur(st.expected_s)]);
    if (m) {
      rows.push(["requests", aiNum(m.requests)]);
      rows.push(["prompt tok", aiNum(m.prompt_tokens) +
                 (m.cached_tokens ? " · " + aiNum(m.cached_tokens) + " reused" : "")]);
      rows.push(["output tok", aiNum(m.generated_tokens)]);
      rows.push(["rate", aiRate(m.prompt_tok_s) + " in / " +
                         aiRate(m.generated_tok_s) + " out tok/s"]);
      if (m.hit_rate !== null && m.hit_rate !== undefined) {
        rows.push(["expert cache", Math.round(m.hit_rate * 1000) / 10 + "% hit (last request)"]);
      }
      if (m.draft_accept !== null && m.draft_accept !== undefined) {
        rows.push(["MTP drafts", Math.round(m.draft_accept * 100) + "% accepted"]);
      }
    }
    card.appendChild(aiKv(rows));
    if (s.loaded && aiLockedTo(s.loaded)) card.appendChild(aiLockLine(s.loaded));
    var lockBtn = aiLockButton(s);
    if (lockBtn) card.appendChild(el("div", { class: "ai-actions" }, [lockBtn]));
    if (m) {
      card.appendChild(el("div", { class: "sub muted",
        text: "server counters, since this Strata job started" }));
    }
    aiGpuFeeds(s).forEach(function (f) {
      var where = String(f.node || "").replace(/^slerm-/, "");
      (f.feed.gpus || []).forEach(function (g) {
        card.appendChild(barRow(g.name + (where ? " · " + where : ""),
          g.util_pct + "% · " + g.mem_used_mb + "/" + g.mem_total_mb + " MB · " +
          g.power_w + " W", g.util_pct, loadClass(g.util_pct)));
      });
    });
    if (s.error) {
      card.appendChild(el("div", { class: "ai-err-text", text: s.error }));
    }
    return card;
  }

  function aiServerCard(s, live, changed) {
    if (s.engine === "strata") return aiStrataCard(s, live, changed);
    // Same shape as a cluster node card: h3 with the name and a pill, then a
    // .meta line, then bars. While it is decoding it gains a ring next to the
    // name and a line saying what it is doing -- the shape of a guest card
    // mid-shutdown on the byteme2 outage test.
    var m = s.metrics;
    var rate = m && (m.generated_tok_s_recent || m.generated_tok_s);
    var card = el("div", { class: "node ai-backend " + (s.ok ? "ok" : "bad") +
                                  (live ? " live" : "") +
                                  (changed ? " just-changed" : "") });

    card.appendChild(el("h3", {}, [
      live ? el("span", { class: "ai-spin" }) : null,
      document.createTextNode(s.name || "server"),
      live ? el("span", { class: "pill busy", text: "generating" })
           : statePill(s.ok ? "running" : "down")
    ]));
    card.appendChild(el("div", { class: "meta",
      text: [s.node, s.partition, s.job_id ? "job " + s.job_id : "",
             s.started ? "up " + dur(Date.now() / 1000 - s.started) : ""
            ].filter(Boolean).join(" · ") }));
    // Filled from the faster /api/ai/live poll; redrawn here from the last
    // reading so the 5s card rebuild does not blank it between live polls.
    var progress = el("div", { class: "ai-progress", "data-job": String(s.job_id) });
    aiProgressFill(progress, state.aiLive[s.job_id]);
    card.appendChild(progress);
    // The one line on the card that is about right now rather than about the
    // run so far. Only drawn when there is something to say.
    if (live) {
      var running = (m && m.requests_processing) || 0;
      var queued = (m && m.requests_deferred) || 0;
      card.appendChild(el("div", { class: "ai-now" }, [
        el("span", { text: running
          ? running + (running === 1 ? " request" : " requests") + " decoding"
          : "GPU busy" }),
        queued ? el("span", { class: "sep", text: "·" }) : null,
        queued ? el("em", { text: queued + " queued" }) : null,
        el("span", { class: "sep", text: "·" }),
        el("span", { text: rate ? aiRate(rate) + " tok/s"
                                : aiGpuBusiest(s) + "% GPU" })
      ]));
    }

    var feeds = aiGpuFeeds(s);
    var rows = [
      ["loaded", s.loaded || "nothing resident"],
      ["can serve", (s.models || []).length + " models"]
    ];
    if ((s.pool || []).length) {
      // Without this a 20B model resident on a 6 GB card reads as impossible.
      var cards = 0;
      feeds.forEach(function (f) { cards += (f.feed.gpus || []).length; });
      var cpus = (s.pool || []).filter(function (w) {
        return w.kind === "cpu";
      }).length;
      var parts = [cards + (cards === 1 ? " GPU" : " GPUs")];
      if (cpus) parts.push(cpus + " CPU " + (cpus === 1 ? "node" : "nodes"));
      rows.push(["pool", parts.join(" + ") +
                 (s.pool_confirmed ? "" : " (assumed)")]);
    }
    if (m) {
      rows.push(["prompt tok", aiNum(m.prompt_tokens)]);
      rows.push(["output tok", aiNum(m.generated_tokens)]);
      rows.push(["rate", aiRate(m.prompt_tok_s) + " in / " +
                         aiRate(m.generated_tok_s) + " out tok/s"]);
      if (!live && (m.requests_processing || m.requests_deferred)) {
        rows.push(["in flight", m.requests_processing +
                   " · " + m.requests_deferred + " queued"]);
      }
    }
    card.appendChild(aiKv(rows));
    if (s.loaded && aiLockedTo(s.loaded)) card.appendChild(aiLockLine(s.loaded));
    if (s.loaded && s.job_id) {
      card.appendChild(el("div", { class: "ai-actions" }, [
        aiLockButton(s),
        aiLockedTo(s.loaded) ? null : aiUnloadButton(s, m)
      ]));
    }
    if (m) {
      // These are the SERVER's own counters and they restart at zero every
      // time a model is swapped in, which happens constantly with one model
      // slot. Saying so is the difference between a reading and a mystery.
      card.appendChild(el("div", { class: "sub muted",
        text: "server counters, since " + (s.loaded || "the model") + " was loaded" }));
      // How full the context window is right now -- the one server counter
      // that is a level rather than a running total, so it is the one worth a
      // bar. It moves during a call, which is exactly when the shimmer runs.
      if (m.kv_cache_ratio !== undefined && m.kv_cache_ratio !== null) {
        var kv = Math.round(m.kv_cache_ratio * 100);
        card.appendChild(barRow("kv cache",
          kv + "%" + (m.kv_cache_tokens ? " · " + aiNum(m.kv_cache_tokens) + " tok" : ""),
          kv, loadClass(kv)));
      }
    }
    feeds.forEach(function (f) {
      // The node matters now that more than one can appear: two cards with the
      // same utilisation are a different story on one node than across two.
      var where = String(f.node || "").replace(/^slerm-/, "");
      (f.feed.gpus || []).forEach(function (g) {
        card.appendChild(barRow(g.name + (where ? " · " + where : ""),
          g.util_pct + "% · " + g.mem_used_mb + "/" + g.mem_total_mb + " MB · " +
          g.power_w + " W", g.util_pct, loadClass(g.util_pct)));
      });
    });
    if (s.error) {
      card.appendChild(el("div", { class: "ai-err-text", text: s.error }));
    }
    return card;
  }

  // The card is rebuilt on every poll, so an unload in progress is remembered
  // here by job, or the button would come back enabled mid-request.
  var aiUnloading = {};

  function aiUnloadButton(s, m) {
    var key = s.job_id + "|" + s.loaded;
    var busy = !!aiUnloading[key];
    var btn = el("button", { class: "tiny danger ai-unload",
      text: busy ? "unloading…" : "Unload " + s.loaded,
      title: "Evict the resident model and free the GPUs. The next request " +
             "for any model loads it again." });
    btn.disabled = busy;
    btn.addEventListener("click", function () {
      var inflight = (m && (m.requests_processing || 0) + (m.requests_deferred || 0)) || 0;
      if (!confirm("Unload " + s.loaded + " from " + s.node + "?" +
                   (inflight ? "\n\n" + inflight + " request(s) in flight will be cut off." : ""))) return;
      aiUnloading[key] = true;
      btn.disabled = true;
      btn.textContent = "unloading…";
      api("/api/ai/unload", { method: "POST",
                              body: { job_id: s.job_id, model: s.loaded } })
        .catch(function (err) { alert("Unload failed: " + err.message); })
        .then(function () {
          delete aiUnloading[key];
          dueReset();
          refreshAI();
        });
    });
    return btn;
  }

  // ------------------------------------------------------------ model lock
  // Held by slerm-orch (2026-10-17): a locked model is never unloaded or idled
  // out, requests for other models are refused, ComfyUI and other GPU jobs
  // wait, and the nodes it runs on stay on. The user's own VMs and the UPS
  // still win, and if it goes away anyway the orchestrator brings it back.
  // Hidden until the running orchestrator knows about locks.
  var aiLockBusy = false;

  function aiLockedTo(model) {
    var lk = state.aiLock;
    return !!(lk && model && lk.model === model);
  }

  function aiLockNodes() {
    var nodes = (state.aiLockNodes || []).map(function (n) { return n.replace(/^slerm-/, ""); });
    return nodes.length ? nodes.join(", ") : "its nodes";
  }

  function aiLockLine(model) {
    var lk = state.aiLock || {};
    var since = lk.since ? new Date(lk.since * 1000) : null;
    return el("div", { class: "ai-lock-line" }, [
      el("span", { class: "pill locked", text: "locked" }),
      el("span", { class: "muted", text: [
        since ? "since " + since.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : null,
        aiLockNodes() + ((state.aiLockNodes || []).length === 1 ? " stays on" : " stay on"),
        "other models refused"].filter(Boolean).join(" · ") })
    ]);
  }

  function aiLockButton(s) {
    if (!state.aiLockSupported || !s.loaded) return null;
    var locked = aiLockedTo(s.loaded);
    if (!locked && state.aiLock) return null;
    var btn = el("button", { class: "tiny ai-lock",
      text: aiLockBusy ? (locked ? "unlocking…" : "locking…") : (locked ? "Unlock" : "Lock model"),
      title: locked ? "Let the idle timer, other models and GPU jobs have the cards again."
                    : "Keep " + s.loaded + " loaded until you unlock it." });
    btn.disabled = aiLockBusy;
    btn.addEventListener("click", function () {
      var policy = (state.aiLastOrch && state.aiLastOrch.policy) || {};
      var msg = locked
        ? "Unlock " + s.loaded + "?\n\nThe idle timer starts again from now: unloaded after " +
          fmtMin(policy.idle_release_s || 1200) + " without requests."
        : "Lock " + s.loaded + "?\n\nUntil you unlock it:\n" +
          "  - it is never unloaded or idled out\n" +
          "  - requests for any other model are refused\n" +
          "  - ComfyUI and other GPU jobs wait\n" +
          "  - " + (s.node || "its node").replace(/^slerm-/, "") + " (and any node it uses) stays powered on\n\n" +
          "Your own VMs and the UPS still take priority; if the model goes away anyway, it is brought back.";
      if (!confirm(msg)) return;
      aiLockBusy = true;
      btn.disabled = true;
      btn.textContent = locked ? "unlocking…" : "locking…";
      api(locked ? "/api/ai/unlock" : "/api/ai/lock", { method: "POST",
                                                         body: locked ? {} : { model: s.loaded } })
        .catch(function (err) { alert((locked ? "Unlock" : "Lock") + " failed: " + err.message); })
        .then(function () {
          aiLockBusy = false;
          dueReset();
          refreshAI();
        });
    });
    return btn;
  }

  // ------------------------------------------------------------ AI progress
  // What a server is doing this second: a model being read in, a prompt
  // being evaluated, tokens coming out. Model loading is an ESTIMATE (nothing
  // reports load progress, so it is elapsed time against the file size at the
  // NFS rate) and is labelled so; prompt progress is the server's own count
  // but only advances once per 2048-token batch, so that bar moves in steps.

  function aiMiB(bytes) {
    if (!bytes) return "?";
    var gib = bytes / 1073741824;
    return gib >= 1 ? gib.toFixed(1) + " GiB" : Math.round(bytes / 1048576) + " MiB";
  }

  function aiIndeterminate() {
    return el("div", { class: "bar indeterminate" }, [el("i")]);
  }

  function aiProgressRow(label, right, pct, sub) {
    var row = el("div", { class: "bar-row ai-prog-row" }, [
      el("div", { class: "bar-label" }, [
        el("span", { text: label }), el("span", { text: right })
      ]),
      pct === null || pct === undefined ? aiIndeterminate() : bar(pct, "")
    ]);
    if (sub) row.appendChild(el("div", { class: "ai-prog-sub", text: sub }));
    return row;
  }

  // Keyed like aiUnloading: the progress block is redrawn every 1.5s, so a
  // stop in flight is remembered here or the button would re-enable.
  var aiStopping = {};

  function aiStopButton(live, l) {
    var key = live.job_id + "|" + l.model;
    var busy = !!aiStopping[key];
    var btn = el("button", { class: "tiny danger ai-stop-load",
      text: busy ? "stopping…" : "Stop loading",
      title: "Kill this load. Anything waiting for " + l.model + " fails; " +
             "the next request for it starts the load again from zero." });
    btn.disabled = busy;
    btn.addEventListener("click", function () {
      if (!confirm("Stop loading " + l.model + " on " + live.node + "?\n\n" +
                   "Requests waiting for it will fail, and a client that " +
                   "retries will start the load over.")) return;
      aiStopping[key] = true;
      btn.disabled = true;
      btn.textContent = "stopping…";
      api("/api/ai/stop-load", { method: "POST",
                                 body: { job_id: live.job_id, model: l.model } })
        .catch(function (err) { alert("Stop failed: " + err.message); })
        .then(function () {
          delete aiStopping[key];
          dueReset();
          refreshAILive();
          refreshAI();
        });
    });
    return el("div", { class: "ai-actions" }, [btn]);
  }

  function aiProgressFill(real, live) {
    var host = real.cloneNode(false);
    aiProgressBuild(host, live);
    morphInto(real, Array.prototype.slice.call(host.childNodes));
  }

  function aiProgressBuild(host, live) {
    if (!live) return;
    (live.loading || []).forEach(function (l) {
      if (l.engine === "strata" && l.phase === "restarting") {
        // The server stays up; only its engine is reloading the weights.
        host.appendChild(aiProgressRow(
          "restarting the engine · " + l.model,
          l.phase_elapsed_s ? dur(l.phase_elapsed_s) : "",
          null,
          [l.detail, l.reason ? "after: " + l.reason : null,
           "requests wait until it is back"].filter(Boolean).join(" · ")));
        return;
      }
      if (l.engine === "strata") {
        var known = l.pct !== null && l.pct !== undefined;
        var el_s = l.elapsed_s || 0;
        host.appendChild(aiProgressRow(
          "starting " + l.model,
          known ? Math.round(l.pct) + "% est." : dur(el_s),
          known ? l.pct : null,
          [(l.text || "starting") + (l.phase_elapsed_s ? " (" + dur(l.phase_elapsed_s) + ")" : ""),
           l.detail,
           dur(el_s) + " elapsed",
           l.expected_s ? (el_s > l.expected_s
             ? "taking longer than the last start (" + dur(l.expected_s) + ")"
             : "~" + dur(l.expected_s - el_s) + " left, going by the last start")
             : "first start of this model: no estimate"
          ].filter(Boolean).join(" · ")));
        return;
      }
      var left = l.expected_s ? Math.max(0, l.expected_s - l.elapsed_s) : null;
      var over = l.expected_s && l.elapsed_s > l.expected_s;
      host.appendChild(aiProgressRow(
        "loading " + l.model,
        l.pct === null ? dur(l.elapsed_s) : Math.round(l.pct) + "% est.",
        l.pct,
        [(l.exact_start ? "" : "≥ ") + dur(l.elapsed_s) + " elapsed",
         l.bytes ? aiMiB(l.bytes) : null,
         over ? "taking longer than the estimate"
              : (left !== null ? "~" + dur(left) + " left" : null),
         "at " + l.rate_mb_s + " MB/s " + (l.rate_measured ? "(measured)" : "(assumed)")
        ].filter(Boolean).join(" · ")));
      host.appendChild(aiStopButton(live, l));
    });
    (live.slots || []).forEach(function (sl) {
      var eta = sl.eta_s !== undefined && sl.eta_s !== null ? "~" + dur(sl.eta_s) + " left" : null;
      var strata = sl.engine === "strata";
      if (sl.phase === "prompt" && strata) {
        // Strata reports the read in chunks, and nothing until the first one.
        var spct = sl.prompt_tokens && sl.prompt_done ? 100 * sl.prompt_done / sl.prompt_tokens : null;
        host.appendChild(aiProgressRow(
          "reading prompt",
          (sl.prompt_done ? aiNum(sl.prompt_done) + " / " : "") + aiNum(sl.prompt_tokens) + " tok" +
            (spct !== null ? " · " + Math.round(spct) + "%" : ""),
          spct,
          [sl.prompt_tok_s ? aiRate(sl.prompt_tok_s) + " tok/s" : null, eta,
           aiNum(sl.n_ctx) + " ctx"].filter(Boolean).join(" · ")));
      } else if (sl.phase === "prompt") {
        var pct = sl.prompt_tokens ? 100 * sl.prompt_done / sl.prompt_tokens : 0;
        host.appendChild(aiProgressRow(
          "reading prompt · task " + sl.task,
          aiNum(sl.prompt_done) + " / " + aiNum(sl.prompt_tokens) + " tok · " + Math.round(pct) + "%",
          pct,
          [sl.prompt_tok_s ? aiRate(sl.prompt_tok_s) + " tok/s" : null, eta,
           sl.prompt_cached ? aiNum(sl.prompt_cached) + " reused from cache" : null,
           "steps every 2048 tok"].filter(Boolean).join(" · ")));
      } else {
        // Strata's max_tokens is the thinking budget (~98k by default): no
        // bar or ETA against it, just the count and the limit.
        var cap = strata ? null : sl.max_new;
        host.appendChild(aiProgressRow(
          strata ? (sl.step || "generating") : "generating · task " + sl.task,
          aiNum(sl.decoded) + (cap ? " / " + aiNum(cap) : "") + " tok",
          cap ? 100 * sl.decoded / cap : null,
          [sl.gen_tok_s ? aiRate(sl.gen_tok_s) + " tok/s" : null,
           strata ? (sl.max_new ? "limit " + aiNum(sl.max_new) : null)
                  : (cap ? eta : "no max_tokens set"),
           aiNum(sl.prompt_tokens + sl.decoded) + " / " + aiNum(sl.n_ctx) + " ctx"
          ].filter(Boolean).join(" · ")));
      }
    });
    if (live.queued) {
      host.appendChild(el("div", { class: "ai-prog-sub ai-prog-queued",
        text: live.queued + (live.queued === 1 ? " request" : " requests") +
              " waiting for the slot" }));
    }
  }

  // ------------------------------------------------------- live model I/O
  // Fed by llm-tap.py in the opencode sandbox (via /data/llm/run/live-opencode.json). The poll carries only
  // the streamed output and a request summary; the full raw request is fetched once per exchange, on open.
  var aiRaw = { key: null, nodes: null, full: {}, fetching: {} };

  function rawKey(e) { return e.sess + ":" + e.id; }

  function rawPre(cls) { return el("pre", { class: cls || "" }); }

  // Keep a pre pinned to the bottom while text streams in, unless the reader scrolled up.
  function rawSet(pre, text, placeholder) {
    var atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 30;
    var next = text || placeholder || "";
    if (pre.textContent !== next) {
      pre.textContent = next;
      pre.classList.toggle("empty", !text);
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    }
  }

  function rawContent(c) {
    if (typeof c === "string") return c;
    if (Array.isArray(c)) return c.map(function (p) {
      return p && p.type === "text" ? p.text : JSON.stringify(p);
    }).join("\n");
    return c === null || c === undefined ? "" : JSON.stringify(c, null, 2);
  }

  function rawRenderInput(host, req) {
    clear(host);
    var msgs = req.messages || [];
    msgs.forEach(function (m, i) {
      var body = rawContent(m.content);
      if (m.tool_calls) body += (body ? "\n\n" : "") + "tool_calls: " + JSON.stringify(m.tool_calls, null, 2);
      if (m.reasoning_content) body = "[reasoning] " + m.reasoning_content + "\n\n" + body;
      var d = el("details", { class: "role-" + (m.role || "other") }, [
        el("summary", {}, [el("b", { text: m.role || "?" }),
          " · " + body.length.toLocaleString() + " chars" + (m.name ? " · " + m.name : "")]),
        el("pre", { text: body })
      ]);
      if (i >= msgs.length - 2 && m.role !== "system") d.open = true;
      host.appendChild(d);
    });
    if (req.tools && req.tools.length) {
      host.appendChild(el("details", {}, [
        el("summary", {}, [el("b", { text: req.tools.length + " tools" }), " · " +
          req.tools.map(function (t) { return (t["function"] || t).name; }).join(", ")]),
        el("pre", { text: JSON.stringify(req.tools, null, 2) })
      ]));
    }
    var raw = el("details", {}, [el("summary", {}, [el("b", { text: "raw request JSON" })]), el("pre")]);
    raw.addEventListener("toggle", function () {
      if (raw.open && !raw.lastChild.textContent) raw.lastChild.textContent = JSON.stringify(req, null, 2);
    });
    host.appendChild(raw);
  }

  function rawFetchFull(key, host) {
    if (aiRaw.full[key]) { rawRenderInput(host, aiRaw.full[key]); return; }
    if (aiRaw.fetching[key]) return;
    aiRaw.fetching[key] = true;
    host.textContent = "loading request…";
    api("/api/ai/raw?full=" + encodeURIComponent(key)).then(function (d) {
      aiRaw.fetching[key] = false;
      var req = d.exchange && d.exchange.request;
      if (!req) { host.textContent = "request body not kept (only the newest 3 are)"; return; }
      aiRaw.full[key] = req;
      if (aiRaw.key === key) rawRenderInput(host, req);
    }).catch(function (e) { aiRaw.fetching[key] = false; host.textContent = "could not load: " + e.message; });
  }

  function rawBuild(host, e) {
    clear(host);
    var n = { meta: el("div", { class: "ai-raw-meta" }), reasoning: rawPre("reasoning"), content: rawPre(),
              tools: rawPre(), input: el("div") };
    var inDetails = el("details", {}, [el("summary", {}, [el("b", { text: "Raw input" }),
      el("span", { class: "ai-raw-insum" })]), n.input]);
    inDetails.addEventListener("toggle", function () {
      if (inDetails.open) rawFetchFull(rawKey(e), n.input);
    });
    n.insum = inDetails.querySelector(".ai-raw-insum");
    n.toolsWrap = el("div", {}, [el("h3", { text: "Tool calls" }), n.tools]);
    host.appendChild(n.meta);
    host.appendChild(el("div", { class: "ai-raw-grid" }, [
      el("div", { class: "ai-raw-col" }, [el("h3", { text: "Input" }), inDetails]),
      el("div", { class: "ai-raw-col" }, [el("h3", { text: "Reasoning" }), n.reasoning,
        el("h3", { text: "Output" }), n.content, n.toolsWrap])
    ]));
    return n;
  }

  function rawUpdate(n, e, age) {
    var s = e.request_summary || {};
    var streaming = !e.done && age < 30;
    clear(n.meta);
    [["", "#" + e.id],
     ["model ", s.model || "?"],
     ["", streaming ? "streaming" : (e.done ? "done" : "no update " + dur(age)), streaming ? "streaming" :
       (e.finish === "disconnected" ? "stopped" : "")],
     ["finish ", e.finish || "—"],
     ["effort ", s.reasoning_effort || "default"],
     ["max ", s.max_tokens || "router cap"],
     ["tokens ", e.usage ? (e.usage.prompt_tokens + " in / " + e.usage.completion_tokens + " out") :
       ((e.reasoning || "").length + (e.content || "").length).toLocaleString() + " chars out"],
     ["started ", when(e.t0)]
    ].forEach(function (f) {
      n.meta.appendChild(el("span", { class: f[2] || "" }, [f[0], el("b", { text: String(f[1]) })]));
    });
    n.insum.textContent = " · " + (s.messages || 0) + " messages, " + (s.tools || 0) + " tools, " +
      (s.chars || 0).toLocaleString() + " chars" + (s.last_user ? " — last: " + s.last_user.slice(-120) : "");
    rawSet(n.reasoning, e.reasoning, "(no reasoning)");
    rawSet(n.content, e.content, streaming ? "(waiting for output…)" : "(no text output)");
    var calls = Object.keys(e.tool_calls || {}).map(function (k) {
      var c = e.tool_calls[k], args = c.arguments;
      try { args = JSON.stringify(JSON.parse(args), null, 2); } catch (x) { /* still streaming */ }
      return c.name + " " + args;
    });
    n.toolsWrap.style.display = calls.length ? "" : "none";
    rawSet(n.tools, calls.join("\n\n"));
  }

  function refreshAIRaw() {
    api("/api/ai/raw").then(function (doc) {
      var host = document.getElementById("ai-raw"), pick = document.getElementById("ai-raw-pick");
      var list = doc.exchanges || [];
      document.getElementById("ai-raw-state").textContent =
        doc.missing ? "" : "updated " + dur(doc.age) + " ago";
      if (!list.length) {
        aiRaw.key = null; aiRaw.nodes = null;
        clear(pick);
        clear(host);
        host.appendChild(el("div", { class: "notice", text: doc.missing
          ? "No live feed yet. It starts with the next opencode launched through the sandbox wrapper."
          : (doc.error || "No exchanges yet.") }));
        return;
      }
      var keys = list.map(rawKey);
      if (pick.options.length !== keys.length ||
          Array.prototype.some.call(pick.options, function (o, i) { return o.value !== keys[i]; })) {
        var keep = pick.value;
        clear(pick);
        list.forEach(function (e) {
          var s = e.request_summary || {};
          pick.appendChild(el("option", { value: rawKey(e),
            text: "#" + e.id + " · " + (s.model || "?") + " · " + when(e.t0) + (e.done ? "" : " · live") }));
        });
        pick.value = keys.indexOf(keep) >= 0 ? keep : keys[0];
      }
      if (document.getElementById("ai-raw-follow").checked) pick.value = keys[0];
      var e = list[keys.indexOf(pick.value)] || list[0];
      if (aiRaw.key !== rawKey(e) || !aiRaw.nodes) {
        aiRaw.key = rawKey(e);
        aiRaw.nodes = rawBuild(host, e);
      }
      rawUpdate(aiRaw.nodes, e, e.t ? Math.max(0, Date.now() / 1000 - e.t) : doc.age);
    }).catch(function () { /* keep the last view; the 5s overview reports reachability */ });
  }

  function refreshAILive() {
    api("/api/ai/live").then(function (data) {
      var next = {};
      (data.servers || []).forEach(function (s) { next[s.job_id] = s; });
      state.aiLive = next;
      strataMonitor(data.servers);
      Array.prototype.forEach.call(
        document.querySelectorAll("#ai-servers .ai-progress"), function (host) {
          aiProgressFill(host, next[host.getAttribute("data-job")]);
        });
    }).catch(function () { /* the 5s overview reports reachability */ });
  }

  // ---------------------------------------------------------------- Strata monitor
  // 2026-10-21 (user: "can i have these kind of analytics in the AI tab", showing Strata's own Monitor page). Drawn
  // from the 1.5 s /api/ai/live poll: llmusage._strata_monitor sends Strata's /metrics (live state, engine, last 15
  // requests) and node04's published feed (cards, CPU, RAM, disk). The sparklines' history is kept here, per job.
  var strataHist = { job: null, rows: [] };
  var STRATA_HIST_MAX = 240;                         // ~6 min at the 1.5 s poll

  function smFeed(mon) {
    var f = mon.feed;
    return f && (f.age_s === undefined || f.age_s === null || f.age_s <= AI_GPU_MAX_AGE) ? f : null;
  }

  function smReqPrefill(r) {
    return r && r.prompt_read && r.prompt_ms ? r.prompt_read / (r.prompt_ms / 1000) : null;
  }

  function smSample(job, mon) {
    if (strataHist.job !== job) strataHist = { job: job, rows: [] };
    var lv = mon.live || {}, f = smFeed(mon), gpus = (f && f.gpus) || [], h = (f && f.host_stats) || {};
    var sum = function (k) { return gpus.reduce(function (a, g) { return a + (g[k] || 0); }, 0); };
    var last = (mon.requests || [])[0];
    strataHist.rows.push({
      decode: lv.state === "generating" ? (lv.tok_s || 0) : 0,
      gpu: gpus.length ? sum("util_pct") / gpus.length : null,
      vram: gpus.length ? sum("mem_used_mb") / 1024 : null,
      temp: gpus.length ? Math.max.apply(null, gpus.map(function (g) { return g.temp_c || 0; })) : null,
      power: gpus.length ? sum("power_w") : null,
      cpu: h.cpu_pct !== undefined ? h.cpu_pct : null,
      ram: h.mem_used_mb !== undefined ? h.mem_used_mb / 1024 : null,
      disk: h.disk_read_mb_s !== undefined ? h.disk_read_mb_s : null,
      pcie: gpus.some(function (g) { return g.pcie_rx_mb_s !== undefined; }) ? sum("pcie_rx_mb_s") : null,
      net: h.net_rx_mb_s !== undefined ? h.net_rx_mb_s + (h.net_tx_mb_s || 0) : null,
      _last: last
    });
    if (strataHist.rows.length > STRATA_HIST_MAX) strataHist.rows.shift();
  }

  function smSpark(key, cls, max) {
    var pts = strataHist.rows.map(function (r) { return r[key]; });
    var vals = pts.filter(function (v) { return v !== null && v !== undefined; });
    if (vals.length < 2) return '<svg class="sm-spark"></svg>';
    // Spread over what is there (at least 40 samples' width), so a fresh page draws a line, not a sliver.
    var span = Math.max(pts.length, 40), top = Math.max(max || 0, Math.max.apply(null, vals)) || 1, n = span - 1;
    var off = span - pts.length, line = "", started = false, first = 0, lastX = 0;
    pts.forEach(function (v, i) {
      if (v === null || v === undefined) return;
      var x = ((off + i) / n * 100).toFixed(2), y = (27 - (v / top) * 25).toFixed(2);
      if (!started) first = x;
      line += (started ? "L" : "M") + x + " " + y + " ";
      started = true; lastX = x;
    });
    return '<svg class="sm-spark ' + (cls || "") + '" viewBox="0 0 100 28" preserveAspectRatio="none">' +
      '<path class="a" d="' + line + "L" + lastX + " 28 L" + first + ' 28 Z"/><path class="l" d="' + line + '"/></svg>';
  }

  function smTile(label, value, unit, sub, spark) {
    return '<div class="sm-tile"><div class="l">' + esc(label) + '</div><div class="v">' + esc(value) +
      (unit ? "<small>" + esc(unit) + "</small>" : "") + '</div><div class="s" title="' + esc(sub || "") + '">' +
      esc(sub || "") + "</div>" + spark + "</div>";
  }

  function smGauge(pct, label) {
    var r = 52, c = 2 * Math.PI * r, arc = c * 0.75, fill = arc * Math.max(0, Math.min(1, pct / 100));
    var cls = pct >= 90 ? "bad" : pct >= 75 ? "warn" : "";
    return '<svg viewBox="0 0 140 140" class="sm-gauge-svg ' + cls + '">' +
      '<circle cx="70" cy="70" r="' + r + '" class="t" stroke-dasharray="' + arc.toFixed(1) + " " + c.toFixed(1) +
      '" transform="rotate(135 70 70)"/>' +
      '<circle cx="70" cy="70" r="' + r + '" class="f" stroke-dasharray="' + fill.toFixed(1) + " " + c.toFixed(1) +
      '" transform="rotate(135 70 70)"/>' +
      '<text x="70" y="70" class="p">' + Math.round(pct) + '%</text><text x="70" y="90" class="u">' + esc(label) +
      "</text></svg>";
  }

  function smBar(label, right, pct, cls) {
    return '<div class="bar-row"><div class="bar-label"><span>' + esc(label) + "</span><span>" + esc(right) +
      '</span></div><div class="bar"><i class="' + (cls || "ok") + '" style="width:' +
      Math.max(0, Math.min(100, pct || 0)).toFixed(1) + '%"></i></div></div>';
  }

  function smRequests(reqs) {
    var finish = { stop: "Done", length: "Max tokens", cancel: "Cancelled", error: "Error" };
    var head = ["Time", "Status", "Prompt", "Reused", "Read", "Output", "Tok/s", "Prefill tok/s", "VRAM hit",
                "MTP", "Duration", "RAM reads", "Disk reads", "Disk MB"];
    var rows = reqs.map(function (r) {
      var t = r.time ? new Date(((r.time || 0) + (r.duration_s || 0)) * 1000) : null;
      var pre = smReqPrefill(r);
      var mtp = r.drafts_offered ? Math.round(100 * r.drafts_accepted / r.drafts_offered) + "%" : "—";
      var hit = r.hit_rate !== null && r.hit_rate !== undefined
        ? (r.hit_rate * 100).toFixed(1) + "%" + (r.pcie_share ? " +" + (r.pcie_share * 100).toFixed(1) + "% PCIe" : "")
        : "—";
      var cells = [t ? t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—",
                   '<span class="sm-st ' + esc(r.finish || "") + '">' + esc(finish[r.finish] || r.finish || "?") + "</span>",
                   aiNum(r.prompt_tokens), aiNum(r.reused), aiNum(r.prompt_read), aiNum(r.output_tokens),
                   r.decode_tok_s ? aiRate(r.decode_tok_s) : "—", pre ? Math.round(pre).toLocaleString() : "—",
                   hit, mtp, r.duration_s !== null && r.duration_s !== undefined ? aiRate(r.duration_s) + " s" : "—",
                   aiNum(r.ram_blobs), aiNum(r.file_blobs), r.file_mb ? aiRate(r.file_mb) : "0"];
      return "<tr>" + cells.map(function (c, i) {
        return "<td" + (i >= 2 && i !== 1 ? ' class="num"' : "") + ">" + (i === 1 ? c : esc(c)) + "</td>";
      }).join("") + "</tr>";
    }).join("");
    return '<div class="table-wrap"><table class="sm-table"><thead><tr>' +
      head.map(function (h, i) { return "<th" + (i >= 2 ? ' class="num"' : "") + ">" + h + "</th>"; }).join("") +
      "</tr></thead><tbody>" + (rows || '<tr><td colspan="14" class="muted">No requests yet in this job.</td></tr>') +
      "</tbody></table></div>";
  }

  function strataMonitor(servers) {
    var host = document.getElementById("ai-strata");
    if (!host) return;
    var s = (servers || []).filter(function (x) { return x.engine === "strata" && x.monitor; })[0];
    if (!s) { if (host.innerHTML) host.innerHTML = ""; return; }
    var mon = s.monitor, lv = mon.live || {}, eng = mon.engine || {}, reqs = mon.requests || [];
    smSample(s.job_id, mon);
    var f = smFeed(mon), gpus = (f && f.gpus) || [], h = (f && f.host_stats) || null;
    var last = reqs[0] || null;
    var stateName = lv.state === "reading" ? "reading" : lv.state === "generating" ? "generating" : "idle";
    var queued = lv.queued || 0;
    var pills = [["idle", "Idle"], ["reading", "Reading"], ["generating", "Generating"], ["queued", "Queued"]]
      .map(function (p) {
        var on = p[0] === "queued" ? queued > 0 : p[0] === stateName;
        return '<span class="sm-pill ' + p[0] + (on ? " on" : "") + '">● ' + p[1] +
          (p[0] === "queued" && queued ? " " + queued : "") + "</span>";
      }).join("");
    var ctx = eng.max_context || 0;
    var used = stateName === "idle"
      ? (last ? (last.prompt_tokens || 0) + (last.output_tokens || 0) : 0)
      : (lv.prompt_tokens || lv.prompt_total || 0) + (lv.generated || 0);
    var ctxPct = ctx ? 100 * used / ctx : 0;
    var what, right, pct, barCls = "ok";
    if (stateName === "reading") {
      var tot = lv.prompt_total || lv.prompt_tokens || 0, done = lv.prompt_read || 0;
      what = "Reading the prompt";
      pct = tot ? 100 * done / tot : 0;
      right = aiNum(done) + " of " + aiNum(tot) + " tokens" +
        (lv.prefill_tok_s_mean ? " · " + Math.round(lv.prefill_tok_s_mean).toLocaleString() + " tok/s" : "");
      barCls = "";
    } else if (stateName === "generating") {
      var ph = String(lv.phase || "answering");
      what = ph.charAt(0).toUpperCase() + ph.slice(1);
      pct = ctxPct;
      right = aiNum(lv.generated) + " tokens · " + aiRate(lv.tok_s) + " tok/s" +
        (lv.elapsed_s ? " · " + dur(lv.elapsed_s) : "");
    } else {
      what = "Idle";
      pct = 0;
      right = last ? "last request " + ago((last.time || 0) + (last.duration_s || 0)) : "";
    }
    // tiles
    var sum = function (k) { return gpus.reduce(function (a, g) { return a + (g[k] || 0); }, 0); };
    var short = function (g) { return String(g.name || "GPU").replace(/^(NVIDIA |Tesla )/, "").replace(/-PCIE-.*$/, ""); };
    var perCard = function (fn) { return gpus.map(fn).join(" · "); };
    var decodeNow = stateName === "generating" ? lv.tok_s : (last && last.decode_tok_s);
    var prefill = lv.state === "reading" ? lv.prefill_tok_s_mean : smReqPrefill(last);
    var tiles = [];
    tiles.push(smTile("Speed", decodeNow ? aiRate(decodeNow) : "—", "tok/s",
      (stateName === "generating" ? "decode now" : "last request") +
      (prefill ? " · prefill " + Math.round(prefill).toLocaleString() + " tok/s" : ""), smSpark("decode", "ok")));
    if (gpus.length) {
      var lim = sum("power_limit_w");
      tiles.push(smTile("GPU load", Math.round(sum("util_pct") / gpus.length), "%",
        perCard(function (g) { return short(g) + " " + g.util_pct + "%"; }), smSpark("gpu", "ok", 100)));
      tiles.push(smTile("VRAM", (sum("mem_used_mb") / 1024).toFixed(1), "/ " + Math.round(sum("mem_total_mb") / 1024) + " GiB",
        eng.expert_cache_mib ? aiNum(eng.expert_slots) + " experts · " + (eng.expert_cache_mib / 1024).toFixed(1) + " GiB" : "",
        smSpark("vram", "", sum("mem_total_mb") / 1024)));
      tiles.push(smTile("GPU temp", Math.max.apply(null, gpus.map(function (g) { return g.temp_c || 0; })), "°C",
        perCard(function (g) { return g.temp_c + "°C"; }), smSpark("temp", "warm", 90)));
      tiles.push(smTile("Power", Math.round(sum("power_w")), "W",
        (lim ? "of " + Math.round(lim) + " W limit · " : "") + perCard(function (g) { return Math.round(g.power_w || 0) + " W"; }),
        smSpark("power", "warm", lim || 0)));
    } else {
      ["GPU load", "VRAM", "GPU temp", "Power"].forEach(function (l) {
        tiles.push(smTile(l, "—", "", "no fresh feed from " + (s.node || "the node"), smSpark("none")));
      });
    }
    var g0 = gpus[0] || {};
    var hasPcie = gpus.some(function (g) { return g.pcie_rx_mb_s !== undefined; });
    tiles.push(smTile("PCIe", g0.pcie_gen ? "Gen" + g0.pcie_gen : "—",
      gpus.filter(function (g) { return g.pcie_width; }).map(function (g) { return "x" + g.pcie_width; }).join(" · "),
      hasPcie ? "to GPU " + Math.round(sum("pcie_rx_mb_s")).toLocaleString() + " MB/s · from " +
                Math.round(sum("pcie_tx_mb_s")).toLocaleString() + " · " +
                perCard(function (g) { return Math.round(g.pcie_rx_mb_s || 0) + ""; })
              : "the node feed adds this at the next job start", smSpark("pcie", "")));
    if (h && h.cpu_pct !== undefined) {
      tiles.push(smTile("CPU", aiRate(h.cpu_pct), "%", (h.cpus || "?") + " CPUs on " + String(s.node || "").replace(/^slerm-/, ""),
        smSpark("cpu", "", 100)));
      tiles.push(smTile("RAM", (h.mem_used_mb / 1024).toFixed(1), "/ " + Math.round(h.mem_total_mb / 1024) + " GiB",
        "page cache " + ((h.cache_mb || 0) / 1024).toFixed(1) + " GiB", smSpark("ram", "", h.mem_total_mb / 1024)));
      tiles.push(smTile("Disk read", aiRate(h.disk_read_mb_s), "MB/s",
        "write " + aiRate(h.disk_write_mb_s) + " MB/s", smSpark("disk", "")));
      tiles.push(smTile("Network", aiRate((h.net_rx_mb_s || 0) + (h.net_tx_mb_s || 0)), "MB/s",
        "in " + aiRate(h.net_rx_mb_s) + " · out " + aiRate(h.net_tx_mb_s), smSpark("net", "")));
    } else {
      ["CPU", "RAM", "Disk read", "Network"].forEach(function (l) {
        tiles.push(smTile(l, "—", "", "the node feed adds this at the next job start", smSpark("none")));
      });
    }
    // context + experts
    var hitPct = last && last.hit_rate !== null && last.hit_rate !== undefined ? last.hit_rate * 100 : null;
    var mtp = last && last.drafts_offered ? 100 * last.drafts_accepted / last.drafts_offered : null;
    var vramTotal = sum("mem_total_mb");
    var side = '<div class="sm-ctx"><div class="sm-sub">Context fill</div>' +
      smGauge(ctxPct, aiToks(used) + " / " + aiToks(ctx)) +
      smBar("Experts on the GPUs", aiNum(eng.expert_slots) + " · " + ((eng.expert_cache_mib || 0) / 1024).toFixed(1) + " GiB",
            vramTotal ? 100 * (eng.expert_cache_mib || 0) / vramTotal : 0, "ok") +
      (hitPct !== null ? smBar("VRAM hit (last)", hitPct.toFixed(1) + "%" +
            (last.pcie_share ? " +" + (last.pcie_share * 100).toFixed(1) + "% PCIe" : ""), hitPct, "ok") : "") +
      (mtp !== null ? smBar("MTP accepted (last)", Math.round(mtp) + "%", mtp, "") : "") +
      '<div class="sm-facts">' + [
        "KV " + (eng.kv || "?") + (eng.kv_resident ? " · " + aiToks(eng.kv_resident) + " resident" : ""),
        eng.conversation_cache_slots ? "conversation cache " + eng.conversation_cache_slots + " × " +
          ((eng.conversation_cache_mib || 0) / 1024).toFixed(1) + " GiB" : null,
        eng.spec ? "spec " + eng.spec + (eng.mtp_max ? " · MTP " + eng.mtp_max : "") : null,
        eng.version ? "engine " + eng.version : null
      ].filter(Boolean).map(esc).join("<br>") + "</div></div>";
    var html = '<div class="sm-head"><h3>Strata monitor</h3>' + pills +
      '<span class="sm-model">' + esc(eng.model || s.loaded || "") + "</span></div>" +
      '<div class="sm-state"><span>' + esc(what) + "</span><span>" + esc(right) + "</span></div>" +
      '<div class="bar"><i class="' + barCls + (stateName !== "idle" ? " live" : "") + '" style="width:' +
      Math.max(stateName === "idle" ? 0 : 1, Math.min(100, pct)).toFixed(1) + '%"></i></div>' +
      '<div class="sm-tiles">' + tiles.join("") + "</div>" +
      '<div class="sm-bottom">' + side + '<div class="sm-reqs"><div class="sm-sub">Recent requests</div>' +
      smRequests(reqs) + "</div></div>";
    if (host.innerHTML !== html) host.innerHTML = html;
  }

  // ---------------------------------------------------------------- ComfyUI
  // ComfyUI shares node04's cards with the LLM router (2026-10-06): comfy-gate on cryoai serves the UI from a
  // CPU-only ComfyUI and starts a one-card Slurm worker when a prompt is queued; the worker exits after an idle
  // spell and the router comes back on the next LLM request. Same card shape as a server, built from
  // /api/ai/overview's `comfy` (comfy.summary()).
  var aiComfyStopping = false;

  var AI_COMFY_PILL = {
    generating: ["busy", "generating"], ready: ["ok", "ready"], starting: ["warn", "starting"],
    waiting: ["warn", "waiting for GPU"], off: ["idle", "off"]
  };

  function aiComfyCard(c) {
    var live = c.phase === "generating";
    var pill = AI_COMFY_PILL[c.phase] || ["idle", c.phase];
    var card = el("div", { class: "node ai-backend ai-comfy " + (c.phase === "off" ? "idle" : "ok") +
                                  (live ? " live" : "") });
    card.appendChild(el("h3", {}, [
      live ? el("span", { class: "ai-spin" }) : null,
      document.createTextNode("ComfyUI"),
      el("span", { class: "pill " + pill[0], text: pill[1] })
    ]));
    var job = c.job || {};
    var since = job.start_time || c.worker_up_since;   // the job's own start, not the gate's view
    card.appendChild(el("div", { class: "meta",
      text: [c.node || "slerm-node04", "1 × P100", job.job_id ? "job " + job.job_id : "",
             since && c.phase !== "waiting" ? "up " + dur(Date.now() / 1000 - since) : ""
            ].filter(Boolean).join(" · ") }));

    // The one line about right now, as on a server card.
    var now = null;
    var q = c.queue || {};
    if (live) {
      var p = c.progress || {};
      var parts = [el("span", { text: (q.running || 1) + " running" })];
      if (q.pending) parts.push(el("span", { class: "sep", text: "·" }), el("em", { text: q.pending + " queued" }));
      if (p.max) parts.push(el("span", { class: "sep", text: "·" }),
                            el("span", { text: "step " + p.value + "/" + p.max }));
      if (p.node) parts.push(el("span", { class: "sep", text: "·" }), el("span", { text: "node " + p.node }));
      now = el("div", { class: "ai-now" }, parts);
    } else if (c.phase === "waiting") {
      var why = job.state === "PENDING"
        ? "job " + job.job_id + " waiting on " + (job.reason || "resources") +
          " · the LLM router yields after 2 idle min"
        : (c.held ? c.held + (c.held === 1 ? " prompt" : " prompts") + " held at the gate" : "starting the worker");
      now = el("div", { class: "ai-now" }, [el("span", { text: why })]);
    } else if (c.phase === "starting") {
      now = el("div", { class: "ai-now" }, [el("span", { text: "ComfyUI booting on the card" +
        (c.held ? " · " + c.held + " held" : "") })]);
    }
    if (now) card.appendChild(now);
    if (live && c.progress && c.progress.max) {
      var pct = Math.round(100 * c.progress.value / c.progress.max);
      card.appendChild(barRow("sampling", c.progress.value + "/" + c.progress.max, pct, "ok"));
    }

    var rows = [["open", el("a", { href: c.open_url, target: "_blank", rel: "noopener",
                                   text: c.open_url.replace(/^https?:\/\//, "").replace(/\/$/, "") })]];
    if (c.phase === "off") rows.push(["GPU", "none held · starts on the next prompt"]);
    if (c.queue) rows.push(["queue", (q.running || 0) + " running · " + (q.pending || 0) + " queued"]);
    if (c.idle_left_s !== null && c.idle_left_s !== undefined) {
      rows.push(["frees the GPU", "in " + dur(c.idle_left_s) + " if nothing is queued"]);
    }
    if (c.last_done) {
      rows.push(["last prompt", (c.last_done.ok ? "done" : "failed") +
                 (c.last_done.seconds ? " in " + c.last_done.seconds + " s" : "") + " · " +
                 dur(Date.now() / 1000 - c.last_done.at) + " ago"]);
    }
    if (c.sessions) rows.push(["browsers", String(c.sessions)]);
    if (!c.gate_reachable) rows.push(["gate", "cryoai:8188 not answering"]);
    card.appendChild(aiKv(rows));

    if (c.gpu && aiFeedFresh(c.gpu)) {
      (c.gpu.gpus || []).forEach(function (g) {
        card.appendChild(barRow(g.name, g.util_pct + "% · " + g.mem_used_mb + "/" + g.mem_total_mb + " MB · " +
          g.power_w + " W", g.util_pct, loadClass(g.util_pct)));
      });
    }
    if (job.job_id) card.appendChild(aiComfyStopButton(c));
    return card;
  }

  function aiComfyStopButton(c) {
    var btn = el("button", { class: "tiny danger ai-unload",
      text: aiComfyStopping ? "stopping…" : "Stop worker",
      title: "Cancel the ComfyUI job now and free the card, instead of waiting for its idle exit. " +
             "The next prompt starts it again." });
    btn.disabled = aiComfyStopping;
    btn.addEventListener("click", function () {
      var q = c.queue || {};
      var busy = (q.running || 0) + (q.pending || 0);
      if (!confirm("Stop the ComfyUI worker (job " + c.job.job_id + ")?" +
                   (busy ? "\n\n" + busy + " prompt(s) running or queued will be dropped." : ""))) return;
      aiComfyStopping = true;
      btn.disabled = true;
      btn.textContent = "stopping…";
      api("/api/ai/comfy/stop", { method: "POST", body: {} })
        .catch(function (err) { alert("Stop failed: " + err.message); })
        .then(function () {
          aiComfyStopping = false;
          dueReset();
          refreshAI();
        });
    });
    return el("div", { class: "ai-actions" }, [btn]);
  }

  function aiServers(servers, comfy) {
    state.aiServersLast = servers || [];
    servers = servers || [];
    var host = document.getElementById("ai-servers");
    if (!servers.length && !comfy) {
      clear(host);
      host.appendChild(el("div", { class: "notice",
        text: "No inference server is running on the cluster. Submit one from " +
              "/data/llm/jobs and it will appear here." }));
      return;
    }
    // Decide every card's state before drawing any of them, so a server that
    // just started or stopped decoding can flash once on the transition.
    var prev = aiMotion.servers, next = {};
    servers.forEach(function (s) { next[s.name] = aiServerLive(s); });
    var cards = servers.map(function (s) {
      var live = next[s.name];
      var changed = prev !== null && prev[s.name] !== undefined &&
                    prev[s.name] !== live;
      return aiServerCard(s, live, changed);
    });
    if (!servers.length) {
      cards.push(el("div", { class: "notice",
        text: "No LLM router is running: the gateway starts one on the next request" +
              (comfy && comfy.phase !== "off" ? " (it waits while ComfyUI holds a card)." : ".") }));
    }
    if (comfy) cards.push(aiComfyCard(comfy));
    morphInto(host, cards);
    aiMotion.servers = next;
    aiReveal(host);
  }

  function aiChart(series) {
    var host = document.getElementById("ai-series");
    var points = (series && series.points) || [];
    var label = document.getElementById("ai-series-label");
    label.textContent = points.length
      ? points.length + " " + (series.bucket || "") + " buckets"
      : "nothing recorded yet";
    if (!points.length) {
      morphInto(host, [el("div", { class: "notice",
        text: "No calls recorded in this window." })]);
      return;
    }
    var peak = 0;
    points.forEach(function (p) {
      if ((p.total_tokens || 0) > peak) peak = p.total_tokens || 0;
    });
    var chart = el("div", { class: "ai-chart" + (aiMotion.fresh ? " grow" : "") });
    points.forEach(function (p, i) {
      var scale = peak ? 100 / peak : 0;
      var col = el("div", { class: "col", title:
        when(p.t) + " — " + aiNum(p.calls) + " calls, " +
        aiNum(p.prompt_tokens) + " prompt + " + aiNum(p.generated_tokens) +
        " generated = " + aiNum(p.total_tokens) + " tokens" });
      var gen = el("i", { class: "seg completion" });
      gen.style.height = ((p.generated_tokens || 0) * scale) + "%";
      var prompt = el("i", { class: "seg prompt" });
      prompt.style.height = ((p.prompt_tokens || 0) * scale) + "%";
      col.appendChild(gen);
      col.appendChild(prompt);
      col.style.setProperty("--i", i);
      // The last bucket is the one still being filled. Marking it only while
      // the cluster is actually working keeps the chart from claiming a
      // leading edge on a window whose newest bucket closed hours ago.
      if (aiMotion.live && i === points.length - 1) col.className = "col now";
      chart.appendChild(col);
    });
    // Morphed, not swapped: the bars ease to their new heights on a poll and
    // only grow up from the axis when the tab or the window is freshly opened.
    if (aiMotion.fresh) clear(host);
    morphInto(host, [chart,
      el("div", { class: "ai-chart-axis" }, [
        el("span", { text: when(points[0].t) }),
        el("span", { text: when(points[points.length - 1].t) })
      ]),
      el("div", { class: "ai-legend" }, [
        el("span", {}, [el("i", { class: "completion" }), "generated"]),
        el("span", {}, [el("i", { class: "prompt" }), "prompt"]),
        el("span", { class: "muted", text: "peak " + aiToks(peak) + " tokens" })
      ])]);
  }

  // Every table is rebuilt and swapped in by id, the way the queue table is:
  // this runs on every poll and must not accumulate.
  function aiTable(id, headers, rows, empty, flags) {
    var built = table(headers, rows, empty);
    built.id = id;
    var trs = built.tBodies[0].rows;
    if (aiMotion.fresh) {
      built.className = "ai-reveal-rows";
      for (var r = 0; r < trs.length; r++) {
        trs[r].style.setProperty("--i", Math.min(r, 12));
      }
    } else if (flags) {
      // Only ever on a later render: on the first one every row is "new" and
      // flagging them all would be a light show that means nothing.
      for (var j = 0; j < trs.length && j < flags.length; j++) {
        if (flags[j]) trs[j].className = "just-added";
      }
    }
    document.getElementById(id).replaceWith(built);
  }

  function aiByModel(rows) {
    aiTable("ai-by-model",
      ["model", "calls", "prompt", "generated", "total", "avg out", "tok/s",
       "GPU time", "last used"],
      (rows || []).map(function (r) {
        return [r.model || "—", aiNum(r.calls),
                aiCell(aiToks(r.prompt_tokens), aiNum(r.prompt_tokens)),
                aiCell(aiToks(r.generated_tokens), aiNum(r.generated_tokens)),
                aiCell(aiToks(r.total_tokens), aiNum(r.total_tokens)),
                aiNum(r.avg_generated_tokens), aiRate(r.generated_tok_s),
                dur(r.busy_s), when(r.last_ts)];
      }), "no calls recorded in this window");
  }

  // Load / stop, per model. Remembered across the 5s table rebuild like the
  // other buttons here, keyed by job and model.
  var aiLoadingReq = {};

  // What the orchestrator says a model needs. "ready" = on the running
  // router; "cold" = reachable, the orchestrator brings up whatever it needs on
  // the first request; "unavailable" = cannot run on anything reachable now.
  function aiAvailPill(r) {
    var a = r.avail;
    if (!a) return el("span", { class: "pill idle", text: "—" });
    var ready = r.job_id && r.state !== "unavailable" && r.state !== "cold";
    var label = !a.reachable ? "unavailable" : ready ? "ready" : "cold";
    var cls = !a.reachable ? "bad" : ready ? "ok" : "warn";
    var tip = (a.reason || "") + (a.topology ? "\ndevices: " + a.topology : "") +
              (a.est_tok_s ? "\n~" + a.est_tok_s + " tok/s" : "");
    var wrap = el("span", { class: "ai-avail" });
    wrap.appendChild(el("span", { class: "pill " + cls, text: label, title: tip }));
    if (a.reachable && a.tested === false) {
      wrap.appendChild(el("span", { class: "pill idle ai-tag", text: "derived split",
        title: "No tested split for the devices available now; one was derived from the full-pool split." }));
    }
    if (r.cached) {
      wrap.appendChild(el("span", { class: "pill ok ai-tag", text: "local copy",
        title: "Kept on the router node's own disk (last 2 models used, 2 h), so it loads without NFS." }));
    }
    if (r.staged) wrap.appendChild(aiStagedPill(r.staged));
    if (!a.reachable && a.reason) {
      wrap.appendChild(el("div", { class: "ai-avail-why", text: a.reason }));
    }
    return wrap;
  }

  // Staging: a copy on every node the model will load on, made ahead of use.
  function aiStagedPill(s) {
    var pct = s.total ? Math.floor(100 * s.done / s.total) : 0;
    var label = s.state === "done" ? "staged" : s.state === "no room" ? "no room to stage"
      : s.state === "error" ? "stage failed" : "staging " + pct + "%";
    var cls = s.state === "done" ? "ok" : s.state === "staging" ? "warn" : "bad";
    var tip = Object.keys(s.nodes || {}).map(function (n) {
      var d = s.nodes[n];
      var p = d.total ? " " + Math.floor(100 * (d.done || 0) / d.total) + "% of " + bytes(d.total) : "";
      return n.replace("slerm-", "") + " (" + (d.role === "router" ? "full copy" : "worker tensors") + "): " +
             d.state + (d.state === "done" ? "" : p) + (d.note ? " · " + d.note : "");
    }).join("\n");
    return el("span", { class: "pill " + cls + " ai-tag", text: label,
      title: tip + "\n\nfor devices " + s.topology });
  }

  // The nodes a model would load on (the orchestrator's best reachable device
  // set): router node first, then each worker with a share of the split.
  function aiNodes(r) {
    var ns = r.avail && r.avail.reachable && r.avail.nodes;
    if (!ns || !ns.length) return el("span", { class: "ai-nodes none", text: "—" });
    var wrap = el("span", { class: "ai-nodes" });
    ns.forEach(function (n) {
      var short = n.node.replace("slerm-", "").replace("node0", "0");
      wrap.appendChild(el("span", { class: "ai-node " + n.role, text: short,
        title: n.node + " · " + (n.role === "router" ? "router (its own GPU/CPU)" :
               n.role === "gpu" ? "GPU worker" : "CPU worker") + " · " + n.share + "% of the model" }));
    });
    return wrap;
  }

  var aiStageReq = {};

  function aiStageButton(r) {
    var un = !!r.staged, busy = !!aiStageReq[r.id];
    var btn = el("button", { class: "tiny ai-stage" + (un ? " quiet" : ""),
      text: busy ? "…" : un ? "Unstage" : "Stage",
      title: un ? "Drop the staged copies of " + r.id + " (ones kept because it was used recently stay)."
                : "Copy " + r.id + " onto every node it will load on (" + ((r.avail && r.avail.topology) || "?") +
                  ") now, so the load reads local disks. Nodes that are off are woken. At most " +
                  (state.aiStageMax || 2) + " models are staged; staging another drops the oldest." });
    btn.disabled = busy;
    btn.addEventListener("click", function () {
      if (!un) {
        var staged = state.aiStaged || [], max = state.aiStageMax || 2;
        if (staged.length >= max && !confirm("Stage " + r.id + "?\n\n" + staged[0].model +
            " is the oldest of " + max + " staged models and will be unstaged.")) return;
      }
      aiStageReq[r.id] = true;
      btn.disabled = true;
      btn.textContent = "…";
      api("/api/ai/stage", { method: "POST", body: { model: r.id, unstage: un } })
        .catch(function (err) { alert((un ? "Unstage" : "Stage") + " failed: " + err.message); })
        .then(function () { delete aiStageReq[r.id]; dueReset(); refreshAI(); });
    });
    return btn;
  }

  var aiBringReq = {};

  function aiBringUpButton(r) {
    var busy = !!aiBringReq[r.id];
    var btn = el("button", { class: "tiny ai-load", text: busy ? "starting…" : "Bring up",
      title: "Have the orchestrator start what " + r.id + " needs (" +
             ((r.avail && r.avail.topology) || "?") + ") and load it." });
    btn.disabled = busy;
    btn.addEventListener("click", function () {
      if (!confirm("Bring up " + r.id + "?\n\nThe router may restart on a different device " +
                   "set, cutting off anything in flight, and machines may be started.")) return;
      aiBringReq[r.id] = true;
      btn.disabled = true;
      btn.textContent = "starting…";
      api("/api/ai/bringup", { method: "POST", body: { model: r.id } })
        .catch(function (err) { alert("Bring-up failed: " + err.message); })
        .then(function () { delete aiBringReq[r.id]; dueReset(); refreshAI(); });
    });
    return btn;
  }

  function aiModelAction(r) {
    var main = aiModelMainAction(r);
    // Staging a model that is already resident gains nothing; unstaging always works.
    // STRATA models read their GGUFs from /data/llm and warm them instead; the orchestrator refuses to stage them.
    var strata = r.engine === "strata" || (r.avail && r.avail.engine === "strata");
    var canStage = !strata && (r.staged || (r.avail && r.avail.reachable && r.state !== "loaded" && r.state !== "loading"));
    if (!canStage) return main;
    var wrap = el("span", { class: "ai-actions" });
    if (main) wrap.appendChild(main);
    wrap.appendChild(aiStageButton(r));
    return wrap;
  }

  function aiModelMainAction(r) {
    if (r.avail && !r.avail.reachable) return "";
    if (state.aiLock && r.id !== state.aiLock.model && (r.state === "cold" || r.state === "unloaded")) {
      return el("span", { class: "muted", text: "locked out",
                          title: "slerm is locked to " + state.aiLock.model + ": unlock it to use this model" });
    }
    if (r.state === "cold") return aiBringUpButton(r);
    if (!r.job_id) return "";
    if (r.engine === "strata") return "";   // the router's load/stop/unload calls mean nothing to Strata
    if (r.state === "loading") {
      return aiStopButton({ job_id: r.job_id, node: r.node }, { model: r.id });
    }
    if (r.state !== "unloaded") return "";
    var key = r.job_id + "|" + r.id;
    var busy = !!aiLoadingReq[key];
    var btn = el("button", { class: "tiny ai-load",
      text: busy ? "starting…" : "Load",
      title: "Load " + r.id + " now. The router holds one model at a time, " +
             "so whatever is resident is evicted first." });
    btn.disabled = busy;
    btn.addEventListener("click", function () {
      var srv = (state.aiServersLast || []).filter(function (s) {
        return String(s.job_id) === String(r.job_id);
      })[0] || {};
      var m = srv.metrics;
      var inflight = (m && (m.requests_processing || 0) + (m.requests_deferred || 0)) || 0;
      var busyLoad = (state.aiLive[r.job_id] && state.aiLive[r.job_id].loading || [])
        .map(function (l) { return l.model; });
      var msg = "Load " + r.id + " on " + (r.node || "the router") + "?";
      if (srv.loaded) {
        msg += "\n\n" + srv.loaded + " will be unloaded first" +
               (inflight ? ", cutting off " + inflight + " request(s) in flight." : ".");
      }
      if (busyLoad.length) {
        msg += "\n\n" + busyLoad.join(", ") + " is still loading; the router " +
               "may refuse until it finishes or is stopped.";
      }
      msg += "\n\nLarge models take several minutes to read in.";
      if (!confirm(msg)) return;
      aiLoadingReq[key] = true;
      btn.disabled = true;
      btn.textContent = "starting…";
      api("/api/ai/load", { method: "POST", body: { job_id: r.job_id, model: r.id } })
        .catch(function (err) { alert("Load failed: " + err.message); })
        .then(function () {
          delete aiLoadingReq[key];
          dueReset();
          refreshAILive();
          refreshAI();
        });
    });
    return btn;
  }

  function aiCatalogue(rows) {
    // A model with zero calls is the point of this table, not a gap in it:
    // it is the difference between what the cluster hosts and what it is
    // actually asked for.
    aiTable("ai-catalogue",
      ["model", "state", "can run", "nodes", "", "context", "size on disk", "params", "quant", "calls", "tokens", "last used"],
      (rows || []).map(function (r) {
        return [r.id || "—", statePill(r.state || "unknown"), aiAvailPill(r), aiNodes(r), aiModelAction(r),
                r.ctx ? aiNum(r.ctx) : "—",
                r.size_bytes ? bytes(r.size_bytes) : "—",
                r.params ? (r.params / 1e9).toFixed(r.params < 1e10 ? 1 : 0) + "B" : "—",
                r.quant || "—", aiNum(r.calls),
                aiCell(aiToks(r.total_tokens), aiNum(r.total_tokens)),
                r.last_ts ? when(r.last_ts) : "never"];
      }), "no models found on any running server");
    // Grey out what cannot run right now; the reason is on the row.
    var tb = document.getElementById("ai-catalogue");
    if (tb && tb.tBodies[0]) {
      (rows || []).forEach(function (r, i) {
        var tr = tb.tBodies[0].rows[i];
        if (tr && r.avail && !r.avail.reachable) tr.classList.add("ai-unavail");
      });
    }
  }

  function aiBySize(rows) {
    aiTable("ai-by-size", ["generated", "calls", "tokens", "GPU time"],
      (rows || []).map(function (r) {
        return [r.bucket, aiNum(r.calls), aiToks(r.generated_tokens),
                dur(r.busy_s)];
      }), "no calls yet");
  }

  function aiByNode(rows) {
    aiTable("ai-by-node", ["node", "model", "calls", "tokens", "GPU time"],
      (rows || []).map(function (r) {
        return [r.node || "—", r.model || "—", aiNum(r.calls),
                aiToks(r.total_tokens), dur(r.busy_s)];
      }), "no calls yet");
  }

  function aiEvents(rows) {
    rows = rows || [];
    var flags = aiSeen("events", rows.map(function (r) {
      return r.id !== undefined ? "i" + r.id : r.ts + "/" + r.model + "/" + r.event;
    }));
    aiTable("ai-events", ["when", "model", "event"],
      rows.map(function (r) {
        return [when(r.ts), r.model || "—", statePill(r.event)];
      }), "no model swaps seen", flags);
  }

  function aiCalls(rows) {
    // A call the cluster served since you last looked slides in at the top.
    // `id` is the row id from the ledger, so the key survives a re-render even
    // though the timestamp is only when the panel READ the record.
    rows = rows || [];
    var flags = aiSeen("calls", rows.map(function (r) {
      return r.id !== undefined ? "i" + r.id : r.ts + "/" + r.model;
    }));
    aiTable("ai-calls",
      ["when", "model", "node", "prompt", "generated", "total", "context",
       "took", "tok/s", ""],
      rows.map(function (r) {
        return [when(r.ts), r.model || "—", r.node || "—",
                aiNum(r.prompt_tokens), aiNum(r.generated_tokens),
                aiNum(r.total_tokens), aiNum(r.context_tokens),
                aiRate(r.total_s) + "s", aiRate(r.generated_tok_s),
                r.truncated ? el("span", { class: "pill bad", text: "truncated" }) : ""];
      }), "no calls recorded in this window", flags);
  }

  function renderAI(data) {
    // Decided once, up front: the tiles, the chart and the view itself all key
    // off the same answer, and asking twice could give two.
    aiMotion.live = (data.servers || []).some(aiServerLive);
    renderOrch(data.orchestrator || null);
    state.aiLastOrch = data.orchestrator || null;
    state.aiLock = (data.orchestrator && data.orchestrator.lock) || null;
    state.aiLockSupported = !!(data.orchestrator && data.orchestrator.lock_supported);
    state.aiLockNodes = (data.orchestrator && data.orchestrator.lock_nodes) || [];
    state.aiStaged = (data.orchestrator && data.orchestrator.staged) || [];
    state.aiStageMax = (data.orchestrator && data.orchestrator.stage_max) || 2;
    document.getElementById("view-ai").classList.toggle("live", aiMotion.live);
    aiTiles(data);
    aiServers(data.servers, data.comfy);
    aiChart(data.series);
    aiByModel(data.by_model);
    aiCatalogue(data.catalogue);
    aiBySize(data.by_size);
    aiByNode(data.by_node);
    aiEvents(data.events);
    aiCalls(data.requests);
    // Whatever this render was, the next one is a poll, not an arrival.
    aiMotion.fresh = false;
  }

  // The window list comes from the API rather than being hard-coded, so
  // adding a window server-side needs no frontend change.
  var AI_WINDOW_LABELS = { "1h": "last hour", "6h": "last 6 hours",
    "24h": "last 24 hours", "7d": "last 7 days", "30d": "last 30 days",
    "90d": "last 90 days", "all": "all time" };

  function initAI() {
    // The section cards ride in on arrival in document order. Set once: they
    // are the only things on this tab that are never rebuilt.
    Array.prototype.forEach.call(
      document.querySelectorAll("#view-ai > section.card"), function (s, i) {
        s.style.setProperty("--i", i);
      });

    var sel = document.getElementById("ai-window");
    sel.addEventListener("change", function () {
      state.aiWindow = sel.value;
      // A different window is a different set of numbers, not a change in the
      // ones on screen. Come in fresh rather than bumping every tile at once.
      aiFresh();
      dueReset();
      refreshAI();
    });
    document.getElementById("ai-sample").addEventListener("click", function (ev) {
      // The sampler already runs on its own interval; this is for when you
      // have just called a model and do not want to wait for the next tick.
      ev.target.disabled = true;
      ev.target.classList.add("working");
      ev.target.textContent = "sampling…";
      api("/api/ai/poll", { method: "POST", body: {} }).catch(function () {})
        .then(function () {
          ev.target.disabled = false;
          ev.target.classList.remove("working");
          ev.target.textContent = "Sample now";
          dueReset();
          refreshAI();
        });
    });
  }

  function aiWindowOptions(windows) {
    var sel = document.getElementById("ai-window");
    if (sel.options.length) return;
    (windows || []).forEach(function (w) {
      sel.appendChild(el("option", { value: w, text: AI_WINDOW_LABELS[w] || w }));
    });
    sel.value = state.aiWindow;
  }

  // Opening the tab is the one moment the whole view is allowed to animate in.
  function aiEnter() {
    var view = document.getElementById("view-ai");
    aiFresh();
    view.classList.add("ai-enter");
    // Off again once it has played, or the class outlives the arrival and the
    // cards animate a second time the next time this tab is shown.
    setTimeout(function () { view.classList.remove("ai-enter"); }, 1400);
    refreshAI();
  }

  function refreshAI() {
    if (state.busy) return;
    state.busy = true;
    api("/api/ai/overview?window=" + encodeURIComponent(state.aiWindow))
      .then(function (data) {
        state.busy = false;
        aiWindowOptions(data.windows);
        renderAI(data);
        stamp();
      }).catch(function (e) {
        state.busy = false;
        stamp("error: " + e.message);
      });
  }
  // ============================================================ end AI

  // ------------------------------------------------------------------- tabs
  function selectTab(id) {
    state.tab = id;
    Array.prototype.forEach.call(document.querySelectorAll(".view"), function (v) {
      v.classList.toggle("active", v.id === "view-" + id);
    });
    Array.prototype.forEach.call(document.querySelectorAll("#tabs button"), function (b) {
      b.classList.toggle("active", b.dataset.tab === id);
    });
    moveTabIndicator();
    // Sections rise in, staggered, once per tab switch (AI has its own entrance).
    var view = document.getElementById("view-" + id) ||
      (state.views[id] && state.views[id].root) || null;
    if (view && id !== "ai") {
      Array.prototype.forEach.call(view.querySelectorAll("section.card"), function (c, i) {
        c.style.setProperty("--i", Math.min(i, 8));
      });
      view.classList.remove("enter"); void view.offsetWidth; view.classList.add("enter");
      clearTimeout(view._enterT);
      view._enterT = setTimeout(function () { view.classList.remove("enter"); }, 1200);
    }
    location.hash = id;
    dueReset();
    if (state.views[id] && state.views[id].kind === "wazuh") {
      refreshWazuh(state.views[id], true);
    } else if (state.views[id]) refreshProject(state.views[id], true);
    else if (id === "ai") aiEnter();
    else refresh();
  }

  function moveTabIndicator() {
    var ind = document.getElementById("tab-ind");
    var b = document.querySelector("#tabs > button.active");
    if (!ind || !b) return;
    ind.style.width = (b.offsetWidth - 20) + "px";
    ind.style.transform = "translateX(" + (b.offsetLeft + 10) + "px)";
  }

  function buildTabs() {
    var tabs = document.getElementById("tabs");
    var entries = [{ id: "cluster", label: "Cluster" }].concat(
      state.projects.map(function (p) { return { id: p.id, label: p.label }; }),
      [{ id: "ai", label: "AI" }, { id: "schedules", label: "Schedules" }]);
    entries.forEach(function (t) {
      tabs.appendChild(el("button", { text: t.label, "data-tab": t.id,
        onclick: function () { selectTab(t.id); } }));
    });
  }

  function stamp(text) {
    document.getElementById("refresh-state").textContent = text ||
      ("updated " + new Date().toLocaleTimeString(undefined, { hour12: false }));
  }

  function refresh() {
    if (state.busy || document.hidden) return;

    if (state.tab === "cluster") {
      if (due("cluster", state.poll.cluster)) {
        state.busy = true;
        api("/api/cluster").then(function (data) {
          renderCluster(data);
          state.busy = false;
          stamp();
        }).catch(function (e) {
          state.busy = false;
          stamp("error: " + e.message);
        });
      }
      if (due("history", state.poll.history)) refreshHistory();
    } else if (state.tab === "ai") {
      if (due("ai", state.poll.ai)) refreshAI();
      if (due("aiLive", state.poll.aiLive)) refreshAILive();
      if (due("aiRaw", state.poll.aiLive)) refreshAIRaw();
    } else if (state.tab === "schedules") {
      if (due("schedules", state.poll.schedules)) { refreshSchedules(); stamp(); }
    } else if (state.views[state.tab] && state.views[state.tab].kind === "wazuh") {
      refreshWazuh(state.views[state.tab]);
    } else if (state.views[state.tab]) {
      refreshProject(state.views[state.tab]);
    }
  }

  // ------------------------------------------------------------ lab switcher
  // Plain links in the template; this only opens and closes the menu.
  (function () {
    var wrap = document.getElementById("labnav"), btn = document.getElementById("labnav-btn"),
        menu = document.getElementById("labnav-menu");
    if (!wrap) return;
    function set(open) {
      menu.hidden = !open;
      wrap.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", open);
    }
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      set(menu.hidden);
      if (!menu.hidden) menu.querySelector(".labnav-item:not(.current)").focus();
    });
    document.addEventListener("click", function (ev) { if (!wrap.contains(ev.target)) set(false); });
    document.addEventListener("keydown", function (ev) {
      if (menu.hidden) return;
      if (ev.key === "Escape") { set(false); btn.focus(); return; }
      if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
      ev.preventDefault();
      var items = Array.prototype.slice.call(menu.querySelectorAll(".labnav-item"));
      var i = items.indexOf(document.activeElement);
      items[(i + (ev.key === "ArrowDown" ? 1 : items.length - 1)) % items.length].focus();
    });
  })();

  // ------------------------------------------------------------------- boot
  api("/api/projects").then(function (data) {
    state.projects = data.projects;
    state.tz = data.display_tz || "UTC";
    if (data.poll) Object.keys(data.poll).forEach(function (k) {
      state.poll[k] = data.poll[k];
    });
    var host = document.getElementById("project-views");
    state.projects.forEach(function (p) { host.appendChild(buildProjectView(p)); });
    buildTabs();
    initSchedules();

    document.getElementById("history-days").addEventListener("change", function (ev) {
      state.historyDays = ev.target.value;
      refreshHistory();
    });
    initAI();
    document.getElementById("ai-raw-pick").addEventListener("change", function () {
      document.getElementById("ai-raw-follow").checked = false;
      state.last.aiRaw = 0; refresh();
    });
    document.getElementById("ai-raw-follow").addEventListener("change", function () {
      state.last.aiRaw = 0; refresh();
    });
    document.getElementById("probe-all").addEventListener("click", function (ev) {
      ev.target.disabled = true;
      ev.target.textContent = "probing…";
      api("/api/cluster/gpu-probe", { method: "POST", body: {} }).catch(function () {});
      setTimeout(function () {
        ev.target.disabled = false;
        ev.target.textContent = "Probe GPUs";
        refresh();
      }, 6000);
    });

    var initial = (location.hash || "#cluster").slice(1);
    selectTab(state.views[initial] || initial === "cluster" ||
      initial === "ai" || initial === "schedules" ? initial : "cluster");
    setInterval(refresh, state.poll.tick);

    window.addEventListener("resize", moveTabIndicator);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(moveTabIndicator);
    window.addEventListener("scroll", function () {
      document.getElementById("top").classList.toggle("scrolled", window.scrollY > 4);
    }, { passive: true });

    // stop polling a tab nobody is looking at
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) { dueReset(); refresh(); }
    });
  }).catch(function (e) {
    document.body.appendChild(el("div", { class: "notice err",
      text: "could not reach the panel API: " + e.message }));
  });
})();
