// AK-1 front-panel behaviour: theme, keys, the HUD, the posts trace and its
// inspector, the in-browser benchmark, and the section scrubber on notes.
// Every number shown is measured here, in the reader's browser.
(function () {
  "use strict";

  var root = document.documentElement;
  var keys = {};

  function $(sel, ctx) { return (ctx || document).querySelector(sel); }
  function $$(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }
  function css(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function store(k, v) {
    try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) {}
    return null;
  }
  function median(xs) {
    var s = xs.slice().sort(function (a, b) { return a - b; });
    var n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }
  function pct(xs, p) {
    var s = xs.slice().sort(function (a, b) { return a - b; });
    return s[Math.min(s.length - 1, Math.floor(p * s.length))];
  }
  function cv(xs) {
    var m = xs.reduce(function (a, b) { return a + b; }, 0) / xs.length;
    var v = xs.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / (xs.length - 1);
    return Math.sqrt(v) / m;
  }
  function ms(x) { return x < 10 ? x.toFixed(2) : x.toFixed(1); }

  /* ---------- theme ---------- */

  var themeHandlers = [];

  function syncTheme() {
    var paper = root.dataset.theme === "paper";
    // mermaid.js and the typo theme's .img-dark/.img-light read these.
    document.body.classList.toggle("dark", !paper);
    document.body.classList.toggle("light", paper);
    $$(".theme-key .value").forEach(function (el) { el.textContent = paper ? "paper" : "bench"; });
    themeHandlers.forEach(function (f) { f(); });
  }

  function toggleTheme() {
    if (root.dataset.theme === "paper") { delete root.dataset.theme; store("theme", "bench"); }
    else { root.dataset.theme = "paper"; store("theme", "paper"); }
    syncTheme();
  }

  $$(".theme-key").forEach(function (b) { b.addEventListener("click", toggleTheme); });
  syncTheme();

  /* ---------- keys ---------- */

  var help = $(".keyhelp");

  function showHelp(on) {
    if (!help) return;
    help.hidden = !on;
    help.classList.toggle("open", on);
    if (on) { var c = $("[data-close]", help); if (c) c.focus(); }
  }

  if (help) {
    help.addEventListener("click", function (e) {
      if (e.target === help || e.target.hasAttribute("data-close")) showHelp(false);
    });
  }

  var gPending = false;
  var gTimer = 0;

  document.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;

    if (e.key === "Escape") { showHelp(false); return; }
    if (gPending) {
      gPending = false;
      if (e.key === "h") location.href = "/";
      else if (e.key === "p") location.href = "/posts/";
      return;
    }
    if (e.key === "g") { gPending = true; clearTimeout(gTimer); gTimer = setTimeout(function () { gPending = false; }, 900); return; }
    if (e.key === "t") { toggleTheme(); return; }
    if (e.key === "?") { showHelp(help && help.hidden); e.preventDefault(); return; }

    var h = keys[e.key];
    if (h && h(e) !== false) e.preventDefault();
  });

  /* ---------- HUD ---------- */

  var hud = $(".hud");
  function hudSet(k, html) { var el = hud && $('[data-hud="' + k + '"]', hud); if (el) el.innerHTML = html; }

  if (hud) {
    var spark = $('[data-hud="spark"]', hud);
    var sctx = spark && spark.getContext("2d");
    var frames = [];
    var last = 0;
    var raf = 0;
    var lastReport = 0;

    if (spark) {
      var dpr = window.devicePixelRatio || 1;
      spark.width = 96 * dpr;
      spark.height = 22 * dpr;
      sctx.scale(dpr, dpr);
    }

    var drawSpark = function () {
      if (!sctx) return;
      var w = 96, h = 22, n = Math.min(frames.length, 48);
      sctx.clearRect(0, 0, w, h);
      var base = css("--line"), ink = css("--ink-3"), hot = css("--accent");
      sctx.fillStyle = base;
      sctx.fillRect(0, h - 1, w, 1);
      var med = median(frames);
      for (var i = 0; i < n; i++) {
        var d = frames[frames.length - n + i];
        var bh = clamp(d / 50 * h, 1, h);
        sctx.fillStyle = d > med * 1.8 && d > 20 ? hot : ink;
        sctx.fillRect(i * 2, h - bh, 1.2, bh);
      }
    };

    var tick = function (now) {
      if (last) {
        frames.push(now - last);
        if (frames.length > 120) frames.shift();
      }
      last = now;
      if (now - lastReport > 500 && frames.length > 10) {
        lastReport = now;
        hudSet("frame", ms(median(frames)) + " / " + ms(pct(frames, 0.99)) + '<span class="u"> ms</span>');
        drawSpark();
      }
      raf = requestAnimationFrame(tick);
    };

    document.addEventListener("visibilitychange", function () {
      if (document.hidden) { cancelAnimationFrame(raf); last = 0; }
      else raf = requestAnimationFrame(tick);
    });
    themeHandlers.push(drawSpark);
    raf = requestAnimationFrame(tick);

    hudSet("cores", navigator.hardwareConcurrency ? String(navigator.hardwareConcurrency) : "–");

    var navTiming = function () {
      var nav = performance.getEntriesByType && performance.getEntriesByType("navigation")[0];
      if (!nav || !nav.loadEventEnd) { setTimeout(navTiming, 250); return; }
      hudSet("load", Math.round(nav.loadEventEnd) + '<span class="u"> ms</span>');
      var bytes = nav.transferSize || 0;
      var body = nav.encodedBodySize || 0;
      performance.getEntriesByType("resource").forEach(function (r) {
        bytes += r.transferSize || 0;
        body += r.encodedBodySize || 0;
      });
      if (bytes > 0) hudSet("xfer", (bytes / 1024).toFixed(0) + '<span class="u"> KiB</span>');
      else if (body > 0) hudSet("xfer", "cached");
    };
    if (document.readyState === "complete") navTiming();
    else window.addEventListener("load", function () { setTimeout(navTiming, 0); });

    // Visitors per hour (written at build time from GoatCounter); the latest hour in accent.
    var vis = $('[data-hud="visitors"]', hud);
    if (vis) {
      var series = vis.dataset.series ? vis.dataset.series.split(",").map(Number) : [];
      var vctx = vis.getContext("2d");
      var vdpr = window.devicePixelRatio || 1;
      vis.width = 96 * vdpr;
      vis.height = 22 * vdpr;
      vctx.scale(vdpr, vdpr);
      var drawVis = function () {
        var w = 96, h = 22, n = series.length || 1, top = Math.max.apply(null, series.concat(1));
        var bw = w / n;
        vctx.clearRect(0, 0, w, h);
        vctx.fillStyle = css("--line");
        for (var dx = 0; dx < w; dx += 3) vctx.fillRect(dx, h - 1, 1.5, 1);
        series.forEach(function (v, i) {
          if (!v) return;
          var bh = Math.max(1.5, v / top * (h - 3));
          vctx.fillStyle = i === n - 1 ? css("--accent") : css("--ink-3");
          vctx.fillRect(i * bw, h - 1 - bh, Math.max(1, bw - 0.6), bh);
        });
      };
      themeHandlers.push(drawVis);
      drawVis();
    }

    var savedRatio = store("ak1-bench");
    if (savedRatio) hudSet("bench", savedRatio + "×");
  }

  /* ---------- revision tables: j/k select, enter open ---------- */

  var rowSelect = null;
  var table = $("table[data-rows]");
  if (table) {
    var rows = $$("tbody tr[data-href]", table);
    var idx = -1;
    rowSelect = function (href, scroll) {
      idx = -1;
      rows.forEach(function (r, i) {
        var on = r.dataset.href === href;
        r.classList.toggle("sel", on);
        if (on) idx = i;
      });
      if (scroll && idx >= 0) rows[idx].scrollIntoView({ block: "nearest" });
    };
    if (!$("[data-trace]")) {
      keys.j = function () { var i = clamp(idx + 1, 0, rows.length - 1); rowSelect(rows[i].dataset.href, true); };
      keys.k = function () { var i = clamp(idx < 0 ? 0 : idx - 1, 0, rows.length - 1); rowSelect(rows[i].dataset.href, true); };
      keys.Enter = function () { if (idx < 0) return false; location.href = rows[idx].dataset.href; };
    }
  }

  /* ---------- trace (front panel) ---------- */

  var traceBox = $("[data-trace]");
  var dataEl = $("#trace-data");
  if (traceBox && dataEl) {
    var posts = JSON.parse(dataEl.textContent);
    posts.forEach(function (p, i) { p.i = i; p.ts = Date.parse(p.date + "T12:00:00Z"); });

    var laneCount = {};
    posts.forEach(function (p) { laneCount[p.lane] = (laneCount[p.lane] || 0) + 1; });
    var lanes = Object.keys(laneCount).sort(function (a, b) { return laneCount[b] - laneCount[a] || a.localeCompare(b); });

    var canvas = $("canvas", traceBox);
    var tip = $(".trace-tip", traceBox);
    var ctx = canvas.getContext("2d");
    var W = 0, H = 0;
    var L = 96, R = 18, TOP = 34, BOT = 14;
    var DAY = 86400000;
    var tMin = posts[0].ts, tMax = posts[posts.length - 1].ts;
    var padAll = Math.max((tMax - tMin) * 0.03, 10 * DAY);
    var full = [tMin - padAll, tMax + padAll];
    var view = full.slice();
    var sel = posts.length - 1;
    var hover = -1;
    var mouseX = -1;
    var rects = [];

    // longest idle gap
    var gap = { d: 0, a: -1 };
    for (var gi = 1; gi < posts.length; gi++) {
      var gd = (posts[gi].ts - posts[gi - 1].ts) / DAY;
      if (gd > gap.d) gap = { d: gd, a: gi - 1 };
    }

    var x = function (t) { return L + (t - view[0]) / (view[1] - view[0]) * (W - L - R); };
    var tAt = function (px) { return view[0] + (px - L) / (W - L - R) * (view[1] - view[0]); };

    var resize = function () {
      var r = traceBox.getBoundingClientRect();
      var dpr = window.devicePixelRatio || 1;
      W = r.width; H = r.height;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      L = W < 520 ? 82 : 96;
      draw();
    };

    var ticks = function () {
      var span = view[1] - view[0];
      var out = [];
      var d = new Date(view[0]);
      if (span > 3 * 365 * DAY) {
        for (var y = d.getUTCFullYear(); ; y++) {
          var t = Date.UTC(y, 0, 1);
          if (t > view[1]) break;
          if (t >= view[0]) out.push({ t: t, s: String(y) });
        }
      } else if (span > 75 * DAY) {
        var stepM = span > 18 * 30 * DAY ? 3 : 1;
        for (var m = d.getUTCFullYear() * 12 + d.getUTCMonth(); ; m += 1) {
          var tm = Date.UTC(Math.floor(m / 12), m % 12, 1);
          if (tm > view[1]) break;
          if (tm >= view[0] && (m % 12) % stepM === 0) out.push({ t: tm, s: new Date(tm).toISOString().slice(0, 7) });
        }
      } else {
        var stepD = span > 40 * DAY ? 14 : span > 14 * DAY ? 7 : 1;
        var t0 = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
        for (var td = t0; td <= view[1]; td += DAY) {
          if (td >= view[0] && Math.round(td / DAY) % stepD === 0) out.push({ t: td, s: new Date(td).toISOString().slice(5, 10) });
        }
      }
      return out;
    };

    var draw = function () {
      if (!W) return;
      var ink = css("--ink"), ink2 = css("--ink-2"), ink3 = css("--ink-3");
      var line = css("--line"), line2 = css("--line-2"), acc = css("--accent"), bg = css("--bg");
      var mono = css("--mono"), sans = css("--sans");
      ctx.clearRect(0, 0, W, H);

      var laneH = (H - TOP - BOT) / lanes.length;

      // lanes
      ctx.font = "10.5px " + mono;
      ctx.textBaseline = "middle";
      lanes.forEach(function (ln, i) {
        var y = TOP + i * laneH;
        ctx.fillStyle = line2;
        ctx.fillRect(L, y + laneH - 0.5, W - L - R, 1);
        ctx.fillStyle = ink3;
        ctx.textAlign = "left";
        ctx.fillText(ln.toUpperCase(), 14, y + laneH / 2);
        if (W >= 520) ctx.fillText(String(laneCount[ln]), L - 22, y + laneH / 2);
      });
      ctx.fillStyle = line;
      ctx.fillRect(L - 0.5, TOP, 1, H - TOP - BOT);

      // ruler
      ctx.fillStyle = line;
      ctx.fillRect(L, TOP - 0.5, W - L - R, 1);
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      var lastLabel = -1e9;
      ticks().forEach(function (tk) {
        var px = x(tk.t);
        ctx.fillStyle = line2;
        ctx.fillRect(Math.round(px), TOP, 1, H - TOP - BOT);
        ctx.fillStyle = line;
        ctx.fillRect(Math.round(px), TOP - 5, 1, 5);
        if (px - lastLabel > 48 && px + 24 < W && px - 24 > L) {
          ctx.fillStyle = ink3;
          ctx.fillText(tk.s, px, TOP - 9);
          lastLabel = px;
        }
      });

      // longest idle gap, drawn as a bracket under the ruler
      if (gap.a >= 0) {
        var ga = x(posts[gap.a].ts), gb = x(posts[gap.a + 1].ts);
        var g0 = clamp(ga, L, W - R), g1 = clamp(gb, L, W - R);
        if (g1 - g0 > 70) {
          ctx.strokeStyle = acc;
          ctx.globalAlpha = 0.55;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(g0 + 6, TOP + 8);
          ctx.lineTo(g1 - 6, TOP + 8);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
          var label = "idle " + Math.round(gap.d).toLocaleString("en-US") + " d";
          ctx.font = "10.5px " + mono;
          var lw = ctx.measureText(label).width + 10;
          ctx.fillStyle = bg;
          ctx.fillRect((g0 + g1) / 2 - lw / 2, TOP + 2, lw, 12);
          ctx.fillStyle = acc;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, (g0 + g1) / 2, TOP + 8.5);
        }
      }

      // spans: width ∝ read time; a lane splits into two rows where spans would overlap
      rects = [];
      var ends = {};
      posts.forEach(function (p) {
        var li = lanes.indexOf(p.lane);
        var w = clamp(p.t * 1.7, 6, 36);
        var px = x(p.ts) - w / 2;
        var e = ends[li] || (ends[li] = [-1e9, -1e9]);
        var r = px >= e[0] + 3 ? 0 : px >= e[1] + 3 ? 1 : 0;
        e[r] = px + w;
        rects.push({ p: p, x: px, w: w, h: 10, li: li, r: r });
      });
      var twoRow = {};
      rects.forEach(function (q) { if (q.r) twoRow[q.li] = true; });
      rects.forEach(function (q) {
        var mid = TOP + q.li * laneH + laneH / 2;
        q.y = twoRow[q.li] ? (q.r ? mid + 2 : mid - 12) : mid - 5;
      });

      ctx.save();
      ctx.beginPath();
      ctx.rect(L, TOP, W - L - R, H - TOP - BOT);
      ctx.clip();

      rects.forEach(function (q, k) {
        var isSel = q.p.i === sel, isHov = q.p.i === hover;
        ctx.fillStyle = isSel ? acc : isHov ? ink : ink2;
        ctx.globalAlpha = isSel || isHov ? 1 : 0.75;
        ctx.fillRect(Math.round(q.x), Math.round(q.y), Math.round(q.w), q.h);
        ctx.globalAlpha = 1;
        if (isSel) {
          ctx.strokeStyle = acc;
          ctx.strokeRect(Math.round(q.x) - 2.5, Math.round(q.y) - 2.5, Math.round(q.w) + 5, q.h + 5);
        }
        // label when there is room before the next span in the same lane
        var next = null;
        for (var n = k + 1; n < rects.length; n++) if (rects[n].li === q.li && rects[n].r === q.r) { next = rects[n]; break; }
        var room = (next ? next.x : W - R) - (q.x + q.w) - 10;
        if (room > 90) {
          ctx.font = "12px " + sans;
          ctx.fillStyle = isSel ? ink : ink3;
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          var s = q.p.title;
          while (s.length > 4 && ctx.measureText(s).width > room) s = s.slice(0, -2);
          if (s !== q.p.title) s = s.replace(/\s*\S?$/, "") + "…";
          ctx.fillText(s, q.x + q.w + 7, q.y + q.h / 2 + 0.5);
        }
      });
      ctx.restore();

      // crosshair
      if (mouseX >= L && mouseX <= W - R) {
        ctx.fillStyle = ink3;
        ctx.globalAlpha = 0.6;
        ctx.fillRect(Math.round(mouseX), TOP, 1, H - TOP - BOT);
        ctx.globalAlpha = 1;
        var ds = new Date(tAt(mouseX)).toISOString().slice(0, 10);
        ctx.font = "10.5px " + mono;
        var tw = ctx.measureText(ds).width + 10;
        var bx = clamp(mouseX - tw / 2, L, W - R - tw);
        ctx.fillStyle = acc;
        ctx.fillRect(bx, TOP - 21, tw, 15);
        ctx.fillStyle = bg;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(ds, bx + tw / 2, TOP - 13);
      }
    };

    var hit = function (mx, my) {
      var best = -1, bestD = 1e9;
      rects.forEach(function (q) {
        if (my < q.y - 6 || my > q.y + q.h + 6) return;
        var cx = clamp(mx, q.x, q.x + q.w);
        var d = Math.abs(mx - cx);
        if (d < 8 && d < bestD) { bestD = d; best = q.p.i; }
      });
      return best;
    };

    var insp = $("[data-inspector]");
    var setI = function (k, v, attr) {
      var el = insp && $('[data-insp="' + k + '"]', insp);
      if (!el) return;
      if (attr) el.setAttribute(attr, v); else el.textContent = v;
    };

    var select = function (i, keepView) {
      sel = clamp(i, 0, posts.length - 1);
      var p = posts[sel];
      setI("an", p.an); setI("lane", p.lane); setI("abs", p.abs);
      setI("link", p.title); setI("link", p.url, "href"); setI("open", p.url, "href");
      setI("date", p.date); setI("t", p.t); setI("words", p.words.toLocaleString("en-US"));
      setI("tags", (p.tags || []).join(", ")); setI("pos", (sel + 1) + " / " + posts.length);
      if (rowSelect) rowSelect(p.url, false);
      if (!keepView && (p.ts < view[0] || p.ts > view[1])) {
        var span = view[1] - view[0];
        view = [p.ts - span / 2, p.ts + span / 2];
      }
      draw();
    };

    var setRange = function (r) {
      $$("[data-range]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.range === r)); });
      if (r === "2025") view = [Date.UTC(2025, 0, 1), tMax + 20 * DAY];
      else if (r === "90") view = [tMax - 90 * DAY, tMax + 6 * DAY];
      else view = full.slice();
      draw();
    };
    $$("[data-range]").forEach(function (b) { b.addEventListener("click", function () { setRange(b.dataset.range); }); });

    var showTip = function (i, mx, my) {
      if (i < 0) { tip.style.opacity = 0; return; }
      var p = posts[i];
      tip.innerHTML = "";
      var b = document.createElement("b");
      b.textContent = p.title;
      tip.appendChild(b);
      tip.appendChild(document.createTextNode("AN-" + p.an + " · " + p.date + " · " + p.t + " min · " + p.words.toLocaleString("en-US") + " words"));
      var tw = tip.offsetWidth, th = tip.offsetHeight;
      tip.style.left = clamp(mx + 14, 4, W - tw - 4) + "px";
      tip.style.top = clamp(my + 14, 4, H - th - 4) + "px";
      tip.style.opacity = 1;
    };

    var drag = null;
    canvas.addEventListener("pointerdown", function (e) {
      drag = { x: e.clientX, v: view.slice(), moved: false, id: e.pointerId };
    });
    canvas.addEventListener("pointermove", function (e) {
      var r = canvas.getBoundingClientRect();
      var mx = e.clientX - r.left, my = e.clientY - r.top;
      if (drag && drag.id === e.pointerId) {
        var dx = e.clientX - drag.x;
        if (!drag.moved && Math.abs(dx) > 4) { drag.moved = true; try { canvas.setPointerCapture(e.pointerId); } catch (err) {} }
        if (drag.moved) {
          var dt = dx / (W - L - R) * (drag.v[1] - drag.v[0]);
          view = [drag.v[0] - dt, drag.v[1] - dt];
          $$("[data-range]").forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
        }
      }
      mouseX = e.pointerType === "mouse" ? mx : -1;
      hover = hit(mx, my);
      canvas.style.cursor = drag && drag.moved ? "grabbing" : hover >= 0 ? "pointer" : "crosshair";
      showTip(drag && drag.moved ? -1 : hover, mx, my);
      draw();
    });
    canvas.addEventListener("pointerup", function (e) {
      if (drag && !drag.moved) {
        var r = canvas.getBoundingClientRect();
        var i = hit(e.clientX - r.left, e.clientY - r.top);
        if (i >= 0) select(i, true);
      }
      drag = null;
    });
    canvas.addEventListener("pointerleave", function () {
      mouseX = -1; hover = -1; showTip(-1); draw();
    });
    canvas.addEventListener("dblclick", function (e) {
      var r = canvas.getBoundingClientRect();
      var i = hit(e.clientX - r.left, e.clientY - r.top);
      if (i >= 0) location.href = posts[i].url; else setRange("all");
    });
    // Zoom only on pinch (ctrlKey) or ctrl/cmd+wheel, so plain scrolling still scrolls the page.
    canvas.addEventListener("wheel", function (e) {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      var r = canvas.getBoundingClientRect();
      var t = tAt(e.clientX - r.left);
      var f = Math.exp(clamp(e.deltaY, -60, 60) * 0.01);
      var span = clamp((view[1] - view[0]) * f, 7 * DAY, (full[1] - full[0]) * 1.5);
      var a = (t - view[0]) / (view[1] - view[0]);
      view = [t - a * span, t + (1 - a) * span];
      $$("[data-range]").forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
      draw();
    }, { passive: false });

    keys.j = keys.ArrowLeft = function () { select(sel - 1); };
    keys.k = keys.ArrowRight = function () { select(sel + 1); };
    keys.Enter = function () { location.href = posts[sel].url; };

    if ("ResizeObserver" in window) new ResizeObserver(resize).observe(traceBox);
    else window.addEventListener("resize", resize);
    themeHandlers.push(draw);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
    resize();
    select(sel, true);
  }

  /* ---------- bench (front panel) ---------- */

  var bench = $("[data-bench]");
  if (bench) {
    var stateEl = $("[data-bench-state]", bench);
    var runBtn = $("[data-bench-run]", bench);
    var verdict = $("[data-bench-verdict]", bench);
    var running = false;

    // Runs in a worker so the page stays responsive. Three untimed warm-up
    // passes let the JIT tier up and touch the memory; each sample then times
    // four passes, because browsers coarsen performance.now(). Results are
    // summed into a sink and posted back, so the JIT cannot delete the loops.
    var src = [
      "self.onmessage = function () {",
      "  var N = 2048, REPS = 15, a = new Int32Array(N * N), sink = 0;",
      "  for (var i = 0; i < a.length; i++) a[i] = i & 1023;",
      "  function rows() { var s = 0; for (var i = 0; i < N; i++) { var o = i * N; for (var j = 0; j < N; j++) s += a[o + j]; } return s | 0; }",
      "  function cols() { var s = 0; for (var j = 0; j < N; j++) { for (var i = 0; i < N; i++) s += a[i * N + j]; } return s | 0; }",
      "  for (var w = 0; w < 3; w++) { sink += rows(); sink += cols(); }",
      "  var r = [], c = [], P = 4;",
      "  for (var k = 0; k < REPS; k++) {",
      "    var t = performance.now(); for (var p = 0; p < P; p++) sink += rows(); r.push((performance.now() - t) / P);",
      "    t = performance.now(); for (p = 0; p < P; p++) sink += cols(); c.push((performance.now() - t) / P);",
      "    self.postMessage({ type: 'progress', k: k + 1, reps: REPS });",
      "  }",
      "  self.postMessage({ type: 'done', rows: r, cols: c, sink: sink });",
      "};"
    ].join("\n");

    var setRow = function (name, med, spread, frac) {
      var row = $('[data-bench-row="' + name + '"]', bench);
      $("[data-v]", row).textContent = ms(med) + " ms";
      $("[data-cv]", row).textContent = (spread * 100).toFixed(1) + "%";
      $(".bar i", row).style.width = (frac * 100).toFixed(1) + "%";
    };

    var run = function () {
      if (running || !window.Worker) {
        if (!window.Worker) stateEl.textContent = "no worker support";
        return;
      }
      running = true;
      runBtn.disabled = true;
      stateEl.textContent = "warming up";
      verdict.textContent = "Running. The warm-up passes aren't timed; they let the browser compile the loops and load the memory.";
      var url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
      var w = new Worker(url);
      w.onmessage = function (e) {
        var m = e.data;
        if (m.type === "progress") { stateEl.textContent = "rep " + m.k + " / " + m.reps; return; }
        w.terminate();
        URL.revokeObjectURL(url);
        running = false;
        runBtn.disabled = false;
        runBtn.textContent = "rerun";
        var rm = median(m.rows), cm = median(m.cols);
        var rcv = cv(m.rows), ccv = cv(m.cols);
        var top = Math.max(rm, cm);
        setRow("rows", rm, rcv, rm / top);
        setRow("cols", cm, ccv, cm / top);
        var ratio = cm / rm;
        var worst = Math.max(rcv, ccv);
        stateEl.textContent = "done · " + new Date().toTimeString().slice(0, 8);
        if (worst > 0.05) {
          verdict.innerHTML = '<span class="warn">Too noisy: spread ' + (worst * 100).toFixed(1) +
            "% &gt; 5%</span>. The samples disagree too much to trust the " + ratio.toFixed(1) +
            "× result. A browser can't pin the worker to one core, so on a phone or a laptop with fast and " +
            "efficient cores it can migrate mid-run, the effect " + (bench.dataset.ref || "the benchmarking note") +
            " §1 is about. Close other tabs and rerun.";
        } else {
          verdict.innerHTML = "On this device, jumping around memory was <b>" + ratio.toFixed(1) + "× slower</b> than reading it in order. " +
            "Spread " + (worst * 100).toFixed(1) + "%, under the 5% line, so the number holds.";
          store("ak1-bench", ratio.toFixed(1));
          hudSet("bench", ratio.toFixed(1) + "×");
        }
      };
      w.onerror = function () {
        running = false;
        runBtn.disabled = false;
        stateEl.textContent = "failed";
        verdict.textContent = "The worker failed to run in this browser.";
      };
      w.postMessage(0);
    };

    runBtn.addEventListener("click", run);
    var idle = window.requestIdleCallback || function (f) { return setTimeout(f, 1200); };
    var start = function () { idle(run, { timeout: 3000 }); };
    if (document.readyState === "complete") start(); else window.addEventListener("load", start);
  }

  /* ---------- notes: contents highlight + section scrubber ---------- */

  var prose = $(".prose");
  var scrub = $("[data-scrub]");
  var tocLinks = $$(".toc a[href^='#']");
  if (prose && (scrub || tocLinks.length)) {
    var byId = {};
    tocLinks.forEach(function (a) { byId[decodeURIComponent(a.hash.slice(1))] = a; });
    var tocHeads = $$("h2[id], h3[id]", prose).filter(function (h) { return byId[h.id]; });
    var h2s = $$("h2[id]", prose);
    var current = null;
    var segs = [];
    var track = scrub && $("[data-track]", scrub);
    var headEl = scrub && $("[data-head]", scrub);
    var tipEl = scrub && $("[data-tip]", scrub);
    var readEl = scrub && $("[data-read]", scrub);
    var minutes = scrub ? +scrub.dataset.minutes || 0 : 0;

    var top0 = function () { return prose.getBoundingClientRect().top + window.scrollY; };

    var build = function () {
      if (!track) return;
      segs.forEach(function (s) { s.el.remove(); });
      segs = [];
      var a = top0(), total = prose.offsetHeight || 1;
      var marks = [{ y: 0, title: "Introduction", id: null }].concat(h2s.map(function (h) {
        return { y: h.getBoundingClientRect().top + window.scrollY - a, title: h.textContent.replace(/^§/, "").trim(), id: h.id };
      }));
      marks.forEach(function (m, i) {
        var end = i + 1 < marks.length ? marks[i + 1].y : total;
        if (end - m.y < 2) return;
        var el = document.createElement("span");
        el.className = "seg";
        el.style.left = (m.y / total * 100) + "%";
        el.style.width = "calc(" + ((end - m.y) / total * 100) + "% - 2px)";
        el.addEventListener("mouseenter", function () {
          tipEl.textContent = m.title;
          tipEl.style.left = ((m.y + end) / 2 / total * 100) + "%";
          tipEl.style.opacity = 1;
        });
        el.addEventListener("mouseleave", function () { tipEl.style.opacity = 0; });
        el.addEventListener("click", function () {
          if (m.id) document.getElementById(m.id).scrollIntoView();
          else window.scrollTo(0, 0);
        });
        track.insertBefore(el, headEl);
        segs.push({ el: el, a: m.y / total, b: end / total });
      });
      update();
    };

    var update = function () {
      // contents rail
      var id = null;
      for (var i = 0; i < tocHeads.length; i++) {
        if (tocHeads[i].getBoundingClientRect().top < window.innerHeight * 0.3) id = tocHeads[i].id;
        else break;
      }
      if (current !== id) {
        if (current && byId[current]) byId[current].classList.remove("on");
        current = id;
        if (id && byId[id]) byId[id].classList.add("on");
      }
      // scrubber
      if (!track) return;
      var p = clamp((window.scrollY + window.innerHeight * 0.35 - top0()) / (prose.offsetHeight || 1), 0, 1);
      headEl.style.left = (p * 100) + "%";
      segs.forEach(function (s) {
        s.el.classList.toggle("done", s.b <= p);
        s.el.classList.toggle("on", s.a <= p && p < s.b);
      });
      var left = Math.max(0, Math.ceil(minutes * (1 - p)));
      readEl.innerHTML = "<b>" + Math.round(p * 100) + "%</b><span class=\"rest\"> · " + (p >= 1 ? "done" : left + " min left") + "</span>";
    };

    var jump = function (dir) {
      var y = window.innerHeight * 0.3 + 2;
      var list = h2s.map(function (h) { return { h: h, t: h.getBoundingClientRect().top }; });
      var target = null;
      if (dir > 0) { for (var i = 0; i < list.length; i++) if (list[i].t > y + 4) { target = list[i].h; break; } }
      else { for (var k = list.length - 1; k >= 0; k--) if (list[k].t < -4) { target = list[k].h; break; } }
      if (target) target.scrollIntoView();
    };
    keys["]"] = function () { jump(1); };
    keys["["] = function () { jump(-1); };

    var q = false;
    window.addEventListener("scroll", function () {
      if (!q) { q = true; requestAnimationFrame(function () { q = false; update(); }); }
    }, { passive: true });
    if ("ResizeObserver" in window) new ResizeObserver(build).observe(prose);
    else window.addEventListener("resize", build);
    build();
  }
})();
