// Theme switcher (:theme), status line position, and VISUAL mode on selection.
(function () {
  var themes = ["gruvbox", "paper", "nord"];
  var root = document.documentElement;

  // mermaid.js and the theme's .img-dark/.img-light classes read these.
  function syncBody(t) {
    var dark = t !== "paper";
    document.body.classList.toggle("dark", dark);
    document.body.classList.toggle("light", !dark);
  }

  function apply(t) {
    root.dataset.theme = t;
    syncBody(t);
    document.querySelectorAll(".theme-key .value").forEach(function (el) {
      el.textContent = t;
    });
  }

  apply(themes.indexOf(root.dataset.theme) >= 0 ? root.dataset.theme : "gruvbox");

  document.querySelectorAll(".theme-key").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var next = themes[(themes.indexOf(root.dataset.theme) + 1) % themes.length];
      apply(next);
      try { localStorage.setItem("theme", next); } catch (e) {}
    });
  });

  var status = document.querySelector(".statusline");
  if (!status) return;

  var pos = status.querySelector(".pos");
  var mode = status.querySelector(".mode");
  var post = status.closest("article");
  var queued = false;

  function update() {
    queued = false;
    var rect = post.getBoundingClientRect();
    var total = rect.height - window.innerHeight;
    if (total <= 0) { pos.textContent = "All"; return; }
    var p = Math.min(1, Math.max(0, -rect.top / total));
    pos.textContent = p <= 0.01 ? "Top" : p >= 0.99 ? "Bot" : Math.round(p * 100) + "%";
  }

  window.addEventListener("scroll", function () {
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }, { passive: true });
  window.addEventListener("resize", update);
  update();

  document.addEventListener("selectionchange", function () {
    var sel = document.getSelection();
    var m = sel && !sel.isCollapsed && post.contains(sel.anchorNode) ? "VISUAL" : "NORMAL";
    mode.textContent = m;
    mode.dataset.mode = m;
  });
})();
