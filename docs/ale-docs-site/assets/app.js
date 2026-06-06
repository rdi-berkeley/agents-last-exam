/* =========================================================================
   App shell: builds the sidebar + topbar chrome into every page from
   window.ALE_NAV, manages theme, active link, mobile drawer, and the
   right-hand TOC (auto-generated from <h2>/<h3> in .article).
   Supports one level of nav nesting via item.children.
   ========================================================================= */
(function () {
  "use strict";

  function norm(p) {
    if (p === "/" || p === "") return "/index.html";
    return p;
  }
  var here = norm(location.pathname);

  // ---- flatten nav (parents then their children) for prev/next + lookups ----
  var FLAT = [];
  (window.ALE_NAV || []).forEach(function (g) {
    g.items.forEach(function (it) {
      FLAT.push({ href: it.href, title: it.title, group: g.label });
      (it.children || []).forEach(function (c) {
        FLAT.push({ href: c.href, title: c.title, group: g.label, parent: it.title });
      });
    });
  });

  // ---- theme ----
  var saved = localStorage.getItem("ale-theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  function toggleTheme() {
    var cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", cur);
    localStorage.setItem("ale-theme", cur);
    setThemeIcon();
  }
  function setThemeIcon() {
    var dark = document.documentElement.getAttribute("data-theme") === "dark";
    var el = document.getElementById("theme-icon");
    if (el) el.textContent = dark ? "☀" : "☾";
  }

  // ---- sidebar ----
  function navLink(it, isChild) {
    var active = norm(it.href) === here ? " active" : "";
    var cls = "nav-link" + (isChild ? " child" : "") + active;
    var badge = it.draft ? '<span class="badge draft">draft</span>' : '';
    return '<a class="' + cls + '" href="' + it.href + '">' +
             '<span class="dot"></span><span>' + it.title + '</span>' + badge +
           '</a>';
  }
  function buildSidebar() {
    var html = '' +
      '<a class="brand" href="/index.html">' +
        '<span class="mark">ALE</span>' +
        '<span><span class="title">Agents\' Last Exam</span><br>' +
        '<span class="subtitle">Framework documentation</span></span>' +
      '</a>';
    (window.ALE_NAV || []).forEach(function (g) {
      html += '<div class="nav-group"><div class="label">' + g.label + '</div>';
      g.items.forEach(function (it) {
        html += navLink(it, false);
        (it.children || []).forEach(function (c) { html += navLink(c, true); });
      });
      html += '</div>';
    });
    return html;
  }

  // ---- TOC from headings ----
  function buildTOC() {
    var art = document.querySelector(".article");
    if (!art) return "";
    var hs = art.querySelectorAll("h2, h3");
    if (!hs.length) return "";
    var out = '<div class="label">On this page</div>';
    hs.forEach(function (h) {
      if (!h.id) h.id = h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      var lvl = h.tagName === "H3" ? " lvl3" : "";
      out += '<a class="' + lvl.trim() + '" href="#' + h.id + '">' + h.textContent + '</a>';
    });
    return out;
  }

  // ---- prev / next ----
  function buildPageNav() {
    var idx = -1;
    for (var i = 0; i < FLAT.length; i++) if (norm(FLAT[i].href) === here) { idx = i; break; }
    if (idx === -1) return "";
    var prev = FLAT[idx - 1], next = FLAT[idx + 1];
    var html = "";
    html += prev ? '<a class="prev" href="' + prev.href + '"><div class="dir">← Previous</div><div class="ttl">' + prev.title + '</div></a>' : '<span style="flex:1"></span>';
    html += next ? '<a class="next" href="' + next.href + '"><div class="dir">Next →</div><div class="ttl">' + next.title + '</div></a>' : '<span style="flex:1"></span>';
    return html;
  }

  // ---- crumbs ----
  function crumbLabel() {
    for (var i = 0; i < FLAT.length; i++) {
      if (norm(FLAT[i].href) === here) {
        var n = FLAT[i];
        var mid = n.parent ? n.group + ' · ' + n.parent : n.group;
        return mid + ' · <b>' + n.title + '</b>';
      }
    }
    return '<b>Documentation</b>';
  }

  // ---- assemble ----
  document.addEventListener("DOMContentLoaded", function () {
    var article = document.querySelector(".article");
    var articleHTML = article ? article.outerHTML : '<div class="article"><p>Empty page.</p></div>';

    var shell = '' +
      '<div class="scrim" id="scrim"></div>' +
      '<div class="layout">' +
        '<aside class="sidebar" id="sidebar">' + buildSidebar() + '</aside>' +
        '<div class="main">' +
          '<div class="topbar">' +
            '<button class="icon-btn menu-btn" id="menu-btn" aria-label="Menu">☰</button>' +
            '<div class="crumbs">' + crumbLabel() + '</div>' +
            '<div class="spacer"></div>' +
            '<button class="icon-btn" id="theme-btn" aria-label="Toggle theme"><span id="theme-icon">☾</span></button>' +
          '</div>' +
          '<div class="content-wrap">' +
            articleHTML +
            '<nav class="toc" id="toc"></nav>' +
          '</div>' +
        '</div>' +
      '</div>';

    document.body.innerHTML = shell;

    var art2 = document.querySelector(".article");
    var pn = buildPageNav();
    if (pn) { var nav = document.createElement("nav"); nav.className = "pagenav"; nav.innerHTML = pn; art2.appendChild(nav); }

    document.getElementById("toc").innerHTML = buildTOC();

    document.getElementById("theme-btn").addEventListener("click", toggleTheme);
    var sb = document.getElementById("sidebar"), scrim = document.getElementById("scrim");
    document.getElementById("menu-btn").addEventListener("click", function () {
      sb.classList.add("open"); scrim.classList.add("show");
    });
    scrim.addEventListener("click", function () { sb.classList.remove("open"); scrim.classList.remove("show"); });

    setThemeIcon();
    initScrollSpy();
  });

  // ---- scrollspy ----
  function initScrollSpy() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
    if (!links.length) return;
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var heads = links.map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); }).filter(Boolean);
    function spy() {
      var pos = window.scrollY + 100, cur = heads[0];
      heads.forEach(function (h) { if (h.offsetTop <= pos) cur = h; });
      links.forEach(function (a) { a.classList.remove("active"); });
      if (cur && map[cur.id]) map[cur.id].classList.add("active");
    }
    window.addEventListener("scroll", spy, { passive: true });
    spy();
  }
})();
