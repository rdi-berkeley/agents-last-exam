# ALE Framework Docs — site

A static documentation site for the `agents-last-exam` (`ale_run`) codebase.
Plain HTML, one file per section, one shared design system. No build step.

## Run it locally

The pages use root-relative asset paths (`/assets/...`), so serve the folder
from its own root:

```bash
cd ale-docs-site
python3 serve.py 5500     # no-cache dev server — reloads always show your edits
# open http://localhost:5500/
```

Use `serve.py`, not plain `python -m http.server`: the latter lets the browser
cache `nav.js`/`app.js`, so structural edits won't appear without a hard refresh
(Cmd/Ctrl+Shift+R). `serve.py` sends no-cache headers so a normal reload is
enough. Pick any free port — on shared/sandbox hosts the 8000–8400 range is
often already taken by a proxy, so `5500` is a safe default here. Opening the `.html` files directly with
`file://` will not work — the shared nav/asset paths need a server root.

## How it's wired

```
ale-docs-site/
├── index.html              Home / overview (root page)
├── pages/                   One HTML file per section
├── assets/
│   ├── style.css            The whole design system (light + dark via CSS vars)
│   ├── nav.js               ← edit this to add/reorder sections (single source of truth)
│   └── app.js               Builds sidebar + topbar + TOC + prev/next into every page
└── README.md
```

Each page only contains its `<article class="article">…</article>` content plus
the two `<script>` tags at the bottom. `app.js` injects all the shared chrome
(sidebar, breadcrumbs, theme toggle, right-hand table of contents, prev/next
footer) at load time, so the layout stays identical across every page.

## Add or edit a section

1. Add an entry to the relevant group in `assets/nav.js` (set `draft: true` for
   a stub). The sidebar and prev/next links update automatically.
2. Copy an existing page in `pages/` as a template and write the `.article` body.
3. Use the shared components: `.note` / `.note warn` / `.note todo` callouts,
   `.diagram` for ASCII diagrams, `.card-grid` + `.card` for hub links,
   `.pill`, tables, and fenced code via `<pre><code>`.

Right-hand TOC entries are generated automatically from the `<h2>`/`<h3>` in
each article — no manual anchors needed.
