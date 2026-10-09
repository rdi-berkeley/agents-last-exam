# ALE framework documentation site

A static documentation site for the `agents-last-exam` (`ale_run`) codebase.
Plain HTML, one file per section, one shared design system. No build step.

## Run it locally

The pages use root-relative asset paths (`/assets/...`), so serve the folder
from its own root. Start in the repository root:

```bash
cd docs/ale-docs-site
uv run python serve.py 5500
# open http://localhost:5500/
```

`serve.py` sends no-cache headers so a normal reload picks up HTML and
navigation changes. Use a free port.
Opening files directly does not resolve the shared root-relative paths.

## Public website integration

The main website's [Docs page](https://agents-last-exam.org/docs) embeds the
static site in an iframe at `/docs/ale/`. Public links use
`/docs?p=pages/run.html`; the wrapper selects that page and updates `p` as the
iframe navigates. The published HTML and navigation rewrite `/pages/`,
`/assets/`, and `/index.html` paths under `/docs/ale/`.

Keep source paths rooted at `/` for local preview. Publish the complete static
directory through the main website's docs integration, including shared JS and
CSS, with that prefix rewrite. Editing Markdown does not update these HTML
pages. This checkout contains the docs source and preview server; it does not
contain the main website wrapper or its deployment configuration.

## Content layout

- Repository Markdown provides practical setup and run commands.
- `pages/providers.html` and provider pages teach setup; `pages/configure.html`
  is the configuration reference, including v1.1 scope and asset pins.
- `pages/run.html` covers launch, revision-aware resume, the short v1 log
  upgrade, output, and cleanup. It pairs with `docs/releases/v1.1.md`.
- Introduction, extension, and reference pages explain the framework contracts.

Update both Markdown and HTML when commands, task lists, image pins, resume,
or output defaults change. Keep one v1.1 release guide and reuse the configure
and run pages for the website.

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
each article. Set an explicit heading ID for a stable cross-page anchor.
