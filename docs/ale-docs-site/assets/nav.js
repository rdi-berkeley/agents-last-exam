/* =========================================================================
   Site information architecture — single source of truth.
   Reader journey: Introduction (overview + architecture: how ALE works) →
   Run experiments (provision + configure + run the benchmark) →
   Build on ALE → Reference → About.
   An item may carry `children` for one level of nesting. `draft: true` = stub.
   ========================================================================= */
window.ALE_NAV = [
  {
    label: "Introduction",
    items: [
      { href: "/index.html",                       title: "Overview" },
      { href: "/pages/run-lifecycle.html",         title: "Run lifecycle" },
      { href: "/pages/arch-environment.html",      title: "Sandbox & provider" },
      { href: "/pages/arch-taskdata.html",         title: "Task spec & data staging" },
      { href: "/pages/arch-executor-deployer.html",title: "Agents & executor" },
      { href: "/pages/data-artifacts.html",        title: "Trajectories & artifacts" },
    ],
  },
  {
    label: "Run experiments",
    items: [
      { href: "/pages/setup-providers.html",       title: "Setup Environment Provider", children: [
        { href: "/pages/setup-gcp.html",           title: "Google Cloud" },
        { href: "/pages/setup-local.html",         title: "VMware / QEMU" },
      ]},
      { href: "/pages/configs.html",               title: "Configure & run a benchmark" },
    ],
  },
  {
    label: "Build on ALE",
    items: [
      { href: "/pages/extend-agents.html",         title: "Add an agent" },
      { href: "/pages/extend-envs.html",           title: "Add an environment" },
      { href: "/pages/extend-tasks.html",          title: "Add a task" },
    ],
  },
  {
    label: "Reference",
    items: [
      { href: "/pages/trajectory-schema.html",     title: "Trajectory schema" },
      { href: "/pages/mcp-tools.html",             title: "MCP tools" },
    ],
  },
  {
    label: "About",
    items: [
      { href: "/pages/about.html",                 title: "About & licensing", draft: true },
    ],
  },
];
