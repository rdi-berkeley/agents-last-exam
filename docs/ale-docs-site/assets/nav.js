/* =========================================================================
   Site information architecture — single source of truth.
   Reader journey: Introduction (overview + architecture: how ALE works) →
   Run experiments (provision + configure + run the benchmark) →
   Build on ALE → Reference.
   An item may carry `children` for one level of nesting. `draft: true` = stub.
   ========================================================================= */
window.ALE_NAV = [
  {
    label: "Introduction",
    items: [
      { href: "/index.html",                       title: "Overview" },
      { href: "/pages/tasks.html",         title: "Task spec & data staging" },
      { href: "/pages/sandbox.html",      title: "Sandbox & provider" },
      { href: "/pages/agents.html",title: "Agents & executor" },
      { href: "/pages/trajectories.html",        title: "Trajectories & artifacts" },
    ],
  },
  {
    label: "Run experiments",
    items: [
      { href: "/pages/providers.html",       title: "Setup Environment Provider", children: [
        { href: "/pages/google-cloud.html",           title: "Google Cloud" },
        { href: "/pages/aws.html",                    title: "AWS" },
        { href: "/pages/local.html",         title: "VMware / QEMU" },
      ]},
      { href: "/pages/configure.html",               title: "Configure & run a benchmark" },
    ],
  },
  {
    label: "Build on ALE",
    items: [
      { href: "/pages/add-agent.html",         title: "Add an agent" },
      { href: "/pages/add-environment.html",           title: "Add an environment" },
      { href: "/pages/add-task.html",          title: "Add a task" },
    ],
  },
  {
    label: "Reference",
    items: [
      { href: "/pages/trajectory-schema.html",     title: "Trajectory schema" },
      { href: "/pages/mcp-tools.html",             title: "MCP tools" },
    ],
  },
];
