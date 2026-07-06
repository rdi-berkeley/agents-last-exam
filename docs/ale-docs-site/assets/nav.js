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
      { href: "/index.html",                    title: "Overview" },
      { href: "/pages/tasks.html",              title: "Task spec & data staging" },
      { href: "/pages/sandbox.html",            title: "Sandbox & provider" },
      { href: "/pages/agents.html",             title: "Agents & executor" },
      { href: "/pages/trajectories.html",       title: "Trajectories & artifacts" },
    ],
  },
  {
    label: "Run experiments",
    items: [
      { href: "/pages/providers.html", title: "Run an experiment", children: [
        { href: "/pages/google-cloud.html",  title: "Google Cloud VMs" },
        { href: "/pages/aws.html",           title: "AWS EC2" },
        { href: "/pages/aliyun.html",        title: "Alibaba Cloud ECS" },
        { href: "/pages/local-docker.html",  title: "Local containers" },
        { href: "/pages/local.html",         title: "QEMU/KVM VMs" },
        { href: "/pages/static.html",        title: "Existing sandbox" },
      ]},
      { href: "/pages/configure.html", title: "Configure an experiment" },
      { href: "/pages/run.html",       title: "Run and collect results" },
    ],
  },
  {
    label: "Build on ALE",
    items: [
      { href: "/pages/add-agent.html",          title: "Add an agent" },
      { href: "/pages/add-task.html",           title: "Add a task" },
      { href: "/pages/build-image.html",        title: "Build a custom image" },
    ],
  },
  {
    label: "Reference",
    items: [
      { href: "/pages/trajectory-schema.html",  title: "Trajectory schema" },
      { href: "/pages/mcp-tools.html",          title: "MCP tools" },
    ],
  },
];
