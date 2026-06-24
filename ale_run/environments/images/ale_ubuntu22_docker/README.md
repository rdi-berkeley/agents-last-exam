# ale-ubuntu22-docker — image build (maintainer notes)

The container form of the `ale-ubuntu22` Linux sandbox, so the `cpu-free-ubuntu`
(no-GPU, no-license) tasks run under the **docker provider** on one host instead
of one GCE VM each. Published **data-less** at
`agentslastexam/ale-ubuntu22-docker:latest`.

To *run* it you need none of this — pull the image and fetch the data
(`scripts/fetch_task_data.sh`); see the **Local Docker** docs page. This
directory is how the maintainers rebuild the image.

## Rebuild the image

A container shares the host kernel, so we don't rebuild packages: export the
`ale-ubuntu22` VM's root filesystem and `docker import` it as one layer, then
bake an entrypoint. The task data (~44 GB) and any secrets are **excluded** — the
data is supplied at runtime by the `local:` source — so the image is ~105 GB and
ships no answers.

```bash
./build.sh                    # export → import → finalize → smoke; resumable
ALE_BUILD_ON_VM=1 ./build.sh  # do it all on the VM (the ~37 GB rootfs never crosses the network)
ALE_PUSH_IMAGE=1  ./build.sh  # ... and docker push from the VM (with ALE_BUILD_ON_VM=1)
```

Env knobs: `ALE_BUILD_VM` / `ALE_BUILD_ZONE` (source VM, default `dev-ubuntu22` /
`us-west2-a`), `ALE_BUILD_IMAGE` (output tag), `ALE_BUILD_WORKDIR` (rootfs-tar
scratch, ~100 GB free), `ALE_FORCE_EXPORT=1`, `ALE_KEEP_VM=1`.

## Scripts

| script | runs on | does |
|--------|---------|------|
| `build.sh` | your host | orchestrates the build — the only entry point |
| `export_rootfs.sh` | the source VM | tars `/` minus kernel/init/logs, the task data, and any baked secrets; clamps out-of-range uids so the import is a plain stream |
| `cleanup.sh` | container, at build | bakes the entrypoint, scrubs VM identity/credentials, makes the empty `/media/user/data` mount point |
| `entrypoint.sh` | container, at runtime | `Xvfb :0` + cua-server on `:5000` (+ optional DinD) |
| `bake_nested_images.sh` | container, at build (optional) | pre-populates `/var/lib/dind` with the DinD task images |

## Optional DinD hooks

A few task evals run `docker` **inside** the sandbox (openroad, k8s_migration,
bpmn ×2). In a container that's Docker-in-Docker: the nested daemon needs
`--privileged` and can't stack overlay2 on overlay2, so it runs on
`fuse-overlayfs` at `/var/lib/dind`, with its images baked in
(`bake_nested_images.sh`) so a pulled image needs no per-start load.

The supported local-container profile intentionally remains a simple rootless
Docker path, so tasks that require Docker, Apptainer, or Singularity inside the
sandbox are excluded from `docker_support.txt`. Run those tasks with the
QEMU/KVM provider, where the nested runtime operates inside a complete Ubuntu
guest.

The DinD path remains available for custom development through
`ALE_ENABLE_DIND=1`, `enable_dind: true`, and `privileged: true`. It is not part
of the supported default task list. At high concurrency, its fuse-overlayfs
daemon can also add significant I/O and delay cua-server readiness.
