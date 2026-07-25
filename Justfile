# Task authoring surface.
#
# Every recipe delegates to the `ale` engine, pinned in ENGINE below. That is why rule
# changes usually reach you as a version bump rather than as a template merge.

ENGINE := "ale-run==0.1.*"

_default:
    @just --list

# Scaffold a new task folder that already passes lint.
new-task path:
    @uvx --from {{ ENGINE }} ale new task tasks/{{ path }}

# Regenerate kits.lock.yaml from the contents of kits/. Never edit that file by hand.
lock:
    @uvx --from {{ ENGINE }} ale kit lock .

# Structural checks: manifests, layout, references, visibility rules.
lint:
    @uvx --from {{ ENGINE }} ale lint .

# Run each task's oracle in place of the agent and require its declared score.
validate:
    @uvx --from {{ ENGINE }} ale validate .

# Everything CI runs.
check: lint validate
    @uvx --from {{ ENGINE }} ale kit lock . --check

# Run one task with a real agent, from this checkout.
run path *args:
    @uvx --from {{ ENGINE }} ale run tasks/{{ path }} {{ args }}
