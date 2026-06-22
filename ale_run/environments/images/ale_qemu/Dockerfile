# ALE QEMU runner
#
# Build from the repository root:
#   docker build \
#     -f ale_run/environments/images/ale_qemu/Dockerfile \
#     -t agentslastexam/ale-qemu:0.1.0 \
#     -t agentslastexam/ale-qemu:latest \
#     .
#
# This image contains QEMU and the container-side networking/display stack.
# Guest operating systems remain in qcow2 files mounted at /storage/data.qcow2.

ARG BASE_IMAGE=trycua/cua-qemu-windows@sha256:d9b0465f75b6cf801bade52e4066260e356e65b5ea0d3dab10508cd6f1ea855c
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="ALE QEMU Runner" \
      org.opencontainers.image.description="QEMU-in-Docker runtime for ALE Ubuntu and Windows full-OS sandboxes" \
      org.opencontainers.image.source="https://github.com/rdi-berkeley/agents-last-exam" \
      org.opencontainers.image.version="0.1.0"

COPY ale_run/environments/images/ale_qemu/entrypoint.sh /usr/local/bin/ale-qemu-entrypoint
COPY ale_run/environments/images/ale_qemu/healthcheck.sh /usr/local/bin/ale-qemu-healthcheck

RUN chmod 0755 /usr/local/bin/ale-qemu-entrypoint /usr/local/bin/ale-qemu-healthcheck

ENV ALE_GUEST_CUA_PORT=5000 \
    INSTALL_WINARENA_APPS=false

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD ["/usr/local/bin/ale-qemu-healthcheck"]

ENTRYPOINT ["/usr/local/bin/ale-qemu-entrypoint"]
