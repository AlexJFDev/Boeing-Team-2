# Sandbox image for model-driven tool calls (WBS 1.8, issue #8).
#
# Everything that can change a result is pinned: the base image by digest, the
# apt package by exact version (apt.lock), and the Python packages by version
# and hash (requirements.lock). If a pin is no longer available the build
# fails; it never falls back to another version.
#
# Bump SANDBOX_VERSION in compose.yaml whenever this file or a lockfile changes.

# python:3.13-slim-trixie, multi-arch index (arm64 and amd64)
FROM python@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY apt.lock /opt/sandbox/

# "pkg=version" makes apt exit non-zero when that exact version is not offered.
RUN apt-get update \
    && xargs -a /opt/sandbox/apt.lock apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /opt/sandbox/
RUN pip install --require-hashes --no-deps -r /opt/sandbox/requirements.lock

COPY sandbox-versions /usr/local/bin/sandbox-versions
RUN chmod 0755 /usr/local/bin/sandbox-versions

# Unprivileged user; tshark only reads capture files here, it never captures.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin sandbox
USER sandbox
WORKDIR /workspace
