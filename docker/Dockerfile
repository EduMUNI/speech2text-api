# syntax = docker/dockerfile:1.0-experimental
FROM python:3.8-buster AS builder

RUN apt-get update \
    && apt-get install -y \
    # must-have-tools-in-each-docker-image-in-production \
    bash less netcat vim curl wget nmap traceroute net-tools iputils-ping openssh-client telnet strace mc dnsutils procps htop ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN --mount=type=secret,id=netrc,dst=/root/.netrc \
    --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY release.tar.gz /tmp/
RUN pip install /tmp/release.tar.gz
CMD ["ga-speech2text-model", "httpapi", "--host", "0.0.0.0", "--port", "5000"]
