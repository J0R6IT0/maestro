# syntax=docker/dockerfile:1

ARG BACKEND=cpu

ARG UBUNTU_IMAGE=ubuntu:24.04
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

FROM ${UBUNTU_IMAGE} AS toolchain

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none

WORKDIR /src

COPY rust-toolchain.toml ./
RUN rustup toolchain install

COPY . .

FROM ${UBUNTU_IMAGE} AS openvino-libs

ARG ORT_OPENVINO_URL=https://files.pythonhosted.org/packages/50/cf/17ba72de2df0fcba349937d2788f154397bbc2d1a2d67772a97e26f6bc5f/onnxruntime_openvino-1.24.1-cp312-cp312-manylinux_2_28_x86_64.whl
ARG ORT_OPENVINO_SHA256=d617fac2f59a6ab5ea59a788c3e1592240a129642519aaeaa774761dfe35150e

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/ort.whl "$ORT_OPENVINO_URL" \
    && echo "$ORT_OPENVINO_SHA256  /tmp/ort.whl" | sha256sum -c - \
    && unzip -q -j /tmp/ort.whl 'onnxruntime/capi/*.so*' -d /tmp/capi \
    && mkdir /out \
    && cd /tmp/capi \
    && mv libonnxruntime.so.1.24.1 /out/libonnxruntime.so.1 \
    && mv libonnxruntime_providers_*.so libopenvino.so.2541 libopenvino_onnx_frontend.so.2541 \
    libopenvino_*_plugin.so libtbb.so.12 libtbbmalloc.so /out/

FROM toolchain AS build-cpu

FROM toolchain AS build-cuda
ENV CARGO_FEATURES=analyzer/cuda

FROM toolchain AS build-openvino

COPY --from=openvino-libs /out/ /opt/onnxruntime/
RUN ln -s libonnxruntime.so.1 /opt/onnxruntime/libonnxruntime.so

ENV CARGO_FEATURES=analyzer/openvino \
    ORT_LIB_LOCATION=/opt/onnxruntime \
    ORT_PREFER_DYNAMIC_LINK=1 \
    RUSTFLAGS="-C link-arg=-Wl,-rpath,\$ORIGIN"

FROM build-${BACKEND} AS build

ARG BACKEND

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/root/.cache/ort.pyke.io \
    --mount=type=cache,target=/src/target,id=maestro-target-${BACKEND} \
    cargo build --release --locked -p analyzer -p server ${CARGO_FEATURES:+--features "$CARGO_FEATURES"} \
    && mkdir -p /out \
    && cp target/release/analyzer target/release/server /out/ \
    && if [ "$BACKEND" = cuda ]; then cp target/release/libonnxruntime*.so* /out/; fi

FROM ${UBUNTU_IMAGE} AS runtime-cpu

FROM ${CUDA_IMAGE} AS runtime-cuda

FROM ${UBUNTU_IMAGE} AS runtime-openvino

RUN apt-get update \
    && apt-get install -y --no-install-recommends intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=openvino-libs /out/ /app/

FROM runtime-${BACKEND} AS maestro

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates procps tzdata \
    && rm -rf /var/lib/apt/lists/*

ARG SUPERCRONIC_VERSION=v0.2.49
ARG SUPERCRONIC_SHA256=a53ae236602c7338aba3fbaff40bda6300eae3b9fedb8261eb06cfe3724430c1
RUN curl -fsSLo /usr/local/bin/supercronic \
    https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-amd64 \
    && echo "${SUPERCRONIC_SHA256}  /usr/local/bin/supercronic" | sha256sum -c - \
    && chmod +x /usr/local/bin/supercronic

WORKDIR /app

RUN mkdir models \
    && for ext in onnx json; do \
    curl -fsSL --retry 5 --retry-all-errors --connect-timeout 30 \
    -o models/discogs-maest-30s-pw-519l-2.$ext \
    https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-519l-2.$ext || exit 1; \
    done

RUN useradd --system --uid 10001 --home-dir /data maestro \
    && mkdir /data && chown maestro /data \
    && ln -s /data/maestro.db /app/maestro.db

COPY --from=build /out/ /app/
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/maestro

ENV PATH=/app:$PATH \
    LISTEN_ADDR=0.0.0.0:3000

EXPOSE 3000
VOLUME /data

CMD ["maestro"]
