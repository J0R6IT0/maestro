# syntax=docker/dockerfile:1

ARG BACKEND=cpu

ARG UBUNTU_IMAGE=ubuntu:24.04
ARG CUDA_IMAGE=nvidia/cuda:12.9.1-base-ubuntu24.04

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

FROM ${UBUNTU_IMAGE} AS ort-libs

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /out

FROM ort-libs AS cpu-libs

ARG ORT_CPU_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-linux-x64-1.31.0.tgz
ARG ORT_CPU_SHA256=cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57

RUN curl -fsSLo /tmp/ort.tgz "$ORT_CPU_URL" \
    && echo "$ORT_CPU_SHA256  /tmp/ort.tgz" | sha256sum -c - \
    && tar -xzf /tmp/ort.tgz -C /out --strip-components=2 --no-same-owner --wildcards '*/lib/libonnxruntime.so.1.*' \
    && mv /out/libonnxruntime.so.1.* /out/libonnxruntime.so.1 \
    && rm /tmp/ort.tgz

FROM ort-libs AS cuda-libs

ARG ORT_CUDA_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-linux-x64-gpu_cuda12-1.31.0.tgz
ARG ORT_CUDA_SHA256=62d3af5269af2143b5d37494337573f1ffb8885f3ad7477ae0cb1cc79c8dedfd

RUN curl -fsSLo /tmp/ort.tgz "$ORT_CUDA_URL" \
    && echo "$ORT_CUDA_SHA256  /tmp/ort.tgz" | sha256sum -c - \
    && tar -xzf /tmp/ort.tgz -C /out --strip-components=2 --no-same-owner --wildcards \
    '*/lib/libonnxruntime.so.1.*' '*/lib/libonnxruntime_providers_shared.so' '*/lib/libonnxruntime_providers_cuda.so' \
    && mv /out/libonnxruntime.so.1.* /out/libonnxruntime.so.1 \
    && rm /tmp/ort.tgz

FROM ort-libs AS openvino-libs

ARG ORT_OPENVINO_URL=https://files.pythonhosted.org/packages/50/cf/17ba72de2df0fcba349937d2788f154397bbc2d1a2d67772a97e26f6bc5f/onnxruntime_openvino-1.24.1-cp312-cp312-manylinux_2_28_x86_64.whl
ARG ORT_OPENVINO_SHA256=d617fac2f59a6ab5ea59a788c3e1592240a129642519aaeaa774761dfe35150e

RUN curl -fsSLo /tmp/ort.whl "$ORT_OPENVINO_URL" \
    && echo "$ORT_OPENVINO_SHA256  /tmp/ort.whl" | sha256sum -c - \
    && unzip -q -j /tmp/ort.whl 'onnxruntime/capi/*.so*' -d /tmp/capi \
    && cd /tmp/capi \
    && mv libonnxruntime.so.1.24.1 /out/libonnxruntime.so.1 \
    && mv libonnxruntime_providers_*.so libopenvino.so.2541 libopenvino_onnx_frontend.so.2541 \
    libopenvino_*_plugin.so libtbb.so.12 libtbbmalloc.so /out/

FROM ${BACKEND}-libs AS libs

FROM ${UBUNTU_IMAGE} AS ffmpeg

ARG FFMPEG_VERSION=9.0.2
ARG FFMPEG_SHA256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl nasm pkg-config xz-utils \
    zlib1g-dev libbz2-dev libopenmpt-dev libgme-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz" \
    && echo "$FFMPEG_SHA256  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
    && mkdir /tmp/ffmpeg \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
    && cd /tmp/ffmpeg \
    && ./configure --disable-autodetect --enable-pthreads --disable-debug --disable-doc --disable-network \
    --disable-ffplay --disable-ffprobe --disable-avdevice --disable-devices --disable-hwaccels \
    --disable-encoders --enable-encoder=pcm_f32le \
    --disable-muxers --enable-muxer=pcm_f32le \
    --disable-protocols --enable-protocol=file,pipe \
    --disable-filters --enable-filter=abuffer,abuffersink,aformat,anull,aresample,atrim \
    --enable-zlib --enable-bzlib --enable-libopenmpt --enable-libgme \
    && make -j"$(nproc)" ffmpeg \
    && install -D -s ffmpeg /out/ffmpeg \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

FROM toolchain AS build

ARG BACKEND

COPY --from=libs /out/ /opt/onnxruntime/
RUN ln -s libonnxruntime.so.1 /opt/onnxruntime/libonnxruntime.so

ENV ORT_LIB_LOCATION=/opt/onnxruntime \
    ORT_PREFER_DYNAMIC_LINK=1 \
    RUSTFLAGS="-C link-arg=-Wl,-rpath,\$ORIGIN"

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/target,id=maestro-target-${BACKEND} \
    if [ "$BACKEND" != cpu ]; then features="--features analyzer/$BACKEND"; fi \
    && cargo build --release --locked -p analyzer -p server $features \
    && mkdir -p /out \
    && cp target/release/analyzer target/release/server /out/

FROM ${UBUNTU_IMAGE} AS runtime-cpu

FROM ${CUDA_IMAGE} AS runtime-cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libcublas-12-9=12.9.2.10-1 libcurand-12-9=10.3.10.19-1 cuda-nvrtc-12-9=12.9.86-1 libcudnn9-cuda-12=9.10.2.21-1 \
    && rm -rf /var/lib/apt/lists/*

FROM ${UBUNTU_IMAGE} AS runtime-openvino

RUN apt-get update \
    && apt-get install -y --no-install-recommends intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-${BACKEND} AS maestro

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates procps tzdata libopenmpt0t64 libgme0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg /out/ffmpeg /usr/local/bin/

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

COPY --from=libs /out/ /app/
COPY --from=build /out/ /app/
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/maestro

ENV PATH=/app:$PATH \
    LISTEN_ADDR=0.0.0.0:3000

EXPOSE 3000
VOLUME /data

CMD ["maestro"]
