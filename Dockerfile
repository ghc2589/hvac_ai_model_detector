################### builder ###################
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git python3-venv pkgconf && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/conan && /opt/conan/bin/pip install -q conan==2.4.* && \
    ln -s /opt/conan/bin/conan /usr/local/bin/conan

WORKDIR /src
COPY . .
WORKDIR /src/Deploy

RUN --mount=type=cache,target=/root/.conan2 bash -euc '\
  conan profile detect --force && \
  conan remote add center2 https://center2.conan.io --force && \
  conan install . --output-folder=build -s build_type=Release \
        -c tools.system.package_manager:mode=install --build=missing && \
  cmake --preset conan-release && \
  cmake --build build --config Release -j$(nproc) && \
  \
  mkdir -p /opt/runtime-libs && \
  find /root/.conan2/p -type f -name "*.so*" -exec cp -an {} /opt/runtime-libs/ \; && \
  ldconfig \
'


################ runtime ################
FROM ubuntu:24.04

WORKDIR /app/Deploy
COPY --from=builder /src/Deploy/build/hvac   /app/Deploy/build/hvac
COPY AI-Models/                               /app/AI-Models/
COPY --from=builder /opt/runtime-libs/        /usr/local/lib/
RUN ldconfig

EXPOSE 18080

CMD ["./build/hvac"]

