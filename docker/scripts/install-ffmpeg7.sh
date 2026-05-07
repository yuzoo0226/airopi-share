#!/bin/bash

set -euo pipefail

FFMPEG_VERSION="${1:-7.1.1}"
BUILD_ROOT="$(mktemp -d)"
TARBALL_URL="https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:/usr/local/share/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH:-}"

cleanup() {
  rm -rf "${BUILD_ROOT}"
}

trap cleanup EXIT

echo "[FFmpeg] Building FFmpeg ${FFMPEG_VERSION} from ${TARBALL_URL}"

cd "${BUILD_ROOT}"
curl -fsSL "${TARBALL_URL}" -o ffmpeg.tar.xz
tar -xf ffmpeg.tar.xz
cd "ffmpeg-${FFMPEG_VERSION}"

./configure \
  --prefix=/usr/local \
  --enable-gpl \
  --enable-shared \
  --disable-static \
  --disable-debug \
  --disable-doc \
  --disable-ffplay \
  --enable-pic

make -j"$(nproc)"
make install
ldconfig

echo "[FFmpeg] Installed $(/usr/local/bin/ffmpeg -version | head -n 1)"
echo "[FFmpeg] libavformat version $(pkg-config --modversion libavformat)"
