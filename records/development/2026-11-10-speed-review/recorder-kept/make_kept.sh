#!/bin/bash
# The engine's base library with the scheduler's kept outputs, built beside the installed engine.
#   make_kept.sh <scratch folder for the build>
# Source: dependencies/llama.cpp-f089c9c/ggml (the installed engine's source, read only), with ggml-backend-eval-kept.patch.
# Only libggml-base.so.0 is made: the scheduler lives there, and every other library and the recorder stay the installed files.
# It is installed alone in dependencies/engine-f089c9c-sm120-kept/, to be put ahead of the recorder's own folder
# (LD_LIBRARY_PATH; the recorder's RUNPATH gives way to it) by a caller that also sets GGML_SCHED_EVAL_KEPT=1.
# Without that variable the patched library takes the installed library's own path through the scheduler.
# Paths in the file's messages are written relative (-ffile-prefix-map), so the bytes do not depend on the scratch folder.
set -eu
ROOT="$(git rev-parse --show-toplevel)"
HERE=$ROOT/records/development/2026-11-10-speed-review/recorder-kept
SOURCE=$ROOT/dependencies/llama.cpp-f089c9c/ggml
INSTALL=$ROOT/dependencies/engine-f089c9c-sm120-kept
[ $# -eq 1 ] || { echo "usage: make_kept.sh <scratch folder for the build>"; exit 2; }
BUILD=$(realpath -m "$1")
[ -e "$BUILD" ] && { echo "$BUILD exists: give a new folder"; exit 2; }
[ -e "$INSTALL/libggml-base.so.0.21.0" ] && { echo "$INSTALL already holds the library: an installed engine is not written over"; exit 2; }
mkdir -p "$BUILD"
cp -r "$SOURCE" "$BUILD/ggml"
( cd "$BUILD" && patch -p1 --quiet < "$HERE/ggml-backend-eval-kept.patch" )
: > "$BUILD/ggml/ggml.pc.in"   # the source folder holds no pkg-config template, and the configure step asks for one
MAP="-ffile-prefix-map=$BUILD/ggml=ggml"
cmake -S "$BUILD/ggml" -B "$BUILD/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_CUDA=OFF \
  -DGGML_STANDALONE=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DCMAKE_C_FLAGS="$MAP" -DCMAKE_CXX_FLAGS="$MAP" > "$BUILD/configure.log" 2>&1
cmake --build "$BUILD/build" --target ggml-base ggml-cpu > "$BUILD/build.log" 2>&1
g++ -O1 -std=c++17 -I "$BUILD/ggml/include" "$HERE/eval_kept_test.cpp" -o "$BUILD/eval_kept_test" -L "$BUILD/build/src" -lggml-cpu -lggml-base -Wl,-rpath,"$BUILD/build/src"
mkdir -p "$INSTALL"
cp "$BUILD/build/src/libggml-base.so.0.21.0" "$INSTALL/libggml-base.so.0.21.0"
ln -s libggml-base.so.0.21.0 "$INSTALL/libggml-base.so.0"
{
  echo "libggml-base.so.0.21.0 sha256 $(sha256sum "$INSTALL/libggml-base.so.0.21.0" | cut -d' ' -f1)"
  echo "source ggml-backend.cpp sha256 $(sha256sum "$SOURCE/src/ggml-backend.cpp" | cut -d' ' -f1)"
  echo "patch sha256 $(sha256sum "$HERE/ggml-backend-eval-kept.patch" | cut -d' ' -f1)"
  echo "compiler $(g++ --version | head -1)"
  echo "cmake $(cmake --version | head -1); -G Ninja Release BUILD_SHARED_LIBS=ON GGML_NATIVE=OFF GGML_CUDA=OFF, file-prefix-map"
  echo "built $(date '+%F %T %Z')"
} > "$INSTALL/BUILD"
cat "$INSTALL/BUILD"
