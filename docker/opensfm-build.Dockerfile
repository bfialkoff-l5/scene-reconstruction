# Rebuild OpenSfM's C++ extensions from our pinned fork
# (Line5-ai/OpenSfM @ c5328439465e6ace011f39077d1077d7b1cdd65d, the exact commit ODM
# 3.6.0 builds) on top of the stock ODM image.
#
# Why this works without rebuilding all of ODM: the opendronemap/odm:gpu runtime image
# already ships the OpenSfM source tree and the prebuilt heavy deps (Ceres, OpenCV) under
# /code/SuperBuild/install, but ODM's Dockerfile deliberately strips the compilers and the
# -dev headers ("to save space"). We restore just those, overlay our forked source, and
# rebuild the extensions in place against the image's existing Ceres/OpenCV.
#
# PROOF phase: the forked source is identical to the image's commit, so the rebuilt image
# must behave like stock ODM. Only after that do we modify ba_helpers.cc.
#
# Build (context = the OpenSfM submodule so the context stays tiny):
#   docker build -f docker/opensfm-build.Dockerfile -t odm-osfm:proof third_party/opensfm
FROM opendronemap/odm:gpu

USER root

# Restore the build toolchain + the -dev headers ODM strips from its runtime image.
# Package names mirror ODM's snap/snapcraft.yaml build deps; Ubuntu 24.04 apt resolves
# them to the same versions ODM's cuda:devel builder used.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config python3-dev wget ca-certificates git \
      libeigen3-dev libgoogle-glog-dev libgflags-dev libsuitesparse-dev liblapack-dev \
      libtbb-dev \
 && rm -rf /var/lib/apt/lists/*

# ODM's runtime image keeps Ceres' headers + CMake config but STRIPS the static lib
# (/code/SuperBuild/install/lib/libceres.a is gone), so OpenSfM can't link. Rebuild the
# exact Ceres ODM 3.6.0 uses (2.0.0 + ODM's ceres.patch, same flags) and install it back
# into the SuperBuild prefix, so the OpenSfM build links it statically just like ODM did.
RUN cd /tmp \
 && wget -q http://ceres-solver.org/ceres-solver-2.0.0.tar.gz \
 && tar xf ceres-solver-2.0.0.tar.gz \
 && cd ceres-solver-2.0.0 \
 && git apply /code/SuperBuild/cmake/ceres.patch \
 && mkdir build && cd build \
 && cmake .. \
      -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC \
      -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
      -DMINIGLOG=ON -DMINIGLOG_MAX_LOG_LEVEL=-100 \
      -DCMAKE_INSTALL_PREFIX:PATH=/code/SuperBuild/install \
 && make "-j$(nproc)" \
 && make install \
 && test -f /code/SuperBuild/install/lib/libceres.a \
 && cd /tmp && rm -rf ceres-solver-2.0.0 ceres-solver-2.0.0.tar.gz

ARG OSFM=/code/SuperBuild/install/bin/opensfm

# Overlay our pinned fork over the image's source copy (build provenance = our submodule).
COPY . ${OSFM}/

# Drop the image's prebuilt extensions so we can be certain the run uses freshly built ones.
RUN rm -f ${OSFM}/opensfm/*.so

# Rebuild against the image's prebuilt Ceres/OpenCV, mirroring ODM's External-OpenSfM.cmake.
# CMake drops the .so into the opensfm package dir via
# CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE = ${opensfm_SOURCE_DIR}/.. (hence -DCMAKE_BUILD_TYPE=Release).
RUN mkdir -p /tmp/osfm-build && cd /tmp/osfm-build \
 && cmake ${OSFM}/opensfm/src \
      -DCMAKE_BUILD_TYPE=Release \
      -DCERES_ROOT_DIR=/code/SuperBuild/install \
      -DOpenCV_DIR=/code/SuperBuild/install/lib/cmake/opencv4 \
      -DADDITIONAL_INCLUDE_DIRS=/code/SuperBuild/install/include \
      -DOPENSFM_BUILD_TESTS=off \
      -DPYTHON_EXECUTABLE=/code/venv/bin/python3 \
 && make "-j$(nproc)" \
 && rm -rf /tmp/osfm-build

# Smoke test: the freshly built extensions import (run from /tmp so /code/opendm/types.py
# can't shadow the stdlib `types` module). Image ENV already sets PYTHONPATH/LD_LIBRARY_PATH.
RUN cd /tmp && python3 -c "from opensfm import io, pymap, pysfm, pybundle, pygeometry; print('opensfm rebuilt OK')"
