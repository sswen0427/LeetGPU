#!/bin/bash

set -e
set -x

rm -rf third_party
git clone https://github.com/sswen0427/cpp3rdlib.git
mv cpp3rdlib third_party
cd third_party
rm .git -rf
mv gtest-1.17.0-debian12-x86_64-gcc12.2.0 gtest

rm abseil-cpp-20260107.1-debian12-x86_64-gcc12.2.0 -rf
rm armadillo-15.2.4-debian12-x86_64-gcc12.2.0 -rf
rm boost-1.90.0-debian12-x86_64-gcc12.2.0 -rf
rm gflags-2.2.2-debian12-x86_64-gcc12.2.0 -rf
rm glog-0.7.1-debian12-x86_64-gcc12.2.0 -rf
rm json-v3.12.0-debian12-x86_64-gcc12.2.0 -rf
rm openblas-0.3.32-debian12-x86_64-gcc12.2.0 -rf
rm re2-2025-11-05-debian12-x86_64-gcc12.2.0 -rf
rm sentencepiece-0.2.1-debian12-x86_64-gcc12.2.0 -rf
rm unordered_dense-4.8.1-debian12-x86_64-gcc12.2.0 -rf
rm unwind-1.8.3-debian12-x86_64-gcc12.2.0 -rf