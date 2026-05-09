#!/bin/bash -ex

find test/ -iname '*.cu' -print0 | xargs -0 clang-format -i --style=Google
