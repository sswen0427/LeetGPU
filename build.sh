#!/bin/bash
set -e
set -x

show_help() {
    echo "Usage: ./build.sh [MODE]"
    echo ""
    echo "Modes:"
    echo "  debug    : Compile in Debug mode (-g -G -O0). Best for cuda-gdb debugging."
    echo "  release  : Compile in Release mode (-O3, NDEBUG). Best for performance testing."
    echo "  help     : Show this help message."
    echo ""
}
# Default to 'release' mode if no argument is provided
MODE=${1:-"release"}

# Get the number of CPU cores for parallel compilation
CORES=$(nproc 2>/dev/null || echo 4)

case $MODE in
    "debug")
        rm -rf build
        mkdir build
        cd build
        cmake .. \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_CUDA_FLAGS="-g -G -O0"
        make -j${CORES}
        echo "Debug build completed successfully!"
        ;;

    "release")
        rm -rf build
        mkdir build
        cd build
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release
        make -j${CORES}
        echo "Release build completed successfully!"
        ;;

    "help"|"-h"|"--help")
        show_help
        exit 0
        ;;

    *)
        echo "Error: Unknown mode '$MODE'"
        show_help
        exit 1
        ;;
esac