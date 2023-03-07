#!/bin/bash

rm -f ./CMakeCache.txt
rm -f ./Makefile ./*/Makefile
rm -rf ./CMakeFiles ./*/CMakeFiles
rm -f ./*.cmake ./*/*.cmake
rm -f ./*.a ./*/*.a
rm -f *.so

cmake .

make
