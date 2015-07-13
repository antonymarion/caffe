#!/usr/bin/env sh
export PATH=/usr/local/cuda-6.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH
../../build/tools/caffe train --solver=../../examples/mnist/lenet_solver.prototxt
