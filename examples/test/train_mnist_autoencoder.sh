#!/usr/bin/env sh

../../build/tools/caffe train \
  --solver=mnist_autoencoder_solver.prototxt
