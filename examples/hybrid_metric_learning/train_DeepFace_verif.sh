#!/usr/bin/env sh

TOOLS=../../build/tools
mkdir log

#GLOG_logtostderr=0 GLOG_alsologtostderr=1 $TOOLS/caffe train --solver=../../examples/DeepFace_verif/DeepFace_verif_solver.prototxt --gpu=1

GLOG_logtostderr=0 GLOG_alsologtostderr=1 $TOOLS/caffe train --solver=../../examples/DeepFace_verif/DeepFace_verif_solver_2.prototxt --gpu=1 --snapshot=../../examples/DeepFace_verif/DeepFace_verif_iter_90000.solverstate

GLOG_logtostderr=0 GLOG_alsologtostderr=1 $TOOLS/caffe train --solver=../../examples/DeepFace_verif/DeepFace_verif_solver_3.prototxt --gpu=1 --snapshot=../../examples/DeepFace_verif/DeepFace_verif_iter_160000.solverstate
