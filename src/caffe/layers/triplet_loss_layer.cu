#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {

    Forward_cpu(bottom, top);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    Backward_cpu(top, propagate_down, bottom);
}
    
INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}