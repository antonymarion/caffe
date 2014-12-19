// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define DEBUG_WUHAO 1
#include <iostream>

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
  own_mask_ = true;
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random numbers

    if (own_mask_) {
      caffe_rng_bernoulli(count, 1. - threshold_, mask);
      #if 0 //DEBUG_WUHAO
      std::cout << "layer " << this->layer_param_.name() << " , mask updated..." << std::endl;
      #endif
    }


#if 0 //DEBUG_WUHAO
  std::cout << "layer " << this->layer_param_.name();
  std::cout << " own_mask_: " << own_mask_ << std::endl;
  for (int i = 0; i < 100; ++i) {
    std::cout << mask[i] << " ";
  }
  std::cout << std::endl;
  std::string input_str;
  std::cout << "pause...";
  std::cin >> input_str;
#endif



    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    if (Caffe::phase() == Caffe::TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = (*bottom)[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);


}  // namespace caffe
