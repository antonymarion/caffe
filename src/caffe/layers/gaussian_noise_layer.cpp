#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <iostream>
#include <string>


#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/debugtool.hpp"

#define DEBUG_WUHAO 1

namespace caffe {

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  std = this->layer_param_.gaussian_noise_param().std();
  LOG(INFO) << "std is: " << std;
}

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), 
                     bottom[0]->height(), bottom[0]->width());
  noise_.resize(bottom[0]->count());
}

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int count = bottom[0]->count();


  if (Caffe::phase() == Caffe::TRAIN) {
    caffe_rng_gaussian(count, Dtype(0.), std, &noise_[0]);
    caffe_add(count, &noise_[0], bottom_data, top_data);

#if 0 //DEBUG_WUHAO
  DebugTool<Dtype> dbg;
  dbg.open("gaussian.bin");
  dbg.write_blob("bottom", *bottom[0], 0);
  dbg.write_blob("top", *((*top)[0]), 0);
  dbg.close();
  std::string input;
  std::cout << "pause..." << std::endl; 
  std::cin >> input;
#endif

  } else {
    caffe_copy(count, bottom_data, top_data);
  }



}

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  caffe_copy(top[0]->count(), top[0]->cpu_diff(), (*bottom)[0]->mutable_cpu_diff());


#if 0 //DEBUG_WUHAO
  DebugTool<Dtype> dbg;
  dbg.open("gaussian.bin");
  dbg.write_blob("bottom", *((*bottom)[0]), 1);
  dbg.write_blob("top", *top[0], 1);
  dbg.close();
  std::string input;
  std::cout << "pause..." << std::endl;   
  std::cin >> input;
#endif

}


INSTANTIATE_CLASS(GaussianNoiseLayer);

}  // namespace caffe
