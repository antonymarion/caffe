#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FractionalMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  FractionalMapParameter fractionalmap_param = this->layer_param_.fractional_map_param();
  if (fractionalmap_param.has_ratio()) {
  	ratio_ = fractionalmap_param.ratio();
  } else {
  	ratio_ = 1;
  }

  CHECK_GT(ratio_, 0) << "The resize ratio cannot be zero.";
}

template <typename Dtype>
void FractionalMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();	
  width_ = bottom[0]->width();
  fractional_height_ = static_cast<int>(ceil(height_*ratio_));	
  fractional_width_ = static_cast<int>(ceil(width_*ratio_));	
  (*top)[0]->Reshape(bottom[0]->num(), channels_, fractional_height_, 
  	fractional_width_);	
  if (top->size() > 1) {  	
  	(*top)[1]->ReshapeLike(*(*top)[0]);
  }
}

template <typename Dtype>
void FractionalMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int top_count = (*top)[0]->count();
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;

  switch (this->layer_param_.fractional_map_param().fraction()) {
  case FractionalMapParameter_FractionMethod_CUBIC:
  	
  	break;
  case FractionalMapParameter_FractionMethod_BOX:

  	break;
  case FractionalMapParameter_FractionMethod_TRIANGLE:

  	break;
  case FractionalMapParameter_FractionMethod_LANCZOS2:

  	break;
  case FractionalMapParameter_FractionMethod_LANCZOS3:

  	break;
  default:
  	LOG(FATAL) << "Unkown fractional_map method.";
  }
}

template <typename Dtype>
void FractionalMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {


  switch (this->layer_param_.fractional_map_param().fraction()) {
  case FractionalMapParameter_FractionMethod_CUBIC:

  	break;
  case FractionalMapParameter_FractionMethod_BOX:

  	break;
  case FractionalMapParameter_FractionMethod_TRIANGLE:

  	break;
  case FractionalMapParameter_FractionMethod_LANCZOS2:

  	break;
  case FractionalMapParameter_FractionMethod_LANCZOS3:

  	break;
  default:
  	LOG(FATAL) << "Unkown fractional_map method.";
  }
}
}

#ifdef CPU_ONLY
STUB_GPU(FractionalMapLayer);
#endif

INSTANTIATE_CLASS(FractionalMapLayer);
}