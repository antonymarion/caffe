#include <algorithm>
#include <cfloat>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/debugtool.hpp"
#define debug 1

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
  Dtype* tmp = NULL;
  tmp = (Dtype*)malloc(sizeof(Dtype)*height_*width_);
  if (NULL == tmp){
  	LOG(FATAL) << "Error malloc memory";
  }

	for (int n = 0; n < bottom[0]->num(); ++n) {
  	for (int c = 0; c < channels_; ++c) {
  		for (int h = 0; h < height_; ++h) {
  			for (int w = 0; w < width_; ++w) {
  				tmp[h * width_ + w] = bottom_data[((n * channels_ + c) * height_ + h) * width_ + w];
  			}
  		}
      cv::Mat cv_img(width_, height_, CV_32FC1, tmp);
      cv::Mat cv_img_out;
      switch (this->layer_param_.fractional_map_param().fraction()) {
        case FractionalMapParameter_FractionMethod_CUBIC:
          cv::resize(cv_img, cv_img_out, cv::Size(fractional_width_, fractional_height_));
          break;
        case FractionalMapParameter_FractionMethod_BOX:
          cv::resize(cv_img, cv_img_out, cv::Size(fractional_width_, fractional_height_));
          break;
        case FractionalMapParameter_FractionMethod_TRIANGLE:
          cv::resize(cv_img, cv_img_out, cv::Size(fractional_width_, fractional_height_));
          break;
        case FractionalMapParameter_FractionMethod_LANCZOS2:
          cv::resize(cv_img, cv_img_out, cv::Size(fractional_width_, fractional_height_));
          break;
        case FractionalMapParameter_FractionMethod_LANCZOS3:
          cv::resize(cv_img, cv_img_out, cv::Size(fractional_width_, fractional_height_));
          break;
        default:
          LOG(FATAL) << "Unkown fractional_map method.";

      }
      for (int ph = 0; ph < fractional_height_; ++ph) {
        for (int pw = 0; pw < fractional_width_; ++pw) {
          int fractional_index = ((n * channels_ + c) * fractional_height_
                                   + ph) * fractional_width_ + pw;
          top_data[fractional_index] = cv_img_out.at<float>(ph, pw);
        }
      }
    }
  }

  #if debug //DEBUG
  DebugTool<Dtype> dbg;
  dbg.open("fractional_map.bin");
  dbg.write_blob("bottom", *bottom[0], 0);
  dbg.write_blob("top", *((*top)[0]), 0);
  dbg.close();
  std::string input;
  std::cout << "pause..." << std::endl; 
  std::cin >> input;
  #endif
}

template <typename Dtype>
void FractionalMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);


  
}


#ifdef CPU_ONLY
STUB_GPU(FractionalMapLayer);
#endif

INSTANTIATE_CLASS(FractionalMapLayer);
}