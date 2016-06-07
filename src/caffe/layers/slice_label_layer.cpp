#include <algorithm>
#include <vector>

#include "caffe/layers/slice_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void SliceLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const SliceLabelParameter& slice_label_param = this->layer_param_.slice_label_param();

    CHECK(!(top.size()/2 == slice_label_param.num)) 
        << "The number of slice_label_layer is error(" << top.size()/2 
        << "vs " << slice_label_param.num << ").";
}

template <typename Dtype>
void SliceLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int batch_size = bottom[0]->num();
    const SliceLabelParameter& slice_label_param = this->layer_param_.slice_label_param();


}


} // namespace