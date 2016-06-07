#include <algorithm>
#include <vector>

#include "caffe/layers/binary_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BinaryLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    neg_target_ = this->layer_param_.binary_label_param().negative_target();
}


template <typename Dtype>
void BinaryLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)  {
    top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void BinaryLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int cnt = top[0]->count();
    const Dtype* bottom1_data = bottom[0]->cpu_data();
    const Dtype* bottom2_data = bottom[1]->cpu_data();

    Dtype* top_data = top[0]->mutable_cpu_data();

    for(int i = 0; i < cnt; ++i) {
        int label1 = static_cast<int>(bottom1_data[i]);
        int label2 = static_cast<int>(bottom2_data[i]);
        top_data[i] = (label1 == label2) ? (1.0) : (neg_target_);
    }

}

template <typename Dtype>
void BinaryLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(BinaryLabelLayer);
#endif

INSTANTIATE_CLASS(BinaryLabelLayer);
REGISTER_LAYER_CLASS(BinaryLabel);

} // namespace caffe