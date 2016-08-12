#include <vector>

#include "caffe/layers/l1_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void L1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  const Dtype* temp = diff_.cpu_data();
  Dtype loss = 0.0;
  for(int i = 0; i < count; ++i) {
    loss += abs(temp[i]);
  }
  loss = loss / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void L1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      const Dtype *temp = diff_.cpu_data();
      Dtype *grad = bottom[i]->mutable_cpu_diff();
      for (int n = 0; n < bottom[i]->count(); ++n){
        grad[i] = alpha*((temp[i] >=0 ) ? 1 : -1);
        if(temp[i] == 0)
          grad[i] = 0;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(L1LossLayer);
#endif

INSTANTIATE_CLASS(L1LossLayer);
REGISTER_LAYER_CLASS(L1Loss);

}  // namespace caffe
