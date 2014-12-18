#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/debugtool.hpp"
#define DEBUG_WUHAO 1

namespace caffe {

template <typename Dtype>
void BinaryLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  neg_target_ = this->layer_param_.binary_label_param().negative_target();
}


template <typename Dtype>
void BinaryLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->ReshapeLike(*bottom[0]);
}
  

template <typename Dtype>
void BinaryLabelLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const int count = (*top)[0]->count();
  const Dtype* bottom1_data = bottom[0]->cpu_data();
  const Dtype* bottom2_data = bottom[1]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  for (int i = 0; i < count; ++i) {
    int label1 = static_cast<int>(bottom1_data[i]);
    int label2 = static_cast<int>(bottom2_data[i]);
    top_data[i] = (label1 == label2) ? (1.0) : (neg_target_);    
  }

#if 0 //DEBUG_WUHAO
  DebugTool<Dtype> dbg;
  dbg.open("binary_label.bin");
  dbg.write_blob("label1", *bottom[0], 0); 
  dbg.write_blob("label2", *bottom[1], 0);
  dbg.write_blob("top", *(*top)[0], 0);
  dbg.close();
  string str;
  std::cout << "pause...";
  std::cin >> str;
#endif


}

template <typename Dtype>
void BinaryLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(BinaryLabelLayer);
#endif

INSTANTIATE_CLASS(BinaryLabelLayer);


}  // namespace caffe
