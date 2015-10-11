#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define DEBUG_WUHAO 1
#include "caffe/debugtool.hpp"

namespace caffe {


template <typename Dtype>
void TripletAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)  {
  Layer<Dtype>::LayerSetUp(bottom, top);

  alpha_ = this->layer_param_.triplet_loss_param().alpha();

}


template <typename Dtype>
void TripletAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width());

  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  diff2_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  euc_diff_.Reshape(bottom[0]->num(), 1, 1, 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void TripletAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  int i;

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[2]->cpu_data(),
      diff2_.mutable_cpu_data());

  const Dtype* diff_data = diff_.cpu_data();
  const Dtype* diff2_data = diff2_.cpu_data();
  Dtype* euc_diff_data = euc_diff_.mutable_cpu_data();
  Dtype dot1, dot2;
  Dtype accuracy = 0;

  for (i = 0; i < num; ++i) {
    //euc_diff_data[i]
    dot1 = caffe_cpu_dot(dim, diff_data + i * dim, diff_data + i * dim);
    dot2 = caffe_cpu_dot(dim, diff2_data + i * dim, diff2_data + i * dim);
    if (dot1 - dot2 + alpha_ < 0) {
      accuracy += 1;
    }
  }

  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;

#if 0 //DEBUG_WUHAO
  //if (Caffe::phase() == Caffe::TRAIN)
  if (accuracy > 0) {
    DebugTool<Dtype> dbg;
    dbg.open("tri_acc.bin");
    dbg.write_blob("bottom", *(bottom[0]), 0);
    dbg.write_blob("bottom2", *(bottom[1]), 0); 
    dbg.write_blob("bottom3", *(bottom[2]), 0);
    dbg.write_blob("top", *(*top)[0], 0);
    dbg.write_blob("diff", diff_, 0);
    dbg.write_blob("diff2", diff2_, 0); 
    dbg.write_blob("euc_diff", euc_diff_, 0);
    std::cout << " acc: " << (*top)[0]->mutable_cpu_data()[0] << std::endl;
    dbg.close();
    string str;
    std::cout << "pause...";
    std::cin >> str;
  }
#endif

}

/*
template <typename Dtype>
void TripletAccuracyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
}
*/

#ifdef CPU_ONLY
STUB_GPU(TripletAccuracyLayer);
#endif

INSTANTIATE_CLASS(TripletAccuracyLayer);

}  // namespace caffe
