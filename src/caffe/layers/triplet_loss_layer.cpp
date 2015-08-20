#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define DEBUG_WUHAO 1
#include "caffe/debugtool.hpp"

namespace caffe {


template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)  {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  alpha_ = this->layer_param_.triplet_loss_param().alpha();

}


template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
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
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
  Dtype loss = 0;

  for (i = 0; i < num; ++i) {
    //euc_diff_data[i]
    dot1 = caffe_cpu_dot(dim, diff_data + i * dim, diff_data + i * dim);
    dot2 = caffe_cpu_dot(dim, diff2_data + i * dim, diff2_data + i * dim);
    euc_diff_data[i] = std::max(dot1 - dot2 + alpha_, Dtype(0));
    loss += (euc_diff_data[i])/(num * 2);
  }

  (*top)[0]->mutable_cpu_data()[0] = loss;

#if 0 //DEBUG_WUHAO
  //if (Caffe::phase() == Caffe::TRAIN)
  DebugTool<Dtype> dbg;
  dbg.open("tri_loss.bin");
  dbg.write_blob("bottom", *(bottom[0]), 0);
  dbg.write_blob("bottom2", *(bottom[1]), 0); 
  dbg.write_blob("bottom3", *(bottom[2]), 0);
  dbg.write_blob("top", *(*top)[0], 0);
  dbg.write_blob("diff", diff_, 0);
  dbg.write_blob("diff2", diff2_, 0); 
  dbg.write_blob("euc_diff", euc_diff_, 0);
  dbg.close();
  string str;
  std::cout << "pause...";
  std::cin >> str;
#endif

}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  int count = (*bottom)[0]->count();
  int num = (*bottom)[0]->num();
  int dim = count / num;

  Dtype loss_weight = top[0]->cpu_diff()[0];
  Dtype scale = loss_weight / num;
  Dtype* euc_diff_data = euc_diff_.mutable_cpu_data();

  Dtype* bottom0_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* bottom1_diff = (*bottom)[1]->mutable_cpu_diff();
  Dtype* bottom2_diff = (*bottom)[2]->mutable_cpu_diff();

  for (int i = 0; i < num; ++i) {
    if (euc_diff_data[i] > 0) {
      euc_diff_data[i] = scale;
    } else {
      euc_diff_data[i] = 0;
    }
  }

  if (propagate_down[0]) {
    caffe_sub(
        count,
        (*bottom)[2]->cpu_data(),
        (*bottom)[1]->cpu_data(),
        bottom0_diff);
    for(int k = 0; k < num; ++k) {
      caffe_scal(dim, euc_diff_data[k], bottom0_diff + dim * k);
    }    
  }

  if (propagate_down[1]) {
    caffe_copy(count, diff_.cpu_data(), bottom1_diff);
    for(int k = 0; k < num; ++k) {
      caffe_scal(dim, - euc_diff_data[k], bottom1_diff + dim * k);
    }    
  }

  if (propagate_down[2]) {
    caffe_copy(count, diff2_.cpu_data(), bottom2_diff);
    for(int k = 0; k < num; ++k) {
      caffe_scal(dim, euc_diff_data[k], bottom2_diff + dim * k);
    }    
  }

#if 0 //DEBUG_WUHAO
  // if (Caffe::phase() == Caffe::TRAIN)
  DebugTool<Dtype> dbg;
  dbg.open("tri_loss.bin");
  dbg.write_blob("bottom", *(*bottom)[0], 0);
  dbg.write_blob("bottom2", *(*bottom)[1], 0); 
  dbg.write_blob("bottom3", *(*bottom)[2], 0);
  dbg.write_blob("top", *top[0], 0);
  dbg.write_blob("diff", diff_, 0);
  dbg.write_blob("diff2", diff2_, 0); 
  dbg.write_blob("euc_diff", euc_diff_, 0);
  dbg.write_blob("bottom", *(*bottom)[0], 1);
  dbg.write_blob("bottom2", *(*bottom)[1], 1); 
  dbg.write_blob("bottom3", *(*bottom)[2], 1);
  dbg.close();
  string str;
  std::cout << "pause...";
  std::cin >> str;
#endif


}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);

}  // namespace caffe
