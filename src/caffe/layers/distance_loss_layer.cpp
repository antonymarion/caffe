#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define DEBUG_WUHAO 1
#ifdef DEBUG_WUHAO
#include "caffe/debugtool.hpp"
#endif

namespace caffe {

template <typename Dtype>
void DistanceLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    M1_ = this->layer_param_.distance_loss_param().m1();
    M2_ = this->layer_param_.distance_loss_param().m2();
    LOG(INFO) << "M1_: " << M1_ << "     M2_: " << M2_;
    switch (this->layer_param_.distance_loss_param().loss_type()) {
    case DistanceLossParameter_LossType_L2:
      LOG(INFO) << "distance loss type: L2";
      break;  
    case DistanceLossParameter_LossType_EXP:
      LOG(INFO) << "distance loss type: EXP";
      break;
    case DistanceLossParameter_LossType_LOG:
      LOG(INFO) << "distance loss type: LOG";
      break;
    default:
      LOG(FATAL) << "Unknown LossType";
    }
}


template <typename Dtype>
void DistanceLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);

  CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[1]->height(), bottom[2]->height());
  CHECK_EQ(bottom[1]->width(), bottom[2]->width());

  (*top)[0]->Reshape(1, 1, 1, 1);

}

template <typename Dtype>
void DistanceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const Dtype* distance_data = bottom[0]->cpu_data();
  const Dtype* label1_data = bottom[1]->cpu_data();
  const Dtype* label2_data = bottom[2]->cpu_data();
  int num = bottom[0]->num();
  Dtype loss = 0;
  int i = 0;

  switch (this->layer_param_.distance_loss_param().loss_type()) {
  case DistanceLossParameter_LossType_L2:
    for (i = 0; i < num; ++i) {
      int label1 = static_cast<int>(label1_data[i]);
      int label2 = static_cast<int>(label2_data[i]);
      Dtype dist = distance_data[i];
      if (label1 == label2) {
        if (dist > M1_) {
          loss += 0.5 * (dist - M1_) * (dist - M1_);
        }
      } else {
        if (dist < M2_) {
          loss += 0.5 * (M2_ - dist) * (M2_ - dist);
        }
      }
    }
    break;  
  case DistanceLossParameter_LossType_EXP:
    LOG(FATAL) << "Not implemented.";
    break;
  case DistanceLossParameter_LossType_LOG:
    LOG(FATAL) << "Not implemented.";
    break;
  default:
    LOG(FATAL) << "Unknown LossType";
  }

  (*top)[0]->mutable_cpu_data()[0] = loss / num;

}

template <typename Dtype>
void DistanceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    int num = (*bottom)[0]->num();
    Dtype* dist_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* distance_data = (*bottom)[0]->cpu_data();
    const Dtype* label1_data = (*bottom)[1]->cpu_data();
    const Dtype* label2_data = (*bottom)[2]->cpu_data();
    int i = 0;

    switch (this->layer_param_.distance_loss_param().loss_type()) {
    case DistanceLossParameter_LossType_L2:
      for (i = 0; i < num; ++i) {
        int label1 = static_cast<int>(label1_data[i]);
        int label2 = static_cast<int>(label2_data[i]);
        Dtype dist = distance_data[i];
        if (label1 == label2) {
          dist_diff[i] = (dist > M1_) ? ((dist - M1_) / num) : 0;
        } else {
          dist_diff[i] = (dist < M2_) ? ((M2_ - dist) / num) : 0;
        }
      }
      break;  
    case DistanceLossParameter_LossType_EXP:
      LOG(FATAL) << "Not implemented.";
      break;
    case DistanceLossParameter_LossType_LOG:
      LOG(FATAL) << "Not implemented.";
      break;
    default:
      LOG(FATAL) << "Unknown LossType";
    }

#if 0 //DEBUG_WUHAO
  DebugTool<Dtype> dbg;
  dbg.open("distance.bin");
  dbg.write_blob("dist", *(*bottom)[0], 0);
  dbg.write_blob("label1", *(*bottom)[1], 0); 
  dbg.write_blob("label2", *(*bottom)[2], 0);
  dbg.write_blob("top", *top[0], 0);
  dbg.write_blob("dist", *(*bottom)[0], 1);  
  dbg.close();
  string str;
  std::cout << "pause...";
  std::cin >> str;
#endif


}

#ifdef CPU_ONLY
STUB_GPU(DistanceLossLayer);
#endif

INSTANTIATE_CLASS(DistanceLossLayer);

}  // namespace caffe
