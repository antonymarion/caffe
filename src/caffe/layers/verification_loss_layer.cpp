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
void VerificationLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    M1_ = this->layer_param_.verification_param().m1();
    M2_ = this->layer_param_.verification_param().m2();
    LOG(INFO) << "VerificationLossLayer:    M1_: " << M1_ << "     M2_: " << M2_;
}


template <typename Dtype>
void VerificationLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  CHECK_EQ(bottom[1]->channels(), bottom[3]->channels());
  CHECK_EQ(bottom[1]->height(), bottom[3]->height());
  CHECK_EQ(bottom[1]->width(), bottom[3]->width());

  (*top)[0]->Reshape(1, 1, 1, 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void VerificationLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const Dtype* feat1_data = bottom[0]->cpu_data();
  const Dtype* feat2_data = bottom[2]->cpu_data();
  const Dtype* label1_data = bottom[1]->cpu_data();
  const Dtype* label2_data = bottom[3]->cpu_data();
  Dtype* diff_data = diff_.mutable_cpu_data();

  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  Dtype loss = 0;

  caffe_sub(count, feat1_data, feat2_data, diff_data);

  for (int i = 0; i < num; ++i) {
    int offset = i * dim;
    int label1 = static_cast<int>(label1_data[i]);
    int label2 = static_cast<int>(label2_data[i]);
    Dtype norm2 = caffe_cpu_dot(dim, diff_data + offset, diff_data + offset);
    Dtype t = 0;
    Dtype norm = sqrt(norm2);
    
    if (label1 == label2) {
      if (norm > M1_) {
        loss += 0.5 * (norm - M1_) * (norm - M1_);
        t = Dtype(1) - M1_ / norm;
        caffe_scal(dim, t, diff_data + offset);
      }
      else {
        caffe_memset(dim * sizeof(Dtype), Dtype(0), diff_data + offset);
      }
    } else {
      if (norm < M2_) {
        loss += 0.5 * (M2_ - norm) * (M2_ - norm);
        t = Dtype(1) - M2_ / norm;
        caffe_scal(dim, t, diff_data + offset);
      } else {
        caffe_memset(dim * sizeof(Dtype), Dtype(0), diff_data + offset);       
      }
    }
  }

  (*top)[0]->mutable_cpu_data()[0] = loss / num;

#if 0 //DEBUG_WUHAO
  LOG(INFO) << "Verif Loss: "  << loss / num;
#endif

}

template <typename Dtype>
void VerificationLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    int count = (*bottom)[0]->count();
    const Dtype alpha = top[0]->cpu_diff()[0] / (*bottom)[0]->num();
    const Dtype* diff_data = diff_.cpu_data();
    if (propagate_down[0]) {
      caffe_cpu_axpby(
          count,              // count
          alpha,                              // alpha
          diff_data,                 // a
          Dtype(0),                           // beta
          (*bottom)[0]->mutable_cpu_diff());  // b
    }
    if (propagate_down[2]) {
      caffe_cpu_axpby(
          count,              // count
          -alpha,                              // -alpha
          diff_data,                 // a
          Dtype(0),                           // beta
          (*bottom)[2]->mutable_cpu_diff());       
    }


#if 0 //DEBUG_WUHAO
  DebugTool<Dtype> dbg;
  dbg.open("verif_loss.bin");
  dbg.write_blob("feat1", *(*bottom)[0], 0);
  dbg.write_blob("label1", *(*bottom)[1], 0); 
  dbg.write_blob("feat2", *(*bottom)[2], 0);
  dbg.write_blob("label2", *(*bottom)[3], 0);
  dbg.write_blob("featd1", *(*bottom)[0], 1);  
  dbg.write_blob("featd2", *(*bottom)[2], 1);
  dbg.write_blob("diff", diff_, 0);
  dbg.close();
  string str;
  std::cout << "pause...";
  std::cin >> str;
#endif


}

#ifdef CPU_ONLY
STUB_GPU(VerificationLossLayer);
#endif

INSTANTIATE_CLASS(VerificationLossLayer);

}  // namespace caffe
