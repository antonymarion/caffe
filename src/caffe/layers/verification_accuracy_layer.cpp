#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <cmath>


#define DEBUG_WUHAO 1
#ifdef DEBUG_WUHAO
#include "caffe/debugtool.hpp"
#endif

namespace caffe {



template <typename Dtype>
void VerificationAccuracyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

    M1_ = this->layer_param_.verification_param().m1();
    M2_ = this->layer_param_.verification_param().m2();
    LOG(INFO) << "VerificationAccuracyLayer:    M1_: " << M1_ << "     M2_: " << M2_;
}


template <typename Dtype>
void VerificationAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  CHECK_EQ(bottom[1]->channels(), bottom[3]->channels());
  CHECK_EQ(bottom[1]->height(), bottom[3]->height());
  CHECK_EQ(bottom[1]->width(), bottom[3]->width());

  (*top)[0]->Reshape(1, 2, 1, 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void VerificationAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
  Dtype accuracy = 0;
  Dtype accuracy_one_thr = 0;

  caffe_sub(count, feat1_data, feat2_data, diff_data);

  for (int i = 0; i < num; ++i) {
    int offset = i * dim;
    int label1 = static_cast<int>(label1_data[i]);
    int label2 = static_cast<int>(label2_data[i]);
    Dtype norm2 = caffe_cpu_dot(dim, diff_data + offset, diff_data + offset);
    Dtype t = 0;
    Dtype norm = sqrt(norm2);
    

    if (isnan(norm)) {
      LOG(INFO) << "norm is nan. norm2: " << norm2;
      std::cout << "diff for this sample is:" << std::endl;
      for (int t = 0; t <  dim; ++t) {
        std::cout << diff_data[offset + t] << " ";
      }
      std::cout << std::endl;
      LOG(FATAL) << "NaN error.";
    }

#if 0 //DEBUG_WUHAO
    static int DBG_cnt = 0;
    if (++DBG_cnt > 10000){
      std::cout << "d:" << norm << std::endl;
    }
#endif


    if (label1 == label2) {
      if (norm > M1_) {
        loss += 0.5 * (norm - M1_) * (norm - M1_);
      } else {
        accuracy += 1;
      }

    } else {
      if (norm < M2_) {
        loss += 0.5 * (M2_ - norm) * (M2_ - norm);
      } else {
        accuracy += 1;
      }
    }

    if (((label1 == label2) && (norm * 2 < M1_ + M2_)) ||
        ((label1 != label2) && (norm * 2 > M1_ + M2_))) {
      accuracy_one_thr += 1;
    }
  }

  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = accuracy_one_thr / num;

#if 0 //DEBUG_WUHAO
  LOG(INFO) << "Verif Loss: "  << loss / num;
  LOG(INFO) << "Verif Accuracy: "  << accuracy / num;
  LOG(INFO) << "Verif Accuracy(one thr): "  << accuracy_one_thr / num;
#endif

#if 0 //DEBUG_WUHAO
  DebugTool<Dtype> dbg;
  dbg.open("verif_loss.bin");
  dbg.write_blob("feat1", *bottom[0], 0);
  dbg.write_blob("label1", *bottom[1], 0); 
  dbg.write_blob("feat2", *bottom[2], 0);
  dbg.write_blob("label2", *bottom[3], 0);
  dbg.write_blob("diff", diff_, 0);
  dbg.close();
  string str;
  std::cout << "pause...";
  std::cin >> str;
#endif

}

#ifdef CPU_ONLY
STUB_GPU(VerificationAccuracyLayer);
#endif

INSTANTIATE_CLASS(VerificationAccuracyLayer);

}  // namespace caffe
