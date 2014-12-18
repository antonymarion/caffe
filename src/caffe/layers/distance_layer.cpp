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
void DistanceLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  switch (this->layer_param_.distance_param().type()) {
  case DistanceParameter_DistanceType_COSINE:
    LOG(INFO) << "distance type: cosine";
    break;
  case DistanceParameter_DistanceType_L2:
    LOG(INFO) << "distance type: L2";
    LOG(FATAL) << "Not implemented.";
    break;
  default:
    LOG(FATAL) << "Unknown distance type.";
  }
  
}


template <typename Dtype>
void DistanceLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  int num = bottom[0]->num();

  (*top)[0]->Reshape(num, 1, 1, 1);

  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  switch (this->layer_param_.distance_param().type()) {
  case DistanceParameter_DistanceType_COSINE:
    norm1_.resize(num);
    norm2_.resize(num);
    dotp_.resize(num);
    break;
  case DistanceParameter_DistanceType_L2:
    diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
    LOG(FATAL) << "Not implemented.";
    break;
  default:
    LOG(FATAL) << "Unknown distance type.";
  }
}

template <typename Dtype>
void DistanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const Dtype* feat1_data = bottom[0]->cpu_data();
  const Dtype* feat2_data = bottom[1]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  int i = 0;

  switch (this->layer_param_.distance_param().type()) {
  case DistanceParameter_DistanceType_COSINE:
    for (i = 0; i < num; ++i) {
      int offset = i * dim; 
      norm1_[i] = caffe_cpu_dot(dim, feat1_data + offset, feat1_data + offset);
      norm2_[i] = caffe_cpu_dot(dim, feat2_data + offset, feat2_data + offset);
      dotp_[i] = caffe_cpu_dot(dim, feat1_data + offset, feat2_data + offset);
      top_data[i] = 1.0 - dotp_[i] / sqrt(norm1_[i] * norm2_[i]);
    }
    break;
  case DistanceParameter_DistanceType_L2:
    LOG(FATAL) << "Not implemented.";
    break;
  default:
    LOG(FATAL) << "Unknown distance type.";
  }


}

template <typename Dtype>
void DistanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  int count = (*bottom)[0]->count();
  int num = (*bottom)[0]->num();
  int dim = count / num;
  int i = 0;

  const Dtype* feat1_data = (*bottom)[0]->cpu_data();
  const Dtype* feat2_data = (*bottom)[1]->cpu_data();

  Dtype* feat1_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* feat2_diff =(*bottom)[1]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  
  if (propagate_down[0]) {
    switch (this->layer_param_.distance_param().type()) {
    case DistanceParameter_DistanceType_COSINE:
      caffe_copy(count, feat2_data, feat1_diff);
      caffe_copy(count, feat1_data, feat2_diff);

      for (i = 0; i < num; ++i) {
        int offset = i * dim;
        caffe_axpy(dim, - dotp_[i] / norm1_[i], feat1_data + offset, feat1_diff + offset);
        caffe_axpy(dim, - dotp_[i] / norm2_[i], feat2_data + offset, feat2_diff + offset);

        // alpha has a minus here !!!
        Dtype alpha =  - top_diff[i] / sqrt(norm1_[i] * norm2_[i]);

        caffe_scal(dim, alpha, feat1_diff + offset);
        caffe_scal(dim, alpha, feat2_diff + offset);
      }
      break;
    case DistanceParameter_DistanceType_L2:
      LOG(FATAL) << "Not implemented.";
      break;
    default:
      LOG(FATAL) << "Unknown distance type.";
    }
  }


#if 0 //DEBUG_WUHAO
  DebugTool<Dtype> dbg;
  dbg.open("distance.bin");
  dbg.write_blob("feat1", *(*bottom)[0], 0);
  dbg.write_blob("feat2", *(*bottom)[1], 0);
  dbg.write_blob("feat1", *(*bottom)[0], 1);  
  dbg.write_blob("feat2", *(*bottom)[1], 1);
  dbg.write_blob("top", *top[0], 0);  
  dbg.write_blob("top", *top[0], 1);  
  dbg.close();
  string str;
  std::cout << "pause...";
  std::cin >> str;
#endif


}

#ifdef CPU_ONLY
STUB_GPU(DistanceLayer);
#endif

INSTANTIATE_CLASS(DistanceLayer);

}  // namespace caffe
