#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"


#define DEBUG_WUHAO 1
#include "caffe/debugtool.hpp"

namespace caffe {

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  
  int dim = bottom[0]->count() / bottom[0]->num();
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  norm_.Reshape(bottom[0]->num(), 1, 1, 1);
  grad_.Reshape(bottom[0]->num(), 1, dim, dim);

  /*
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  */
}

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();

  const Dtype EPS = 1e-7; 
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int i = 0, k = 0;

  for (i = 0; i < num; ++i) {
    norm_data[i] = EPS + sqrt(caffe_cpu_dot(dim, bottom_data + i * dim, bottom_data + i * dim));
    for (k = 0; k < dim; ++k) {
      top_data[i * dim + k] = bottom_data[i * dim + k] / norm_data[i];
    }
  }

#if 0 //DEBUG_WUHAO
  DebugTool<Dtype> dbg;
  dbg.open("l2_normalize_fprop.bin");
  dbg.write_blob("bot", *bottom[0], 0);
  dbg.write_blob("m_norm", norm_, 0);  
  dbg.write_blob("top", *(*top)[0], 0);  
  dbg.close();
  string str;
  std::cout << "pause...";
  std::cin >> str;
#endif


}

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  //const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* norm_data = norm_.cpu_data();
  Dtype* grad_data = grad_.mutable_cpu_data();

  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / num;
  int i = 0, j = 0, k = 0;

  for (i = 0; i < num; ++i) {
    Dtype s = norm_data[i];
    const Dtype* x = bottom_data + i * dim;
    for (j = 0; j < dim; ++j) {
      for (k = 0; k < dim; ++k) {
        if (j != k) {
          grad_data[j * dim + k] = - x[j] * x[k] / (s*s*s);
        } else {
          grad_data[j * dim + k] = (s * s - x[j] * x[k]) / (s*s*s);
        }
      }
    }

    caffe_cpu_gemv<Dtype>(CblasNoTrans, dim, dim, Dtype(1), grad_data, top_diff + i * dim, Dtype(0), bottom_diff + i * dim);
  }

#if 0 //DEBUG_WUHAO
  DebugTool<Dtype> dbg;
  dbg.open("l2_normalize_bprop.bin");
  dbg.write_blob("bot", *(*bottom)[0], 0);
  dbg.write_blob("m_norm", norm_, 0);  
  dbg.write_blob("top", *top[0], 0);

  dbg.write_blob("bot", *(*bottom)[0], 1);  
  dbg.write_blob("top", *top[0], 1); 
  dbg.write_blob("m_grad", grad_, 0);   
  dbg.close();

  string str;
  std::cout << "pause...";
  std::cin >> str;
#endif

}


#ifdef CPU_ONLY
STUB_GPU(L2NormalizeLayer);
#endif

INSTANTIATE_CLASS(L2NormalizeLayer);


}  // namespace caffe
