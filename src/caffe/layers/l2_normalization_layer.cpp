#include <vector>

#include "caffe/layers/l2_normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    int dim = bottom[0]->count() / bottom[0]->num();
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), 
            bottom[0]->height(), bottom[0]->width());
    norm_.Reshape(bottom[0]->num(), 1, 1, 1);
    grad_.Reshape(bottom[0]->num(), 1, dim, dim);
}

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data          = top[0]->mutable_cpu_data();
    Dtype* norm_data         = norm_.mutable_cpu_data();

    const Dtype eps = 1e-7;
    int num = bottom[0]->num();
    int dim = bottom[0]->count()/num;

    for(int i = 0; i < num; ++i) {
        norm_data[i] = eps + sqrt(caffe_cpu_dot(dim, bottom_data + i*dim, bottom_data + i*dim));
        for (int k = 0; k < dim; ++k) {
            top_data[i*dim + k] = bottom_data[i*dim + k] / norm_data[i];
        }
    }
}

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff    = top[0]->cpu_diff();
    const Dtype* top_data    = top[0]->cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* norm_data   = norm_.cpu_data();
    Dtype* bottom_diff       = bottom[0]->mutable_cpu_diff();
    Dtype* grad_data         = grad_.mutable_cpu_data();

    int num = bottom[0]->num();
    int dim = bottom[0]->count() / num;

    for (int i = 0; i < num; ++i) {
        Dtype dot = caffe_cpu_dot(dim, top_data + i*dim, top_diff + i*dim);
        caffe_cpu_scale(dim, dot, top_data + i*dim, bottom_diff + i*dim);
        caffe_sub(dim, top_diff + i*dim, bottom_diff + i*dim, bottom_diff + i*dim);
        caffe_cpu_scale(dim, norm_data[i], bottom_diff + i*dim, bottom_diff + i*dim);
    }
/*    for (int i = 0; i < num; ++i) {
        Dtype s = norm_data[i];
        const Dtype* x = bottom_data + i*dim;
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < dim; ++k) {
                if(j != k) {
                    grad_data[j*dim + k] = -x[j] * x[k] / (s*s*s);
                } else {
                    grad_data[j*dim + k] = (s*s - x[j] * x[k]) / (s*s*s);
                }
            }
        } //for (int i = 0; i < num; ++i)

        caffe_cpu_gemv<Dtype>(CblasNoTrans, dim, dim, Dtype(1), grad_data, top_diff + i*dim, Dtype(0), bottom_diff + i*dim);
    } //for (int i = 0; i < num; ++i)*/

}

#ifdef CPU_ONLY
STUB_GPU(L2NormalizationLayerL2NormalizationLayer);
#endif

INSTANTIATE_CLASS(L2NormalizationLayer);
REGISTER_LAYER_CLASS(L2Normalization);
} // namespace caffe
