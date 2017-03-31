#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/orthogonal_constraint_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OrthogonalConstraintLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype *top_data = top[0]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        M_, N_, K_, (Dtype)1.,
        bottom_data, weight, (Dtype)0., top_data);

    if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
            bias_multiplier_.gpu_data(),
            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
    }

    if(lambda_ > 0) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
            N_, N_, K_,
            (Dtype)1., weight, weight,
            (Dtype)0., WW_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
            N_, K_, N_, 
            (Dtype)1., WW_.gpu_data(), weight, 
            (Dtype)0., W_gradient_.mutable_gpu_data());
        caffe_gpu_axpby<Dtype>(N_*K_, (Dtype)(-1), weight, 
            (Dtype)1., W_gradient_.mutable_gpu_data());
    }
}

template <typename Dtype>
void OrthogonalConstraintLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (this->param_propagate_down_[0]) { 
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_, K_, M_,
            (Dtype)1., top_diff, bottom_data,
            (Dtype)0., this->blobs_[0]->mutable_gpu_diff());

        if(lambda_ > 0) {
            caffe_gpu_axpby(N_*K_, lambda_, W_gradient_.gpu_data(), 
                (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
        }
    }

    if (bias_term_ && this->param_propagate_down_[1]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        // Gradient with respect to bias
        caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
            bias_multiplier_.gpu_data(), (Dtype)1.,
            this->blobs_[1]->mutable_gpu_diff());
    }

    if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        // Gradient with respect to bottom data
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_, K_, N_,
            (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
            (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
    
}


INSTANTIATE_LAYER_GPU_FUNCS(OrthogonalConstraintLayer);

} // namespace caffe