#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/relevance_constraint_layer.hpp"

namespace caffe {

template <typename Dtype>
void RelevanceConstraintLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {

    //Forward_cpu(bottom, top);

    const Dtype* bottom_data_1 = bottom[0]->gpu_data();
    const Dtype* bottom_data_2 = bottom[1]->gpu_data();

    Dtype* top_data_1 = top[0]->mutable_gpu_data();
    Dtype* top_data_2 = top[1]->mutable_gpu_data();

    const Dtype* W_1 = this->blobs_[0]->gpu_data();
    const Dtype* W_2 = this->blobs_[1]->gpu_data();

    if (M_ == 1) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                             W_1, bottom_data_1, (Dtype)0., top_data_1);
        caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                             W_2, bottom_data_2, (Dtype)0., top_data_2);
        if (bias_term_) {
          caffe_gpu_axpy<Dtype>(N_, bias_multiplier_1_.cpu_data()[0],
                                this->blobs_[2]->gpu_data(), top_data_1);
          caffe_gpu_axpy<Dtype>(N_, bias_multiplier_2_.cpu_data()[0],
                                this->blobs_[3]->gpu_data(), top_data_2);
        }
    } else {
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                              CblasTrans,
                              M_, N_, K_, (Dtype)1.,
                              bottom_data_1, W_1, (Dtype)0., top_data_1);
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                              CblasTrans,
                              M_, N_, K_, (Dtype)1.,
                              bottom_data_2, W_2, (Dtype)0., top_data_2);
        if (bias_term_){
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                                bias_multiplier_1_.gpu_data(),
                                this->blobs_[2]->gpu_data(), (Dtype)1., top_data_1);
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                                bias_multiplier_2_.gpu_data(),
                                this->blobs_[3]->gpu_data(), (Dtype)1., top_data_2);
        }
    }

    if(lambda_ > 0) {
        // compute sum oof WW^T
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
            N_, N_, K_,
            (Dtype)1., W_1, W_1,
            (Dtype)0., WW_1.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
            N_, N_, K_,
            (Dtype)1., W_2, W_2,
            (Dtype)0., WW_2.mutable_cpu_data());

        caffe_copy(N_*N_, WW_2.cpu_data(), WW_sum.mutable_cpu_data());
        caffe_axpy(N_*N_, (Dtype)1., WW_1.cpu_data(), WW_sum.mutable_cpu_data());

        // SVD
        int info = caffe_cpu_gesvd<Dtype>(N_, N_, WW_sum.mutable_cpu_data(),  S.mutable_cpu_data(), 
                    U.mutable_cpu_data(),  VT.mutable_cpu_data());

        //caffe_sqrt<Dtype>(N_, S.cpu_data(), S.mutable_cpu_data());

        // compute inverse of gamma
        Dtype* diag_S_data = diag_S.mutable_cpu_data();
        Dtype* inverse_gamma_data = inverse_gamma_.mutable_cpu_data();
        caffe_set(N_*N_, Dtype(0), diag_S_data);
        for(int i = 0; i < N_; ++i) {
            diag_S_data[i * N_ + i] = sqrt(S.cpu_data()[i]);
        }

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, N_, N_, (Dtype)1., U.cpu_data(), 
            diag_S.cpu_data(), (Dtype)0., inverse_gamma_data);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, N_, N_, (Dtype)1., inverse_gamma_.cpu_data(), 
            VT.cpu_data(), (Dtype)0., inverse_gamma_data);

        Dtype trace = caffe_cpu_asum<Dtype>(N_, diag_S.cpu_data());
        for(int i = 0; i < N_; ++i) {
            for(int j = 0; j < N_; ++j) {
                inverse_gamma_data[i * N_ + j] = inverse_gamma_.cpu_data()[i] / (trace + 1e-5);
            }
        }

        info = caffe_cpu_getri<Dtype>(N_, inverse_gamma_data);
    }
}

template <typename Dtype>
void RelevanceConstraintLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    //Backward_cpu(top, propagate_down, bottom);
    const Dtype* W_1 = this->blobs_[0]->gpu_data();
    const Dtype* W_2 = this->blobs_[1]->gpu_data();
    const Dtype* inverse_gamma_data = inverse_gamma_.cpu_data();

    const Dtype* bottom_data_1 = bottom[0]->gpu_data();
    const Dtype* bottom_data_2 = bottom[1]->gpu_data();

    //update W
    if (this->param_propagate_down_[0]) {
        const Dtype* top_diff_1 = top[0]->gpu_diff();
        // Gradient with respect to weight
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff_1, bottom_data_1,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());

        if(lambda_ > 0) {
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                N_, K_, N_,
                lambda_, inverse_gamma_data, W_1,
                (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
        }
    }

    if (this->param_propagate_down_[1]) {
        const Dtype* top_diff_2 = top[1]->gpu_diff();
        // Gradient with respect to weight
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff_2, bottom_data_2,
          (Dtype)1., this->blobs_[1]->mutable_gpu_diff());

        if(lambda_ > 0) {
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                N_, K_, N_,
                lambda_, inverse_gamma_data, W_2,
                (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
        }
    }

    // update b
    if (bias_term_ && this->param_propagate_down_[2]) {
        const Dtype* top_diff_1 = top[0]->gpu_diff();
        // Gradient with respect to bias
        caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff_1,
            bias_multiplier_1_.gpu_data(), (Dtype)1.,
            this->blobs_[2]->mutable_gpu_diff());
    }

    if (bias_term_ && this->param_propagate_down_[3]) {
        const Dtype* top_diff_2 = top[1]->gpu_diff();
        // Gradient with respect to bias
        caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff_2,
            bias_multiplier_2_.gpu_data(), (Dtype)1.,
            this->blobs_[3]->mutable_gpu_diff());
    }

    // propagation data
    if (propagate_down[0]) {
        const Dtype* top_diff_1 = top[0]->gpu_diff();
        // Gradient with respect to bottom data
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_, K_, N_,
            (Dtype)1., top_diff_1, this->blobs_[0]->gpu_data(),
            (Dtype)0., bottom[0]->mutable_gpu_diff());
    }

    if (propagate_down[1]) {
        const Dtype* top_diff_2 = top[1]->gpu_diff();
        // Gradient with respect to bottom data
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_, K_, N_,
            (Dtype)1., top_diff_2, this->blobs_[1]->gpu_data(),
            (Dtype)0., bottom[1]->mutable_gpu_diff());
    }

}
    
INSTANTIATE_LAYER_GPU_FUNCS(RelevanceConstraintLayer);

}