#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/relevance_constraint_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{


template <typename Dtype>
void RelevanceConstraintLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int num_output = this->layer_param_.relevance_constraint_param().num_output();
    bias_term_           = this->layer_param_.relevance_constraint_param().bias_term();
    const int axis       = bottom[0]->CanonicalAxisIndex(1);

    N_ = num_output;
    K_ = bottom[0]->count(axis);

    if(this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";
    } else {
        if(bias_term_) {
            this->blobs_.resize(4);
        } else {
            this->blobs_.resize(2);
        }
    }

    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;

    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.relevance_constraint_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    weight_filler->Fill(this->blobs_[1].get());

    if(bias_term_) {
        vector<int> bias_shape(1, N_);
        this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
        this->blobs_[3].reset(new Blob<Dtype>(bias_shape));
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.relevance_constraint_param().bias_filler()));
        bias_filler->Fill(this->blobs_[2].get());
        bias_filler->Fill(this->blobs_[3].get());
    }

    this->param_propagate_down_.resize(this->blobs_.size(), true);

    lambda_ = this->layer_param_.relevance_constraint_param().lambda();

    // inverse of matrix gamma
    vector<int> inverse_gamma_shape(2);
    inverse_gamma_shape[0] = N_;
    inverse_gamma_shape[1] = N_;
    inverse_gamma_.Reshape(inverse_gamma_shape);

    WW_1.Reshape(inverse_gamma_shape);
    WW_2.Reshape(inverse_gamma_shape);
    WW_sum.Reshape(inverse_gamma_shape);

    vector<int> S_shape(2);
    S_shape[0] = 1;
    S_shape[1] = N_;

    U.Reshape(inverse_gamma_shape);
    VT.Reshape(inverse_gamma_shape);
    S.Reshape(S_shape);
    diag_S.Reshape(inverse_gamma_shape);
}

template <typename Dtype>
void RelevanceConstraintLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int axis = bottom[0]->CanonicalAxisIndex(1);

    M_ = bottom[0]->count(0, axis);

    vector<int> top_shape = bottom[0]->shape();
    top_shape.resize(axis + 1);
    top_shape[axis] = N_;

    top[0]->Reshape(top_shape);
    top[1]->Reshape(top_shape);

    if(bias_term_) {
        vector<int> bias_shape(1, M_);
        bias_multiplier_1_.Reshape(bias_shape);
        caffe_set(M_, Dtype(1), bias_multiplier_1_.mutable_cpu_data());
        bias_multiplier_2_.Reshape(bias_shape);
        caffe_set(M_, Dtype(1), bias_multiplier_2_.mutable_cpu_data());
    }
}

template <typename Dtype>
void RelevanceConstraintLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data_1 = bottom[0]->cpu_data();
    const Dtype* bottom_data_2 = bottom[1]->cpu_data();

    Dtype* top_data_1 = top[0]->mutable_cpu_data();
    Dtype* top_data_2 = top[1]->mutable_cpu_data();

    const Dtype* W_1 = this->blobs_[0]->cpu_data();
    const Dtype* W_2 = this->blobs_[1]->cpu_data();

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data_1, W_1, (Dtype)0., top_data_1); 
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data_2, W_2, (Dtype)0., top_data_2); 

    if(bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
            bias_multiplier_1_.cpu_data(),
            this->blobs_[2]->cpu_data(), (Dtype)1., top_data_1); 
       caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
            bias_multiplier_2_.cpu_data(),
            this->blobs_[3]->cpu_data(), (Dtype)1., top_data_2); 
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
void RelevanceConstraintLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    const Dtype* W_1 = this->blobs_[0]->cpu_data();
    const Dtype* W_2 = this->blobs_[1]->cpu_data();
    const Dtype* inverse_gamma_data = inverse_gamma_.cpu_data();

    const Dtype* bottom_data_1 = bottom[0]->cpu_data();
    const Dtype* bottom_data_2 = bottom[1]->cpu_data();

    // update W_1
    if(this->param_propagate_down_[0]) {
        const Dtype* top_diff_1 = top[0]->cpu_diff();
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_, K_, M_,
            (Dtype)1, top_diff_1, bottom_data_1,
            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());

        if(lambda_ > 0) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                N_, K_, N_,
                lambda_, inverse_gamma_data, W_1,
                (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
        }
    }

    // update W_2
    if(this->param_propagate_down_[1]) {
        const Dtype* top_diff_2 = top[1]->cpu_diff();
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_, K_, M_,
            (Dtype)1, top_diff_2, bottom_data_2,
            (Dtype)1., this->blobs_[1]->mutable_cpu_diff());

        if(lambda_ > 0) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                N_, K_, N_,
                lambda_, inverse_gamma_data, W_2,
                (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
        }
    }


    // update b
    if (bias_term_ && this->param_propagate_down_[2]) {
        const Dtype* top_diff_1 = top[0]->cpu_diff();
        caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1, top_diff_1,
            bias_multiplier_1_.cpu_data(), (Dtype)1.,
            this->blobs_[2]->mutable_cpu_diff());
    }

    if (bias_term_ && this->param_propagate_down_[3]) {
        const Dtype* top_diff_2 = top[1]->cpu_diff();
        caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1, top_diff_2,
            bias_multiplier_2_.cpu_data(), (Dtype)1.,
            this->blobs_[3]->mutable_cpu_diff());
    }

    // propagate data
    if(propagate_down[0]) {
        const Dtype* top_diff_1 = top[0]->cpu_diff();

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_, K_, N_,
            (Dtype)1., top_diff_1, this->blobs_[0]->cpu_data(),
            (Dtype)0., bottom[0]->mutable_cpu_diff());
    }

    if(propagate_down[0]) {
        const Dtype* top_diff_2 = top[1]->cpu_diff();

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_, K_, N_,
            (Dtype)1., top_diff_2, this->blobs_[1]->cpu_data(),
            (Dtype)0., bottom[1]->mutable_cpu_diff());
    }
}

#ifdef CPU_ONLY
STUB_GPU(RelevanceConstraintLayer);
#endif

INSTANTIATE_CLASS(RelevanceConstraintLayer);
REGISTER_LAYER_CLASS(RelevanceConstraint);

}