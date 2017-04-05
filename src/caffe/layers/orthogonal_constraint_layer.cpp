#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/orthogonal_constraint_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OrthogonalConstraintLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int num_output = this->layer_param_.orthogonal_constraint_param().num_output();
    bias_term_           = this->layer_param_.orthogonal_constraint_param().bias_term();
    const int axis       = bottom[0]->CanonicalAxisIndex(1);

    N_ = num_output;
    K_ = bottom[0]->count(axis);

    if(this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";
    } else {
        if(bias_term_) {
            this->blobs_.resize(2);
        } else {
            this->blobs_.resize(1);
        }
    }

    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;

    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.orthogonal_constraint_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    W_gradient_.Reshape(weight_shape);

    if(bias_term_) {
        vector<int> bias_shape(1, N_);
        this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.orthogonal_constraint_param().bias_filler()));
        bias_filler->Fill(this->blobs_[1].get());
    }

    this->param_propagate_down_.resize(this->blobs_.size(), true);
    lambda_ = this->layer_param_.orthogonal_constraint_param().lambda();
}

template <typename Dtype>
void OrthogonalConstraintLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int axis = bottom[0]->CanonicalAxisIndex(1);

    M_ = bottom[0]->count(0, axis);

    vector<int> top_shape = bottom[0]->shape();
    top_shape.resize(axis + 1);
    top_shape[axis] = N_;

    top[0]->Reshape(top_shape);

    if(bias_term_) {
        vector<int> bias_shape(1, M_);
        bias_multiplier_.Reshape(bias_shape);
        caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }

    // orthogonal constraint
    vector<int> matrix_shape(2);
    matrix_shape[0] = N_;
    matrix_shape[1] = N_;

    WW_.Reshape(matrix_shape);
}

template <typename Dtype>
void OrthogonalConstraintLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        M_, N_, K_, (Dtype)1.,
        bottom_data, weight, (Dtype)0., top_data);

    if (bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
            bias_multiplier_.cpu_data(),
            this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
    }
}

template <typename Dtype>
void OrthogonalConstraintLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {


    if (this->param_propagate_down_[0]) { 
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* weight = this->blobs_[0]->cpu_data();
        
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_, K_, M_,
            (Dtype)1., top_diff, bottom_data,
            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());

        if(lambda_ > 0) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                N_, N_, K_,
                (Dtype)1., weight, weight,
                (Dtype)0., WW_.mutable_cpu_data());
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
                N_, K_, N_, 
                (Dtype)1., WW_.cpu_data(), weight, 
                (Dtype)0., W_gradient_.mutable_cpu_data());
            caffe_cpu_axpby<Dtype>(N_*K_, (Dtype)(-1), weight, 
                (Dtype)1., W_gradient_.mutable_cpu_data());
            caffe_cpu_axpby(N_*K_, lambda_, W_gradient_.cpu_data(), 
                (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
        }
    }

    if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        M_, K_, N_,
        (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
        (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
    
}

#ifdef CPU_ONLY
STUB_GPU(OrthogonalConstraintLayer);
#endif

INSTANTIATE_CLASS(OrthogonalConstraintLayer);
REGISTER_LAYER_CLASS(OrthogonalConstraint);

} // namespace caffe