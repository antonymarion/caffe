#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/multi_view_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
void MultiViewLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int num_output_shared = this->layer_param_.multi_view_param().num_output_shared();
    const int num_output_unique = this->layer_param_.multi_view_param().num_output_unique();
    bias_term_shared_           = this->layer_param_.multi_view_param().bias_term_shared();

    N_shared_   = num_output_shared;
    N_unique_1_ = num_output_unique;
    N_unique_2_ = num_output_unique;

    const int axis = bottom[0]->CanonicalAxisIndex(1);

    K_shared_   = bottom[0]->count(axis);
    K_unique_1_ = bottom[0]->count(axis);
    K_unique_2_ = bottom[0]->count(axis);

    if(this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";
    } else {
        if(bias_term_shared_) {
            this->blobs_.resize(4);
        } else {
            this->blobs_.resize(3);
        }
    }

    vector<int> weight_shape_shared(2);
    vector<int> weight_shape_unique(2);

    weight_shape_shared[0] = N_shared_;
    weight_shape_shared[1] = K_shared_;

    weight_shape_unique[0] = N_unique_1_;
    weight_shape_unique[1] = K_unique_1_;

    this->blobs_[0].reset(new Blob<Dtype>(weight_shape_shared));
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape_unique));
    this->blobs_[2].reset(new Blob<Dtype>(weight_shape_unique));
    shared_ptr<Filler<Dtype> > weight_filler_shared(GetFiller<Dtype>(
        this->layer_param_.multi_view_param().weight_filler_shared()));
    weight_filler_shared->Fill(this->blobs_[0].get());

    shared_ptr<Filler<Dtype> > weight_filler_unique_1(GetFiller<Dtype>(
        this->layer_param_.multi_view_param().weight_filler_unique_1()));
    weight_filler_unique_1->Fill(this->blobs_[1].get());

    shared_ptr<Filler<Dtype> > weight_filler_unique_2(GetFiller<Dtype>(
        this->layer_param_.multi_view_param().weight_filler_unique_2()));
    weight_filler_unique_2->Fill(this->blobs_[2].get());

    if(bias_term_shared_) {
        vector<int> bias_shape(1, N_shared_);
        this->blobs_[3].reset(new Blob<Dtype>(bias_shape));
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.multi_view_param().bias_filler_shared()));
        bias_filler->Fill(this->blobs_[3].get());
    }

    this->param_propagate_down_.resize(this->blobs_.size(), true);

    lambda_ = this->layer_param_.multi_view_param().lambda();
}

template <typename Dtype>
void MultiViewLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const int axis = bottom[0]->CanonicalAxisIndex(1);
    // const int new_K = bottom[0]->count(axis);

    M_shared_   = bottom[0]->count(0, axis);
    M_unique_1_ = M_shared_;
    M_unique_2_ = M_shared_;

    vector<int> top_shape_shared   = bottom[0]->shape();
    vector<int> top_shape_unique_1 = bottom[0]->shape();
    vector<int> top_shape_unique_2 = bottom[0]->shape();

    top_shape_shared.resize(axis + 1);
    top_shape_unique_1.resize(axis + 1);
    top_shape_unique_2.resize(axis + 1);

    top_shape_shared[axis]   = N_shared_;
    top_shape_unique_1[axis] = N_unique_1_;
    top_shape_unique_2[axis] = N_unique_2_;

    top[0]->Reshape(top_shape_shared);
    top[1]->Reshape(top_shape_unique_1);
    top[2]->Reshape(top_shape_shared);
    top[3]->Reshape(top_shape_unique_2);

    if(bias_term_shared_) {
        vector<int> bias_shape(1, M_shared_);
        bias_multiplier_shared_.Reshape(bias_shape);
        caffe_set(M_shared_, Dtype(1), bias_multiplier_shared_.mutable_cpu_data());
    }

    vector<int> temp_W_shape(2);
    vector<int> temp_V_shape(2);
    temp_W_shape[0] = N_shared_;
    temp_W_shape[1] = N_shared_;
    temp_V_shape[0] = N_unique_1_;
    temp_V_shape[1] = N_unique_1_;
    temp_W.Reshape(temp_W_shape);
    temp_V1.Reshape(temp_V_shape);
    temp_V2.Reshape(temp_V_shape);
}

template <typename Dtype>
void MultiViewLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data_1 = bottom[0]->cpu_data();
    const Dtype* bottom_data_2 = bottom[1]->cpu_data();


    Dtype* top_data_shared_1 = top[0]->mutable_cpu_data();
    Dtype* top_data_unique_1 = top[1]->mutable_cpu_data();
    Dtype* top_data_shared_2 = top[2]->mutable_cpu_data();
    Dtype* top_data_unique_2 = top[3]->mutable_cpu_data();

    const Dtype* W  = this->blobs_[0]->cpu_data();
    const Dtype* V1 = this->blobs_[1]->cpu_data();
    const Dtype* V2 = this->blobs_[2]->cpu_data();

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      M_shared_, N_shared_, K_shared_, (Dtype)1.,
      bottom_data_1, W, (Dtype)0., top_data_shared_1); 
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      M_shared_, N_shared_, K_shared_, (Dtype)1.,
      bottom_data_2, W, (Dtype)0., top_data_shared_2);   

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      M_unique_1_, N_unique_1_, K_unique_1_, (Dtype)1.,
      bottom_data_1, V1, (Dtype)0., top_data_unique_1); 
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      M_unique_2_, N_unique_2_, K_unique_2_, (Dtype)1.,
      bottom_data_2, V2, (Dtype)0., top_data_unique_2); 

    if(bias_term_shared_) {
       caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_shared_, N_shared_, 1, (Dtype)1.,
            bias_multiplier_shared_.cpu_data(),
            this->blobs_[3]->cpu_data(), (Dtype)1., top_data_shared_1); 
       caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_shared_, N_shared_, 1, (Dtype)1.,
            bias_multiplier_shared_.cpu_data(),
            this->blobs_[3]->cpu_data(), (Dtype)1., top_data_shared_2); 
    }
}

template <typename Dtype>
void MultiViewLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    const Dtype* W  = this->blobs_[0]->cpu_data();
    const Dtype* V1 = this->blobs_[1]->cpu_data();
    const Dtype* V2 = this->blobs_[2]->cpu_data();

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        N_shared_, N_shared_, K_shared_,
        (Dtype)1., W, W,
        (Dtype)1., temp_W.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        N_unique_1_, N_unique_1_, K_unique_1_,
        (Dtype)1., V1, V1,
        (Dtype)1., temp_V1.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        N_unique_2_, N_unique_2_, K_unique_2_,
        (Dtype)1., V2, V2,
        (Dtype)1., temp_V2.mutable_cpu_data());

    const Dtype* bottom_data_1 = bottom[0]->cpu_data();
    const Dtype* bottom_data_2 = bottom[1]->cpu_data();

    // update W
    if(this->param_propagate_down_[0]) {
        const Dtype* top_diff_shared_1 = top[0]->cpu_diff();
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_shared_, K_shared_, M_shared_,
            (Dtype)0.5, top_diff_shared_1, bottom_data_1,
            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());

        const Dtype* top_diff_shared_2 = top[2]->cpu_diff();
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_shared_, K_shared_, M_shared_,
            (Dtype)0.5, top_diff_shared_2, bottom_data_2,
            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            N_unique_1_, K_shared_, N_unique_1_,
            lambda_, temp_V1.cpu_data(), W,
            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            N_unique_2_, K_shared_, N_unique_2_,
            lambda_, temp_V2.cpu_data(), W,
            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }

    // update V1
    if(this->param_propagate_down_[1]) {
        const Dtype* top_diff_unique_1 = top[1]->cpu_diff();
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_unique_1_, K_unique_1_, M_unique_1_,
            (Dtype)1., top_diff_unique_1, bottom_data_1,
            (Dtype)1., this->blobs_[1]->mutable_cpu_diff());

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            N_shared_, K_unique_1_, N_shared_,
            lambda_, temp_W.cpu_data(), V1,
            (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
    }

    // update V2
    if(this->param_propagate_down_[2]) {
        const Dtype* top_diff_unique_2 = top[3]->cpu_diff();
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_unique_2_, K_unique_2_, M_unique_2_,
            (Dtype)1., top_diff_unique_2, bottom_data_2,
            (Dtype)1., this->blobs_[2]->mutable_cpu_diff());

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            N_shared_, K_unique_2_, N_shared_,
            lambda_, temp_W.cpu_data(), V2,
            (Dtype)1., this->blobs_[2]->mutable_cpu_diff());
    }

    //update b
    if (bias_term_shared_ && this->param_propagate_down_[3]) {
        const Dtype* top_diff_shared_1 = top[0]->cpu_diff();
        // Gradient with respect to bias
        caffe_cpu_gemv<Dtype>(CblasTrans, M_shared_, N_shared_, (Dtype)0.5, top_diff_shared_1,
            bias_multiplier_shared_.cpu_data(), (Dtype)1.,
            this->blobs_[3]->mutable_cpu_diff());

        const Dtype* top_diff_shared_2 = top[2]->cpu_diff();
        caffe_cpu_gemv<Dtype>(CblasTrans, M_shared_, N_shared_, (Dtype)0.5, top_diff_shared_2,
            bias_multiplier_shared_.cpu_data(), (Dtype)1.,
            this->blobs_[3]->mutable_cpu_diff());
    }

    // propagate data
    if(propagate_down[0]) {
        const Dtype* top_diff_shared_1 = top[0]->cpu_diff();
        const Dtype* top_diff_unique_1 = top[1]->cpu_diff();

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_shared_, K_shared_, N_shared_,
            (Dtype)1., top_diff_shared_1, this->blobs_[0]->cpu_data(),
            (Dtype)0., bottom[0]->mutable_cpu_diff());

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_unique_1_, K_unique_1_, N_unique_1_,
            (Dtype)1., top_diff_unique_1, this->blobs_[1]->cpu_data(),
            (Dtype)1., bottom[0]->mutable_cpu_diff());
    }

    if(propagate_down[1]) {
        const Dtype* top_diff_shared_2 = top[2]->cpu_diff();
        const Dtype* top_diff_unique_2 = top[3]->cpu_diff();

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_shared_, K_shared_, N_shared_,
            (Dtype)1., top_diff_shared_2, this->blobs_[0]->cpu_data(),
            (Dtype)0., bottom[1]->mutable_cpu_diff());

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_unique_2_, K_unique_2_, N_unique_2_,
            (Dtype)1., top_diff_unique_2, this->blobs_[2]->cpu_data(),
            (Dtype)1., bottom[1]->mutable_cpu_diff());
    }

}


#ifdef CPU_ONLY
STUB_GPU(MultiViewLayer);
#endif

INSTANTIATE_CLASS(MultiViewLayer);
REGISTER_LAYER_CLASS(MultiView);

} // namespace caffe