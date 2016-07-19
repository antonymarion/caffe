#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"


namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::LayerSetUp(bottom, top);

    margin = this->layer_param_.triplet_loss_param().margin();
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);

    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    CHECK_EQ(bottom[0]->num(), bottom[2]->num());

    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());

    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());

    diff_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
        bottom[0]->height(), bottom[0]->width());
    diff_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
        bottom[0]->height(), bottom[0]->width());
    diff_pn_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
        bottom[0]->height(), bottom[0]->width());

    euc_diff_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    int num   = bottom[0]->num();
    int dim   = count / num;

    caffe_sub(
        count,
        bottom[0]->cpu_data(),  // a
        bottom[1]->cpu_data(),  // p
        diff_ap_.mutable_cpu_data());  // a_i-p_i
    caffe_sub(
        count,
        bottom[0]->cpu_data(),  // a
        bottom[2]->cpu_data(),  // n
        diff_an_.mutable_cpu_data());  // a_i-n_i
    caffe_sub(
        count,
        bottom[1]->cpu_data(),  // p
        bottom[2]->cpu_data(),  // n
        diff_pn_.mutable_cpu_data());  // p_i-n_i

    const Dtype* diff_ap_data = diff_ap_.cpu_data();
    const Dtype* diff_an_data = diff_an_.cpu_data();
    Dtype* euc_diff_data      = euc_diff_.mutable_cpu_data();
    Dtype loss                = 0.0;
    Dtype dot1, dot2;

    for(int i = 0; i < num; ++i) {
        dot1 = caffe_cpu_dot(dim, diff_ap_data + (i*dim), diff_ap_data + (i*dim));
        dot2 = caffe_cpu_dot(dim, diff_an_data + (i*dim), diff_an_data + (i*dim));
        euc_diff_data[i] = std::max(dot1 - dot2 + margin, Dtype(0));
        loss += (euc_diff_data[i]) / (num * 2);
    }

    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    int count = bottom[0]->count();
    int num   = bottom[0]->num();
    int dim   = count / num;

    Dtype loss_weight    = top[0]->cpu_diff()[0];
    Dtype scale          = loss_weight / num;
    Dtype *euc_diff_data = euc_diff_.mutable_cpu_data();

    Dtype* bottom0_diff = bottom[0]->mutable_cpu_diff();
    Dtype* bottom1_diff = bottom[1]->mutable_cpu_diff();
    Dtype* bottom2_diff = bottom[2]->mutable_cpu_diff();

    for(int i = 0; i < num; ++i){
        if(euc_diff_data[i] > 0) {
            euc_diff_data[i] = scale;
        } else {
            euc_diff_data[i] = 0;
        }
    }

    if(propagate_down[0]) {
        for(int k = 0; k < num; ++k) {
            caffe_copy(count, diff_pn_.cpu_data(), bottom0_diff);
            caffe_scal(dim, - euc_diff_data[k], bottom0_diff + dim * k);
        }
    }

    if(propagate_down[1]) {
        for(int k = 0; k < num; ++k) {
            caffe_copy(count, diff_ap_.cpu_data(), bottom1_diff);
            caffe_scal(dim, - euc_diff_data[k], bottom1_diff + dim * k);
        }
    }

    if(propagate_down[2]) {
        for(int k = 0; k < num; ++k) {
            caffe_copy(count, diff_an_.cpu_data(), bottom2_diff);
            caffe_scal(dim, euc_diff_data[k], bottom2_diff + dim * k);
        }
    }

}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

} //namespace caffe
