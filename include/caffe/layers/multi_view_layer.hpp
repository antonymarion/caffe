#ifndef CAFFE_MULTI_VIEW_LAYER_HPP_
#define CAFFE_MULTI_VIEW_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class MultiViewLayer : public Layer<Dtype> {
 public:
  explicit MultiViewLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiView"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_shared_;
  int K_shared_;
  int N_shared_;
  bool bias_term_shared_;
  Blob<Dtype> bias_multiplier_shared_;

  int M_unique_1_;
  int K_unique_1_;
  int N_unique_1_;

  int M_unique_2_;
  int K_unique_2_;
  int N_unique_2_;

  float lambda_;
  Blob<Dtype> temp_V1, temp_V2, temp_W;
};

}  // namespace caffe

#endif  // CAFFE_MULTI_VIEW_LAYER_HPP_
