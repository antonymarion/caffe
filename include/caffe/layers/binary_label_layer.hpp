#ifndef CAFFE_BINARY_LABEL_HPP_
#define CAFFE_BINARY_LABEL_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BinaryLabelLayer : public Layer<Dtype> {
  public:
    explicit BinaryLabelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
    virtual inline const char* type() const { return "BinaryLabel"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype neg_target_;
};

} // namespace caffe

#endif // CAFFE_BINARY_LABEL_HPP_