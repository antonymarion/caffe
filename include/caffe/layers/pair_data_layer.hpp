#ifndef CAFFE_PAIR_DATA_LAYER_HPP_
#define CAFFE_PAIR_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class PairDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PairDataLayer(const LayerParameter& param);
  virtual ~PairDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline const char* type() const { return "PairData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 4; }
  virtual inline int MaxTopBlobs() const { return 4; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
};

}  // namespace caffe

#endif  // CAFFE_PAIR_DATA_LAYER_HPP_
