#ifndef CAFFE_VOXELS_DATA_LAYER_HPP_
#define CAFFE_VOXELS_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/prediction.hpp"

#include <opencv2/core/core.hpp>

namespace caffe {

/**
 * @brief Provides data to the Net from depth files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DepthDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DepthDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~DepthDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DepthData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
	void UnpackDepth(const cv::Mat & cv_img, cv::Mat & data,cv::Mat & depth,cv::Mat & gt);

  vector<std::string>  lines_;
  int lines_id_;
};


}  // namespace caffe

#endif  // CAFFE_Depth_DATA_LAYER_HPP_
