#ifndef CAFFE_HDF5_DATA_3DSKETCH_LAYER_HPP_
#define CAFFE_HDF5_DATA_3DSKETCH_LAYER_HPP_

#include "hdf5.h"

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"
#include  <ctime>
#include  <random>

namespace caffe {

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5Data3DSketchLayer : public Layer<Dtype> {
 public:
  explicit HDF5Data3DSketchLayer(const LayerParameter& param)
	  : Layer<Dtype>(param),  generator_(time(NULL)), distribution_2nd_view(0,12) , distribution_n_views(1,param.hdf5_data_3dsketch_param().nviews()){}
  virtual ~HDF5Data3DSketchLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "HDF5Data3DSketch"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void LoadHDF5FileData(const char* filename);

  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;
  std::vector<shared_ptr<Blob<Dtype> > > hdf_blobs_;
  std::vector<unsigned int> data_permutation_;
  std::vector<unsigned int> file_permutation_;

	shared_ptr<Blob<Dtype> > data_update_;
	std::default_random_engine generator_;
	std::uniform_int_distribution<int> distribution_2nd_view;
	std::uniform_int_distribution<int> distribution_n_views;
};

}  // namespace caffe

#endif  // CAFFE_HDF5_DATA_3DSKETCH_LAYER_HPP_
