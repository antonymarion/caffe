#ifndef CAFFE_UTIL_PREDICTION_H_
#define CAFFE_UTIL_PREDICTION_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>
#include "caffe/blob.hpp"

namespace caffe {

	template <typename Dtype>
	using Slice = Eigen::Matrix<Dtype,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>;
	template <typename Dtype>
	using Grid = std::vector<Slice<Dtype> >;

	template <typename Dtype>
	Grid<Dtype> rotate_voxels_prediction(const Grid<Dtype> &vox,
									   const Eigen::Matrix<Dtype,4,4> &model,
									   const Eigen::Matrix<Dtype,4,4> &view,
									   const Eigen::Matrix<Dtype,4,4> &proj);

	template <typename Dtype>
	Dtype rotated_proba_value(const Grid<Dtype> &vox,
							  const Eigen::Matrix<Dtype,4,4> &model,
							  const Eigen::Matrix<Dtype,4,4> &view,
							  const Eigen::Matrix<Dtype,4,4> &proj,
							  Dtype z,Dtype x, Dtype y);

	template <typename Dtype>
	Dtype z_to_depth(int z, int size, const Eigen::Matrix<Dtype,4,4> &proj);
	template <typename Dtype>
	int depth_to_z(Dtype depth, int size, const Eigen::Matrix<Dtype,4,4> &proj);

	template <typename Dtype>
	Eigen::Matrix<Dtype,3,1> rotate_coords(Dtype x, Dtype y, Dtype z,
								  const Eigen::Matrix<Dtype,4,4> &model, 
								  const Eigen::Matrix<Dtype,4,4> &view,
								  const Eigen::Matrix<Dtype,4,4> &proj);
	template <typename Dtype>
	void rotate_blobs(const Blob<Dtype> * pred,
					  const Dtype* viewpoint1,
					  const Dtype* viewpoint2,
					  const Dtype* view_mat,
					  const Dtype* proj_mat,
					  Dtype * output) ;
	
}

#endif
