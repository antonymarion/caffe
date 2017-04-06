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
										 const Eigen::Matrix<Dtype,4,4> &model1,
										 const Eigen::Matrix<Dtype,4,4> &view1,
										 const Eigen::Matrix<Dtype,4,4> &model2,
										 const Eigen::Matrix<Dtype,4,4> &view2,
										 const Eigen::Matrix<Dtype,4,4> &proj);

	template <typename Dtype>
	Dtype rotated_proba_value(const Grid<Dtype> &vox,
							  const Eigen::Matrix<Dtype,4,4> &model1,
							  const Eigen::Matrix<Dtype,4,4> &view1,
							  const Eigen::Matrix<Dtype,4,4> &model2,
							  const Eigen::Matrix<Dtype,4,4> &view2,
							  const Eigen::Matrix<Dtype,4,4> &proj,
							  Dtype z,Dtype x, Dtype y);

	template <typename Dtype>
	Dtype z_to_depth(int z, int size, const Eigen::Matrix<Dtype,4,4> &proj);
	template <typename Dtype>
	int depth_to_z(Dtype depth, int size, const Eigen::Matrix<Dtype,4,4> &proj);

	template <typename Dtype>
	Eigen::Matrix<Dtype,3,1> rotate_coords(Dtype x, Dtype y, Dtype z,
										   const Eigen::Matrix<Dtype,4,4> &model1,
										   const Eigen::Matrix<Dtype,4,4> &view1,
										   const Eigen::Matrix<Dtype,4,4> &model2,
										   const Eigen::Matrix<Dtype,4,4> &view2,
										   const Eigen::Matrix<Dtype,4,4> &proj);
	template <typename Dtype>
	void rotate_blobs(const Blob<Dtype> * pred,
					  const Dtype* model1,
					  const Dtype* view_mat1,
					  const Dtype* model2,
					  const Dtype* view_mat2,
					  const Dtype* proj_mat,
					  Dtype * output) ;

		template <typename Dtype>
		Grid<Dtype> unpack_pred_in_image( cv::Mat &image, int grid_rows, int grid_cols);
	template <typename Dtype>
	int CV_type();

	
}

#endif
