#ifndef CAFFE_CMD_LOSS_LAYER_HPP_
#define CAFFE_CMD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
	/**
	* Computes the CMD loss
	*/
	template <typename Dtype>
	class CMDLossLayer : public LossLayer<Dtype> {
	public:
		explicit CMDLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "CMDLoss"; }
		/**
		* Unlike most loss layers, in the CMDLossLayer we can backpropagate
		* to the first two inputs.
		*/
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		//virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		//	const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	//	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		//	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> X_mean_, Y_mean_, dis_mean_;
		Blob<Dtype> X_moments_, Y_moments_, dis_moments_;
		Dtype norm_dis_mean_;
		Blob<Dtype> pow_X_submean_, pow_Y_submean_;
		vector<Dtype> norm_dis_moments_;
		

		// extra temporarary variables is used to carry out sums
		Blob<Dtype> X_batch_sum_multiplier_, Y_batch_sum_multiplier_;
		Blob<Dtype> tmp_, tmp2_;
	};

}  // namespace caffe

#endif  // CAFFE_CMD_LOSS_LAYER_HPP_
