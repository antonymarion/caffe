#include <algorithm>
#include <vector>

#include "caffe/layers/CMD_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void CMDLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
	CHECK_EQ(bottom[0]->height(), bottom[1]->height());
	CHECK_EQ(bottom[0]->width(), bottom[1]->width());

	//CMDLossParameter param = this->layer_param_.cmd_loss_param();
	int K_ = this->layer_param_.cmd_loss_param().k();
	CHECK_GE(K_, 1);

	X_mean_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
	Y_mean_.ReshapeLike(X_mean_);
	dis_mean_.ReshapeLike(X_mean_);
	tmp_.ReshapeLike(X_mean_);

	if (K_ > 1){
		X_moments_.Reshape(K_ - 1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
		Y_moments_.ReshapeLike(X_moments_);
		dis_moments_.ReshapeLike(X_moments_);
		norm_dis_moments_.resize(K_ - 1);
		tmp2_.ReshapeLike(X_moments_);
		pow_X_submean_.Reshape((K_ - 1)*bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
		pow_Y_submean_.Reshape((K_ - 1)*bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
	}


	// vector of ones used to sum along channels
	vector<int> sz;
	sz.push_back(bottom[0]->shape(0));
	X_batch_sum_multiplier_.Reshape(sz);//X_batch_sum_multiplier: batch求和乘子 size = N1 x 1， 用于X沿N方向求和
	caffe_set(X_batch_sum_multiplier_.count(), Dtype(1),
		X_batch_sum_multiplier_.mutable_cpu_data());

	sz[0] = bottom[1]->shape(0);
	Y_batch_sum_multiplier_.Reshape(sz);//Y_batch_sum_multiplier: batch求和乘子 size = N2 x 1， 用于Y沿N方向求和
	caffe_set(Y_batch_sum_multiplier_.count(), Dtype(1),
		Y_batch_sum_multiplier_.mutable_cpu_data());

}

template <typename Dtype>
void CMDLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* X_data = bottom[0]->cpu_data();
	const Dtype* Y_data = bottom[1]->cpu_data();
	Dtype loss(0.0);

	int K_ = this->layer_param_.cmd_loss_param().k();
	Dtype eps_ = this->layer_param_.cmd_loss_param().eps();
	Dtype decay_ = this->layer_param_.cmd_loss_param().decay();
	Dtype moments_decay = decay_;
		
	//compute X_mean_ & X_moments_
	int num = bottom[0]->shape(0);
	int feat_dim = bottom[0]->count(1);//feat_dim: C x H x W
	caffe_cpu_gemv(CblasTrans, num, feat_dim,
		Dtype(1. / num), X_data, X_batch_sum_multiplier_.cpu_data(), 
		Dtype(0),	X_mean_.mutable_cpu_data());//X_mean_: 1 x C x H x W, X_data沿N方向求和结果

	Blob<Dtype> X_submean, norm_X_submean;// X_submean = X-E(X)
	X_submean.ReshapeLike(*bottom[0]);
	norm_X_submean.ReshapeLike(*bottom[0]);

	X_submean.CopyFrom(*bottom[0]);
	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, feat_dim, 1, 
		Dtype(-1), X_batch_sum_multiplier_.cpu_data(), X_mean_.cpu_data(), 
		Dtype(1), X_submean.mutable_cpu_data());
	norm_X_submean.CopyFrom(X_submean);
	for (int i = 0; i < K_ - 1; ++i){
		caffe_copy(norm_X_submean.count(), norm_X_submean.cpu_data(), pow_X_submean_.mutable_cpu_data()+pow_X_submean_.offset(i*num));
		caffe_mul(X_submean.count(), X_submean.cpu_data(), norm_X_submean.cpu_data(), norm_X_submean.mutable_cpu_data());
		caffe_cpu_gemv(CblasTrans, num, feat_dim,
			Dtype(1. / num), norm_X_submean.cpu_data(), X_batch_sum_multiplier_.cpu_data(),
			Dtype(0), X_moments_.mutable_cpu_data()+X_moments_.offset(i));
	}

	//compute Y_mean_ & Y_moments_
	num = bottom[1]->shape(0);
	caffe_cpu_gemv(CblasTrans, num, feat_dim,
		Dtype(1. / num), Y_data, Y_batch_sum_multiplier_.cpu_data(),
		Dtype(0), Y_mean_.mutable_cpu_data());//Y_mean_: 1 x C x H x W, Y_data沿N方向求和结果

	Blob<Dtype> Y_submean, norm_Y_submean;// Y_submean = Y-E(Y)
	Y_submean.ReshapeLike(*bottom[1]);
	norm_Y_submean.ReshapeLike(*bottom[1]);

	Y_submean.CopyFrom(*bottom[1]);
	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, feat_dim, 1,
		Dtype(-1), Y_batch_sum_multiplier_.cpu_data(), Y_mean_.cpu_data(),
		Dtype(1), Y_submean.mutable_cpu_data());
	norm_Y_submean.CopyFrom(Y_submean);

	for (int i = 0; i < K_ - 1; ++i){
		caffe_copy(norm_Y_submean.count(), norm_Y_submean.cpu_data(), pow_Y_submean_.mutable_cpu_data() + pow_Y_submean_.offset(i*num));
		caffe_mul(Y_submean.count(), Y_submean.cpu_data(), norm_Y_submean.cpu_data(), norm_Y_submean.mutable_cpu_data());
		caffe_cpu_gemv(CblasTrans, num, feat_dim,
			Dtype(1. / num), norm_Y_submean.cpu_data(), Y_batch_sum_multiplier_.cpu_data(),
			Dtype(0), Y_moments_.mutable_cpu_data() + Y_moments_.offset(i));
	}
		
	//Loss = norm(E(x)-E(y)) + norm(var(x)-var(y)) + ...
	caffe_sub(feat_dim, X_mean_.cpu_data(), Y_mean_.cpu_data(), dis_mean_.mutable_cpu_data());
	norm_dis_mean_ = dis_mean_.sumsq_data() + eps_;
	caffe_powx(1, &norm_dis_mean_, Dtype(0.5), &norm_dis_mean_);
	loss += norm_dis_mean_;

	if (K_ > 1){
		caffe_sub(X_moments_.count(), X_moments_.cpu_data(), Y_moments_.cpu_data(), dis_moments_.mutable_cpu_data());
		caffe_powx(X_moments_.count(), dis_moments_.cpu_data(), Dtype(2), tmp2_.mutable_cpu_data());
	}
	for (int i = 0; i < K_ - 1; ++i){
		Dtype tt = caffe_cpu_asum(feat_dim, tmp2_.mutable_cpu_data() + tmp2_.offset(i));
		tt += eps_;
		caffe_powx(1, &tt, Dtype(0.5), &tt);
		norm_dis_moments_[i] = tt;
		loss += moments_decay * norm_dis_moments_[i];
		moments_decay *= decay_;
	}

	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CMDLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int K_ = this->layer_param_.cmd_loss_param().k();
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	Dtype decay_ = this->layer_param_.cmd_loss_param().decay();
	Dtype moments_decay = decay_;

	if (propagate_down[0]) {
		Dtype* X_diff = bottom[0]->mutable_cpu_diff();
		int num = bottom[0]->shape(0);
		int count = bottom[0]->count();
		int feat_dim = bottom[0]->count(1);//feat_dim: C x H x W

		Blob<Dtype> diff_tmp, diff_moments_tmp;
		diff_tmp.ReshapeLike(*bottom[0]);
		//pow_sum_tmp.Reshape(1, bottom[0]->channels, bottom[0]->height(), bottom[0]->width());
		diff_moments_tmp.ReshapeLike(*bottom[0]);

		// replicate dis_mean_ to input size
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, feat_dim, 1,
			Dtype(1. / (num*norm_dis_mean_)), X_batch_sum_multiplier_.cpu_data(), dis_mean_.cpu_data(),
			Dtype(0), X_diff);
		for (int i = 0; i < K_ - 1; ++i){
			int k = 2 + i;
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, feat_dim, 1,
				Dtype(k/(num*norm_dis_moments_[i])), X_batch_sum_multiplier_.cpu_data(), 
				dis_moments_.cpu_data()+dis_moments_.offset(i),
				Dtype(0), diff_tmp.mutable_cpu_data());
			caffe_copy(count, pow_X_submean_.cpu_data() + pow_X_submean_.offset(i*num), diff_moments_tmp.mutable_cpu_data());
			if (k > 2)
				caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, feat_dim, 1,
					Dtype(-1), X_batch_sum_multiplier_.cpu_data(), X_moments_.cpu_data() + X_moments_.offset(i-1),
					Dtype(1), diff_moments_tmp.mutable_cpu_data());
				
			caffe_mul(count, diff_moments_tmp.cpu_data(), diff_tmp.cpu_data(), diff_tmp.mutable_cpu_data());
			caffe_scal(count, moments_decay, diff_tmp.mutable_cpu_data());
			caffe_add(count, X_diff, diff_tmp.cpu_data(), X_diff);
			moments_decay *= decay_;
		}
			// Scale down gradient
		caffe_scal(count, loss_weight / (2 * num), X_diff);
	}

	moments_decay = decay_;
	if (propagate_down[1]) {
		Dtype* Y_diff = bottom[1]->mutable_cpu_diff();
		int num = bottom[1]->shape(0);
		int count = bottom[1]->count();
		int feat_dim = bottom[1]->count(1);//feat_dim: C x H x W

		Blob<Dtype> diff_tmp, diff_moments_tmp;
		diff_tmp.ReshapeLike(*bottom[1]);
		diff_moments_tmp.ReshapeLike(*bottom[1]);

		// replicate dis_mean_ to input size
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, feat_dim, 1,
			Dtype(-1. / (num*norm_dis_mean_)), Y_batch_sum_multiplier_.cpu_data(), dis_mean_.cpu_data(),
			Dtype(0), Y_diff);
		for (int i = 0; i < K_ - 1; ++i){
			int k = 2 + i;
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, feat_dim, 1,
				Dtype(-k / (num*norm_dis_moments_[i])), Y_batch_sum_multiplier_.cpu_data(),
				dis_moments_.cpu_data() + dis_moments_.offset(i),
				Dtype(0), diff_tmp.mutable_cpu_data());

			caffe_copy(count, pow_Y_submean_.cpu_data() + pow_Y_submean_.offset(i*num), diff_moments_tmp.mutable_cpu_data());
			if (k > 2){
				caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, feat_dim, 1,
					Dtype(-1), Y_batch_sum_multiplier_.cpu_data(), Y_moments_.cpu_data() + Y_moments_.offset(i-1),
					Dtype(1), diff_moments_tmp.mutable_cpu_data());
			}
			caffe_mul(count, diff_moments_tmp.cpu_data(), diff_tmp.cpu_data(), diff_tmp.mutable_cpu_data());
			caffe_scal(count, moments_decay, diff_tmp.mutable_cpu_data());
			caffe_add(count, Y_diff, diff_tmp.cpu_data(), Y_diff);
			moments_decay *= decay_;
		}
		// Scale down gradient
		caffe_scal(count, loss_weight / (2*num), Y_diff);
	}

}

#ifdef CPU_ONLY
	STUB_GPU(CMDLossLayer);
#endif

	INSTANTIATE_CLASS(CMDLossLayer);
	REGISTER_LAYER_CLASS(CMDLoss);

}  // namespace caffe