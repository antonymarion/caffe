#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void DetectionAccLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	thr_ = this->layer_param_.detection_acc_param().thr();
}

template <typename Dtype>
void DetectionAccLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 3, 1, 1);
}

template <typename Dtype>
void DetectionAccLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  int cnt = bottom[1]->count();
  for (int ind = 0; ind < cnt; ind++)
  {
  	int label = static_cast<int>(bottom_label[ind]);
  	if (label > 0)
  	{
  		true_positive += (bottom_data[ind] >=0);
  		false_negative += (bottom_data[ind] < 0);
  		count_pos++;
  	}
  	else
  	{
  		true_negative += (bottom_data[ind] < 0);
  		false_positive += (bottom_data[ind] >=0);
  		count_neg++;
  	}
  }

  Dtype recall = (count_pos > 0)? (true_positive / count_pos) : 0;
  Dtype percision = (true_positive > 0)? (true_positive / (true_positive + false_positive)) : 0;
  Dtype acc = ((count_pos+count_neg) > 0)? ((true_positive+true_negative) / (count_pos + count_neg)) : 0;

  (*top)[0]->mutable_cpu_data()[0] = acc;
  (*top)[0]->mutable_cpu_data()[1] = percision; 
  (*top)[0]->mutable_cpu_data()[2] = recall;
}

INSTANTIATE_CLASS(DetectionAccLayer);
}
