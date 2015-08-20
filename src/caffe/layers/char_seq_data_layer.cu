#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void CharSeqDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());

  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
        (*top)[1]->mutable_gpu_data());
  }

  //(*top)[1]->cpu_data();

  for (int i=0; i < max_length_; i++)
  {
    Dtype* temp_label = (*top)[i+2]->mutable_cpu_data();
    for (int nbatch=0; nbatch < this->layer_param_.char_seq_data_param().batch_size(); nbatch++)
    {
      //this operation is ill-posed, it may be fixed latter. 
      int idx = (int) (*top)[1]->data_at(nbatch, 0, 0, 0);

      *(temp_label + nbatch) = *(character_label_ + idx*max_length_+i);
#if 0//DEBUG
      std::cout << "label1:" << *(character_label_ + nbatch*max_length_+i)<<"\n";
      std::cout << "temp: " << *(temp_label + nbatch) << "\n";
      std::cout << "top: " << *((*top)[i+2]->mutable_cpu_data()+nbatch)<<"\n";
      std::cout << "pause..";
      string str;
      std::cin >> str;
#endif
    }
    (*top)[i+2]->gpu_data();
  }

  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(CharSeqDataLayer);

}  // namespace caffe
