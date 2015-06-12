#include <vector>

#include "caffe/data_layers.hpp"

#define DEBUG_WUHAO 1
#ifdef DEBUG_WUHAO
#include "caffe/debugtool.hpp"
#endif
namespace caffe {

template <typename Dtype>
void PairDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());
  caffe_copy(prefetch_data2_.count(), prefetch_data2_.cpu_data(),
      (*top)[2]->mutable_gpu_data());
  caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
      (*top)[1]->mutable_gpu_data());
  caffe_copy(prefetch_label2_.count(), prefetch_label2_.cpu_data(),
      (*top)[3]->mutable_gpu_data());


#ifdef DEBUG_WUHAO
  /*
  LOG(INFO) << "in PairDataLayer::Forward_gpu ";

  DebugTool<Dtype> dbg;
  dbg.open("data.bin");
  dbg.write_blob("data", *(*top)[0], 0);
  dbg.write_blob("data2", *(*top)[2], 0);
  dbg.write_blob("label", *(*top)[1], 0);
  dbg.write_blob("label2",*(*top)[3], 0);  
  dbg.close();

  string str;
  std::cout << std::flush << "pause..." << std::flush;
  std::cin >> str;
  */
#endif

  // Start a new prefetch thread
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
}

INSTANTIATE_CLASS(PairDataLayer);

}  // namespace caffe
