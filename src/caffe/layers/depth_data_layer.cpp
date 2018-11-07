#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/depth_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/prediction.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
DepthDataLayer<Dtype>::~DepthDataLayer<Dtype>() {
  this->StopInternalThread();
}


cv::Mat img_to_depth(const cv::Mat& rgb_img)
{
cv::Mat out(rgb_img.size(),CV_32FC1);
        for (uint y = 0; y < out.cols; ++y) {
            for (uint x = 0; x < out.rows; ++x) {
                unsigned char * p = reinterpret_cast<unsigned char *>(&out.at<float>(x, y));
                for (std::size_t i = 0; i != sizeof(float); ++i) {
                    p[i] = rgb_img.at(x, y)[i];
                }
            }
        }
        return out;
}


template <typename Dtype>
	void	DepthDataLayer<Dtype>::UnpackDepth(const cv::Mat & cv_img,
												 cv::Mat & data,cv::Mat & depth,cv::Mat & gt) {
		int size = this->layer_param_.depth_data_param().img_size();
		//int dim = this->layer_param_.depth_data_param().depth_size();
		CHECK(cv_img.rows == size) << "#rows in image should be the size of input image";
		cv::Mat data_alpha = cv_img(cv::Rect(0,0,size,size));
		int from_to[3*2] = {0,0,1,1,2,2};
		data = cv::Mat(data_alpha.size(),data_alpha.depth()+8*2);
		cv::mixChannels(&data_alpha,1,&data,1,from_to,3);
		
		// LOG(INFO) << "data of size "  << data.rows << "," << data.cols << "," << data.channels() ;
  depth = img_to_depth(cv_img(cv::Rect(size,0,size,size)));
		  depth = img_to_depth(cv_img(cv::Rect(size*2,0,size,size)));

	}

	
template <typename Dtype>
void DepthDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.depth_data_param().new_height();
  const int new_width  = this->layer_param_.depth_data_param().new_width();
  const bool is_color  = this->layer_param_.depth_data_param().is_color();
  string root_folder = this->layer_param_.depth_data_param().root_folder();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.depth_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  while (std::getline(infile, line)) {
    lines_.push_back(line);
  }

  CHECK(!lines_.empty()) << "File is empty";
 
  if (this->layer_param_.depth_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  } else {
	  if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 ) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_],
                                    new_height, new_width, is_color); 
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_];
  //TODO unpack depth
  cv::Mat  data, depth,gt;
  //Grid<Dtype> vox;
  this->UnpackDepth(cv_img,data,depth,gt);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(data);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.depth_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  
  top_shape = this->data_transformer_->InferBlobShape(depth);
  // Reshape prefetch_data and top[1] according to the batch_size.
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->depth_.Reshape(top_shape);
  }
  top[1]->Reshape(top_shape);
  LOG(INFO) << "output depth size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  
  
  top_shape = this->data_transformer_->InferBlobShape(gt);
  // Reshape prefetch_data and top[1] according to the batch_size.
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->depth_.Reshape(top_shape);
  }
  top[2]->Reshape(top_shape);
  LOG(INFO) << "output gt size: " << top[2]->num() << ","
      << top[2]->channels() << "," << top[2]->height() << ","
      << top[2]->width();

  }

template <typename Dtype>
void DepthDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void DepthDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  DepthDataParameter depth_data_param = this->layer_param_.depth_data_param();
  const int batch_size = depth_data_param.batch_size();
  const int new_height = depth_data_param.new_height();
  const int new_width = depth_data_param.new_width();
  const bool is_color = depth_data_param.is_color();
  string root_folder = depth_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_],
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_];


       cv::Mat  data, depth,gt;
  //Grid<Dtype> vox;
  this->UnpackDepth(cv_img,data,depth,gt);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(data);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  top_shape[0] = batch_size;
batch->data_.Reshape(top_shape);

  top_shape = this->data_transformer_->InferBlobShape(data);
  // Reshape prefetch_data and top[1] according to the batch_size.
  top_shape[0] = batch_size;
batch->depth_.Reshape(top_shape);


  top_shape = this->data_transformer_->InferBlobShape(data);
  // Reshape prefetch_data and top[1] according to the batch_size.
  top_shape[0] = batch_size;
batch->gt_.Reshape(top_shape);


  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_],
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_];
	this->UnpackDepth(cv_img,data,vox);
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(data, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

	offset = batch->label_.offset(item_id);
	Dtype * label_data = prefetch_label+offset;
	for(int c = 0; c < vox_size; c++)
	{
		std::memcpy(label_data, vox[c].data() ,vox_size * vox_size * sizeof(Dtype));
		label_data += vox_size*vox_size;
	}
	
    //prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.depth_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DepthDataLayer);
REGISTER_LAYER_CLASS(DepthData);

}  // namespace caffe
#endif  // USE_OPENCV
