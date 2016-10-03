/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_data_pred_layer.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/prediction.hpp"
#include <cv.h>
#include <highgui.h>

namespace caffe {

template <typename Dtype>
HDF5DataPredLayer<Dtype>::~HDF5DataPredLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataPredLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size()-1; //WARNING change -1 
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
  }

  //WARNING load data for rotation
  viewpoint_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  view_mat_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  proj_mat_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  hdf5_load_nd_dataset(file_id, "viewpoint",
        MIN_DATA_DIM, MAX_DATA_DIM, viewpoint_.get());
  hdf5_load_nd_dataset(file_id, "view_mat",
        MIN_DATA_DIM, MAX_DATA_DIM, view_mat_.get());
  hdf5_load_nd_dataset(file_id, "proj_mat",
        MIN_DATA_DIM, MAX_DATA_DIM, proj_mat_.get());

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0);
  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_pred_param().shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }
}

template <typename Dtype>
void HDF5DataPredLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_pred_param().source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << source;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_pred_param().shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // Load the first HDF5 file and initialize the line counter.
  LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str());
  current_row_ = 0;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_pred_param().batch_size();
  const int top_size = this->layer_param_.top_size()-1; //WARNING -1 to avoid last blob
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
	  top_shape.resize(hdf_blobs_[i]->num_axes());
    top_shape[0] = batch_size;
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = hdf_blobs_[i]->shape(j);
    }
    top[i]->Reshape(top_shape);
  }
  //WARNING set last blob, which is the prediction.
  //Same size as the gt (considered to be in top_size -1 pos)
  top_shape.resize(hdf_blobs_[top_size-1]->num_axes());
  top_shape[0] = batch_size;
  for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = hdf_blobs_[top_size-1]->shape(j);
  }
  top[top_size]->Reshape(top_shape);
 
  //WARNING init prediction net
  pred_net_.CopyTrainedLayersFrom(this->layer_param_.hdf5_data_pred_param().trained_file());

}
#define mod(a,b) ((a)<0?(a)+(b):(a)%(b))

template <typename Dtype>
void HDF5DataPredLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_pred_param().batch_size();
  //parcours images du batch
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == hdf_blobs_[0]->shape(0)) {
		//si fin du fichier
      if (num_files_ > 1) {
		  //prochain fichier
        ++current_file_;
        if (current_file_ == num_files_) {
			//si dernier fichier, revenir a 0 et shuffle
          current_file_ = 0;
          if (this->layer_param_.hdf5_data_pred_param().shuffle()) {
            std::random_shuffle(file_permutation_.begin(),
                                file_permutation_.end());
          }
          DLOG(INFO) << "Looping around to first file.";
        }
		//load fichier
        LoadHDF5FileData(
            hdf_filenames_[file_permutation_[current_file_]].c_str());
      }
	  //shuffle sur les lignes
      current_row_ = 0;
      if (this->layer_param_.hdf5_data_pred_param().shuffle())
        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }
	//donner donnees
    for (int j = 0; j < this->layer_param_.top_size()-1; ++j) { //WARNING -1
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    }
	//WARNING retrieve id_view (%8 ?) and choose first view (random puis /8 + idv1)
	int idv2 = data_permutation_[current_row_];
	int v2 = idv2 % 8;
	bool is_top = v2 > 3; //true if idv1 between 4 and 7
	int dir = 1;//distribution_2nd_view(generator_);
	int v1 = v2;
	if(dir != 0)
	{
		if (is_top) v1 -= 4;
		v1 = mod(v1 + dir,4);
		if (is_top) v1 += 4;
	} else
	{
		if (is_top) v1 -= 4;
		else v1 += 4;
	}
	int idv1 = (idv2/8)*8+v1; //id (in dbase) of the first view to use
	std::cout<<idv2<<" -> view "<<v2<<std::endl;
	std::cout<<" -> dir "<<dir<<std::endl;
	std::cout<<v1<<" -> id "<<idv1<<std::endl;

	std::cout<<hdf_blobs_[0]->shape_string()<<std::endl;

	//WARNING faire prediction
	Blob<Dtype>* input_layer = pred_net_.input_blobs()[0];
	int data_dim = top[0]->count() / top[0]->shape(0);
	caffe_copy(data_dim,
          &hdf_blobs_[0]->cpu_data()[idv1 * data_dim],
			   input_layer->mutable_cpu_data());
	
	cv::Mat im1 = cv::Mat(256,256,CV_32FC1, input_layer->mutable_cpu_data());
  	cv::namedWindow( "sk1", CV_WINDOW_NORMAL );
	cv::imshow("sk1",im1);
cv::Mat im2 = cv::Mat(256,256,CV_32FC1, &hdf_blobs_[0]->mutable_cpu_data()[idv2 * data_dim]);
  	cv::namedWindow( "sk2", CV_WINDOW_NORMAL );
	cv::imshow("sk2",im2);
	cv::waitKey(0);
	pred_net_.Forward();
	std::cout<<pred_net_.output_blobs()[0]->shape_string()<<std::endl;

	// cv::Mat pred = cv::Mat(10*64,64,CV_64FC1);//, &pred_net_.output_blobs()[0]->mutable_cpu_data()[64*30]);
	// //pred *= 255;
	// for(int r = 0; r < 64*10; r++)
	// {
	// 	for(int c = 0; c < 64; c++)
	// 	{
	// 		pred.at<double>(r,c) = pred_net_.output_blobs()[0]->data_at(0,0,64*25+r, c);
	// 	}

	// }

	// cv::namedWindow( "pred", CV_WINDOW_NORMAL );
	// cv::imshow("pred",pred);
	// cv::waitKey(0);
	int last_blob = this->layer_param_.top_size()-1;
	//WARNING rotate pred and put it in network
	rotate_blobs(pred_net_.output_blobs()[0],
				 &viewpoint_->cpu_data()[idv1 * 16],
				 &viewpoint_->cpu_data()[idv2 * 16],
				 &view_mat_->cpu_data()[idv1 * 16],
				 &proj_mat_->cpu_data()[idv1 * 16],
				 &top[last_blob]->mutable_cpu_data()[i * data_dim]);

  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataPredLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataPredLayer);
REGISTER_LAYER_CLASS(HDF5DataPred);

}  // namespace caffe
