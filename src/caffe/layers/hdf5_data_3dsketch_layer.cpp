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
#include <unordered_set>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_data_3dsketch_layer.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/prediction.hpp"

#include <opencv2/core/core.hpp>

//#include <cv.h>
//#include <highgui.h>

namespace caffe {

template <typename Dtype>
HDF5Data3DSketchLayer<Dtype>::~HDF5Data3DSketchLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5Data3DSketchLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size(); //WARNING change -1 
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  //load single view sketch and voxel gt
  for (int i = 0; i < top_size; ++i) {
	DLOG(INFO) << "Loading dataset  " << this->layer_param_.top(i).c_str();
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
						 MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get(),true);
  }


  //WARNING load data for rotation
  data_update_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  hdf5_load_nd_dataset(file_id, "sketch3D_update",
        MIN_DATA_DIM, MAX_DATA_DIM, data_update_.get(),true);



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
  if (this->layer_param_.hdf5_data_3dsketch_param().shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }
}

template <typename Dtype>
void HDF5Data3DSketchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_3dsketch_param().source();
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
  if (this->layer_param_.hdf5_data_3dsketch_param().shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // Load the first HDF5 file and initialize the line counter.
  LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str());
  current_row_ = 0;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_3dsketch_param().batch_size();
  const int top_size = this->layer_param_.top_size(); //WARNING -1 to avoid last blob
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
	  top_shape.resize(hdf_blobs_[i]->num_axes());
    top_shape[0] = batch_size;
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = hdf_blobs_[i]->shape(j);
    }
    top[i]->Reshape(top_shape);
  }

}
#define mod(a,b) ((a)<0?(a)+(b):(a)%(b))

template <typename Dtype>
void HDF5Data3DSketchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_3dsketch_param().batch_size();
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
          if (this->layer_param_.hdf5_data_3dsketch_param().shuffle()) {
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
      if (this->layer_param_.hdf5_data_3dsketch_param().shuffle())
        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }

	//LOG(INFO) << "copy voxel_64";

	//donner donnees (sauf premier blob : combine sketches)
    for (int j = 1; j < this->layer_param_.top_size(); ++j) { //WARNING 1 to avoid first blob
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    }

	//choose number of other views
	int nviews = distribution_n_views(generator_);
	//std::cout<<"nviews " <<nviews<<std::endl;
	int id_blob = 0;
	//WARNING retrieve id_view (%8 ?) and choose first view (random puis /8 + idv1)
	int idv1 = data_permutation_[current_row_];
	int v1 = idv1%8;
	int id_obj = idv1/8;
	std::unordered_set<int> views;
	views.insert(v1);
	
	int data_dim = top[id_blob]->count() / top[id_blob]->shape(0);
	Dtype * top_data = &top[id_blob]->mutable_cpu_data()[i * data_dim];
	shared_ptr<Blob<Dtype> > aggreg_sketches;
	//LOG(INFO) << "copy initial 3dsketch";
	caffe_copy(data_dim,
				&hdf_blobs_[id_blob]->cpu_data()[idv1  * data_dim],
				top_data);
		   
	for(int n = 1; n < nviews; n++)
	{
		//choose other views in the 13 update ones
		int v2 = distribution_2nd_view(generator_);
		auto insertion = views.insert(v2);
		while(!(insertion.second))
		{
			v2 = distribution_2nd_view(generator_);
			insertion = views.insert(v2);
		}
		int idv2 = id_obj*13+v2; //id (in dbase) of the first view to use
		//std::cout<<"obj "<<id_obj<<" v1 : " <<v1<<", v2 : " <<v2<<std::endl;
		//std::cout<<" idv1 : " <<idv1<<", idv2 : " <<idv2<<std::endl;
		
		//WARNING aggreagate values
		caffe_add(data_dim,
				  top_data,
				  &data_update_->cpu_data()[idv2  * data_dim],
				  top_data);
		
	}
	if(nviews > 1)
		caffe_scal<Dtype>(data_dim,
				   1.0/(nviews),
				   top_data);
	// caffe_copy(data_dim,
	// 		   aggreg_sketches->cpu_data(),
	// 		   &top[id_blob]->mutable_cpu_data()[i * data_dim]);
  }
}


INSTANTIATE_CLASS(HDF5Data3DSketchLayer);
REGISTER_LAYER_CLASS(HDF5Data3DSketch);

}  // namespace caffe
