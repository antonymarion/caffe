#include <leveldb/db.h>
#include <stdint.h>

#include <stdio.h>
#include <string>
#include <vector>

#include <fstream>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/debugtool.hpp"
#define DEBUG 1

namespace caffe {


template <typename Dtype>
CharSeqDataLayer<Dtype>::~CharSeqDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.char_seq_data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
  if (character_label_)
    delete [] character_label_;
}

template <typename Dtype>
void CharSeqDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.char_seq_data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.char_seq_data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.char_seq_data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.char_seq_data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.char_seq_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.char_seq_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.char_seq_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.char_seq_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.char_seq_data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.char_seq_data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.char_seq_data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.char_seq_data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.char_seq_data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.char_seq_data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.char_seq_data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.char_seq_data_param().batch_size(),
        1, 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();

  // load source list
  /*ifstream source(this->layer_param_.char_seq_data_param().source_list().c_str());
  if (!source) {
    LOG(FATAL) << "Error opening image list file: ";
    return;
  }  
  string filename;
  int label;
  while(source >> filename >> label) {
    image_paths_.push_back(filename);
  }

  LOG(INFO) << "loading image list done. Total images: " << image_paths_.size();  */


  // load char_seq_label list
  LOG(INFO) << "Opening character sequence label file...";
  std::ifstream ifs(this->layer_param_.char_seq_data_param().char_seq_list().c_str());
  if(!ifs)
  {
    LOG(FATAL) << "Character Sequence Label File loading failed..";
    return;
  }

  int num_samples = 0;
  int temp = 0;
  max_length_ = this->layer_param_.char_seq_data_param().max_length();
  ifs >> num_samples;
  CHECK(num_samples > 0);
  character_label_ = new Dtype[max_length_ * num_samples];
  if (character_label_ == NULL)
  {
  	LOG(INFO) << "Memory Allocate Failed..";
  	return;
  }
  for(int i = 0; i < num_samples * max_length_; i++)
  {
  	ifs >> temp;
  	*(character_label_ + i) = temp;
  }

#if 0//DEBUG
    string str;

    for(int i=0; i < num_samples; i++)
    {
      for(int j=0; j < max_length_; j++)
      {
        std::cout << *(character_label_ + i*max_length_+j)<<" ";
      }
      std::cout << "pause..";
      std::cin >> str;
    }

 #endif

  for (int i = 0; i < max_length_; i++)
  {
    (*top)[i+2]->Reshape(this->layer_param_.char_seq_data_param().batch_size(), 1, 1, 1);
  }

}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void CharSeqDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.char_seq_data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.char_seq_data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum.ParseFromString(iter_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }

    // go to the next iter
    switch (this->layer_param_.char_seq_data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}

template <typename Dtype>
void CharSeqDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  //BasePrefetchingDataLayer::JoinPrefetchThread();
  this->JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }

#if 0//DEBUG
    string str;

    for(int i=0; i < this->layer_param_.char_seq_data_param().batch_size(); i++)
    {
      for(int j=0; j < max_length_; j++)
      {
        std::cout << *(character_label_ + i*max_length_+j)<<" ";
      }
      std::cout << "pause..";
      std::cin >> str;
    }
#endif

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
  }


  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(CharSeqDataLayer, Forward);
#endif


INSTANTIATE_CLASS(CharSeqDataLayer);

}  // namespace caffe
