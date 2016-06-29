#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
#include <vector>
#include <lmdb.h>
#include <boost/thread.hpp>

#include <fstream>

#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layers/pair_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/blocking_queue.hpp"

using std::ifstream;
using std::pair;


namespace caffe {

template <typename Dtype>
PairDataLayer<Dtype>::PairDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param) {
}

template <typename Dtype>
PairDataLayer<Dtype>::~PairDataLayer<Dtype>() {
    //this->JoinPrefetchThread();
    // clean up the database resources
    switch (this->layer_param_.pair_data_param().backend()) {
    case PairDataParameter_DB_LEVELDB:
        break;  // do nothing
    case PairDataParameter_DB_LMDB:
        mdb_cursor_close(mdb_cursor_);
        mdb_close(mdb_env_, mdb_dbi_);
        mdb_txn_abort(mdb_txn_);
        mdb_env_close(mdb_env_);
        break;
    default:
        LOG(FATAL) << "Unknown database backend";
    }
}

template <typename Dtype>
void PairDataLayer<Dtype>::load_pairs(string fn) {
    LOG(INFO) << "opening pairs file: " << fn;
    ifstream ifs(fn.c_str());
    if(!ifs) {
        LOG(FATAL) << "Error opening pairs list file: " << fn;
        return;
    }
    
    int num_pairs = 0;
    int idx1      = 0;
    int idx2      = 0;

    while(ifs >> idx1 >> idx2) {
        pairs_.push_back(make_pair(idx1, idx2));
    }
    LOG(INFO) << "Loading pairs done. Total pairs: " << pairs_.size();
}

template <typename Dtype>
void PairDataLayer<Dtype>::load_list(string fn) {
    LOG(INFO) << "opening image list file: " << fn;
    ifstream ifs(fn.c_str());

    if (!ifs) {
        LOG(FATAL) << "Error opening image list file: " << fn;
        return;
    }

    string filename;
    int label;
    while(ifs >> filename >> label) {
        image_paths_.push_back(filename);
    }

    LOG(INFO) << "loading image list done. Total images: " << image_paths_.size();
}

template <typename Dtype>
void PairDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {
    // Initialize DB
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.pair_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(ERROR) << "Opening lmdb " << this->layer_param_.pair_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    Datum datum;
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);

    // image
    int crop_size = this->layer_param_.transform_param().crop_size();
    if(crop_size > 0){
        top[0]->Reshape(this->layer_param_.pair_data_param().batch_size(),
                        datum.channels(), crop_size, crop_size);
        top[2]->Reshape(this->layer_param_.pair_data_param().batch_size(),
                        datum.channels(), crop_size, crop_size);
        this->batch1->data_.Reshape(this->layer_param_.pair_data_param().batch_size(),
                        datum.channels(), crop_size, crop_size);
        this->batch2->data_.Reshape(this->layer_param_.pair_data_param().batch_size(),
                        datum.channels(), crop_size, crop_size); 
    } else {
        top[0]->Reshape(this->layer_param_.pair_data_param().batch_size(), 
                        datum.channels(), datum.height(), datum.width());
        top[2]->Reshape(this->layer_param_.pair_data_param().batch_size(), 
                        datum.channels(), datum.height(), datum.width());
        this->batch1->data_.Reshape(this->layer_param_.pair_data_param().batch_size(),
                        datum.channels(), datum.height(), datum.width());
        this->batch2->data_.Reshape(this->layer_param_.pair_data_param().batch_size(),
                        datum.channels(), datum.height(), datum.width());
    }

    // label
    top[1]->Reshape(this->layer_param_.pair_data_param().batch_size(), 1, 1, 1);
    top[3]->Reshape(this->layer_param_.pair_data_param().batch_size(), 1, 1, 1);    
    this->batch1->label_.Reshape(this->layer_param_.pair_data_param().batch_size(),
            1, 1, 1);
    this->batch2->label_.Reshape(this->layer_param_.pair_data_param().batch_size(),
            1, 1, 1); 

    this->datum_channels_ = datum.channels();
    this->datum_height_   = datum.height();
    this->datum_width_    = datum.width();
    this->datum_size_     = datum.channels() * datum.height() * datum.width();

    LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

    cur_pair_ = 0;
    load_pairs(this->layer_param_.pair_data_param().pairs_list());
    load_list(this->layer_param_.pair_data_param().source_list());
}

template <typename Dtype>
void PairDataLayer<Dtype>::get_cur_key(int pair_channel, string& keystr){
    int idx =  0;
    string key_cstr;

  
    CHECK((pair_channel == 1) || (pair_channel == 0));
    if (pair_channel == 0) {
        idx = pairs_[cur_pair_].first;
    } else if (pair_channel == 1) {
        idx = pairs_[cur_pair_].second;
    }

    CHECK(idx < image_paths_.size());
    key_cstr = format_int(idx, 8) + "_" + image_paths_[idx];
    keystr.clear();
    keystr = key_cstr;

}

template <typename Dtype>
void PairDataLayer<Dtype>::get_value(string& keystr, Datum& datum){
    string value;

    switch (this->layer_param_.pair_data_param().backend()) {
    case PairDataParameter_DB_LEVELDB:
        break;
    case PairDataParameter_DB_LMDB:
        mdb_key_.mv_size = keystr.size();
        mdb_key_.mv_data = reinterpret_cast<void*>(&keystr[0]);  
        CHECK_EQ(mdb_get(mdb_txn_, mdb_dbi_, &mdb_key_, &mdb_value_), MDB_SUCCESS);   
        datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);  
        break;
    default:
        LOG(FATAL) << "Unknown database backend";
    }
}

// This function is used to create a thread for prefetching the data
template <typename Dtype>
void PairDataLayer<Dtype>::InternalThreadEntry() {
    Datum datum;
    CHECK(this->batch1->data_.count());
    Dtype* top_data   = this->batch1->data_.mutable_cpu_data();
    Dtype* top_label  = this->batch1->label_.mutable_cpu_data();
    Dtype* top_data2  = this->batch2->data_.mutable_cpu_data();
    Dtype* top_label2 = this->batch2->label_.mutable_cpu_data();

    const int batch_size = this->layer_param_.pair_data_param().batch_size();
    string keystr;

    for (int item_id = 0; item_id < batch_size; ++item_id) {

        int offset1 = batch1->data_.offset(item_id);
        this->batch1->data_.set_cpu_data(top_data + offset1);
        get_cur_key(0, keystr);
        get_value(keystr, datum);
        this->data_transformer_->Transform(datum, &(this->batch1->data_));
        top_label[item_id] = datum.label();

        LOG(ERROR) << keystr;

        int offset2 = batch2->data_.offset(item_id);
        this->batch2->data_.set_cpu_data(top_data2 + offset2);
        get_cur_key(1, keystr);
        get_value(keystr, datum);
        this->data_transformer_->Transform(datum, &(this->batch2->data_));
        top_label2[item_id] = datum.label();

        string str;
        std::cin >> str;

        cur_pair_++;   
    }

}

template <typename Dtype>
void PairDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

}

template <typename Dtype>
void PairDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                                        vector<Blob<Dtype>*>& top) {
    // Copy the data
/*    caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
            top[0]->mutable_cpu_data());
    caffe_copy(this->prefetch_data2_.count(), this->prefetch_data2_.cpu_data(),
            top[2]->mutable_cpu_data());

    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
            top[1]->mutable_cpu_data());
    caffe_copy(this->prefetch_label2_.count(), this->prefetch_label2_.cpu_data(),
            top[3]->mutable_cpu_data());*/
  
}

template <typename Dtype>
void PairDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        vector<Blob<Dtype>*>& top){

}

INSTANTIATE_CLASS(PairDataLayer);
REGISTER_LAYER_CLASS(PairData);

} // namespace caffe