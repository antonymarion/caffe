#ifndef CAFFE_PAIR_DATA_LAYER_HPP_
#define CAFFE_PAIR_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class PairDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
    explicit PairDataLayer(const LayerParameter& param);
    virtual ~PairDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*> & bottom, 
        const vector<Blob<Dtype>*>& top);

    virtual inline bool ShareInParallel() const { return false; }
    virtual inline const char* type() const { return "PairData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int MinTopBlobs() const { return 4; }
    virtual inline int MaxTopBlobs() const { return 4; }

    void Forward_cpu(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top); 
    void Forward_gpu(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top);

protected:
    virtual void InternalThreadEntry();
    virtual void load_batch(Batch<Dtype>* batch);

    // Blob<Dtype> prefetch_data_;
    // Blob<Dtype> prefetch_label_;
    // Blob<Dtype> prefetch_data2_;
    // Blob<Dtype> prefetch_label2_;

    Batch<Dtype>* batch1;
    Batch<Dtype>* batch2;

    void load_pairs(string fn);
    void load_list(string fn);

    void get_cur_key(int pair_channel, string& keystr);
    void get_value(string& keystr, Datum& datum);

    // LMDB
    MDB_env* mdb_env_;
    MDB_dbi mdb_dbi_;
    MDB_txn* mdb_txn_;
    MDB_cursor* mdb_cursor_;
    MDB_val mdb_key_, mdb_value_;

    vector<pair<int, int> > pairs_;
    int cur_pair_;
    vector<string> image_paths_;

};

} // namespace

#endif