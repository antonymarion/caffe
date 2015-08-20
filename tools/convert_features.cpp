// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

int _MAX_PATH = 10000;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb", "The backend for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

// get size of features
static int get_file_size(const char *filename)
{
  int fsize;
  FILE *fp;

  fp = fopen(filename, "rb");
  if(fp == NULL)
    return 0;
  fseek(fp, 0, SEEK_END);
  fsize = ftell(fp);
  fclose(fp);

  return fsize;
}

// read features
static unsigned char * read_file_buffer(const char *filename, int *size)
{
  int fsize;
  FILE *fp;
  unsigned char *pnchBuf = NULL;

  *size = 0;

  fsize = get_file_size(filename);
  if(fsize == 0)
    return NULL;

  if((fp = fopen(filename, "rb")) != NULL )
  {
    pnchBuf = (unsigned char *) malloc(fsize * sizeof(unsigned char));
    fread(pnchBuf, fsize, 1, fp);
    fclose(fp);
  }

  *size = fsize;

  return pnchBuf;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetLogDestination(0, argv[1]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] LOG_PATH ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 5) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  bool is_color = !FLAGS_gray;
  std::ifstream infile(argv[3]);
  std::vector<std::pair<string, int> > lines;
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(ERROR) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(ERROR) << "A total of " << lines.size() << " images.";

  // const string& db_backend = FLAGS_backend;
  const string& db_backend = "lmdb";
  const char* db_path = argv[4];

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Preload features
  string root_folder(argv[2]);
  char file_name[_MAX_PATH];
  string str_tmp;
  int i, file_size, fea_len;
  float **features;

  features = (float **) malloc(lines.size() * sizeof(float *));
  LOG(ERROR) << "Loading features...";
  for(i=0; i<lines.size(); i++)
  {
    str_tmp = root_folder + lines[i].first;
    strcpy(file_name, str_tmp.c_str());
    features[i] = (float *) read_file_buffer(file_name, &file_size);

    if(i % 10000 == 0)
    {
      LOG(ERROR) << "Loaded " << i << " features.";
    }
  }
  fea_len = file_size / sizeof(float);
  LOG(ERROR) << "feature length: " << fea_len;

  // Open new db
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  // leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;

  // Open db
  if (db_backend == "leveldb") {  // leveldb
    LOG(INFO) << "Opening leveldb " << db_path;
    leveldb::Status status = leveldb::DB::Open(
        options, db_path, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_path
        << ". Is it already existing?";
    batch = new leveldb::WriteBatch();
  } else if (db_backend == "lmdb") {  // lmdb
    LOG(INFO) << "Opening lmdb " << db_path;
    CHECK_EQ(mkdir(db_path, 0744), 0)
        << "mkdir " << db_path << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed. Does the lmdb already exist? ";
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }

  // Storing to db
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size;
  bool data_size_initialized = false;

  bool output_datum_size = true;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    // if (!ReadImageToDatum(root_folder + lines[line_id].first,
    //     lines[line_id].second, resize_height, resize_width, is_color, &datum)) {
    //   continue;
    // }

    // if (output_datum_size) {
    //   LOG(ERROR) << "datum size: " << datum.channels()
    //       << ", " << datum.height() << ", " << datum.width();
    //   output_datum_size = false;
    // }
    datum.set_channels(fea_len);
    datum.set_height(1);
    datum.set_width(1);
    datum.set_label(lines[line_id].second);

    // LOG(INFO) << datum.data().size();
    datum.clear_data();
    datum.clear_float_data();
    // LOG(INFO) << datum.data().size();
    // LOG(INFO) << line_id;

    for(int j = 0; j < fea_len; j++)
    {
      datum.add_float_data(features[line_id][j]);
    }

    // LOG(INFO) << datum.data().size() << data_size;
    if (!data_size_initialized) {
      data_size = datum.channels() * datum.height() * datum.width();
      data_size_initialized = true;
    } else {
      CHECK_EQ(datum.float_data_size(), data_size) << "Incorrect data field size "
          << datum.float_data_size();
    }
    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.c_str());
    string value;
    datum.SerializeToString(&value);
    string keystr(key_cstr);

    // Put in db
    if (db_backend == "leveldb") {  // leveldb
      batch->Put(keystr, value);
    } else if (db_backend == "lmdb") {  // lmdb
      mdb_data.mv_size = value.size();
      mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
      mdb_key.mv_size = keystr.size();
      mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
      CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
          << "mdb_put failed";
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }

    if (++count % 1000 == 0) {
      // Commit txn
      if (db_backend == "leveldb") {  // leveldb
        db->Write(leveldb::WriteOptions(), batch);
        delete batch;
        batch = new leveldb::WriteBatch();
      } else if (db_backend == "lmdb") {  // lmdb
        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
      } else {
        LOG(FATAL) << "Unknown db backend " << db_backend;
      }
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    if (db_backend == "leveldb") {  // leveldb
      db->Write(leveldb::WriteOptions(), batch);
      delete batch;
      delete db;
    } else if (db_backend == "lmdb") {  // lmdb
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
      mdb_close(mdb_env, mdb_dbi);
      mdb_env_close(mdb_env);
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }
    LOG(ERROR) << "Processed " << count << " files.";
  }

  for(i=0; i < lines.size(); i++)
  {
    free(features[i]);
  }

  free(features);

  return 0;
}
