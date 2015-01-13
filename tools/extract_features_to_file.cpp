#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

static int _MAX_PATH = 1000;

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

static int create_directory(const char *directory)
{
  int i;
  int len;
  char dir[_MAX_PATH], temp_dir[_MAX_PATH];

  memcpy(temp_dir, directory, _MAX_PATH);

  len = (int)strlen(temp_dir);
  for(i=0; i<len; i++)
  {
    if(temp_dir[i] == '/')
      temp_dir[i] = '\\';
  }
  if(temp_dir[len-1] != '\\')
  {
    temp_dir[len] = '\\';
    temp_dir[len+1] = 0;
    len++;
  }
  memset(dir, 0, _MAX_PATH);
  for(i=0; i<len; i++)
  {
    dir[i] = temp_dir[i];
    if(temp_dir[i] == '\\')
    {
      if(i > 0)
      {
        if(temp_dir[i-1] == ':')
          continue;
        else
        {
          if(access(dir, 0) == 0)
            continue;
          else /* create it */
          {
            if(mkdir(dir, S_IRWXO) != 0)
              return -1;
          }
        }
      }
    }
  }

  return 0;
}


static char ** chTwoDMalloc(int iImageX, int iImageY)
{
  int i, j;
  char **pchMem = NULL;
  if((pchMem = (char **)malloc(iImageY*sizeof(char *))) == NULL)
    return NULL;
  for(i=0; i<iImageY; i++)
  {
    pchMem[i] = NULL;
    if((pchMem[i] = (char *)malloc(iImageX)) == NULL)
    {
      for(j=0; j<i; j++)
      {
        free(pchMem[j]);
      }
      free(pchMem);
      return NULL;
    }
  }
  return pchMem;
}

static int replace_ext_name(char *id, const char *ext)
{
  int i, len, idx;
  bool found = false;

  len = (int) strlen(id);
  for(i=len-1; i>=0; i--)
  {
    if(id[i] == '.')
    {
      found = true;
      break;
    }
  }

  if(!found)
    return -1;

  idx = i + 1;
  len = (int) strlen(ext);
  for(i=0; i<len; i++)
  {
    id[idx+i] = ext[i];
  }
  id[idx+len] = 0;

  return 0;
}

static int get_dir_from_filename(const char *file_name, char *dir)
{
  int len;
  int i;

  len = (int) strlen(file_name);
  for(i=len-1; i>=0; i--)
  {
    if(file_name[i] == '\\' || file_name[i] == '/')
    {
      break;
    }
  }

  strcpy(dir, file_name);
  dir[i+1] = 0;

  return 0;
}

static int create_file(const char *file_name, const char type)
{
  FILE *fp;
  char dir[_MAX_PATH];
  char mode[5];

  if(type == 'b')
  {
    strcpy(mode, "wb");
  }
  else
  {
    strcpy(mode, "w");
  }

  fp = fopen(file_name, mode);
  if(fp == NULL)
  {
    get_dir_from_filename(file_name, dir);
    create_directory(dir);
    fp = fopen(file_name, mode);
    if(fp == NULL)
      return -1;
  }
  fclose(fp);

  return 0;
}

static void TwoDMFree(int iImageY, void **pMem)
{
  int i;
  for(i=0; i<iImageY; i++)
  {
    if(pMem[i]!=NULL)
      free(pMem[i]);
  }
  free(pMem);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) 
{
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 7;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  feature_list_file  save_feature_leveldb_name1[,name2,...]  "
    "  num_mini_batches  [CPU/GPU]  [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and leveldb names seperated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and leveldbs must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  Caffe::set_phase(Caffe::TEST);

  arg_pos = 0;  // the name of the executable
  string pretrained_binary_proto(argv[++arg_pos]);

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  string extract_feature_blob_names(argv[++arg_pos]);
  vector<string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  // read feature list
  int num_feat, feat_index, ret;
  char feat_list_with_label[_MAX_PATH];
  FILE *fp;
  strcpy(feat_list_with_label, argv[++arg_pos]);

  fp = fopen(feat_list_with_label, "r");
  char c_id[128], c_label[16], **id_list;
  if(NULL == fp)
  {
      LOG(INFO) << "Open file" << feat_list_with_label << "failed";
      return -1;
  }
  else
  {
      num_feat = 0;
      while(!feof(fp))
      {
        ret = fscanf(fp, "%s %s", c_id, c_label);
        if( 2 != ret)
          break;
        else
        {
            num_feat++;
        }
      }
      fclose(fp);
  }
  id_list = chTwoDMalloc(128, num_feat);

  fp = fopen(feat_list_with_label, "r");
  if(NULL == fp)
  {
      LOG(INFO) << "Open file" << feat_list_with_label << "failed";
      return -1;
  }
  else
  {
      feat_index = 0;
      while(!feof(fp))
      {
        ret = fscanf(fp, "%s %s", id_list[feat_index], c_label);
        if(2 != ret)
          break;
        else
        {
          feat_index++;
        }
      }
      fclose(fp);
  }

  // save path
  char save_feature_path[_MAX_PATH], feature_fn[_MAX_PATH];
  strcpy(save_feature_path, argv[++arg_pos]);
  LOG(INFO)<< "Create feature path " << save_feature_path;
  ret = create_directory(save_feature_path);
  CHECK(ret == 0) << "Failed to create feature path " << save_feature_path;

  int batch_size = atoi(argv[++arg_pos]);
  int num_mini_batches = num_feat / batch_size + 1;

  // string save_feature_leveldb_names(argv[++arg_pos]);
  // vector<string> leveldb_names;
  // boost::split(leveldb_names, save_feature_leveldb_names,
  //              boost::is_any_of(","));
  // CHECK_EQ(blob_names.size(), leveldb_names.size()) <<
  //     " the number of blob names and leveldb names must be equal";
  // size_t num_features = blob_names.size();


  // for (size_t i = 0; i < num_features; i++) {
  //   CHECK(feature_extraction_net->has_blob(blob_names[i]))
  //       << "Unknown feature blob name " << blob_names[i]
  //       << " in the network " << feature_extraction_proto;
  // }

  // leveldb::Options options;
  // options.error_if_exists = true;
  // options.create_if_missing = true;
  // options.write_buffer_size = 268435456;
  // vector<shared_ptr<leveldb::DB> > feature_dbs;
  // for (size_t i = 0; i < num_features; ++i) {
  //   LOG(INFO)<< "Opening leveldb " << leveldb_names[i];
  //   leveldb::DB* db;
  //   leveldb::Status status = leveldb::DB::Open(options,
  //                                              leveldb_names[i].c_str(),
  //                                              &db);
  //   CHECK(status.ok()) << "Failed to open leveldb " << leveldb_names[i];
  //   feature_dbs.push_back(shared_ptr<leveldb::DB>(db));
  // }

  // int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Extacting Features";

  feat_index = 0;
  vector<Blob<float>*> input_vec;

  for (int batch_index = 0; batch_index < num_mini_batches && feat_index < num_feat; ++batch_index) 
  {
    feature_extraction_net->Forward(input_vec);
    const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
      ->blob_by_name(extract_feature_blob_names);
    int num_features = feature_blob->num();
    int dim_features = feature_blob->count() / num_features;
    Dtype* feature_blob_data;

    for (int n = 0; n < num_features && feat_index<num_feat; ++n) 
    {
      feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob->offset(n);

      //write to file
      sprintf(feature_fn, "%s/%s", save_feature_path, id_list[feat_index]);
      //replace_ext_name(feature_fn, "feat");
      create_file(feature_fn, 'b');

      printf("%s\n", feature_fn);

      fp = fopen(feature_fn, "wb");
      fwrite(feature_blob_data, dim_features*sizeof(Dtype), 1, fp);
      fclose(fp);

      LOG(INFO) << "Extract " << feature_fn << "Finished(" << feat_index << "/" << num_feat << ").";

      feat_index++;
    }


  } 

  TwoDMFree(num_feat, (void **)id_list);

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

