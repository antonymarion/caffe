/***

***/
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <stdarg.h>
#include <sys/stat.h> // for mkdir

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/opencv.hpp>

static int _MAX_PATH = 1024;

#define ACCESS access
#define MKDIR(a) mkdir((a), 0755)


using namespace caffe;
using namespace std;
using namespace cv;

static int create_directory(const char *directory);
static int get_dir_from_filename(const char *file_name, char *dir);
static int create_file(const char *file_name, const char type);
static int replace_ext_name(char *id, const char *ext);
static int create_file(const char *file_name, const char type);

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    const int num_required_args = 7;
    if (argc < num_required_args) {
      LOG(ERROR)<<
      "This program takes in a trained network and an input of images"
      " extract features of the input data produced by the net.\n"
      "Usage: extract_features_to_file  feature_extraction_proto_file"
      "  pretrained_net_param  imgaes_root_path, images_list_file"
      "  extract_feature_blob_name save_feature_root_path"
      "  [CPU/GPU] [DEVICE_ID=0]\n"
      "Note: you can extract multiple features in one pass by specifying"
      " multiple feature blob names and dataset names separated by ','."
      " The names cannot contain white space characters and the number of blobs"
      " and datasets must be equal.";
      return 1;
    }

    // choose device
    int arg_pos = num_required_args;
    if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
      LOG(ERROR)<< "Using GPU";
      int device_id = 0;
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

    // read image list
    vector<string> img_path;
    string path;
    vector<string> str;

    ifstream ifs(argv[4]);
    if(!ifs) {
      LOG(ERROR) << "Error opening file: " << string(argv[4]);
    } else {
      LOG(INFO) << "Image List file " << string(argv[4]) << " opened.";
    }

    while(getline(ifs, path)){
      vector<string>().swap(str);
      boost::split(str, path, boost::is_any_of(" "));
      img_path.push_back(str[0]);
    }
    LOG(ERROR) << "Loaded image list: " << img_path.size() << " images.";
    ifs.close();

    std::string img_dir = argv[3];
    if (img_dir[img_dir.length() - 1] != '/') {
      img_dir += "/";
    }

    // save dir
    string save_dir = argv[6];
    if (save_dir[save_dir.length() - 1] != '/') {
      save_dir += "/";
    }
    int ret = create_directory(argv[6]);
    if(0 != ret) {
      LOG(FATAL) << "Create dir " << argv[6] << " failed.";
    }

    // network init
    string feature_extraction_proto(argv[1]);
    string pretrained_binary_proto(argv[2]);
    NetParameter net_param;
    ReadProtoFromTextFile(feature_extraction_proto, &net_param);

    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(pretrained_binary_proto, &trained_net_param);

    Net<Dtype> feature_extraction_net(net_param);
    feature_extraction_net.CopyTrainedLayersFrom(trained_net_param);

    // feature blob 
    string blob_name(argv[5]);
    CHECK(feature_extraction_net.has_blob(blob_name))
        << "Unknown feature blob name " << blob_name
        << " in the network " << feature_extraction_proto;

    // Tips: The name of "Input" layer must be "data".  
    const boost::shared_ptr<Blob<Dtype> > data_blob = 
                  feature_extraction_net.blob_by_name("data");    

    if(!data_blob) {
      LOG(ERROR) << "No Input layer found..";
    }

    LOG(ERROR) << "Input layer init";

    LOG(ERROR) << data_blob->num() << " "
              << data_blob->channels() << " "
              << data_blob->width() << " "
              << data_blob->height() << " ";

    const int batch_size = data_blob->num();
    const int net_width = data_blob->width();
    const int net_height = data_blob->height();
    const int data_dim = data_blob->count() / data_blob->num();

    LOG(ERROR) << "Feature extraction...";
    char save_feature_path[_MAX_PATH];
    vector<string> feature_path;
    int cnt = 0;
    int feat_index = 1;

    for (int i = 0; i < img_path.size(); ++i) {
      path = img_dir + img_path[i];
      sprintf(save_feature_path, "%s%s", save_dir.c_str(), img_path[i].c_str());
      replace_ext_name(save_feature_path, "feat");
      feature_path.push_back(string(save_feature_path));

      Mat cv_img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
      if(!cv_img.data){
        LOG(ERROR) << "Could not open or find file." << path; 
        return false;
      }

      // resize 
      if((net_height != cv_img.rows) || (net_width != cv_img.cols )){
          resize(cv_img, cv_img, Size(net_width, net_height));
      }

      Dtype* top_data = data_blob->mutable_cpu_data();
      
      for (int y = 0; y < net_height; ++y){
        for (int x = 0; x < net_width; ++x) {
          top_data[cnt * data_dim + y * cv_img.cols + x] = 
              Dtype(cv_img.at<uchar>(y, x) / Dtype(256));
        } // for (int y = 0; y < cv_img.rows; ++y) 
      } // for (int x = 0; x < cv_img.cols; ++x)

      cnt++;
      if((cnt == batch_size) || i == img_path.size() - 1) {
        // LOG(ERROR) << feature_path.size() << " " << feature_path[0];
        feature_extraction_net.Forward();

        const boost::shared_ptr<Blob<Dtype> > feature_blob =  
                feature_extraction_net.blob_by_name(blob_name);

        const int num_features = feature_blob->num();
        const int dim_features = feature_blob->count() / num_features;

        Dtype* feature_blob_data;
        for (int n = 0; n < feature_path.size(); ++n) {
          feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob->offset(n);
          create_file(feature_path[n].c_str(), 'b');
          FILE* fp = fopen(feature_path[n].c_str(), "wb");

          // nomalization
          float norm = 0.0;
          for (int d = 0; d < dim_features; ++d) {
            norm += feature_blob_data[d] * feature_blob_data[d];
          }
          for (int d = 0; d < dim_features; ++d) {
            feature_blob_data[d]  = feature_blob_data[d] / sqrt(norm);
          }

          fwrite(feature_blob_data, dim_features*sizeof(Dtype), 1, fp);
          fclose(fp);

          LOG(ERROR) << "Extract " << feature_path[n] << " Finished(" 
                     << feat_index << "/" << img_path.size() << ").";
          feat_index++;
        } // for (int batch_index = 0; batch_index < num_features; ++batch_index)

        vector<string>().swap(feature_path);
        cnt = 0;
      } // if((cnt == batch_size) || i == img_path.size() - 1)
    } // for (int i = 0; i < img_path.size(); ++i)
}



static int create_directory(const char *directory)
{
  int i;
  int len;
  char dir[_MAX_PATH], temp_dir[_MAX_PATH];

  memcpy(temp_dir, directory, _MAX_PATH);

  len = (int)strlen(temp_dir);
  for(i = 0; i < len; i++) {
    if(temp_dir[i] == '\\')
      temp_dir[i] = '/';
  }
  if(temp_dir[len - 1] != '/') {
    temp_dir[len] = '/';
    temp_dir[len + 1] = 0;
    len++;
  }
  memset(dir, 0, _MAX_PATH);
  for(i = 0; i < len; i++) {
    dir[i] = temp_dir[i];
    if(temp_dir[i] == '/') {
      if(i > 0) {
        if(temp_dir[i - 1] == ':')
          continue;
        else {
          if(ACCESS(dir, 0) == 0)
            continue;
          else {
            /* create it */
            if(MKDIR(dir) != 0)
              return -1;
          }
        }
      }
    }
  }

  return 0;
}

static int get_dir_from_filename(const char *file_name, char *dir)
{
  int len;
  int i;

  len = (int) strlen(file_name);
  for(i = len - 1; i >= 0; i--) {
    if(file_name[i] == '\\' || file_name[i] == '/') {
      break;
    }
  }
  strcpy(dir, file_name);
  dir[i + 1] = 0;
  return 0;
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
