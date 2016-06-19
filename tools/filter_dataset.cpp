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

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    const int num_required_args = 5;
    if (argc < num_required_args) {
      LOG(ERROR)<<
      "This program takes in a trained network and an input of images"
      " extract features of the input data produced by the net.\n"
      "Usage: extract_features_to_file  feature_extraction_proto_file"
      "  pretrained_net_param, images_list_file"
      "  save_list_path [CPU/GPU] [DEVICE_ID=0]\n"
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

    vector<string> img_path;
    vector<string> label;
    string path;
    vector<string> str;

    ifstream ifs(argv[3]);
    if(!ifs) {
      LOG(ERROR) << "Error opening file: " << string(argv[4]);
    } else {
      LOG(INFO) << "Image List file " << string(argv[4]) << " opened.";
    }

    while(getline(ifs, path)){
      vector<string>().swap(str);
      boost::split(str, path, boost::is_any_of(" "));
      img_path.push_back(str[0]);
      label.push_back(str[1]);
    }
    LOG(ERROR) << "Loaded image list: " << img_path.size() << " images.";
    ifs.close();


    string feature_extraction_proto(argv[1]);
    string pretrained_binary_proto(argv[2]);
    NetParameter net_param;
    ReadProtoFromTextFile(feature_extraction_proto, &net_param);

    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(pretrained_binary_proto, &trained_net_param);

    Net<Dtype> feature_extraction_net(net_param);
    feature_extraction_net.CopyTrainedLayersFrom(trained_net_param);

    

    const boost::shared_ptr<Blob<Dtype> > argmax_blob =
              feature_extraction_net.blob_by_name("argmax");

    if (!argmax_blob) {
        LOG(FATAL) << "no argmax blob found.";
    }

    LOG(ERROR) << "Feature extraction...";

    const int batch_size = argmax_blob->num();

    int cnt = 0;
    int feat_index = 1;
    vector<string> feature_path;
    vector<string> feature_label;
    const Dtype* argmax_data = argmax_blob->cpu_data();

    ofstream outfile(argv[4]);


    for (int i = 0; i < img_path.size(); ++i) {
        feature_path.push_back(img_path[i]);
        feature_label.push_back(label[i]);
        cnt++;
        if((cnt == batch_size) || i == img_path.size() - 1) {
            feature_extraction_net.Forward();
            argmax_data = argmax_blob->cpu_data();
            int argmax_dim = argmax_blob->count()/argmax_blob->num();

            for(int n = 0; n < feature_path.size(); ++n) {
                outfile << feature_path[n] << " " << feature_label[n] << " " << argmax_data[n*argmax_dim] 
                        << " " << argmax_data[n*argmax_dim+1] << endl;

                LOG(ERROR) << "Extract " << feature_path[n] << " Finished(" 
                           << feat_index << "/" << img_path.size() << ").";
                feat_index++;
                //cout << feature_path[n] << " " << feature_label[n] << endl;
                //cout << argmax_data[n*argmax_dim] << " " << argmax_data[n*argmax_dim+1] << endl;
            }
            vector<string>().swap(feature_path);
            vector<string>().swap(feature_label);
            cnt = 0;
        }
    }

    outfile.close();
    return 0;
}