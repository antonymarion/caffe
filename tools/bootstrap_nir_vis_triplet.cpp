#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  
#include <string>
#include <utility>
#include <vector>
#include <stdlib.h> 
#include <time.h> 
#include <math.h>
#include <boost/format.hpp>     
#include <boost/tokenizer.hpp>     
#include <boost/algorithm/string.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace std;
using boost::replace_all;

DEFINE_string(alpha, "400", "The threshold");
DEFINE_int32(num_pos_sample, 100, "The number of positive samples");
DEFINE_int32(num_neg_sample, 800, "The number of negative samples");

int _MAX_PATH = 10000;

// get size of features
int get_file_size(const string filename) {
    int fsize;
    FILE *fp;

    fp = fopen(filename.c_str(), "rb");
    if(fp == NULL) {
        fsize = 0;
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    fsize = ftell(fp);
    fclose(fp);

    return fsize;
}

unsigned char * read_file_buffer(const string filename, int &size) {
    int fsize;
    FILE *fp;
    unsigned char *pnchBuf = NULL;

    size = 0;

    fsize = get_file_size(filename);
    
    if(fsize == 0)
        return NULL;

    if((fp = fopen(filename.c_str(), "rb")) != NULL )
    {
        pnchBuf = (unsigned char *) malloc(fsize * sizeof(unsigned char));
        fread(pnchBuf, fsize, 1, fp);
        fclose(fp);
    }

    size = fsize;
    return pnchBuf;
}

// L2 distance compute
static float compute_L2_distance(float* feat1, float* feat2, int feat_len)
{
  float distance = 0;
  float norm_feat1 = 0;
  float norm_feat2 = 0;
  // for (int i = 0; i < feat_len; ++i)
  // {
  //   norm_feat1 += feat1[i] * feat1[i];
  //   norm_feat2 += feat2[i] * feat2[i];  
  // }

  // norm_feat1 = sqrt(norm_feat1);
  // norm_feat2 = sqrt(norm_feat2);

  for (int i = 0; i < feat_len; ++i )
  {
    distance += sqrt((feat1[i] - feat2[i])*(feat1[i] - feat2[i]));
  }
  return distance;
}

int ReadFeatureList(const string feature_list, vector<pair<string, int> > &list) { 
    ifstream ifs(feature_list.c_str());

    if(!ifs) {
        cout << "Error opening file: " << feature_list << endl;
        return -1;
    } else {
        cout << "Feature list file " << feature_list << " opened." << endl;
    }

    string filename;
    int label;

    while(ifs >> filename >> label) {
        list.push_back(make_pair(filename, label));
    }

    cout << "Load feature list: " << list.size() << " features." << endl;
    ifs.close();
    return 0;
}

int LoadFeatures(const string root_path, vector<pair<string, int> > &list, float **features, int &feat_len) {
    const int num = list.size();
    int feat_size;

#ifdef _OPENMP
    const int num_thread = 2*omp_get_num_procs();
    omp_set_num_threads(num_thread);
    #pragma omp parallel for
#endif
    for (int i = 0; i < num; ++i) {
        string filename = root_path + list[i].first;
        replace_all(filename, ".jpg", ".feat");
        replace_all(filename, ".bmp", ".feat");

        features[i] = (float *) read_file_buffer(filename, feat_size);        

        if(i % 10000 == 0) {
            cout << "Load " << i << " features. " << endl;
        }
    }

    feat_len = feat_size / sizeof(float);
    cout << "Loading features is finished. Feature length: " << feat_len << endl;

    return 0;
}

int main(int argc, char** argv) {

  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  // bootstrap_triplet_pairs feature_dir feature_list save_pairs_data_list
  const int num_required_args = 3;
  string probe(argv[2]);
  string gallary(argv[3]);
  
  // read feature list
  vector<pair<string, int> > probe_lines;
  vector<pair<string, int> > gallary_lines;
  string filename;
  int label;

  ReadFeatureList(probe, probe_lines);
  ReadFeatureList(gallary, gallary_lines);

  // load features
  string root_folder(argv[1]);
  float **probe_features;
  float **gallary_features;
  int feat_len;

  probe_features = (float **) malloc(probe_lines.size() * sizeof(float *));
  gallary_features = (float **) malloc(gallary_lines.size() * sizeof(float *));

  LoadFeatures(root_folder, probe_lines, probe_features, feat_len);
  LoadFeatures(root_folder, gallary_lines, gallary_features, feat_len);

  float alpha = atof(FLAGS_alpha.c_str());
  int num_pos_sample = FLAGS_num_pos_sample;
  int num_neg_sample = FLAGS_num_neg_sample;

  LOG(ERROR) << "alpha = " << alpha << ", num_pos = " << num_pos_sample << ", num_neg = " << num_neg_sample;
  
  string save_path(argv[4]);
  string save_path_anchor = save_path + "_anchor.txt";
  string save_path_pos = save_path + "_pos.txt";
  string save_path_neg = save_path + "_neg.txt";
  std::ofstream outfile_anchor(save_path_anchor.c_str());
  std::ofstream outfile_pos(save_path_pos.c_str());
  std::ofstream outfile_neg(save_path_neg.c_str());
  
  // nir-vis-nir
  for(int idx_anchor = 0; idx_anchor < probe_lines.size(); ++idx_anchor) {
    float* feat_anchor = probe_features[idx_anchor];
    int label_anchor = probe_lines[idx_anchor].second;
    for(int idx_pos = 0; idx_pos < gallary_lines.size(); ++idx_pos) {
        int label_pos = gallary_lines[idx_pos].second;
        if(label_anchor != label_pos)
            continue;
        float* feat_pos = gallary_features[idx_pos];
        float score_pos = compute_L2_distance(feat_anchor, feat_pos, 256);

        int count = 0;
        while(count < num_neg_sample) {
            int idx_neg = rand() % probe_lines.size();
            int label_neg = probe_lines[idx_neg].second;
            if(label_anchor == label_neg)
                continue;

            float* feat_neg = probe_features[idx_neg];
            float score_neg = compute_L2_distance(feat_anchor, feat_neg, 256);

            if(score_pos + alpha > score_neg) {
                outfile_anchor << probe_lines[idx_anchor].first << " " << probe_lines[idx_anchor].second << endl;
                outfile_pos << gallary_lines[idx_pos].first << " " << gallary_lines[idx_pos].second << endl;
                outfile_neg << probe_lines[idx_neg].first << " " << probe_lines[idx_neg].second << endl;
            }
            count++;
        }
    }
  }

  // vis-nir-vis
  for(int idx_anchor = 0; idx_anchor < gallary_lines.size(); ++idx_anchor) {
    float* feat_anchor = gallary_features[idx_anchor];
    int label_anchor = gallary_lines[idx_anchor].second;
    for(int idx_pos = 0; idx_pos < probe_lines.size(); ++idx_pos) {
        int label_pos = probe_lines[idx_pos].second;
        if(label_anchor != label_pos)
            continue;
        float* feat_pos = probe_features[idx_pos];
        float score_pos = compute_L2_distance(feat_anchor, feat_pos, 256);

        int count = 0;
        while(count < num_neg_sample) {
            int idx_neg = rand() % gallary_lines.size();
            int label_neg = gallary_lines[idx_neg].second;
            if(label_anchor == label_neg)
                continue;

            float* feat_neg = gallary_features[idx_neg];
            float score_neg = compute_L2_distance(feat_anchor, feat_neg, 256);

            if(score_pos < score_neg && score_pos + alpha > score_neg) {
                outfile_anchor << gallary_lines[idx_anchor].first << " " << gallary_lines[idx_anchor].second << endl;
                outfile_pos << probe_lines[idx_pos].first << " " << probe_lines[idx_pos].second << endl;
                outfile_neg << gallary_lines[idx_neg].first << " " << gallary_lines[idx_neg].second << endl;
            }
            count++;
        }
    }
  }


  outfile_anchor.close();
  outfile_pos.close();
  outfile_neg.close();



}