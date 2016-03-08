#ifndef CAFFE_MOBILE_HPP_
#define CAFFE_MOBILE_HPP_

#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>

using std::string;
using std::vector;

namespace caffe {

class CaffeMobile {
public:
  ~CaffeMobile();

  static CaffeMobile *Get();
  static CaffeMobile *Get(const string &model_path, const string &weights_path);

  void SetMean(const string &mean_file);

  void SetMean(const vector<float> &mean_values);

  void SetScale(const float scale);

  vector<int> PredictTopK(const string &img_path, int k);
  float timePrediction(const vector<string>& img_paths);

  vector<vector<float>> ExtractFeatures(const string &img_path,
                                        const string &str_blob_names);

  shared_ptr<Net<float>> net_;
private:
  static CaffeMobile *caffe_mobile_;
  static string model_path_;
  static string weights_path_;

  CaffeMobile(const string &model_path, const string &weights_path);

  void Preprocess(const cv::Mat &img, vector<cv::Mat> *input_channels);

  void WrapInputLayer(std::vector<cv::Mat> *input_channels, int layeri = 0);

  vector<float> Forward(const string &filename);

  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  float scale_;
};

} // namespace caffe

#endif
