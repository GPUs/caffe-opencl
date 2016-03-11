#include <algorithm>
#include <string>
#include <vector>
#include <iostream>

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"

#include "caffe_mobile.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>

using std::clock;
using std::clock_t;
using std::string;
using std::vector;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::MemoryDataLayer;

namespace caffe {


template <typename T> vector<int> argmax(vector<T> const &values, int N) {
  vector<size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));
  std::partial_sort(indices.begin(), indices.begin() + N, indices.end(),
                    [&](size_t a, size_t b) { return values[a] > values[b]; });
  return vector<int>(indices.begin(), indices.begin() + N);
}

CaffeMobile *CaffeMobile::caffe_mobile_ = 0;
string CaffeMobile::model_path_ = "";
string CaffeMobile::weights_path_ = "";

CaffeMobile *CaffeMobile::Get() {
  CHECK(caffe_mobile_);
  return caffe_mobile_;
}

CaffeMobile *CaffeMobile::Get(const string &model_path,
                              const string &weights_path) {
  if (!caffe_mobile_ || model_path != model_path_ ||
      weights_path != weights_path_) {
    caffe_mobile_ = new CaffeMobile(model_path, weights_path);
    model_path_ = model_path;
    weights_path_ = weights_path;
  }
  return caffe_mobile_;
}

CaffeMobile::CaffeMobile(const string &model_path, const string &weights_path) {
  CHECK_GT(model_path.size(), 0) << "Need a model definition to score.";
  // CHECK_GT(weights_path.size(), 0) << "Need model weights to score.";

  clock_t t_start = clock();
  net_.reset(new Net<float>(model_path, caffe::TEST, Caffe::GetDefaultDevice()));
  if(weights_path != "") {
    net_->CopyTrainedLayersFrom(weights_path);
  }
  clock_t t_end = clock();
  std::cout << "Loading time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC
          << " ms." << std::endl;

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float> *input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  scale_ = 0.0;
}

CaffeMobile::~CaffeMobile() { net_.reset(); }

void CaffeMobile::SetMean(const vector<float> &mean_values) {
  CHECK_EQ(mean_values.size(), num_channels_)
      << "Number of mean values doesn't match channels of input layer.";

  cv::Scalar channel_mean(0);
  double *ptr = &channel_mean[0];
  for (int i = 0; i < num_channels_; ++i) {
    ptr[i] = mean_values[i];
  }
  mean_ = cv::Mat(input_geometry_, (num_channels_ == 3 ? CV_32FC3 : CV_32FC1),
                  channel_mean);
}

void CaffeMobile::SetMean(const string &mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float *data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

void CaffeMobile::SetScale(const float scale) {
  CHECK_GT(scale, 0);
  scale_ = scale;
}

void CaffeMobile::Preprocess(const cv::Mat &img,
                             std::vector<cv::Mat> *input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  if (!mean_.empty()) {
    cv::subtract(sample_float, mean_, sample_normalized);
  } else {
    sample_normalized = sample_float;
  }

  if (scale_ > 0.0) {
    sample_normalized *= scale_;
  }

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float *>(input_channels->at(0).data) ==
        net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}

void CaffeMobile::WrapInputLayer(std::vector<cv::Mat> *input_channels, int layeri) {
  Blob<float> *input_layer = net_->input_blobs()[layeri];

  int width = input_layer->width();
  int height = input_layer->height();
  float *input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void CaffeMobile::loadImage(const string& filename) {
  cv::Mat img = cv::imread(filename, -1);
  CHECK(!img.empty()) << "Unable to decode image " << filename;

  Blob<float> *input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_, input_geometry_.height,
                       input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

}

vector<float> CaffeMobile::Forward(const string &filename) {
  clock_t t_start = clock();
  net_->ForwardPrefilled();
  clock_t t_end = clock();
  VLOG(1) << "Forwarding time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC
          << " ms.";

  /* Copy the output layer to a std::vector */
  Blob<float> *output_layer = net_->output_blobs()[0];
  const float *begin = output_layer->cpu_data();
  const float *end = begin + output_layer->channels();
  return vector<float>(begin, end);
}

vector<int> CaffeMobile::PredictTopK(const string &img_path, int k) {
  const vector<float> probs = Forward(img_path);
  k = std::min<int>(std::max(k, 1), probs.size());
  return argmax(probs, k);
}

float CaffeMobile::timePrediction(const vector<string> &img_paths) {
  float total_time = 0.;
  const size_t batchsize = net_->input_blobs().size();

  for(size_t bi = 0; bi < img_paths.size(); bi += batchsize) { // batching.
    for(size_t i = bi; i < bi + batchsize; i++) {
      if(i >= img_paths.size()) break;
      auto img_path = img_paths[i];
      cv::Mat img = cv::imread(img_path, -1);
      CHECK(!img.empty()) << "Unable to decode image " << img_path;
      Blob<float> *input_layer = net_->input_blobs()[i - bi];

      input_layer->Reshape(1, num_channels_, input_geometry_.height,
                           input_geometry_.width);
      net_->Reshape();

      vector<cv::Mat> input_channels;
      WrapInputLayer(&input_channels, i - bi);
      Preprocess(img, &input_channels);

    }

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    net_->ForwardPrefilled();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = end - start;

    total_time += fp_ms.count();
  }

  return total_time;
}

} // namespace caffe


