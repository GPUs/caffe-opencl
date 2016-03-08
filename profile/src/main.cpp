#include <string.h>
#include <string>
#include <vector>
#include <memory>

#include "cmdoptions.hpp"
#include "caffe_mobile.hpp"
#include "caffe/caffe.hpp"

using namespace std;

#ifdef USE_EIGEN
#include <omp.h>
#else
#include <cblas.h>
#endif

using caffe::CaffeMobile;
using caffe::Caffe;

void setNumThreads(int numThreads) {
  int num_threads = numThreads;
#ifdef USE_EIGEN
  omp_set_num_threads(num_threads);
#else
  openblas_set_num_threads(num_threads);
#endif
}


void loadModel(string modelPath, string weightsPath) {
  CaffeMobile::Get(modelPath, 
                   weightsPath);
}

void setMeanWithMeanFile(string meanFile) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  caffe_mobile->SetMean(meanFile);
}

int setMode(string mode) {
  cout << "[mode]" << mode << endl;
  if(mode == "gpu") {
    vector<int> gpus;
    gpus.push_back(0);
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevices(gpus);  // TODO: not sure we need to set twice.
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpus[0]);
    return 0;
  }else if(mode == "cpu") {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
    return 0;
  }
  LOG(ERROR) << "caffe mode undefined: " << mode << endl;
  return 1;
}

int main(int argc, const char* argv[]) {
  auto cmdparser = make_shared<CmdParserMain>(argc, argv);
  cmdparser->parse();
  
  // Immediatly exit if user wanted to see the usage information only.
  if(cmdparser->help.isSet())
  {
    return 0;
  }
  if(setMode(cmdparser->mode.getValue())) {
    return 1;
  }

  // Set the number of threads.
  if(cmdparser->mode.getValue() == "cpu") {
    size_t numThreads = cmdparser->numThreads.getValue();
    setNumThreads(numThreads);
  }
  return 0;
}

