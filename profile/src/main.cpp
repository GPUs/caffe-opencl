#include <string.h>
#include <string>
#include <vector>
#include <memory>
#include <ctime>

#include "basic.hpp"
#include "cmdoptions.hpp"
#include "caffe_mobile.hpp"
#include "caffe/caffe.hpp"
#include "energy.hpp"

#include "caffe/customcl/customcl_math_functions.hpp"

using namespace std;

#ifdef USE_EIGEN
#include <omp.h>
#else
#include <cblas.h>
#endif

using caffe::CaffeMobile;
using caffe::Blob;
using caffe::Caffe;


shared_ptr<CmdParserMain> cmdparser;

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
    Caffe::SetDevices(gpus);  // TODO: not sure we need to set twice.
    Caffe::set_mode(Caffe::GPU);
    Caffe::Get().USE_CUSTOM_GPU_KERNEL = true;

    return 0;
  }else if(mode == "viennacl") {
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
  cmdparser = make_shared<CmdParserMain>(argc, argv);
  cmdparser->parse();
  // Immediatly exit if user wanted to see the usage information only.
  if(cmdparser->help.isSet())
  {
    return 0;
  }

  std::cout << "[load] set mode" << std::endl; 
  if(setMode(cmdparser->mode.getValue())) {
    return 1;
  }
  
  // Set the number of threads.
  if(cmdparser->mode.getValue() == "cpu") {
    size_t numThreads = cmdparser->numThreads.getValue();
    setNumThreads(numThreads);
  }


  // load model.
  std::cout << "[load] model" << std::endl;
  CaffeMobile* mobile = CaffeMobile::Get(
      cmdparser->model_path.getValue(), 
      cmdparser->weights_path.getValue()
  );
  
  // run feedforward.
  cout << "[test] warm up run" << endl;
  mobile->net_->Forward(vector<Blob<float>*>(), 0);
  cout << "[test] running" << endl;
  size_t iterations = cmdparser->iterations.getValue();
  std::pair<double, double> first_time_energy = Mocha::getCurrentEnergy();
  double start, end;
  if(cmdparser->input.getValue() == "") { // no image input.
    start = time_stamp();
    for(size_t it = 0; it < iterations; it++) {
      mobile->net_->Forward(vector<Blob<float>*>(), 0);
    }
    end = time_stamp();
  }else{
    mobile->loadImage(cmdparser->input.getValue());
    start = time_stamp();
    for(size_t it = 0; it < iterations; it++) {
      mobile->net_->ForwardPrefilled();
    }
    end = time_stamp();
  }
  //auto queue = viennacl::ocl::get_context(0).get_queue().handle().get();
  //cl_int err = clFinish(queue);
  //SAMPLE_CHECK_ERRORS(err);

  cout << "[test] quiescence" << endl;
  sleep(2); 
  std::pair<double, double> last_time_energy = Mocha::getCurrentEnergy();

  // aggregate result.
  double time = (end - start) / iterations;
  std::cout << "energy before: " << first_time_energy.second << std::endl;
  std::cout << "energy after: " << last_time_energy.second << std::endl;
  double energy = (last_time_energy.second - first_time_energy.second) / 1e3 / iterations;
  double power = energy / time;
  cout << "feedforward" << endl;
  cout << "\ttime = " << time << " secs" << endl;
  cout << "\tenergy = " << energy << " mJ" << endl;
  cout << "\tpower = " << energy / time << " mW" << endl;

  // clean up.
  delete mobile;
  return 0;
}

