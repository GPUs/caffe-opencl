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

int main(int argc, const char* argv[]) {
  auto cmdparser = make_shared<CmdParserMain>(argc, argv);
  cmdparser->parse();

  // Immediatly exit if user wanted to see the usage information only.
  if(cmdparser->help.isSet())
  {
    return 0;
  }
  cout << "mode: " << cmdparser->mode.getValue() << endl;
}

