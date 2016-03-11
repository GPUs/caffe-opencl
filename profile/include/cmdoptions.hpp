#ifndef _MOCHA_PROFILE_CMDOPTIONS_HPP_
#define _MOCHA_PROFILE_CMDOPTIONS_HPP_

#include "cmdparser.hpp"

// All command-line options for GEMM sample
class CmdParserMain : public CmdParserCommon
{
public:
  CmdOption<string> mode;
  CmdOption<string> input;
  CmdOption<size_t> numThreads;
  CmdOption<size_t> iterations;
  
  CmdOption<string> cl_program, arithmetic;

  CmdOption<string> model_path, weights_path;

  CmdParserMain (int argc, const char** argv);

  virtual void parse ();

private:
};


#endif  // end of the include guard
