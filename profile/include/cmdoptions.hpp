#ifndef _MOCHA_PROFILE_CMDOPTIONS_HPP_
#define _MOCHA_PROFILE_CMDOPTIONS_HPP_

#include "cmdparser.hpp"

// All command-line options for GEMM sample
class CmdParserMain : public CmdParserCommon
{
public:
  CmdOption<string> mode;
  
  CmdParserMain (int argc, const char** argv);

  virtual void parse ();

private:
};


#endif  // end of the include guard
