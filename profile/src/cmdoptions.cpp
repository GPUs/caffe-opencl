#include <limits>
#include <cmath>

#include "cmdoptions.hpp"

using namespace std;


#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4355)    // 'this': used in base member initializer list
#endif

CmdParserMain::CmdParserMain (int argc, const char** argv) :
    CmdParserCommon(argc, argv),
    mode(
        *this,
        'm',
        "mode",
        "",
        "which mode to run caffe (gpu / cpu / viennacl)",
        "gpu"
    ),
    numThreads(
        *this,
        't',
        "thread",
        "<integer>",
        "the number of threads (cpu mode only)",
        4
    )
{
}

#ifdef _MSC_VER
#pragma warning (pop)
#endif

void CmdParserMain::parse ()
{
    CmdParserCommon::parse();
}



