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
    ),
    iterations(
        *this,
        'i',
        "iterations",
        "<integer>",
        "the total number of iterations to run the benchmark",
        1
    ),
    model_path(
        *this,
        0,
        "model",
        "",
        "path to caffe model",
        "/sdcard/model/bvlc_reference_caffenet/deploy.prototxt"
    ),
    weights_path(
        *this,
        0,
        "weights",
        "",
        "weights of the caffe model",
        ""  // by default, we choose to skip loading weights for the model.
    ),
    cl_program(
        *this,
        0,
        "cl_program",
        "",
        "name of the cl_program to use",
        "blocking-2-v4"
    ),
    arithmetic(
        *this,
        0,
        "arithmetic",
        "",
        "precision of the arithmetics",
        "float"
    ),
    input(
        *this,
        0,
        "input",
        "",
        "input test image",
        "" // default no test image.
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



