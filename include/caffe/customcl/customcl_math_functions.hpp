#ifndef CUSTOMCL_MATH_FUNCTIONS_HPP
#define CUSTOMCL_MATH_FUNCTIONS_HPP

#include <memory>
#include <iostream>
#include <utility>
#include <unordered_map>

#include "caffe/customcl/basic.hpp"
#include "caffe/customcl/oclobject.hpp"
#include "viennacl/ocl/context.hpp"

namespace caffe{

static std::shared_ptr<OpenCLBasic> oclobjects;
static std::shared_ptr<OpenCLProgramOneKernel> gemm_exec, transpose_exec;
static cl_int err;

// TODO: hack! instead of initalizing transpose plates in caffe model.
#define TRANSPOSE_BUFFER_DIM 4096
static char transpose_buffer[TRANSPOSE_BUFFER_DIM * TRANSPOSE_BUFFER_DIM * 8];
static void* transpose_ptr = 0;

static void customcl_setup(
    std::string cl_program = "gemm-blocking-2x2-vload4.cl", 
    std::string arithmetic = "float") {

    err = 0;
    // build options for opencl.
    std::string cl_build_options =
        "-DT=" + arithmetic +
        " -DT4=" + arithmetic + "4" + 
        " -DT8=" + arithmetic + "8" + 
        " -DT16=" + arithmetic + "16" + 
        " " + (arithmetic == "double" ? " -DSAMPLE_NEEDS_DOUBLE" : "") + 
        " " + (arithmetic == "half" ? " -DSAMPLE_NEEDS_HALF" : "");

    // clkernel name.
    std::string clkernel_path = "clkernel/";

    if(cl_program == "blocking-2-v4") {
      clkernel_path += "gemm-blocking-2x2-vload4.cl";
    }else if(cl_program == "blocking-4-v4") {
      clkernel_path += "gemm-blocking-4x4-vload4.cl";
    }else if(cl_program == "noblock-v8") {
      clkernel_path += "gemm-noblock-vload8.cl";
    }


    oclobjects = std::make_shared<OpenCLBasic>(
            "0",
            "gpu",
            "0"
        );

    std::wstring clkernel_path_w;
    clkernel_path_w.assign(clkernel_path.begin(), clkernel_path.end());

    gemm_exec = std::make_shared<OpenCLProgramOneKernel>(
          *oclobjects,
          clkernel_path_w, 
          "",
          "gemm",
          cl_build_options
        );
    
    transpose_exec = std::make_shared<OpenCLProgramOneKernel>(
          *oclobjects,
          clkernel_path_w, 
          "",
          "transpose",
          cl_build_options
        );

    transpose_ptr = (void*)clCreateBuffer(
        viennacl::ocl::current_context().handle().get(),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        TRANSPOSE_BUFFER_DIM * TRANSPOSE_BUFFER_DIM * 8,
        transpose_buffer,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);
}

template<typename Dtype>
static void customcl_gpu_gemm(const int ctx_id, const int M,
                       const int N, const int K ,
                       const Dtype* A, const Dtype* B, Dtype* C) {
  // implement transpose.
  std::cout << "addr " << B << std::endl;
  std::cout << "MNK " << M << " " << N << " " << K << std::endl;
  Dtype* transB = (Dtype*)transpose_ptr;
   
}

}

#endif
