#ifndef CUSTOMCL_MATH_FUNCTIONS_HPP
#define CUSTOMCL_MATH_FUNCTIONS_HPP

#include <memory>
#include <cmath>
#include <iostream>
#include <utility>
#include <unordered_map>
#include <fstream>
#include <streambuf>

#include "caffe/customcl/basic.hpp"
#include "caffe/customcl/oclobject.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/backend.hpp"

namespace caffe{

static std::shared_ptr<OpenCLBasic> oclobjects;
static viennacl::ocl::kernel gemm_exec, transpose_exec;
static cl_int err;
static std::string cl_program;
static viennacl::ocl::program cl_prog;

static bool customcl_is_setup = false;

// TODO: hack! instead of initalizing transpose plates in caffe model.
#define TRANSPOSE_BUFFER_DIM 4096
static char transpose_buffer[TRANSPOSE_BUFFER_DIM * TRANSPOSE_BUFFER_DIM * 8];
static void* transpose_ptr = 0;

static void customcl_setup(
    std::string cl_program = "blocking-2-v4", 
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
    
    caffe::cl_program = cl_program;
    if(cl_program == "blocking-2-v4") {
      clkernel_path += "gemm-blocking-2x2-vload4.cl";
    }else if(cl_program == "blocking-4-v4") {
      clkernel_path += "gemm-blocking-4x4-vload4.cl";
    }else if(cl_program == "noblock-v8") {
      clkernel_path += "gemm-noblock-vload8.cl";
    }

    std::ifstream kernel_file(clkernel_path);
    std::string kernel_str((std::istreambuf_iterator<char>(kernel_file)),
                           std::istreambuf_iterator<char>());
    viennacl::ocl::current_context().build_options(
        "-DT=" + arithmetic +
        " -DT4=" + arithmetic + "4" + 
        " -DT8=" + arithmetic + "8" + 
        " -DT16=" + arithmetic + "16" + 
        " " + (arithmetic == "double" ? " -DSAMPLE_NEEDS_DOUBLE" : "") + 
        " " + (arithmetic == "half" ? " -DSAMPLE_NEEDS_HALF" : ""));
        

    cl_prog = viennacl::ocl::get_context(0).add_program(
        kernel_str, "gemm_program");

    gemm_exec = cl_prog.get_kernel("gemm");
    transpose_exec = cl_prog.get_kernel("transpose"); 

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
  //std::cout << "addr " << B << std::endl;
  //std::cout << "MNK " << M << " " << N << " " << K << std::endl;
  if(!customcl_is_setup) {
    caffe::customcl_setup();
    customcl_is_setup = true;
  }
  Dtype* transB = (Dtype*)transpose_ptr;
  

  //transpose_exec.arg(0, viennacl::ocl::handle<Dtype*>(transB, 
  //      viennacl::ocl::current_context()));
  cl_kernel kernel = transpose_exec.handle().get();

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &transB);
  SAMPLE_CHECK_ERRORS(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
  SAMPLE_CHECK_ERRORS(err);
  err = clSetKernelArg(kernel, 2, sizeof(int), &K);
  SAMPLE_CHECK_ERRORS(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &N);
  SAMPLE_CHECK_ERRORS(err);
  
  size_t local_size[2];
  local_size[0] = 16;
  local_size[1] = 16;

  size_t global_size[2]; // TODO: ceil.
  global_size[0] = int(double(K) / local_size[0] + 1) * local_size[0];
  global_size[1] = int(double(N) / local_size[1] + 1) * local_size[1];


  auto queue = viennacl::ocl::get_context(0).get_queue().handle().get();
  err = clEnqueueNDRangeKernel(
      queue,
      kernel,
      2,
      0,
      global_size,
      local_size,
      0, 0, NULL
  );
  SAMPLE_CHECK_ERRORS(err);

  err = clFinish(queue);
  SAMPLE_CHECK_ERRORS(err);

}

}

#endif
