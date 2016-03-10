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
static viennacl::ocl::kernel gemm_exec, transpose_exec, copy_exec;
static cl_int err;
static std::string cl_program;
static viennacl::ocl::program cl_prog;

static bool customcl_is_setup = false;

// TODO: hack! instead of initalizing transpose plates in caffe model.
#define TRANSPOSE_BUFFER_DIM 4096
#define MAX_BUFFER_DIM (TRANSPOSE_BUFFER_DIM * TRANSPOSE_BUFFER_DIM * 8)
#define CUSTOM_GEMM_VERIFICATION false


static char host_trans_buffer[MAX_BUFFER_DIM];
static void* transpose_ptr = 0;
static char host_copy_buffer[MAX_BUFFER_DIM];
static void* copy_ptr = 0;
static char host_result_buffer[MAX_BUFFER_DIM];
static void* result_ptr = 0;

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
    copy_exec = cl_prog.get_kernel("copy"); 

    transpose_ptr = (void*)clCreateBuffer(
        viennacl::ocl::current_context().handle().get(),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        TRANSPOSE_BUFFER_DIM * TRANSPOSE_BUFFER_DIM * 8,
        host_trans_buffer,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    copy_ptr = (void*)clCreateBuffer(
        viennacl::ocl::current_context().handle().get(),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        TRANSPOSE_BUFFER_DIM * TRANSPOSE_BUFFER_DIM * 8,
        host_copy_buffer,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    result_ptr = (void*)clCreateBuffer(
        viennacl::ocl::current_context().handle().get(),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        TRANSPOSE_BUFFER_DIM * TRANSPOSE_BUFFER_DIM * 8,
        host_result_buffer,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);
}


template<typename Dtype>
static void customcl_copy_matrix(const int ctx_id, const Dtype* source, const int width, const int height,
    Dtype* target, const int output_width, const int output_height) {
   
  cl_kernel kernel = copy_exec.handle().get();

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &target);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 1, sizeof(int), &output_width);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 2, sizeof(int), &output_height);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &source);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 4, sizeof(int), &width);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 5, sizeof(int), &height);
  SAMPLE_CHECK_ERRORS(err);

  auto queue = viennacl::ocl::get_context(ctx_id).get_queue().handle().get();

  size_t local_size[2] = {16, 16};
  size_t global_size[2] = {(output_width + 16 - 1) / 16 * 16, 
                           (output_height + 16 - 1) / 16 * 16};

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

template<typename Dtype>
static void customcl_transpose_matrix(const int ctx_id, const Dtype* source, int width, int height,
    Dtype* target, int output_width, int output_height) {

  cl_kernel kernel = transpose_exec.handle().get();

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &target);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 1, sizeof(int), &output_width);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 2, sizeof(int), &output_height);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &source);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 4, sizeof(int), &width);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 5, sizeof(int), &height);
  SAMPLE_CHECK_ERRORS(err);
  
  size_t local_size[2] = {16, 16};
  size_t global_size[2] = {(output_width + 16 - 1) / 16 * 16, 
                           (output_height + 16 - 1) / 16 * 16};

  auto queue = viennacl::ocl::get_context(ctx_id).get_queue().handle().get();

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

template<typename Dtype>
inline void assertEq(Dtype a, Dtype b) {
  if(a != b) {
    std::cout << "[assert failed] a = " << a << " b = " << b << std::endl;
    throw "verification failed";
  }
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
  auto queue = viennacl::ocl::get_context(ctx_id).get_queue().handle().get();

  const int align = 32;
  int oK = (K + align - 1) / align * align;
  int oM = (M + align - 1) / align * align;
  int oN = (N + align - 1) / align * align;

  Dtype* copy_buffer = (Dtype*)copy_ptr;
  if(sizeof(Dtype) * oK * oM > MAX_BUFFER_DIM) {
    throw "customcl_gpu_gemm: maximum buffer size exceeded.";
  }

  if(sizeof(Dtype) * oN * oK > MAX_BUFFER_DIM) {
    throw "customcl_gpu_gemm: maximum buffer size exceeded.";
  }

  customcl_copy_matrix(ctx_id,
      A, K, M,
      copy_buffer, oK, oM);

#if CUSTOM_GEMM_VERIFICATION == true
  clEnqueueMapBuffer(
      queue,
      (cl_mem) copy_buffer,
      CL_TRUE,    // blocking map
      CL_MAP_READ,
      0,
      oK * oM * sizeof(Dtype),
      0, 0, 0,
      &err
  );
  SAMPLE_CHECK_ERRORS(err);

  std::cout << "[verify copy] " << std::endl;
  for(size_t i = 0; i < oK; i++) {
    for(size_t j = 0; j < oM; j++) {
      if(i < K and j < M) {
        assertEq(((Dtype*)host_copy_buffer)[j * oK + i], A[j * K + i]);
      }else{
        assertEq(((Dtype*)host_copy_buffer)[j * oK + i], (Dtype)0.);
      }
    }
  }
#endif

  Dtype* trans_buffer = (Dtype*)transpose_ptr;

  customcl_transpose_matrix(ctx_id,
      B, K, N,
      trans_buffer, oN, oK);

#if CUSTOM_GEMM_VERIFICATION == true
  clEnqueueMapBuffer(
      queue,
      (cl_mem) trans_buffer,
      CL_TRUE,    // blocking map
      CL_MAP_READ,
      0,
      N * K * sizeof(Dtype),
      0, 0, 0,
      &err
  );
  SAMPLE_CHECK_ERRORS(err);

  std::cout << "[verifying] " << std::endl;
  for(size_t i = 0; i < N; i++) {
    for(size_t j = 0; j < K; j++) {
      if(B[i * K + j] != host_trans_buffer[j * oN + i]) {
        throw "verifcation failed";
      }
    }
  }
#endif


  cl_kernel kernel = gemm_exec.handle().get();

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &copy_buffer);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &trans_buffer);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_ptr);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 3, sizeof(int), &oM);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 4, sizeof(int), &oK);
  SAMPLE_CHECK_ERRORS(err);

  err = clSetKernelArg(kernel, 5, sizeof(int), &oN);
  SAMPLE_CHECK_ERRORS(err);

  size_t local_size[2] = {16, 16};
  size_t global_size[2] = {oM / 2, oN / 2};

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

  // copy to output mem.
  customcl_copy_matrix(ctx_id,
      (Dtype*)result_ptr, oN, oM,
      C, N, M);
}

}

#endif
