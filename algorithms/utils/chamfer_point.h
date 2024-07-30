#pragma once

#include <algorithm>
#include <iostream>

#include "../bench/parse_command_line.h"
#include "NSGDist.h"
#include "euclidian_point.h"
#include "mips_point.h"
#include "parlay/internal/file_map.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "types.h"
#include <torch/torch.h>

#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <unistd.h>
#include <random>

#define COMPILER_BARRIER() __asm__ __volatile__ ("" ::: "memory")
std::mt19937 rng(42);

template <class Point_> struct Chamfer_Point {

  // static_assert(!std::is_floating_point<Point_>::value,
  // "Float types are NOT allowed here!");
  using T = decltype(Point_::val);
  using distanceType = float;
  constexpr static int SAMPLING_COUNT = 10;
  std::vector<int>comparable_indices;
  torch::Tensor sampled_member_vectors;
  torch::Tensor member_vectors;
  // template<typename C, typename range> friend struct Quantized_Mips_Point;

  struct parameters {
    int dims;
    int num_vectors;
    parameters() : dims(0), num_vectors(0) {}
    parameters(int dims) : dims(dims), num_vectors(1) {}
    parameters(int dims, int num_vectors)
        : dims(dims), num_vectors(num_vectors) {}
  };

  static distanceType d_min() { return Point_::d_min(); }
  static bool is_metric() { return false; }
  T operator[](long i) const {
    return *(values + i);
  } // I feel like this should probably return the ith vector
  template <bool use_samples>
  float brute(const Chamfer_Point<Point_> &x) const{
    // this distance is asymmetric! we iterate over curr vector.
    int x_num_vecs = x.params.num_vectors;
    int curr_num_vecs = params.num_vectors;
    int curr_dim = params.dims;
    float return_dist1 = 0.;
    auto find_optimal_b = [&](T* curr_vec){
      float curr_min = std::numeric_limits<float>::infinity();
      for (int j = 0; j < x_num_vecs; j++) {
        T *x_vec = x.values + j * curr_dim;
        if constexpr (std::is_same_v<Point_, Mips_Point<T>>) {
          curr_min =
              std::min(curr_min, mips_distance(curr_vec, x_vec, curr_dim));
        } else {
          curr_min =
              std::min(curr_min, euclidian_distance(curr_vec, x_vec, curr_dim));
        }
      }
      return curr_min;
    };
    if constexpr (use_samples) {
      for (int i : comparable_indices) {
        T *curr_vec = values + i * curr_dim;
        return_dist1 += find_optimal_b(curr_vec);
      }
    }else{
      for (int i = 0; i < curr_num_vecs; i++) {
        T *curr_vec = values + i * curr_dim;
        return_dist1 += find_optimal_b(curr_vec);
      }
    }
    return return_dist1;
  }

  float vectorized(const Chamfer_Point<Point_> &x) const{
    torch::NoGradGuard no_grad;
    torch::Tensor tensor_inter, max_indices;
    std::tie(tensor_inter, max_indices) = torch::max(torch::matmul(member_vectors,x.member_vectors.transpose(0, 1)), 1);
    float tensor_result = -1*torch::sum(tensor_inter).item<float>();
    return tensor_result;
  }
  class ThreadSafeVector {
public:
    void push_back(double value) {
        std::lock_guard<std::mutex> lock(mutex_);
        vec_.push_back(value);
        if(s_!="error"){
          double a = 0;
          for(auto &i:vec_) a+=i;
          std::cout<<s_<<" "<<vec_.size()<<" "<<a/vec_.size()<<"\n";
          std::cout.flush();
        }
        if(vec_.size()%1000 == 0 && s_=="error"){
          std::cout<<" DECILES PERCENTAGE ERROR \n";
          std::sort(vec_.begin(),vec_.end());
          for(int i=0;i<10;i++){
            std::cout<<vec_[ ((i + 1)*(vec_.size() - 1))/10 ]<<" ";
          }
          std::cout<<"\n";
          std::cout.flush();
        }
    }

    std::vector<double> get_all() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return vec_;
    }
    
    ThreadSafeVector(std::string s) : s_(s){}
private:
    mutable std::mutex mutex_;
    std::vector<double> vec_;
    std::string s_;
};

  float distance(const Chamfer_Point<Point_> &x) const{
    static ThreadSafeVector bruteV("baseline");
    static ThreadSafeVector vec("sampled");
    static ThreadSafeVector error("error");
    COMPILER_BARRIER();
    auto start = std::chrono::high_resolution_clock::now();
    auto brute_result = brute<false>(x);
    COMPILER_BARRIER();
    auto end = std::chrono::high_resolution_clock::now();
    COMPILER_BARRIER();
    // Calculate the duration
    std::chrono::duration<double> elapsed_brute = end - start;
    bruteV.push_back(elapsed_brute.count());
    // float return_dist2 = 0;
    // for (int i = 0; i < x_num_vecs; i++) {
    //   T *x_vec = x.values + i * curr_dim;
    //   float curr_min = std::numeric_limits<float>::infinity();
    //   for (int j = 0; j < curr_num_vecs; j++) {
    //     T *curr_vec = values + j * curr_dim;
    //     if constexpr (std::is_same_v<Point_, Mips_Point<T>>) {
    //       curr_min =
    //           std::min(curr_min, mips_distance(curr_vec, x_vec, curr_dim));
    //     } else {
    //       curr_min =
    //           std::min(curr_min, euclidian_distance(curr_vec, x_vec,
    //           curr_dim));
    //     }
    //   }
    //   return_dist2 += curr_min;
    // }
    //
    // return (return_dist1 + return_dist2) / 2;
    COMPILER_BARRIER();
    auto start_v = std::chrono::high_resolution_clock::now();
    auto tensor_result = brute<true>(x);
    COMPILER_BARRIER();
    auto end_v = std::chrono::high_resolution_clock::now();
    COMPILER_BARRIER();
    std::chrono::duration<double> elapsed_v = end_v - start_v;
    vec.push_back(elapsed_v.count());
    // disable autograd 
    // if(abs(tensor_result - brute_result) > 1e-4 ){
    //   exit(-1);
    // }
    error.push_back(((tensor_result*params.num_vectors)/comparable_indices.size() - brute_result)/brute_result);
    return tensor_result;
  }

  void prefetch() const {
    int l = (params.dims * params.num_vectors * sizeof(T) - 1) / 64 + 1;
    for (int i = 0; i < l; i++)
      __builtin_prefetch((char *)values + i * 64);
  }

  long id() const { return id_; }

  Chamfer_Point() : values(nullptr), id_(-1), params(0) {}

  Chamfer_Point(T *values, long id, parameters params)
      : values(values), id_(id), params(params) {
      // random sampling/kmeans for comparable indices
      {
        for(int i=0;i<params.num_vectors;i++)
          if(rng()%params.num_vectors < SAMPLING_COUNT)
            comparable_indices.push_back(i);
      }
      // make a tensor from the chosen indices
      torch::NoGradGuard no_grad;
      auto member_options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(false);
      member_vectors = torch::from_blob(values, {params.num_vectors, params.dims}, member_options);
      auto sampling_options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(torch::kCPU).requires_grad(false);
      sampled_member_vectors = member_vectors.index({torch::tensor(comparable_indices, torch::kInt64)});
    }

  bool operator==(const Chamfer_Point<Point_> &q) const {
    if (q.params.num_vectors != params.num_vectors) {
      return false;
    }
    for (int i = 0; i < params.dims * params.num_vectors; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }

  bool same_as(const Chamfer_Point<Point_> &q) const {
    return values == q.values;
  }

  void normalize() {
    double norm = 0.0;
    for (int j = 0; j < params.dims; j++)
      norm += values[j] * values[j];
    norm = std::sqrt(norm);
    if (norm == 0)
      norm = 1.0;
    for (int j = 0; j < params.dims; j++)
      values[j] = values[j] / norm;
  }

  static void translate_point(T *values, const Point_ &p,
                              const parameters &params) {
    for (int j = 0; j < params.dims; j++)
      values[j] = (T)p[j];
  }

  template <typename PR>
  static parameters generate_parameters(const PR &pr, int num_vectors) {
    return parameters(pr.dimension(), num_vectors);
  }

private:
  T *values;
  long id_;
  parameters params;
};
