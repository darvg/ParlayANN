#pragma once

#include <algorithm>
#include <iostream>

#include "../bench/parse_command_line.h"
#include "NSGDist.h"
#include "euclidian_point.h"
#include "mips_point.h"
#include "threadlocal.h"
#include "parlay/internal/file_map.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "types.h"

#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <unistd.h>

#include <cblas.h>

auto chamfer_buffer = threadlocal::buffer<float, 300 * 300 * 4, 161>();

template <class Point_> struct Chamfer_Point {

  // static_assert(!std::is_floating_point<Point_>::value,
  // "Float types are NOT allowed here!");
  using T = decltype(Point_::val);
  using distanceType = float;
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

  float distance(const Chamfer_Point<Point_> &x) const {
    // this distance is asymmetric! we iterate over curr vector.
    int x_num_vecs = x.params.num_vectors;
    int curr_num_vecs = params.num_vectors;
    int curr_dim = params.dims;
    float return_dist1 = 0.;
    float return_dist2 = 0.;
    for (int i = 0; i < curr_num_vecs; i++) {
      T *curr_vec = values + i * curr_dim;
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
      return_dist1 += curr_min;
    }
    return return_dist1;
    // do a matmul to get pairwise distances
    if constexpr (std::is_same_v<Point_, Mips_Point<float>>) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, curr_num_vecs,
                  x_num_vecs, curr_dim, -1.0, values, curr_dim, x.values,
                  curr_dim, 0.0, chamfer_buffer.get(), x_num_vecs);
      for (int i = 0; i < curr_num_vecs; i++) {
        return_dist2 += *std::min_element(
            chamfer_buffer.get() + i * x_num_vecs,
            chamfer_buffer.get() + (i + 1) * x_num_vecs);
      }
    } else {
      raise("Not implemented");
    }
    // if(abs(return_dist2 - return_dist1) > 1e-4){
    //   std::cout<<return_dist2<<" "<<return_dist1<<std::endl;
    //   exit(-1);
    // }else{
    //   std::cout<<"OKAY!\n";
    // }

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
    return return_dist1;
  }

  void prefetch() const {
    int l = (params.dims * params.num_vectors * sizeof(T) - 1) / 64 + 1;
    for (int i = 0; i < l; i++)
      __builtin_prefetch((char *)values + i * 64);
  }

  long id() const { return id_; }

  Chamfer_Point() : values(nullptr), id_(-1), params(0) {}

  Chamfer_Point(T *values, long id, parameters params)
      : values(values), id_(id), params(params) {}

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
