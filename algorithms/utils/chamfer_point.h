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
#ifdef INVERT_CHAMFER
    return x.distance_impl(this*);
#elif defined SUM_CHAMFER
    return x.distance_impl(this*) + distance_impl(x);
#else
    return  distance_impl(x);
#endif
  }
  float distance_impl(const Chamfer_Point<Point_> &x) const {
    // this distance is asymmetric! we iterate over curr vector.
    float factor = 1;
    int x_num_vecs = x.params.num_vectors;
#ifndef CHAMFER_SAMPLING
    int curr_num_vecs = params.num_vectors;
#elif
    constexpr int32_t sampling_count = 50;
    factor = (1.0 * params.num_vectors) / std::min(params.num_vectors, sampling_count);
    int curr_num_vecs = std::min(params.num_vectors, sampling_count);
#endif
    int curr_dim = params.dims;
    float return_dist1 = 0.;

    // do a matmul to get pairwise distances
    if constexpr (std::is_same_v<Point_, Mips_Point<float>>) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, curr_num_vecs,
                  x_num_vecs, curr_dim, -1.0, values, curr_dim, x.values,
                  curr_dim, 0.0, chamfer_buffer.get(), x_num_vecs);
      for (int i = 0; i < curr_num_vecs; i++) {
        return_dist1 += *std::min_element(
            chamfer_buffer.get() + i * x_num_vecs,
            chamfer_buffer.get() + (i + 1) * x_num_vecs);
      }
    } else {
      raise("Not implemented");
    }

    return return_dist1 * factor;
  }

  void prefetch() const {
#ifndef CHAMFER_SAMPLING
    int l = (params.dims * params.num_vectors * sizeof(T) - 1) / 64 + 1;
#else
    int l = (params.dims * std::min(params.num_vectors, sampling_count) * sizeof(T) - 1) / 64 + 1;
#endif
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

  void randomize(){
    std::mt19937 rng(id);

    std::vector <int32_t> permutation (params.num_vectors);
    std::vector <int32_t> rpermutation (params.num_vectors);
	  std::vector <int32_t> cpermutation (params.num_vectors);

    for(int i = 0; i < params.num_vectors; i++)
      permutation[i] = rpermutation[i] = cpermutation[i] = i;
    
    std::shuffle(permutation.begin(), permutation.end(), rng);

    for(int i = 0; i < params.num_vectors; i++){

	    int32_t location_to_swap_with = rpermutation[permutation[i]];

      if(location_to_swap_with == i) continue;

	    auto swap_memory_location = [](void* const a, void* const b, const size_t n ){
  		  unsigned char* p;
  		  unsigned char* q;
  		  unsigned char* const sentry = (unsigned char*)a + n;
  		  for ( p = (unsigned char*)a, q = (unsigned char*)b; p < sentry; ++p, ++q ) {
     			const unsigned char t = *p;
     			*p = *q;
     			*q = t;
  		  }
	    };

 	    int32_t vector_size = params.dims;
	    swap_memory_location(values + i * vector_size , values + location_to_swap_with * vector_size, vector_size);

	    rpermutation[cpermutation[i]] = location_to_swap_with;
	    rpermutation[cpermutation[location_to_swap_with]] = i;
	    std::swap(cpermutation[i], cpermutation[location_to_swap_with]);
    }
  }
private:
  T *values;
  long id_;
  parameters params;
#ifdef CHAMFER_SAMPLING
  constexpr int32_t sampling_count = 50;
#endif
};
