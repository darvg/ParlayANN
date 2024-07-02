#pragma once

#include <algorithm>
#include <iostream>
#include <sys/mman.h>

#include "../bench/parse_command_line.h"
#include "chamfer_point.h"
#include "parlay/internal/file_map.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "types.h"

#include <assert.h>
#include <fcntl.h>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

template <typename T_, class Point_> struct ChamferPointRange {
  using T = T_;
  using Point = Point_;
  using parameters = typename Point::parameters;

  long dimension() const { return dims; }

  ChamferPointRange() : values(std::shared_ptr<T[]>(nullptr, std::free)) {
    n = 0;
  }

  template <typename PR>
  ChamferPointRange(const PR &pr, const parameters &p) : params(p) {
    n = pr.size();
    dims = pr.dimension();
    long num_bytes = n * dims * sizeof(T);
    T *ptr = (T *)aligned_alloc(1l << 21, num_bytes);
    madvise(ptr, num_bytes, MADV_HUGEPAGE);
    values = std::shared_ptr<T[]>(ptr, std::free);
    T *vptr = values.get();
    parlay::parallel_for(0, n, [&](long i) {
      Point::translate_point(vptr + i * dims, pr[i], params);
    });
  }

  template <typename PR>
  ChamferPointRange(PR &pr)
      : ChamferPointRange(pr, Point::generate_parameters(pr)) {}

  ChamferPointRange(char *filename)
      : values(std::shared_ptr<T[]>(nullptr, std::free)) {
    if (filename == NULL) {
      n = 0;
      dims = 0;
      return;
    }
    std::ifstream reader(filename);
    assert(reader.is_open());

    // read num points and max degree
    unsigned int num_points;
    unsigned int d;
    reader.read((char *)(&num_points), sizeof(unsigned int));
    n = num_points;
    reader.read((char *)(&d), sizeof(unsigned int));
    dims = d;
    params = parameters(d);
    std::cout << "Detected " << num_points << " points with dimension " << d
              << std::endl;

    prefix_sums = std::make_shared<unsigned int[]>(num_points + 1);
    reader.read((char *)(&prefix_sums),
                sizeof(unsigned int) * (num_points + 1));

    long num_bytes = n * dims * sizeof(T) * prefix_sums[num_points];
    T *ptr = (T *)malloc(num_bytes);
    madvise(ptr, num_bytes, MADV_HUGEPAGE);
    values = std::shared_ptr<T[]>(ptr, std::free);
    reader.read((char *)(&values), sizeof(T) * (num_bytes));
    // size_t BLOCK_SIZE = 1000000;
    // size_t index = 0;
    // while (index < n) {
    //   size_t floor = index;
    //   size_t ceiling = index + BLOCK_SIZE <= n ? index + BLOCK_SIZE : n;
    //   T *data_start = new T[(ceiling - floor) * dims];
    //   reader.read((char *)(data_start), sizeof(T) * (ceiling - floor) *
    //   dims); T *data_end = data_start + (ceiling - floor) * dims;
    //   parlay::slice<T *, T *> data = parlay::make_slice(data_start,
    //   data_end); int data_bytes = dims * sizeof(T);
    //   parlay::parallel_for(floor, ceiling, [&](size_t i) {
    //     for (int j = 0; j < dims; j++)
    //       values.get()[i * aligned_dims + j] = data[(i - floor) * dims + j];
    //     // std::memmove(values.get() + i*aligned_dims, data.begin() +
    //     // (i-floor)*dims, data_bytes);
    //   });
    //   delete[] data_start;
    //   index = ceiling;
    // }
  }

  size_t size() const { return n; }

  unsigned int get_dims() const { return dims; }

  Point operator[](long i) const {
    if (i > n) {
      std::cout << "ERROR: point index out of range: " << i << " from range "
                << n << ", " << std::endl;
      abort();
    }
    int num_vectors = prefix_sums[i + 1] - prefix_sums[i];
    return Point(values.get() + prefix_sums[i] * dims, i,
                 Point::parameters(dims, num_vectors));
  }

  parameters params;

private:
  std::shared_ptr<T[]> values;
  std::shared_ptr<unsigned int[]> prefix_sums;
  unsigned int dims;
  size_t n;
};
