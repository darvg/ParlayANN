#pragma once

#include <algorithm>
#include <fstream>
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

    std::cout << "setting up" << std::endl;

    // read num points and max degree
    uint32_t num_points;
    uint32_t d;
    reader.read((char *)(&num_points), sizeof(uint32_t));
    n = num_points;
    reader.read((char *)(&d), sizeof(uint32_t));
    dims = d;
    params = parameters(d);
    std::cout << "Detected " << num_points << " points with dimension " << d
              << std::endl;

    prefix_sums = std::shared_ptr<uint32_t[]>(
        new uint32_t[num_points + 1], std::default_delete<uint32_t[]>());

    std::vector<uint32_t> prefix_sums_vec(num_points + 1);

    assert(num_points == 500000);

    // confirm the alloc worked
    if (prefix_sums == nullptr) {
      std::cout << "ERROR: could not allocate memory for prefix sums"
                << std::endl;
      abort();
    }

    prefix_sums[0] = 0;

    volatile int tmp2;
    // std::cout << "reader position: " << reader.tellg() << std::endl;

    size_t reader_position = reader.tellg();
    // reader.seekg(0, std::ios::end);

    // std::cout << "reader end position: " << reader.tellg() << std::endl;

    // reader.seekg(reader_position);

    // read prefix sums
    for (int i = 0; i < num_points; i++) {
      // reader.read((char *)(prefix_sums.get() + i + 1), sizeof(uint32_t));
      // separate the read and write for debug
      uint32_t tmp;

      reader_position = reader.tellg();

      reader.read((char *)(&tmp), sizeof(uint32_t));

      prefix_sums_vec[i + 1] = tmp;

      // if (reader.tellg() > 500000 ){
      //   std::cout << "prev reader position: " << reader_position <<
      //   std::endl;
      // }
      // tmp2 = tmp;
      // prefix_sums[i + 1] = tmp;
    }
    // std::cout << "Reader state: " << reader.rdstate() << std::endl;

    // reader.read(reinterpret_cast<char*>(prefix_sums.get() + 1),
    //             sizeof(uint32_t) * (num_points));

    memcpy(prefix_sums.get(), prefix_sums_vec.data(),
           sizeof(uint32_t) * (num_points + 1));

    long num_bytes = dims * sizeof(T) * prefix_sums[num_points];
    T *ptr = (T *)malloc(num_bytes);
    // madvise(ptr, num_bytes, MADV_HUGEPAGE);
    values = std::shared_ptr<T[]>(ptr, std::default_delete<T[]>());
    reader.read((char *)(values.get()), num_bytes);
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

  uint32_t get_dims() const { return dims; }

  Point operator[](long i) const {
    if (i > n) {
      std::cout << "ERROR: point index out of range: " << i << " from range "
                << n << ", " << std::endl;
      abort();
    }
    int num_vectors = prefix_sums[i + 1] - prefix_sums[i];
    auto x = Point(values.get() + prefix_sums[i] * dims, i,
                   typename ChamferPointRange<T_, Point_>::Point::parameters(
                       dims, num_vectors));
    return x;
  }

  parameters params;

private:
  std::shared_ptr<T[]> values;
  std::shared_ptr<uint32_t[]> prefix_sums;
  uint32_t dims;
  size_t n;
};
