// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>
#include <set>

#include "beamSearch.h"
#include "csvfile.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parse_results.h"
#include "stats.h"
#include "types.h"

template <typename Point, typename PointRange, typename QPointRange,
          typename indexType>
nn_result checkRecall(Graph<indexType> &G, PointRange &Base_Points,
                      PointRange &Query_Points, QPointRange &Q_Base_Points,
                      QPointRange &Q_Query_Points, groundTruth<indexType> GT,
                      bool random, long start_point, long k, QueryParams &QP,
                      bool verbose) {
  if (GT.size() > 0 && k > GT.dimension()) {
    std::cout << k << "@" << k << " too large for ground truth data of size "
              << GT.dimension() << std::endl;
    abort();
  }

  parlay::sequence<parlay::sequence<indexType>> all_ngh;

  parlay::internal::timer t;
  float query_time;
  stats<indexType> QueryStats(Query_Points.size());
  QueryStats.clear();
  // to help clear the cache between runs
  auto volatile xx = parlay::random_permutation<long>(5000000);
  t.next_time();
  if (random) {
    all_ngh = beamSearchRandom<Point, PointRange, indexType>(
        Query_Points, G, Base_Points, QueryStats, QP);
  } else {
    all_ngh = qsearchAll<Point, PointRange, QPointRange, indexType>(
        Query_Points, Q_Query_Points, G, Base_Points, Q_Base_Points, QueryStats,
        start_point, QP);
  }
  query_time = t.next_time();

  float recall = 0.0;
  float recall_1_100 = 0.0;
  float ndcg = 0.0;
  float max_max_approximation = 0.0;
  float max_mean_approximation = 0.0;
  float max_avg_approximation = 0.0;
  float mean_max_approximation = 0.0;
  float mean_mean_approximation = 0.0;
  float mean_avg_approximation = 0.0;
  // TODO deprecate this after further testing
  bool dists_present = true;
  if (GT.size() > 0 && !dists_present) {
    size_t n = Query_Points.size();
    int numCorrect = 0;
    for (indexType i = 0; i < n; i++) {
      std::set<indexType> reported_nbhs;
      for (indexType l = 0; l < k; l++)
        reported_nbhs.insert((all_ngh[i])[l]);
      for (indexType l = 0; l < k; l++) {
        if (reported_nbhs.find((GT.coordinates(i, l))) != reported_nbhs.end()) {
          numCorrect += 1;
        }
      }
    }
    recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
  } else if (GT.size() > 0 && dists_present) {
    size_t n = Query_Points.size();

    float numCorrect = 0;
    float numCorrect_1_100 = 0;

    for (indexType i = 0; i < n; i++) {
      float mean_approx = 0.0;
      float max_approx = 0.0;
      float avg_approx_num = 0.0;
      float avg_approx_den = 0.0;
      float avg_approx = 0.0;
      parlay::sequence<int> results_with_ties;
      for (indexType l = 0; l < k; l++) {
        if(GT.coordinates(i, l) < 1e9)
          results_with_ties.push_back(GT.coordinates(i, l));
      }
      Point qp = Query_Points[i];
      std::set<int> reported_nbhs;
      // create 2 vector of sorted points in all_ngh[i] by their distance to q, and points in GT.coordinates(i,l)
      // Calculate the maximum, mean, and average of the ratio of distances 
      // between corresponding elements in two sorted vectors:
      // 1. Sort the points in all_ngh[i] based on their distance to the query point qp.
      // 2. Sort the points in GT.coordinates(i, l) similarly.
      // 3. Compute the ratio of distances for each index and calculate the max, mean, and avg of sums.
      // This helps evaluate the quality of the nearest neighbor approximation.

      // Sort the points in all_ngh[i] based on their distance to the query point qp.
      std::vector<std::pair<float, indexType>> sorted_ngh;
      for (indexType l = 0; l < QP.k; l++) {
        float dist = qp.distance(Base_Points[(all_ngh[i])[l]]);
        sorted_ngh.push_back(std::make_pair(dist, (all_ngh[i])[l]));
      }
      std::sort(sorted_ngh.begin(), sorted_ngh.end(),
                [](const std::pair<float, indexType> &a,
                   const std::pair<float, indexType> &b) {
                  return a.first < b.first;
                });
      // Sort the points in GT.coordinates(i, l) similarly.
      std::vector<std::pair<float, indexType>> sorted_gt;
      for (indexType l = 0; l < results_with_ties.size(); l++) {
        float dist = qp.distance(Base_Points[results_with_ties[l]]);
        sorted_gt.push_back(std::make_pair(dist, results_with_ties[l]));
      }
      std::sort(sorted_gt.begin(), sorted_gt.end(),
                [](const std::pair<float, indexType> &a,
                   const std::pair<float, indexType> &b) {
                  return a.first < b.first;
                });
      // Calculate the maximum, mean, and average of the ratio of distances
      // between corresponding elements in two sorted vectors:
      for (indexType l = 0; l < sorted_ngh.size(); l++) {
        float dist_ngh = sorted_ngh[l].first;
        float dist_gt = sorted_gt[l].first;
        if (dist_gt > 0) {
          float ratio = dist_ngh / dist_gt;
          max_approx = std::max(max_approx, ratio);
          mean_approx += ratio;
          avg_approx_num += dist_ngh;
          avg_approx_den += dist_gt;
        }
      }
      mean_approx /= sorted_ngh.size();
      avg_approx   = avg_approx_num / avg_approx_den;
      max_max_approximation = std::max(max_max_approximation, max_approx);
      max_mean_approximation = std::max(max_mean_approximation, mean_approx);
      max_avg_approximation = std::max(max_avg_approximation, avg_approx);
      mean_max_approximation  += max_approx;
      mean_mean_approximation += mean_approx;
      mean_avg_approximation  += avg_approx;
      for (indexType l = 0; l < QP.k; l++) {
        reported_nbhs.insert((all_ngh[i])[l]);
      }
      for (indexType l = 0; l < results_with_ties.size(); l++) {
        if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end())
        {
          numCorrect += 1;
        }
      }
      float curr_correct = 0;
      for (indexType l = 0; l < results_with_ties.size(); l++) {
        if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end())
        {
          curr_correct += 1;
        }
      }
      numCorrect_1_100 += curr_correct/std::min(reported_nbhs.size(),results_with_ties.size());
    }
    recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
    recall_1_100 = static_cast<float>(numCorrect_1_100) / static_cast<float>(1 * n);
    mean_max_approximation  /= n;
    mean_mean_approximation /= n;
    mean_avg_approximation  /= n;
  }
  float QPS = Query_Points.size() / query_time;
  if (verbose)
    std::cout << "search: Q=" << QP.beamSize << ", k=" << QP.k
              << ", limit=" << QP.limit << ", dlimit=" << QP.degree_limit
              << ", recall=" << recall
              << ", visited=" << QueryStats.visited_stats()[0]
              << ", comparisons=" << QueryStats.dist_stats()[0]
              << ", QPS=" << QPS << std::endl;

  auto stats_ = {QueryStats.dist_stats(), QueryStats.visited_stats()};
  parlay::sequence<indexType> stats = parlay::flatten(stats_);
  nn_result N(recall, recall_1_100, stats, QPS, k, QP.beamSize, QP.cut, Query_Points.size(),
              QP.limit, QP.degree_limit, k, ndcg, max_max_approximation,
              max_mean_approximation, max_avg_approximation,
              mean_max_approximation, mean_mean_approximation,
              mean_avg_approximation);
  return N;
}

void write_to_csv(std::string csv_filename, parlay::sequence<float> buckets,
                  parlay::sequence<nn_result> results, Graph_ G) {
  csvfile csv(csv_filename);
  csv << "GRAPH"
      << "Parameters"
      << "Size"
      << "Build time"
      << "Avg degree"
      << "Max degree" << endrow;
  csv << G.name << G.params << G.size << G.time << G.avg_deg << G.max_deg
      << endrow;
  csv << endrow;
  csv << "Num queries"
      << "Target recall"
      << "Actual recall"
      << "QPS"
      << "Average Cmps"
      << "Tail Cmps"
      << "Average Visited"
      << "Tail Visited"
      << "k"
      << "Q"
      << "cut" << endrow;
  for (int i = 0; i < results.size(); i++) {
    nn_result N = results[i];
    csv << N.num_queries << buckets[i] << N.recall << N.QPS << N.avg_cmps
        << N.tail_cmps << N.avg_visited << N.tail_visited << N.k << N.beamQ
        << N.cut << endrow;
  }
  csv << endrow;
  csv << endrow;
}

parlay::sequence<long> calculate_limits(size_t upper_bound) {
  parlay::sequence<long> L(6);
  for (float i = 0; i < 6; i++) {
    L[i] = (long)((4 + i) * ((float)upper_bound) * .1);
    // std::cout << L[i - 1] << std::endl;
  }
  // auto limits = parlay::remove_duplicates(L);
  return L; // limits;
}

template <typename Point, typename PointRange, typename QPointRange,
          typename indexType>
void search_and_parse(Graph_ G_, Graph<indexType> &G, PointRange &Base_Points,
                      PointRange &Query_Points, QPointRange &Q_Base_Points,
                      QPointRange &Q_Query_Points, groundTruth<indexType> GT,
                      char *res_file, long k, bool random = true,
                      indexType start_point = 0, bool verbose = false) {
  parlay::sequence<nn_result> results;
  std::vector<long> beams;
  std::vector<long> allr;
  std::vector<double> cuts;

  QueryParams QP;
  QP.limit = (long)G.size();
  QP.degree_limit = (long)G.max_degree();
  beams = {20,50,100};
  std::vector<int> num_cluster = {1};
  if (k == 0)
    allr = {10};
  else
    allr = {k};
  cuts = {500};
  for (auto nc : num_cluster){
  for (long r : allr) {
    results.clear();
    for (float cut : cuts) {
      QP.cut = cut;
      for (float Q : beams) {
        QP.k = Q;
        QP.beamSize = Q;
        QP.num_clusters = nc;
          results.push_back(
              checkRecall<Point, PointRange, QPointRange, indexType>(
                  G, Base_Points, Query_Points, Q_Base_Points, Q_Query_Points,
                  GT, random, start_point, Q, QP, verbose));
      }
    }
    for (auto result : results) {
      result.print();
    }
  }
  std::cout<<"Results for " << nc << " clusters\n";
  }
}

template <typename Point, typename PointRange, typename indexType>
void search_and_parse(Graph_ G_, Graph<indexType> &G, PointRange &Base_Points,
                      PointRange &Query_Points, groundTruth<indexType> GT,
                      char *res_file, long k, bool random = true,
                      indexType start_point = 0, bool verbose = false) {
  search_and_parse<Point>(G_, G, Base_Points, Query_Points, Base_Points,
                          Query_Points, GT, res_file, k, random, start_point,
                          verbose);
}
