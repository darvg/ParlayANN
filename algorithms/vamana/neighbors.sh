#!/bin/bash
make -j


# echo the graph path befor running each command

echo "GREP_ME spacev1_unsorted"
./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -k 100 -data_type int8 -dist_func Euclidian \
    -query_path /scratch/spacev1/query/query.i8bin \
    -gt_path /scratch/spacev1/gts/space_gt.bin -res_path test.csv \
    -base_path /scratch/spacev1/base/spacev100m_base.i8bin \
    -graph_path /scratch/spacev1/indices/spacev1_unsorted.bin

echo "GREP_ME spacev1_sorted"
./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -k 100 -data_type int8 -dist_func Euclidian \
    -query_path /scratch/spacev1/query/query.i8bin \
    -gt_path /scratch/spacev1/gts/space_gt.bin -res_path test.csv \
    -base_path /scratch/spacev1/base/spacev100m_base.i8bin \
    -graph_path /scratch/spacev1/indices/spacev1_sorted.bin

echo "GREP_ME bigann_unsorted"
./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -k 100 -data_type float -dist_func Euclidian \
    -query_path /scratch/bigann/query/query.public.10K.u8bin \
    -gt_path /scratch/bigann/gts/bigann_gt.bin -res_path test.csv \
    -base_path /scratch/bigann/base/learn.100M.u8bin \
    -graph_path /scratch/bigann/indices/bigann_unsorted.bin

echo "GREP_ME bigann_sorted"
./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -k 100 -data_type int8 -dist_func Euclidian \
    -query_path /scratch/bigann/query/query.public.10K.u8bin \
    -gt_path /scratch/bigann/gts/bigann_gt.bin -res_path test.csv \
    -base_path /scratch/bigann/base/learn.100M.u8bin \
    -graph_path /scratch/bigann/indices/bigann_sorted.bin

./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -k 100 -data_type floatqq -dist_func Euclidian \
    -query_path /scratch/sift/query/sift_query.bin \
    -gt_path /scratch/sift/gts/euclidean_gt.bin -res_path test.csv \
    -base_path /scratch/sift/base/sift_base.bin \
    -graph_path /scratch/spacev1/indices//scratch/sift/indcies/unsorted_alpha_0_83

./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -k 100 -data_type float -dist_func Euclidian \
    -query_path /scratch/sift/query/sift_query.bin \
    -gt_path /scratch/sift/gts/euclidean_gt.bin -res_path test.csv \
    -base_path /scratch/sift/base/sift_base.bin \
    -graph_path /scratch/spacev1/indices//scratch/sift/indcies/sorted_alpha_0_83