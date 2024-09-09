#!/bin/bash
make -j

P=/ssd1/data/bigann
# ./neighbors -R 64 -L 128 -alpha 1.2 -data_type uint8 -dist_func Euclidian -base_path $P/base.1B.u8bin.crop_nb_1000000

# # PARLAY_NUM_THREADS=1
# ./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -data_type uint8 -dist_func Euclidian -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-1M -res_path test.csv -base_path $P/base.1B.u8bin.crop_nb_1000000
# # ./neighbors -R 64 -L 128 -alpha 1.2 -data_type uint8 -dist_func Euclidian -graph_path $P/graph-10M -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-10M -res_path test.csv -base_path $P/base.1B.u8bin.crop_nb_10000000

# Q=/ssd1/data/text2image1B
# ./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -two_pass 0 -dist_func mips -query_path $Q/query.public.10K.fbin -gt_path $Q/text2image-10K-1M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_1000000

# V=/ssd1/data/MSSPACEV1B
# ./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -data_type int8 -dist_func Euclidian -query_path $V/query.i8bin -gt_path $V/msspacev-1M -res_path test.csv -base_path $V/spacev1b_base.i8bin.crop_nb_1000000

# ./neighbors -R 64 -L 128 -a 1.2 -data_type uint8 -dist_func Euclidian -query_path /ssd1/data/bigann/query.public.10K.u8bin -gt_path /ssd1/data/bigann/bigann-1M -res_path test.csv -base_path /ssd1/data/bigann/base.1B.u8bin.crop_nb_1000000

# P=/ssd1/data/FB_ssnpp
# make
# ./neighbors -R 128 -L 256 -alpha 1.0 -data_type uint8 -two_pass 0 -dist_func Euclidian -k 10 -query_path $P/ssnpp_nonzero.u8bin -gt_path $P/ssnpp-nonzero-1M -res_path test.csv -base_path $P/FB_ssnpp_database.u8bin.crop_nb_1000000

T=/ssd1/data/gist
# ./neighbors -R 200 -L 600 -alpha 1.0 -two_pass 0 -data_type float -dist_func chamfer \
# 	-graph_path /scratch/msmarco-colbert/indices/graph_chamfer2 \
# 	-query_path /scratch/msmarco-colbert/query/msmarco_query_tiny.bin \
# 	-gt_path /scratch/msmarco-colbert/gts/msmarco_gt_b_small_q_tiny.bin \
# 	-res_path test_tiny.csv -base_path /scratch/msmarco-colbert/base/msmarco_base_small.bin

# ./neighbors -R 200 -L 600 -alpha 1.0 -two_pass 0  -data_type float -dist_func chamfer \
#       -graph_path /scratch/msmarco-colbert/indices/graph_chamfer_nosym_newneworder \
#       -query_path /scratch/msmarco-colbert/query/msmarco_query_tiny.bin \
#       -gt_path /scratch/msmarco-colbert/gts/msmarco_gt_b_small100k_q_tiny.bin \
#       -res_path test_tiny.csv -base_path /scratch/msmarco-colbert/base/msmarco_base_small100k.bin

./neighbors -R 200 -L 600 -alpha 0.83 -two_pass 0 -k 100 -data_type float -dist_func chamfer \
	        -graph_path /scratch/msmarco-colbert/indices/graph_chamfer_100k_150q_alpha_0_5 \
		        -query_path /scratch/msmarco-colbert/query/msmarco_query_tiny.bin \
			        -gt_path /scratch/msmarco-colbert/gts/msmarco_gt_b_small100k_q_tiny.bin \
				        -res_path test_tiny.csv -base_path /scratch/msmarco-colbert/base/msmarco_base_small100k.bin
