# c=(50000 75000 100000 125000 150000 175000 200000 250000)
# t=(10000 25000 50000 75000 10000 125000 150000 175000)
# c=(1000 4000 8000 16000 32000 64000 128000 256000)
# t=(1000 2000 3000 4000 8000 12000 16000 20000)
c=(60000)
t=(10000 11000 12000 13000 14000 15000)
for i in "${c[@]}"; do
	for j in "${t[@]}"; do
		echo $i $j
		python3 /home/t-gollapudis/ParlayANN/python/test.py --base_data ~/datasets/yfcc/data/base.bin --base_labels ~/datasets/yfcc/data/base_labels_parlay.txt --query_data ~/datasets/yfcc/data/query.bin --query_labels ~/datasets/yfcc/data/query_labels_parlay.txt --gt_file ~/datasets/yfcc/gts/gt_common.bin \
			--bitvector_cutoff 10000 --tiny_cutoff ${i} --target_points ${t} >>outputyfcc_10000_x.txt
	done
done
