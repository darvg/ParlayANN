# c=(50000 75000 100000 125000 150000 175000 200000 250000)
# t=(10000 25000 50000 75000 10000 125000 150000 175000)
c=(1000 4000 8000 16000 32000 64000 128000 256000)
t=(1000 2000 3000 4000 8000 12000 16000 20000)
c=(256000)
t=(20000)
#python3 test.py --build --base_data ~/datasets/wikipedia_large_35/data/base.bin --base_labels ~/datasets/wikipedia_large_35/data/base_labels_parlay.txt --query_data ~/datasets/wikipedia_large_35/data/query.bin --query_labels ~/datasets/wikipedia_large_35/data/query_labels_parlay.txt --gt_file ~/datasets/wikipedia_large_35/gts/gt_common.bin \
#	--bitvector_cutoff 20 --tiny_cutoff 20 --target_points 20

for i in "${c[@]}"; do
	for j in "${t[@]}"; do
		echo $i $j
		python3 test.py --base_data ~/datasets/wikipedia_large_35/data/base.bin --base_labels ~/datasets/wikipedia_large_35/data/base_labels_parlay.txt --query_data ~/datasets/wikipedia_large_35/data/query.bin --query_labels ~/datasets/wikipedia_large_35/data/query_labels_parlay.txt --gt_file ~/datasets/wikipedia_large_35/gts/gt_common.bin \
			--bitvector_cutoff $i --tiny_cutoff $i --target_points $j >>output35_10000_3.txt
	done
done
