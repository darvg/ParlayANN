c=(50000 75000 100000 125000 150000 175000 200000 250000)
t=(10000 25000 50000 75000 10000 125000 150000 175000)

for i in "${c[@]}"; do
	for j in "${t[@]}"; do
		taskset --cpu-list 1 python3 test.py --base_data ~/datasets/wikipedia_large/data/base.bin --base_labels ~/datasets/wikipedia_large/data/base_labels_parlay.txt --query_data ~/datasets/wikipedia_large/data/query.bin --query_labels ~/datasets/wikipedia_large/data/query_labels_parlay.txt --gt_file ~/datasets/wikipedia_large/gts/gt_common.bin \
			--bitvector_cutoff "$i" --tiny_cutoff "$i" --target_points "$j" >>output35.txt
	done
done
