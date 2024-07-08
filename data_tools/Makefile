include ../algorithms/bench/parallelDefsANN


vec_to_bin : vec_to_bin.cpp
	$(CC) $(CFLAGS) -o vec_to_bin vec_to_bin.cpp $(LFLAGS) 

compute_groundtruth : compute_groundtruth.cpp ../algorithms/utils/*.h
	$(CC) $(CFLAGS) -o compute_groundtruth compute_groundtruth.cpp $(LFLAGS) -g -O3

blocked_groundtruth: blocked_groundtruth.cpp ../algorithms/utils/*.h
	$(CC) $(CFLAGS) -o blocked_groundtruth blocked_groundtruth.cpp $(LFLAGS) -g -O3

compute_range_groundtruth : compute_range_groundtruth.cpp
	$(CC) $(CFLAGS) -o compute_range_groundtruth compute_range_groundtruth.cpp $(LFLAGS) 

crop : crop.cpp
	$(CC) $(CFLAGS) -o crop crop.cpp $(LFLAGS) 

random_sample : random_sample.cpp
	$(CC) $(CFLAGS) -o random_sample random_sample.cpp $(LFLAGS) 

clean:
	rm -f vec_to_bin compute_groundtruth blocked_groundtruth compute_range_groundtruth crop random_sample