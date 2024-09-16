#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include <cblas.h>

const int nbits = 6;
const int dproj = 10240;
const int b = 1 << nbits;
const int rreps = 40;
const bool query = false;

// Function declarations
void load_data(const std::string& filename, std::vector<std::vector<float>>& pts, std::vector<int>& shape);
void generate_down_proj(std::vector<float>& down_proj, int rows, int cols);
void generate_lsh_partitions(std::vector<std::vector<float>>& lsh_partitions, int nbits, int d, int rreps);
void process_point(const std::vector<float>& pt, const std::vector<float>& down_proj, 
                   const std::vector<std::vector<float>>& lsh_partitions, std::vector<float>& final_emb, 
                   int d, long long dproj, bool query);

int main() {
    std::vector<std::vector<float>> pts;
    std::vector<int> shape(2);

    // Load data
    if(query)
        load_data("./quora_query_base.bin", pts, shape);
    else
        load_data("./quora_base", pts, shape);
    
    int d = shape[1];
    int dlarge = b * rreps * d;
    long long dproj_size = (1LL * dproj) * dlarge;
    
    std::cout << "Loaded " << (query ? "query" : "base") << " data with shape " << shape[0] << " x " << shape[1] << std::endl;
    std::cout << "Loaded " << pts.size() << " multivectors" << std::endl;
    
    // Generate down projection matrix
    std::vector<float> down_proj(dproj_size);
    generate_down_proj(down_proj, dproj, dlarge);
    
    std::cout << "Generated down projection matrix with dimensions " << dproj << " x " << dlarge << std::endl;
    
    // Generate LSH partitions
    std::vector<std::vector<float>> lsh_partitions(rreps, std::vector<float>(nbits * d));
    generate_lsh_partitions(lsh_partitions, nbits, d, rreps);
    
    std::cout << "Generated " << rreps << " LSH tables to generate embs of dimension " << dlarge << std::endl;
    
    std::string filename;
    if(query)
        filename = "./quora_query_muvera.bin";
    else
        filename = "./quora_base_muvera.bin";
    std::ofstream outfile(filename, std::ios::binary);
    
    shape[0] = pts.size();
    shape[1] = dproj;
    outfile.write(reinterpret_cast<char*>(shape.data()), sizeof(int) * 2);
    outfile.close();
    // Process points
    int64_t factor = 10;
    int64_t batch_size = (pts.size() + factor - 1) / factor;
    std::vector<std::vector<float>> final_embs(batch_size, std::vector<float>(dproj, 0.0f));
    

    for(size_t j = 0; j < factor ; j++){
        int32_t left = std::min(pts.size(), size_t( batch_size* (j + 1) )) - batch_size * j;
        #pragma omp parallel for num_threads(128)
        for (size_t i = batch_size*j ; i < std::min(pts.size(), size_t(batch_size*(j + 1))); ++i) {
            process_point(pts[i], down_proj, lsh_partitions, final_embs[i - batch_size*j], d, dproj, query);
        }
        std::cout << "Generated " << final_embs.size() << " embs with size " << dproj << " " << j << std::endl;
        std::string filename;
        if(query)
            filename = "./msmarco_query_tiny_muvera.bin";
        else
            filename = "./msmarco_base_muvera.bin";
        std::ofstream outfile(filename, std::ios::binary | std::ios_base::app);
        
        for (size_t i = batch_size*j ; i < std::min(pts.size(), size_t(batch_size*(j + 1))); ++i) {
            auto &emb = final_embs[i - batch_size*j];
            outfile.write(reinterpret_cast<const char*>(emb.data()), sizeof(float) * dproj);
        }
        outfile.close();
        final_embs.assign(batch_size, std::vector<float>(dproj, 0.0f));
    }
    // Save results
    return 0;
}

void load_data(const std::string& filename, std::vector<std::vector<float>>& pts, std::vector<int>& shape) {
    std::ifstream file_meta_data(filename + ".bin", std::ios::binary);
    file_meta_data.read(reinterpret_cast<char*>(shape.data()), sizeof(int) * 2);
    
    std::vector<int> psums(shape[0] + 1);
    file_meta_data.read(reinterpret_cast<char*>(psums.data() + 1), sizeof(int) * (shape[0]));
    pts.resize(shape[0]);
    for (int i = 0; i < shape[0]; ++i) {
        pts[i].resize(psums[i+1] * (1LL* shape[1]) - psums[i] * (1LL* shape[1]));
        file_meta_data.read(reinterpret_cast<char*>(pts[i].data()), sizeof(float) * pts[i].size());
    }
    file_meta_data.close();
}

void generate_down_proj(std::vector<float>& down_proj, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 1);
    
    for (long long i = 0; i < (1ll * rows) * cols; ++i) {
        down_proj[i] = dis(gen) == 0 ? -1.0f : 1.0f;
    }
}

void generate_lsh_partitions(std::vector<std::vector<float>>& lsh_partitions, int nbits, int d, int rreps) {
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rreps; ++i) {
        for (int j = 0; j < nbits * d; ++j) {
            lsh_partitions[i][j] = dis(gen);
        }
    }
}

void process_point(const std::vector<float>& pt, const std::vector<float>& down_proj, 
                   const std::vector<std::vector<float>>& lsh_partitions, std::vector<float>& final_emb, 
                   int d, long long dproj, bool query) {
    size_t num_curr_vecs = pt.size() / d;
    std::vector<float> codes(num_curr_vecs * nbits); // pt.size() / d = num vecs in pt
    std::vector<int> a(nbits);
    for (int i = 0; i < nbits; ++i) {
        a[i] = 1 << (nbits - 1 - i);
    }
    
    for (int i = 0; i < rreps; ++i) {
        // Compute vector - plane dotproducts
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    num_curr_vecs, nbits, d,
                    1.0f, pt.data(), d,
                    lsh_partitions[i].data(), d,
                    0.0f, codes.data(), nbits);

        // convert bitstrings into plain numbers
        std::vector<int> bucket_codes(num_curr_vecs);
        for (long long j = 0; j < num_curr_vecs ; ++j) {
            int code = 0;
            for (int k = 0; k < nbits; ++k) {
                code += (codes[j * nbits + k] > 0) * a[k];
            }
            bucket_codes[j] = code;
        }
        
        // Process codes
        std::vector<float> y(d, 0.0f);
        std::vector<std::vector<int>> codes_to_inds(b);
        for (size_t j = 0; j < bucket_codes.size(); ++j) {
            codes_to_inds[bucket_codes[j]].push_back(j);
        }
        
        for (int j = 0; j < b; ++j) {
            if (!codes_to_inds[j].empty()) {
                std::fill(y.begin(), y.end(), 0.0f);
                for (int idx : codes_to_inds[j]) {
                    cblas_saxpy(d, 1.0f, pt.data() + idx * d, 1, y.data(), 1);
                }
                
                if (!query) {
                    float scale = 1.0f / codes_to_inds[j].size();
                    cblas_sscal(d, scale, y.data(), 1);
                }
                
                cblas_sgemv(CblasRowMajor, CblasNoTrans, dproj, d, 1.0f, 
                            down_proj.data() + i * b * d * dproj + j * d * dproj, d, 
                            y.data(), 1, 1.0f, final_emb.data(), 1);
            }
        }
    }
}
