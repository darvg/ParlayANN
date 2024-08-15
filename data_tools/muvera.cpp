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
                   int d, int dproj, bool query);

int main() {
    std::vector<std::vector<float>> pts;
    std::vector<int> shape(2);

    // Load data
    load_data("./msmarco_base_small100k.bin", pts, shape);
    
    int d = shape[1];
    int dlarge = b * rreps * d;
    
    std::cout << "Loaded " << (query ? "query" : "base") << " data with shape " << shape[0] << " x " << shape[1] << std::endl;
    std::cout << "Loaded " << pts.size() << " multivectors" << std::endl;
    
    // Generate down projection matrix
    std::vector<float> down_proj(dproj * dlarge);
    generate_down_proj(down_proj, dproj, dlarge);
    
    std::cout << "Generated down projection matrix with dimensions " << dproj << " x " << dlarge << std::endl;
    
    // Generate LSH partitions
    std::vector<std::vector<float>> lsh_partitions(rreps, std::vector<float>(nbits * d));
    generate_lsh_partitions(lsh_partitions, nbits, d, rreps);
    
    std::cout << "Generated " << rreps << " LSH tables to generate embs of dimension " << dlarge << std::endl;
    
    // Process points
    std::vector<std::vector<float>> final_embs(pts.size(), std::vector<float>(dproj, 0.0f));
    
    #pragma omp parallel for
    for (size_t i = 0; i < pts.size(); ++i) {
        process_point(pts[i], down_proj, lsh_partitions, final_embs[i], d, dproj, query);
    }
    
    std::cout << "Generated " << final_embs.size() << " embs with size " << dproj << std::endl;
    
    // Save results
    std::ofstream outfile("./msmarco_base_small100k_muvera.bin", std::ios::binary);
    shape[0] = pts.size();
    shape[1] = dproj;
    outfile.write(reinterpret_cast<char*>(shape.data()), sizeof(int) * 2);
    for (const auto& emb : final_embs) {
        outfile.write(reinterpret_cast<const char*>(emb.data()), sizeof(float) * dproj);
    }
    outfile.close();
    
    return 0;
}

void load_data(const std::string& filename, std::vector<std::vector<float>>& pts, std::vector<int>& shape) {
    std::ifstream file(filename, std::ios::binary);
    file.read(reinterpret_cast<char*>(shape.data()), sizeof(int) * 2);
    
    std::vector<int> psums(shape[0] + 1);
    file.read(reinterpret_cast<char*>(psums.data()), sizeof(int) * (shape[0] + 1));
    
    std::vector<float> data(psums.back() * shape[1]);
    file.read(reinterpret_cast<char*>(data.data()), sizeof(float) * data.size());
    file.close();
    
    pts.resize(shape[0]);
    for (int i = 0; i < shape[0]; ++i) {
        pts[i].assign(data.begin() + psums[i] * shape[1], data.begin() + psums[i+1] * shape[1]);
    }
}

void generate_down_proj(std::vector<float>& down_proj, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    
    for (int i = 0; i < rows * cols; ++i) {
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
                   int d, int dproj, bool query) {
    std::vector<float> codes(pt.size() / d);
    std::vector<int> a(nbits);
    for (int i = 0; i < nbits; ++i) {
        a[i] = 1 << (nbits - 1 - i);
    }
    
    for (int i = 0; i < rreps; ++i) {
        // Compute codes
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                    pt.size() / d, nbits, d, 
                    1.0f, pt.data(), d, 
                    lsh_partitions[i].data(), d, 
                    0.0f, codes.data(), nbits);
        
        for (float& code : codes) {
            code = code > 0 ? 1.0f : 0.0f;
        }
        
        std::vector<int> bucket_codes(codes.size());
        for (size_t j = 0; j < codes.size(); ++j) {
            int code = 0;
            for (int k = 0; k < nbits; ++k) {
                code += static_cast<int>(codes[j * nbits + k]) * a[k];
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
