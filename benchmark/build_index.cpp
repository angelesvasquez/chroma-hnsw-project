#include "../hnswlib/hnswlib/hnswlib.h"
#include "../hnswlib/hnswlib/hnswalg.h"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

static const size_t ef_construction = 200;
static const size_t M = 16;

std::vector<float> loadFvecs(const std::string &path, size_t &n_out, size_t &dim_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("No se pudo abrir " + path);

    int32_t dim;
    f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    f.seekg(0, std::ios::end);
    size_t file_size = (size_t)f.tellg();
    size_t record_size = sizeof(int32_t) + (size_t)dim * sizeof(float);
    size_t n = file_size / record_size;

    f.seekg(0, std::ios::beg);
    std::vector<float> data(n * (size_t)dim);
    for (size_t i = 0; i < n; i++) {
        int32_t d;
        f.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&data[i * (size_t)dim]), (size_t)dim * sizeof(float));
    }
    n_out = n;
    dim_out = (size_t)dim;
    return data;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <original|skip|ads|skipads>\n";
        return 1;
    }
    std::string mode = argv[1];
    bool enable_skip = (mode == "skip" || mode == "skipads");
    bool enable_ads  = (mode == "ads"  || mode == "skipads");

    size_t n_base, dim;
    auto base = loadFvecs("sift1M/sift_base.fvecs", n_base, dim);
    std::cout << "n_base=" << n_base << " dim=" << dim << "\n";

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, n_base, M, ef_construction, 42);

    if (enable_skip) index.configureSkipConnections(true, 4);
    if (enable_ads) {
        index.configureADSampling(true, 32, -0.20f);
        index.trainPCA(base.data(), n_base, dim);
    }

    auto t0 = Clock::now();
    for (size_t i = 0; i < n_base; i++) {
        index.addPoint(&base[i * dim], (hnswlib::labeltype)i);
    }
    auto t1 = Clock::now();
    std::cout << "build_s=" << std::chrono::duration<double>(t1 - t0).count() << "\n";

    std::string out_path = "sift1M/index_" + mode + ".bin";
    index.saveIndex(out_path);
    std::cout << "Guardado: " << out_path << "\n";
    return 0;
}