#include "../hnswlib/hnswlib/hnswlib.h"
#include "../hnswlib/hnswlib/hnswalg.h"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <unordered_set>

using Clock = std::chrono::high_resolution_clock;
static const size_t K = 10;

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

std::vector<std::vector<int>> loadIvecsGroundtruth(const std::string &path, size_t k) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("No se pudo abrir " + path);

    int32_t dim;
    f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    f.seekg(0, std::ios::end);
    size_t file_size = (size_t)f.tellg();
    size_t record_size = sizeof(int32_t) + (size_t)dim * sizeof(int32_t);
    size_t n = file_size / record_size;

    f.seekg(0, std::ios::beg);
    std::vector<std::vector<int>> gt(n);
    for (size_t i = 0; i < n; i++) {
        int32_t d;
        f.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        std::vector<int32_t> row(d);
        f.read(reinterpret_cast<char*>(row.data()), (size_t)d * sizeof(int32_t));
        gt[i].assign(row.begin(), row.begin() + std::min((size_t)d, k));
    }
    return gt;
}

void runMode(const std::string &mode, hnswlib::L2Space &space,
             const std::vector<float> &queries, size_t n_query, size_t dim,
             const std::vector<std::vector<int>> &gt,
             const std::vector<float> &base, size_t n_base,
             const std::vector<size_t> &ef_values) {

    bool enable_skip = (mode == "skip" || mode == "skipads");
    bool enable_ads  = (mode == "ads"  || mode == "skipads");
    std::string path = "sift1M/index_" + mode + ".bin";

    hnswlib::HierarchicalNSW<float> index(&space, path);

    if (enable_skip) index.configureSkipConnections(true, 4);
    if (enable_ads) { index.configureADSampling(true, 32, -0.20f); index.trainPCA(base.data(), n_base, dim); }

    std::cout << "\n---- " << mode << " ----\n";
    for (size_t ef : ef_values) {
        index.setEf(ef);
        index.resetADSamplingCounters();

        size_t hits = 0;
        auto t0 = Clock::now();
        for (size_t q = 0; q < n_query; q++) {
            auto result = index.searchKnn(&queries[q * dim], K);
            std::unordered_set<int> found;
            while (!result.empty()) { found.insert((int)result.top().second); result.pop(); }
            for (int gt_id : gt[q]) if (found.count(gt_id)) hits++;
        }
        auto t1 = Clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "ef_search=" << ef
                   << "  recall@10=" << (double)hits / (double)(n_query * K)
                   << "  QPS=" << (double)n_query / seconds;
        if (enable_ads) std::cout << "  prune_rate=" << index.getADSamplingPruneRate();
        std::cout << "\n";
    }
}

int main() {
    size_t n_query, dim_q;
    auto queries = loadFvecs("sift1M/sift_query.fvecs", n_query, dim_q);
    auto gt = loadIvecsGroundtruth("sift1M/sift_groundtruth.ivecs", K);

    size_t n_base, dim;
    auto base = loadFvecs("sift1M/sift_base.fvecs", n_base, dim);

    hnswlib::L2Space space(dim);
    std::vector<size_t> ef_values = {5, 10, 20, 50, 100, 150, 200};

    for (const std::string &mode : {"original", "skip", "ads", "skipads"}) {
        runMode(mode, space, queries, n_query, dim, gt, base, n_base, ef_values);
    }
    return 0;
}