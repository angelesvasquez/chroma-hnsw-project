#include "../hnswlib/hnswlib/hnswlib.h"
#include "../hnswlib/hnswlib/hnswalg.h"
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <stdexcept>
#include <fstream>

using Clock = std::chrono::high_resolution_clock;

static size_t DIM = 128;
static size_t N_BASE = 10000;
static size_t N_QUERY = 100;
static const size_t K = 10;
static const size_t ef_construction = 200;
static const size_t M = 16;

std::vector<float> loadFvecs(const std::string &path, size_t &n_out, size_t &dim_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("No se pudo abrir " + path);

    int32_t dim;
    f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    f.seekg(0, std::ios::end);
    size_t file_size = f.tellg();
    size_t record_size = sizeof(int32_t) + dim * sizeof(float);
    size_t n = file_size / record_size;

    f.seekg(0, std::ios::beg);
    std::vector<float> data(n * dim);
    for (size_t i = 0; i < n; i++) {
        int32_t d;
        f.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(float));
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
    size_t file_size = f.tellg();
    size_t record_size = sizeof(int32_t) + dim * sizeof(int32_t);
    size_t n = file_size / record_size;

    f.seekg(0, std::ios::beg);
    std::vector<std::vector<int>> gt(n);
    for (size_t i = 0; i < n; i++) {
        int32_t d;
        f.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        std::vector<int32_t> row(d);
        f.read(reinterpret_cast<char*>(row.data()), d * sizeof(int32_t));
        gt[i].assign(row.begin(), row.begin() + std::min((size_t)d, k));
    }
    return gt;
}

std::vector<std::vector<int>> bruteForceGroundTruth(
    const std::vector<float> &base, const std::vector<float> &queries,
    size_t n_base, size_t n_query, size_t dim, size_t k) {

    std::vector<std::vector<int>> gt(n_query);
    for (size_t q = 0; q < n_query; q++) {
        std::vector<std::pair<float, int>> dists(n_base);
        const float *qv = &queries[q * dim];
        for (size_t i = 0; i < n_base; i++) {
            const float *bv = &base[i * dim];
            float s = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                float diff = qv[d] - bv[d];
                s += diff * diff;
            }
            dists[i] = {s, (int)i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt[q].resize(k);
        for (size_t j = 0; j < k; j++) gt[q][j] = dists[j].second;
    }
    return gt;
}

struct BenchResult {
    double recall_at_k;
    double qps;
    double prune_rate;
};

hnswlib::HierarchicalNSW<float>* buildIndex(
    hnswlib::L2Space &space,
    const std::vector<float> &base,
    bool enable_skip, bool enable_adsampling,
    double &build_seconds, float epsilon = -0.20f) {

    auto *index = new hnswlib::HierarchicalNSW<float>(&space, N_BASE, M, ef_construction, 42);

    if (enable_skip) index->configureSkipConnections(true, 4);
    if (enable_adsampling) {
        index->configureADSampling(true, 32, epsilon);
        index->trainPCA(base.data(), N_BASE, DIM);
    }

    auto t0 = Clock::now();
    for (size_t i = 0; i < N_BASE; i++) {
        index->addPoint(&base[i * DIM], (hnswlib::labeltype)i);
    }
    auto t1 = Clock::now();
    build_seconds = std::chrono::duration<double>(t1 - t0).count();

    return index;
}

BenchResult evaluateIndex(
    hnswlib::HierarchicalNSW<float> *index,
    const std::vector<float> &queries,
    const std::vector<std::vector<int>> &ground_truth,
    size_t ef_search) {

    index->setEf(ef_search);
    index->resetADSamplingCounters();

    size_t hits = 0;
    auto t0 = Clock::now();
    for (size_t q = 0; q < N_QUERY; q++) {
        auto result = index->searchKnn(&queries[q * DIM], K);
        std::unordered_set<int> found;
        while (!result.empty()) {
            found.insert((int)result.top().second);
            result.pop();
        }
        for (int gt_id : ground_truth[q])
            if (found.count(gt_id)) hits++;
    }
    auto t1 = Clock::now();
    double search_seconds = std::chrono::duration<double>(t1 - t0).count();

    BenchResult r;
    r.recall_at_k = (double)hits / (double)(N_QUERY * K);
    r.qps = (double)N_QUERY / search_seconds;
    r.prune_rate = index->getADSamplingPruneRate();
    return r;
}

int main(int argc, char** argv) {
    std::string base_path = argc > 1 ? argv[1] : "siftsmall/siftsmall_base.fvecs";
    std::string query_path = argc > 2 ? argv[2] : "siftsmall/siftsmall_query.fvecs";
    std::string gt_path = argc > 3 ? argv[3] : "siftsmall/siftsmall_groundtruth.ivecs";

    std::cout << "Cargando dataset: " << base_path << "\n";
    auto base = loadFvecs(base_path, N_BASE, DIM);
    size_t dim_q;
    auto queries = loadFvecs(query_path, N_QUERY, dim_q);
    if (dim_q != DIM) throw std::runtime_error("Dimension de queries no coincide con la base");

    std::cout << "Base: " << N_BASE << " vectores, Queries: " << N_QUERY << ", dim=" << DIM << "\n";

    auto gt = loadIvecsGroundtruth(gt_path, K);

    std::vector<size_t> ef_search = {10, 20, 50, 100};
    hnswlib::L2Space space(DIM);
    double build_time;

    std::cout << "\n---- HNSW original ----\n";
    auto *original = buildIndex(space, base, false, false, build_time);
    for (size_t ef : ef_search) {
        auto r = evaluateIndex(original, queries, gt, ef);
        std::cout << "ef_search=" << ef << "  recall@10=" << r.recall_at_k << "  QPS=" << r.qps << "  build_s=" << build_time << "\n";
    }
    delete original;

    std::cout << "\n---- MEJORADO (Skip Connections + ADSampling/PCA) ----\n";
    auto *mejorado = buildIndex(space, base, true, true, build_time);
    for (size_t ef : ef_search) {
        auto r = evaluateIndex(mejorado, queries, gt, ef);
        std::cout << "ef_search=" << ef << "  recall@10=" << r.recall_at_k << "  QPS=" << r.qps << "  build_s=" << build_time << "\n";
    }
    delete mejorado;

    std::cout << "\n---- Solo Skip Connections -----\n";
    auto *skip = buildIndex(space, base, true, false, build_time);
    for (size_t ef : ef_search) {
        auto r = evaluateIndex(skip, queries, gt, ef);
        std::cout << "ef_search=" << ef << "  recall@10=" << r.recall_at_k << "  QPS=" << r.qps << "  build_s=" << build_time << "\n";
    }
    delete skip;

    std::cout << "\n---- Solo ADSampling/PCA ----\n";
    auto *ads = buildIndex(space, base, false, true, build_time);
    for (size_t ef : ef_search) {
        auto r = evaluateIndex(ads, queries, gt, ef);
        std::cout << "ef_search=" << ef << "  recall@10=" << r.recall_at_k << "  QPS=" << r.qps << "  build_s=" << build_time << "  prune_rate=" << r.prune_rate << "\n";
    }
    delete ads;

    std::cout << "\n---- ADSampling/PCA: variando epsilon (ef_search=50) ----\n";
    std::vector<float> epsilons = {0.5f, 0.3f, 0.15f, 0.05f, 0.0f, -0.1f, -0.2f, -0.35f, -0.5f, -0.65f};
    for (float eps : epsilons) {
        auto *idx = buildIndex(space, base, false, true, build_time, eps);
        auto r = evaluateIndex(idx, queries, gt, 50);
        std::cout << "epsilon=" << eps << "  recall@10=" << r.recall_at_k << "  QPS=" << r.qps << "  prune_rate=" << r.prune_rate << "\n";
        delete idx;
    }

    std::cout << "\n---- ADSampling/PCA: variando epsilon (ef_search=100) ----\n";
    for (float eps : epsilons) {
        auto *idx = buildIndex(space, base, false, true, build_time, eps);
        auto r = evaluateIndex(idx, queries, gt, 100);
        std::cout << "epsilon=" << eps << "  recall@10=" << r.recall_at_k << "  QPS=" << r.qps << "  prune_rate=" << r.prune_rate << "\n";
        delete idx;
    }

    return 0;
}