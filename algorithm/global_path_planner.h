#ifndef OBJVIEWBENCH_GLOBAL_PATH_PLANNER_H_
#define OBJVIEWBENCH_GLOBAL_PATH_PLANNER_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <gurobi_c++.h>

#include "objview_geometry.h"

namespace objview {

struct HamiltonianPathConfig {
    std::vector<Vec3> view_positions;
    std::vector<int> view_ids;
    int start_view_id = -1;
    int end_view_id = -1;
    Vec3 obstacle_center{0.0, 0.0, 0.0};
    double obstacle_radius = 0.0;
    double time_limit_sec = -1.0;
    bool silent = true;
};

struct HamiltonianPathResult {
    bool solved = false;
    double total_length = -1.0;
    double model_build_runtime_sec = 0.0;
    double optimize_runtime_sec = 0.0;
    double solution_runtime_sec = 0.0;
    double billable_runtime_sec = 0.0;
    std::vector<int> path_view_ids;
};

inline GRBEnv& sharedGurobiEnv() {
    static std::once_flag init_flag;
    static std::unique_ptr<GRBEnv> env;
    std::call_once(init_flag, []() {
        env = std::make_unique<GRBEnv>(true);
        env->set(GRB_IntParam_OutputFlag, 0);
        env->start();
    });
    return *env;
}

inline void warmupGurobi() {
    GRBEnv& env = sharedGurobiEnv();
    GRBModel model(env);
    model.set(GRB_IntParam_OutputFlag, 0);
    GRBVar x = model.addVar(0.0, 1.0, 1.0, GRB_BINARY, "warmup_x");
    GRBLinExpr objective = x;
    model.setObjective(objective, GRB_MAXIMIZE);
    model.optimize();
}

inline double segmentPointMinDistance(const Vec3& p, const Vec3& q, const Vec3& x) {
    const Vec3 pq = q - p;
    const double denom = pq.squaredNorm();
    if (denom <= 1e-18) {
        return (x - p).norm();
    }

    double t = (x - p).dot(pq) / denom;
    t = std::max(0.0, std::min(1.0, t));
    return (x - (p + t * pq)).norm();
}

inline double collisionAvoidUnitSphereDistanceFromOrigin(const Vec3& p, const Vec3& q, double radius) {
    constexpr double pi = 3.141592653589793238462643383279502884;

    if (radius <= 0.0) {
        return (p - q).norm();
    }

    const double r1 = p.norm();
    const double r2 = q.norm();
    if (r1 <= radius + 1e-9 || r2 <= radius + 1e-9) {
        return (p - q).norm();
    }

    if (segmentPointMinDistance(p, q, Vec3::Zero()) >= radius - 1e-9) {
        return (p - q).norm();
    }

    const double tan1 = std::sqrt(std::max(r1 * r1 - radius * radius, 0.0));
    const double tan2 = std::sqrt(std::max(r2 * r2 - radius * radius, 0.0));

    double cos_theta = p.dot(q) / (r1 * r2);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    const double theta = std::acos(cos_theta);

    const double gamma1 = std::acos(std::max(-1.0, std::min(1.0, radius / r1)));
    const double gamma2 = std::acos(std::max(-1.0, std::min(1.0, radius / r2)));
    const double two_pi = 2.0 * pi;

    auto normAng = [&](double a) {
        a = std::fmod(a, two_pi);
        if (a < 0.0) a += two_pi;
        return a;
    };

    const double ta[2] = {normAng(gamma1), normAng(-gamma1)};
    const double tb[2] = {normAng(theta + gamma2), normAng(theta - gamma2)};

    double best = std::numeric_limits<double>::infinity();
    for (double a : ta) {
        for (double b : tb) {
            double delta = std::abs(b - a);
            delta = std::min(delta, two_pi - delta);
            best = std::min(best, tan1 + tan2 + radius * delta);
        }
    }
    return best;
}

inline double collisionAvoidSphereDistance(
    const Vec3& p,
    const Vec3& q,
    const Vec3& center,
    double radius) {
    return collisionAvoidUnitSphereDistanceFromOrigin(p - center, q - center, radius);
}

inline void findSubtour(int n, double** sol, int* tour_len, int* tour) {
    std::vector<char> seen(n, 0);
    int bestind = -1;
    int bestlen = n + 1;
    int start = 0;

    while (start < n) {
        int node = 0;
        for (; node < n; ++node) {
            if (!seen[node]) break;
        }
        if (node == n) break;

        for (int len = 0; len < n; ++len) {
            tour[start + len] = node;
            seen[node] = 1;

            int i = 0;
            for (; i < n; ++i) {
                if (sol[node][i] > 0.5 && !seen[i]) {
                    node = i;
                    break;
                }
            }

            if (i == n) {
                ++len;
                if (len < bestlen) {
                    bestlen = len;
                    bestind = start;
                }
                start += len;
                break;
            }
        }
    }

    for (int i = 0; i < bestlen; ++i) {
        tour[i] = tour[bestind + i];
    }
    *tour_len = bestlen;
}

class SubtourElimCallback : public GRBCallback {
public:
    SubtourElimCallback(std::vector<std::vector<GRBVar>>* xvars, int node_count)
        : vars_(xvars), n_(node_count) {}

protected:
    void callback() override {
        try {
            if (where != GRB_CB_MIPSOL) return;

            double** x = new double*[n_];
            int* tour = new int[n_];
            for (int i = 0; i < n_; ++i) {
                x[i] = getSolution((*vars_)[i].data(), n_);
            }

            int len = 0;
            findSubtour(n_, x, &len, tour);

            if (len < n_) {
                GRBLinExpr expr = 0;
                for (int i = 0; i < len; ++i) {
                    for (int j = i + 1; j < len; ++j) {
                        expr += (*vars_)[tour[i]][tour[j]];
                    }
                }
                addLazy(expr <= len - 1);
            }

            for (int i = 0; i < n_; ++i) {
                delete[] x[i];
            }
            delete[] x;
            delete[] tour;
        }
        catch (const GRBException& e) {
            std::cerr << "Gurobi callback error " << e.getErrorCode()
                      << ": " << e.getMessage() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown error during subtour callback." << std::endl;
        }
    }

private:
    std::vector<std::vector<GRBVar>>* vars_;
    int n_;
};

inline HamiltonianPathResult solveHamiltonianPath(const HamiltonianPathConfig& cfg) {
    if (cfg.view_positions.empty()) {
        throw std::runtime_error("view_positions is empty.");
    }
    if (cfg.view_ids.empty()) {
        throw std::runtime_error("view_ids is empty.");
    }
    if (cfg.start_view_id < 0) {
        throw std::runtime_error("start_view_id is required.");
    }

    auto containsView = [&](int vid) {
        return std::find(cfg.view_ids.begin(), cfg.view_ids.end(), vid) != cfg.view_ids.end();
    };
    if (!containsView(cfg.start_view_id)) {
        throw std::runtime_error("start_view_id must be included in view_ids.");
    }
    if (cfg.end_view_id != -1 && !containsView(cfg.end_view_id)) {
        throw std::runtime_error("end_view_id must be -1 or included in view_ids.");
    }
    if (cfg.end_view_id == cfg.start_view_id) {
        throw std::runtime_error("end_view_id must differ from start_view_id.");
    }
    for (int vid : cfg.view_ids) {
        if (vid < 0 || vid >= static_cast<int>(cfg.view_positions.size())) {
            throw std::runtime_error("view_ids contains an out-of-range view id.");
        }
    }
    if (cfg.view_ids.size() == 1) {
        HamiltonianPathResult result;
        result.solved = true;
        result.total_length = 0.0;
        result.path_view_ids = cfg.view_ids;
        return result;
    }

    const int real_n = static_cast<int>(cfg.view_ids.size());
    const int copy_node = real_n;
    const int n = real_n + 1;

    auto localToViewId = [&](int local_id) {
        if (local_id == copy_node) return -1;
        return cfg.view_ids[local_id];
    };
    auto viewIdToLocal = [&](int view_id) {
        for (int i = 0; i < real_n; ++i) {
            if (cfg.view_ids[i] == view_id) return i;
        }
        return -1;
    };

    GRBEnv& env = sharedGurobiEnv();
    const auto model_build_start = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> graph(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == copy_node || j == copy_node) {
                graph[i][j] = 0.0;
                continue;
            }

            const int u = localToViewId(i);
            const int v = localToViewId(j);
            graph[i][j] = collisionAvoidSphereDistance(
                cfg.view_positions[u],
                cfg.view_positions[v],
                cfg.obstacle_center,
                cfg.obstacle_radius);
        }
    }

    GRBModel model(env);
    if (cfg.silent) {
        model.set(GRB_IntParam_OutputFlag, 0);
    }
    model.set(GRB_IntParam_LazyConstraints, 1);
    if (cfg.time_limit_sec > 0.0) {
        model.set(GRB_DoubleParam_TimeLimit, cfg.time_limit_sec);
    }

    std::vector<std::vector<GRBVar>> vars(n, std::vector<GRBVar>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            vars[i][j] = model.addVar(
                0.0,
                1.0,
                graph[i][j],
                GRB_BINARY,
                "x_" + std::to_string(i) + "_" + std::to_string(j));
            vars[j][i] = vars[i][j];
        }
    }

    for (int i = 0; i < n; ++i) {
        GRBLinExpr expr = 0;
        for (int j = 0; j < n; ++j) {
            expr += vars[i][j];
        }
        model.addConstr(expr == 2, "deg2_" + std::to_string(i));
        vars[i][i].set(GRB_DoubleAttr_UB, 0.0);
    }

    vars[copy_node][viewIdToLocal(cfg.start_view_id)].set(GRB_DoubleAttr_LB, 1.0);
    if (cfg.end_view_id != -1) {
        vars[copy_node][viewIdToLocal(cfg.end_view_id)].set(GRB_DoubleAttr_LB, 1.0);
    }

    SubtourElimCallback cb(&vars, n);
    model.setCallback(&cb);
    const auto optimize_start = std::chrono::steady_clock::now();
    const double model_build_runtime_sec =
        std::chrono::duration<double>(optimize_start - model_build_start).count();
    model.optimize();
    const auto solution_start = std::chrono::steady_clock::now();
    const double optimize_runtime_sec =
        std::chrono::duration<double>(solution_start - optimize_start).count();

    HamiltonianPathResult result;
    result.model_build_runtime_sec = model_build_runtime_sec;
    result.optimize_runtime_sec = optimize_runtime_sec;
    if (model.get(GRB_IntAttr_SolCount) <= 0) {
        result.billable_runtime_sec = result.model_build_runtime_sec + result.optimize_runtime_sec;
        return result;
    }

    double** sol = new double*[n];
    for (int i = 0; i < n; ++i) {
        sol[i] = model.get(GRB_DoubleAttr_X, vars[i].data(), n);
    }

    int* tour = new int[n];
    int len = 0;
    findSubtour(n, sol, &len, tour);
    if (len != n) {
        for (int i = 0; i < n; ++i) delete[] sol[i];
        delete[] sol;
        delete[] tour;
        throw std::runtime_error("Internal error: optimized TSP solution still contains a subtour.");
    }

    std::vector<int> ordered(tour, tour + len);
    for (int i = 0; i < n; ++i) delete[] sol[i];
    delete[] sol;
    delete[] tour;

    auto copy_it = std::find(ordered.begin(), ordered.end(), copy_node);
    if (copy_it == ordered.end()) {
        throw std::runtime_error("Internal error: copy node is missing from solution.");
    }
    std::rotate(ordered.begin(), copy_it, ordered.end());
    ordered.erase(ordered.begin());

    if (ordered.empty() || localToViewId(ordered.front()) != cfg.start_view_id) {
        std::reverse(ordered.begin(), ordered.end());
    }

    result.solved = true;
    result.total_length = 0.0;
    result.path_view_ids.reserve(ordered.size());
    for (int local_id : ordered) {
        result.path_view_ids.push_back(localToViewId(local_id));
    }
    for (size_t i = 1; i < ordered.size(); ++i) {
        result.total_length += graph[ordered[i - 1]][ordered[i]];
    }
    result.solution_runtime_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solution_start).count();
    result.billable_runtime_sec =
        result.model_build_runtime_sec + result.optimize_runtime_sec + result.solution_runtime_sec;

    return result;
}

}  // namespace objview

#endif  // OBJVIEWBENCH_GLOBAL_PATH_PLANNER_H_
