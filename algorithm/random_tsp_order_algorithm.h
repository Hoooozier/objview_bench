#ifndef OBJVIEWBENCH_RANDOM_TSP_ORDER_ALGORITHM_H_
#define OBJVIEWBENCH_RANDOM_TSP_ORDER_ALGORITHM_H_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "objview_algorithm.h"
#include "objview_view_io.h"

namespace objview {

struct RandomTspOrderConfig {
    int budget = 5;
    int seed = 42;  // Base seed; per-episode seed is derived from benchmark context.
    std::string views_path = "../Tammes_sphere/360_xyz.txt";
    double view_radius = 3.0;
    double obstacle_radius = 1.0;
    double tsp_time_limit_sec = -1.0;
    bool silent = true;
};

class RandomTspOrderAlgorithm : public Algorithm {
public:
    explicit RandomTspOrderAlgorithm(RandomTspOrderConfig cfg) : cfg_(cfg) {
        if (cfg_.budget < 0) throw std::runtime_error("RandomTspOrder budget must be non-negative.");
    }

    std::vector<ViewEntry> candidateViewSpace() const override {
        const auto positions = loadViewPositions(cfg_.views_path, cfg_.view_radius);
        std::vector<ViewEntry> views;
        views.reserve(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            ViewEntry entry;
            entry.view_idx = static_cast<int>(i);
            entry.pose.v = {
                positions[i].x(),
                positions[i].y(),
                positions[i].z(),
                0.0,
                0.0,
                0.0,
                0.0,
            };
            views.push_back(entry);
        }
        return views;
    }

    AlgorithmDecision decideNext(const AlgorithmContext& ctx) override {
        double reported_runtime_sec = 0.0;
        if (!planned_) {
            const PlanResult plan = makePlan(ctx);
            planned_path_ = plan.path;
            cursor_ = 0;
            planned_ = true;
            reported_runtime_sec = plan.billable_runtime_sec;

            if (!cfg_.silent) {
                std::cout << "Random-TSPOrder path:";
                for (int vid : planned_path_) std::cout << " " << vid;
                std::cout << " seed=" << deriveEpisodeSeed(ctx)
                          << " runtime_sec=" << reported_runtime_sec << std::endl;
            }
        }

        if (cursor_ >= planned_path_.size()) {
            auto decision = AlgorithmDecision::stop(
                planned_path_.size() < static_cast<size_t>(cfg_.budget)
                    ? "candidate_exhausted"
                    : "plan_end");
            decision.withRuntime(reported_runtime_sec);
            return decision;
        }

        auto decision = AlgorithmDecision::move(planned_path_[cursor_++]);
        decision.withRuntime(reported_runtime_sec);
        return decision;
    }

private:
    struct PlanResult {
        std::vector<int> path;
        double billable_runtime_sec = 0.0;
    };

    RandomTspOrderConfig cfg_;
    bool planned_ = false;
    std::vector<int> planned_path_;
    size_t cursor_ = 0;

    PlanResult makePlan(const AlgorithmContext& ctx) const {
        const auto plan_start = std::chrono::steady_clock::now();
        std::vector<int> candidates;
        candidates.reserve(ctx.candidate_views.size());
        for (const auto& v : ctx.candidate_views) {
            if (samePose(v.pose, ctx.current_pose)) continue;
            candidates.push_back(v.view_idx);
        }

        std::mt19937 rng(deriveEpisodeSeed(ctx));
        std::shuffle(candidates.begin(), candidates.end(), rng);
        if (static_cast<int>(candidates.size()) > cfg_.budget) {
            candidates.resize(cfg_.budget);
        }
        if (candidates.empty()) {
            PlanResult plan;
            plan.billable_runtime_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - plan_start).count();
            return plan;
        }
        std::sort(candidates.begin(), candidates.end());

        std::vector<Vec3> positions = loadViewPositions(cfg_.views_path, cfg_.view_radius);
        for (int vid : candidates) {
            if (vid < 0 || vid >= static_cast<int>(positions.size())) {
                throw std::runtime_error("Candidate view id is out of range for views_path: " +
                                         std::to_string(vid));
            }
        }
        const int virtual_start_id = static_cast<int>(positions.size());
        positions.push_back(cameraPosition(ctx.current_pose));

        std::vector<int> tsp_ids;
        tsp_ids.reserve(candidates.size() + 1);
        tsp_ids.push_back(virtual_start_id);
        tsp_ids.insert(tsp_ids.end(), candidates.begin(), candidates.end());

        HamiltonianPathConfig tsp_cfg;
        tsp_cfg.view_positions = positions;
        tsp_cfg.view_ids = tsp_ids;
        tsp_cfg.start_view_id = virtual_start_id;
        tsp_cfg.end_view_id = -1;
        tsp_cfg.obstacle_center = Vec3(0.0, 0.0, 0.0);
        tsp_cfg.obstacle_radius = cfg_.obstacle_radius;
        tsp_cfg.time_limit_sec = cfg_.tsp_time_limit_sec;
        tsp_cfg.silent = cfg_.silent;

        const auto solve_start = std::chrono::steady_clock::now();
        const auto result = solveHamiltonianPath(tsp_cfg);
        const auto solve_end = std::chrono::steady_clock::now();
        if (!result.solved) {
            throw std::runtime_error("RandomTspOrder TSP planner failed to find a solution.");
        }

        PlanResult plan;
        plan.path.reserve(candidates.size());
        for (int vid : result.path_view_ids) {
            if (vid == virtual_start_id) continue;
            plan.path.push_back(vid);
        }
        const double pre_solve_runtime_sec =
            std::chrono::duration<double>(solve_start - plan_start).count();
        const double post_solve_runtime_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_end).count();
        plan.billable_runtime_sec =
            pre_solve_runtime_sec + result.billable_runtime_sec + post_solve_runtime_sec;
        return plan;
    }

    uint32_t deriveEpisodeSeed(const AlgorithmContext& ctx) const {
        uint64_t hash = 1469598103934665603ull;

        auto mixByte = [&](uint8_t byte) {
            hash ^= static_cast<uint64_t>(byte);
            hash *= 1099511628211ull;
        };
        auto mixString = [&](const std::string& value) {
            for (unsigned char c : value) mixByte(c);
            mixByte(0xff);
        };
        auto mixInt = [&](int value) {
            const uint32_t v = static_cast<uint32_t>(value);
            for (int i = 0; i < 4; ++i) {
                mixByte(static_cast<uint8_t>((v >> (i * 8)) & 0xffu));
            }
            mixByte(0xfe);
        };

        const std::string constraint =
            ctx.episode_config["viewspace_constraint"].get("name", "").asString();
        const int start_view_id =
            ctx.episode_config["start_state"].get("start_view_id", -1).asInt();
        const std::string method_name =
            ctx.episode_config.get("method_name", "random_tsp_order").asString();

        mixString("objview_random_tsp_order_seed_v1");
        mixInt(cfg_.seed);
        mixString(ctx.uid);
        mixString(constraint);
        mixInt(start_view_id);
        mixInt(cfg_.budget);
        mixString(method_name);

        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdull;
        hash ^= hash >> 33;
        hash *= 0xc4ceb9fe1a85ec53ull;
        hash ^= hash >> 33;
        return static_cast<uint32_t>(hash & 0xffffffffu);
    }
};

}  // namespace objview

#endif  // OBJVIEWBENCH_RANDOM_TSP_ORDER_ALGORITHM_H_
