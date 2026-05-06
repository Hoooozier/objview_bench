#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gurobi_c++.h>

#include "random_tsp_order_algorithm.h"

namespace {

objview::Pose7d poseFromPosition(const objview::Vec3& p) {
    objview::Pose7d pose;
    pose.v = {p.x(), p.y(), p.z(), 0.0, 0.0, 0.0, 0.0};
    return pose;
}

objview::AlgorithmContext makeContext(const std::string& views_path, double view_radius) {
    objview::RandomTspOrderConfig cfg;
    cfg.views_path = views_path;
    cfg.view_radius = view_radius;
    objview::RandomTspOrderAlgorithm algorithm(cfg);

    objview::AlgorithmContext ctx;
    ctx.uid = "unit_test_uid";
    ctx.episode_id = "unit_test_episode";
    ctx.step_index = 0;
    ctx.visited_view_num = 1;
    ctx.candidate_views = algorithm.candidateViewSpace();
    ctx.current_pose = poseFromPosition(objview::Vec3(view_radius, 0.0, 0.0));
    ctx.episode_config["uid"] = ctx.uid;
    ctx.episode_config["method_name"] = "simple_random_tsporder";
    ctx.episode_config["viewspace_constraint"]["name"] = "whole";
    ctx.episode_config["start_state"]["start_view_id"] = 0;
    return ctx;
}

std::vector<int> drainMoves(objview::Algorithm& algorithm, const objview::AlgorithmContext& ctx, int expected_budget) {
    std::vector<int> path;
    for (int i = 0; i < expected_budget; ++i) {
        const auto d = algorithm.decideNext(ctx);
        if (d.type != objview::AlgorithmDecision::Type::Move) {
            throw std::runtime_error("Expected Move before budget was exhausted.");
        }
        path.push_back(d.view_id);
    }

    const auto stop = algorithm.decideNext(ctx);
    if (stop.type != objview::AlgorithmDecision::Type::Stop) {
        throw std::runtime_error("Expected Stop after budget was exhausted.");
    }
    if (stop.stop_reason != "plan_end") {
        throw std::runtime_error("Expected stop_reason=plan_end, got " + stop.stop_reason);
    }
    return path;
}

void checkPath(const std::vector<int>& path, int budget, int num_views) {
    if (static_cast<int>(path.size()) != budget) {
        throw std::runtime_error("Path length does not match budget.");
    }

    std::set<int> unique(path.begin(), path.end());
    if (unique.size() != path.size()) {
        throw std::runtime_error("Path contains duplicate view ids.");
    }

    for (int vid : path) {
        if (vid < 0 || vid >= num_views) {
            throw std::runtime_error("Path contains out-of-range view id.");
        }
    }
}

std::vector<int> runOnce(const objview::AlgorithmContext& ctx, int budget, const std::string& views_path) {
    objview::RandomTspOrderConfig cfg;
    cfg.budget = budget;
    cfg.seed = 42;
    cfg.views_path = views_path;
    cfg.view_radius = 3.0;
    cfg.obstacle_radius = 1.0;
    cfg.silent = true;

    objview::RandomTspOrderAlgorithm algorithm(cfg);
    return drainMoves(algorithm, ctx, budget);
}

void testBudget(const objview::AlgorithmContext& ctx, int budget, const std::string& views_path) {
    const auto path1 = runOnce(ctx, budget, views_path);
    const auto path2 = runOnce(ctx, budget, views_path);

    checkPath(path1, budget, static_cast<int>(ctx.candidate_views.size()));
    if (path1 != path2) {
        throw std::runtime_error("Seed=42 is not deterministic for budget " + std::to_string(budget));
    }

    std::cout << "[ok] budget=" << budget
              << " start_pose=("
              << ctx.current_pose.v[0] << ", "
              << ctx.current_pose.v[1] << ", "
              << ctx.current_pose.v[2] << ")"
              << " end=free"
              << " submit_path:";
    for (int vid : path1) std::cout << " " << vid;
    std::cout << "\n";
}

void testContextDerivedSeed(const objview::AlgorithmContext& ctx, const std::string& views_path) {
    auto uid_ctx = ctx;
    uid_ctx.uid = "different_uid";
    uid_ctx.episode_config["uid"] = uid_ctx.uid;

    auto constraint_ctx = ctx;
    constraint_ctx.episode_config["viewspace_constraint"]["name"] = "upper";

    auto start_ctx = ctx;
    start_ctx.episode_config["start_state"]["start_view_id"] = 2;

    const auto base_path = runOnce(ctx, 10, views_path);
    const auto uid_path = runOnce(uid_ctx, 10, views_path);
    const auto constraint_path = runOnce(constraint_ctx, 10, views_path);
    const auto start_path = runOnce(start_ctx, 10, views_path);
    const auto budget_path = runOnce(ctx, 11, views_path);

    if (base_path == uid_path) {
        throw std::runtime_error("Derived seed did not change for uid.");
    }
    if (base_path == constraint_path) {
        throw std::runtime_error("Derived seed did not change for constraint.");
    }
    if (base_path == start_path) {
        throw std::runtime_error("Derived seed did not change for start_view_id.");
    }
    if (std::equal(base_path.begin(), base_path.end(), budget_path.begin())) {
        throw std::runtime_error("Derived seed did not change for budget.");
    }

    std::cout << "[ok] context-derived seed changes across uid/constraint/start/budget\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string views_path = "../Tammes_sphere/360_xyz.txt";
        if (argc > 1) views_path = argv[1];

        const auto ctx = makeContext(views_path, 3.0);
        if (ctx.candidate_views.size() != 360) {
            throw std::runtime_error("Expected 360 candidate views.");
        }

        for (int budget : {5, 10, 30, 50}) {
            testBudget(ctx, budget, views_path);
        }
        testContextDerivedSeed(ctx, views_path);

        std::cout << "All RandomTspOrderAlgorithm tests passed.\n";
        return 0;
    }
    catch (const GRBException& e) {
        std::cerr << "Gurobi error " << e.getErrorCode() << ": " << e.getMessage() << "\n";
        return 2;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
