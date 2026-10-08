// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minisat/minisatup.h"
#include "rust/cxx.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace sat_interface::minisat {

struct ExternalPropagator : public MiniSatUP::ExternalPropagator {
    uint8_t *state;
    rust::Fn<void(uint8_t *, rust::Slice<const int32_t>)> rust_notify_assignment;
    rust::Fn<void(uint8_t *)> rust_notify_new_decision_level;
    rust::Fn<void(uint8_t *, size_t)> rust_notify_backtrack;
    rust::Fn<bool(uint8_t *, rust::Slice<const int32_t>)> rust_cb_check_found_model;
    rust::Fn<int32_t(uint8_t *)> rust_cb_decide;
    rust::Fn<int32_t(uint8_t *)> rust_cb_propagate;
    rust::Fn<int32_t(uint8_t *, int32_t)> rust_cb_add_reason_clause_lit;
    rust::Fn<bool(uint8_t *, bool *)> rust_cb_has_external_clause;
    rust::Fn<int32_t(uint8_t *)> rust_cb_add_external_clause_lit;

    ExternalPropagator(
        uint8_t *state, bool is_lazy, bool are_reasons_forgettable,
        rust::Fn<void(uint8_t *, rust::Slice<const int32_t>)> rust_notify_assignment,
        rust::Fn<void(uint8_t *)> rust_notify_new_decision_level,
        rust::Fn<void(uint8_t *, size_t)> rust_notify_backtrack,
        rust::Fn<bool(uint8_t *, rust::Slice<const int32_t>)> rust_cb_check_found_model,
        rust::Fn<int32_t(uint8_t *)> rust_cb_decide,
        rust::Fn<int32_t(uint8_t *)> rust_cb_propagate,
        rust::Fn<int32_t(uint8_t *, int32_t)> rust_cb_add_reason_clause_lit,
        rust::Fn<bool(uint8_t *, bool *)> rust_cb_has_external_clause,
        rust::Fn<int32_t(uint8_t *)> rust_cb_add_external_clause_lit)
        : state(state),
          rust_notify_assignment(rust_notify_assignment),
          rust_notify_new_decision_level(rust_notify_new_decision_level),
          rust_notify_backtrack(rust_notify_backtrack),
          rust_cb_check_found_model(rust_cb_check_found_model),
          rust_cb_decide(rust_cb_decide),
          rust_cb_propagate(rust_cb_propagate),
          rust_cb_add_reason_clause_lit(rust_cb_add_reason_clause_lit),
          rust_cb_has_external_clause(rust_cb_has_external_clause),
          rust_cb_add_external_clause_lit(rust_cb_add_external_clause_lit) {
        this->is_lazy = is_lazy;
        this->are_reasons_forgettable = are_reasons_forgettable;
    }

    void notify_assignment(const std::vector<int> &lits) override {
        rust_notify_assignment(state, rust::Slice<const int32_t>(lits.data(), lits.size()));
    }
    void notify_new_decision_level() override { rust_notify_new_decision_level(state); }
    void notify_backtrack(size_t new_level) override { rust_notify_backtrack(state, new_level); }
    bool cb_check_found_model(const std::vector<int> &model) override {
        return rust_cb_check_found_model(state,
                                         rust::Slice<const int32_t>(model.data(), model.size()));
    }
    int cb_decide() override { return rust_cb_decide(state); }
    int cb_propagate() override { return rust_cb_propagate(state); }
    int cb_add_reason_clause_lit(int propagated_lit) override {
        return rust_cb_add_reason_clause_lit(state, propagated_lit);
    }
    bool cb_has_external_clause(bool &is_forgettable) override {
        return rust_cb_has_external_clause(state, &is_forgettable);
    }
    int cb_add_external_clause_lit() override { return rust_cb_add_external_clause_lit(state); }
};

struct Terminator : public MiniSatUP::Terminator {
    uint8_t *state;
    rust::Fn<bool(uint8_t *)> rust_terminate;

    Terminator(uint8_t *state, rust::Fn<bool(uint8_t *)> rust_terminate)
        : state(state), rust_terminate(rust_terminate) {}

    bool terminate() override { return rust_terminate(state); }
};

class Solver {
public:
    void add_clause(rust::Slice<const int32_t> lits);
    int32_t solve();
    int32_t val(int32_t lit);
    void add_observed_var(int32_t var);
    void connect_external_propagator(std::unique_ptr<ExternalPropagator> propagator);
    void disconnect_external_propagator();
    void connect_terminator(std::unique_ptr<Terminator> terminator);
    void disconnect_terminator();
    void interrupt();
    bool set_option(rust::Str name, int32_t val);

private:
    MiniSatUP::Solver solver;
    std::unique_ptr<ExternalPropagator> propagator;
    std::unique_ptr<Terminator> terminator;
};

std::unique_ptr<Solver> new_solver();

std::unique_ptr<Terminator> new_terminator(uint8_t *state, rust::Fn<bool(uint8_t *)> terminate);

std::unique_ptr<ExternalPropagator> new_external_propagator(
    uint8_t *state, bool is_lazy, bool are_reasons_forgettable,
    rust::Fn<void(uint8_t *, rust::Slice<const int32_t>)> notify_assignment,
    rust::Fn<void(uint8_t *)> notify_new_decision_level,
    rust::Fn<void(uint8_t *, size_t)> notify_backtrack,
    rust::Fn<bool(uint8_t *, rust::Slice<const int32_t>)> cb_check_found_model,
    rust::Fn<int32_t(uint8_t *)> cb_decide,
    rust::Fn<int32_t(uint8_t *)> cb_propagate,
    rust::Fn<int32_t(uint8_t *, int32_t)> cb_add_reason_clause_lit,
    rust::Fn<bool(uint8_t *, bool *)> cb_has_external_clause,
    rust::Fn<int32_t(uint8_t *)> cb_add_external_clause_lit);

} // namespace sat_interface::minisat
