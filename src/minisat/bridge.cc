// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sat-interface/src/minisat/bridge.hpp"

#include <string>

namespace sat_interface::minisat {

void Solver::add_clause(rust::Slice<const int32_t> lits) {
    for (int32_t lit : lits) {
        solver.add(lit);
    }
    solver.add(0);
}

int32_t Solver::solve() { return solver.solve(); }

int32_t Solver::val(int32_t lit) { return solver.val(lit); }

void Solver::add_observed_var(int32_t var) { solver.add_observed_var(var); }

void Solver::connect_external_propagator(std::unique_ptr<ExternalPropagator> p) {
    propagator = std::move(p);
    solver.connect_external_propagator(propagator.get());
}

void Solver::disconnect_external_propagator() {
    solver.disconnect_external_propagator();
    propagator.reset();
}

void Solver::connect_terminator(std::unique_ptr<Terminator> t) {
    terminator = std::move(t);
    solver.connect_terminator(terminator.get());
}

void Solver::disconnect_terminator() {
    solver.disconnect_terminator();
    terminator.reset();
}

void Solver::interrupt() { solver.terminate(); }

bool Solver::set_option(rust::Str name, int32_t val) {
    return solver.set(std::string(name).c_str(), val);
}

std::unique_ptr<Solver> new_solver() { return std::make_unique<Solver>(); }

std::unique_ptr<Terminator> new_terminator(uint8_t *state, rust::Fn<bool(uint8_t *)> terminate) {
    return std::make_unique<Terminator>(state, terminate);
}

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
    rust::Fn<int32_t(uint8_t *)> cb_add_external_clause_lit) {
    return std::make_unique<ExternalPropagator>(
        state, is_lazy, are_reasons_forgettable, notify_assignment, notify_new_decision_level,
        notify_backtrack, cb_check_found_model, cb_decide, cb_propagate, cb_add_reason_clause_lit,
        cb_has_external_clause, cb_add_external_clause_lit);
}

} // namespace sat_interface::minisat
