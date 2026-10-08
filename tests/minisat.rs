// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "minisat")]

use sat_interface::ExternalPropagator;
use sat_interface::minisat::{MiniSat, Status};

#[test]
fn sat_and_model() {
    let mut s = MiniSat::new();
    s.add_clause(&[1, 2]);
    s.add_clause(&[-1]);
    assert_eq!(s.solve(), Status::Satisfiable);
    assert_eq!(s.val(1), -1);
    assert_eq!(s.val(2), 2);
}

#[test]
fn unsat() {
    let mut s = MiniSat::new();
    s.add_clause(&[1, 2]);
    s.add_clause(&[-1, 2]);
    s.add_clause(&[1, -2]);
    s.add_clause(&[-1, -2]);
    assert_eq!(s.solve(), Status::Unsatisfiable);
}

#[test]
fn unknown_options_are_rejected() {
    let mut s = MiniSat::new();
    assert!(s.set("phase-saving", 1));
    assert!(!s.set("elevate", 3));
}

struct ForceNegThree {
    assigned: Vec<i32>,
    level: usize,
    pending: Vec<i32>,
    rejected: usize,
}

impl ExternalPropagator for ForceNegThree {
    fn notify_assignment(&mut self, lits: &[i32]) {
        self.assigned.extend_from_slice(lits);
    }

    fn notify_new_decision_level(&mut self) {
        self.level += 1;
    }

    fn notify_backtrack(&mut self, new_level: usize) {
        self.level = new_level;
        self.assigned.clear();
    }

    fn cb_check_found_model(&mut self, model: &[i32]) -> bool {
        if model.contains(&3) {
            self.rejected += 1;
            self.pending = vec![-3, 0];
            false
        } else {
            true
        }
    }

    fn cb_has_external_clause(&mut self, is_forgettable: &mut bool) -> bool {
        *is_forgettable = false;
        !self.pending.is_empty()
    }

    fn cb_add_external_clause_lit(&mut self) -> i32 {
        if self.pending.is_empty() {
            0
        } else {
            self.pending.remove(0)
        }
    }
}

#[test]
fn external_propagator_steers_model() {
    let mut p = ForceNegThree {
        assigned: vec![],
        level: 0,
        pending: vec![],
        rejected: 0,
    };
    let mut s = MiniSat::new();
    s.add_clause(&[1, 2]);
    s.add_clause(&[1, 3]);
    s.connect_external_propagator(&mut p);
    s.add_observed_var(1);
    s.add_observed_var(2);
    s.add_observed_var(3);
    assert_eq!(s.solve(), Status::Satisfiable);
    s.disconnect_external_propagator();
    assert_eq!(s.val(1), 1);
    assert_eq!(s.val(3), -3);
    assert!(p.rejected >= 1, "model with 3=true must have been rejected");
}

#[test]
fn fresh_variables_during_search() {
    struct Grow {
        pending: Vec<i32>,
        done: bool,
    }
    impl ExternalPropagator for Grow {
        fn notify_assignment(&mut self, _lits: &[i32]) {}
        fn notify_new_decision_level(&mut self) {}
        fn notify_backtrack(&mut self, _new_level: usize) {}
        fn cb_check_found_model(&mut self, _model: &[i32]) -> bool {
            if self.done {
                return true;
            }
            self.done = true;
            self.pending = vec![5, 0, -5, -1, 0];
            false
        }
        fn cb_has_external_clause(&mut self, is_forgettable: &mut bool) -> bool {
            *is_forgettable = false;
            !self.pending.is_empty()
        }
        fn cb_add_external_clause_lit(&mut self) -> i32 {
            self.pending.remove(0)
        }
    }
    let mut p = Grow {
        pending: vec![],
        done: false,
    };
    let mut s = MiniSat::new();
    s.add_clause(&[1]);
    s.connect_external_propagator(&mut p);
    s.add_observed_var(1);
    s.add_observed_var(5);
    assert_eq!(s.solve(), Status::Unsatisfiable);
}

#[test]
fn terminator_stops_search() {
    struct Now;
    impl sat_interface::Terminator for Now {
        fn terminated(&mut self) -> bool {
            true
        }
    }
    let mut s = MiniSat::new();
    s.add_clause(&[1, 2]);
    s.add_clause(&[-1, 2]);
    s.connect_terminator(&mut Now);
    assert_eq!(s.solve(), Status::Unknown);
    s.disconnect_terminator();
    assert_eq!(s.solve(), Status::Satisfiable);
}
