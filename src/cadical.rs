// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module implements the traits with the CaDiCal solver

use crate::{AddConstraints, Clause, Formula, Model, SatSolver, SolverState};
use cadical_sys::{CaDiCal, Status};

impl AddConstraints<Clause> for CaDiCal {
    fn insert(&mut self, constraints: &Clause) {
        self.clause6(&constraints.0)
    }
}

impl AddConstraints<Formula> for CaDiCal {
    fn insert(&mut self, constraints: &Formula) {
        for c in constraints.iter() {
            self.insert(c);
        }
    }
}

impl SolverState for Status {
    fn is_sat(&self) -> bool {
        *self == Status::SATISFIABLE
    }

    fn is_unsat(&self) -> bool {
        *self == Status::UNSATISFIABLE
    }

    fn is_unknown(&self) -> bool {
        *self == Status::UNKNOWN
    }
}

impl SatSolver for CaDiCal {
    type Status = Status;

    fn new() -> Self {
        CaDiCal::new()
    }

    fn solve(&mut self) -> Status {
        CaDiCal::solve(self)
    }

    fn val(&mut self, lit: i32) -> bool {
        CaDiCal::val(self, lit) == lit
    }

    fn load_model(&mut self) -> Model {
        let max = self.vars();
        Model::get_model(max as usize, |i| SatSolver::val(self, i as i32))
    }

    fn block_model(&mut self) {
        let max = self.vars();
        let blocking_clause = Clause::new((1..max + 1).map(|i| -self.val(i)).collect());
        self.insert(&blocking_clause);
    }
}
