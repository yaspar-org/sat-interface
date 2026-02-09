// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module union multiple solvers as one solver by using these solvers in a round-robin manner.

use crate::{AddConstraints, Model, SatSolver};

/// treats multiple [SatSolver]s as one [SatSolver]
pub struct MultiSolver<T> {
    solvers: Vec<T>,
    idx: usize,
}

impl<T> MultiSolver<T> {
    /// rotate to the next solver
    pub fn next(&mut self) {
        self.idx += 1;
        if self.idx >= self.solvers.len() {
            self.idx = 0;
        }
    }

    /// return a mutable slice of [SatSolver]s
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.solvers.as_mut_slice()
    }
}

impl<T> MultiSolver<T>
where
    T: SatSolver,
{
    /// n > 0
    pub fn new_by_size(n: usize) -> Self {
        let mut solvers = vec![];
        for _ in 0..n {
            solvers.push(T::new());
        }
        Self { solvers, idx: 0 }
    }
}

impl<T, C> AddConstraints<C> for MultiSolver<T>
where
    T: AddConstraints<C>,
{
    fn insert(&mut self, constraints: &C) {
        self.as_mut_slice().insert(constraints)
    }
}

impl<T> SatSolver for MultiSolver<T>
where
    T: SatSolver,
{
    type Status = T::Status;

    fn new() -> Self {
        Self::new_by_size(1)
    }

    fn solve(&mut self) -> Self::Status {
        self.solvers[self.idx].solve()
    }

    fn val(&mut self, lit: i32) -> bool {
        SatSolver::val(&mut self.solvers[self.idx], lit)
    }

    fn load_model(&mut self) -> Model {
        self.solvers[self.idx].load_model()
    }

    fn block_model(&mut self) {
        let model = self.load_model();
        let negated = model.negate();
        for s in self.solvers.iter_mut() {
            s.insert(&negated);
        }
    }
}
