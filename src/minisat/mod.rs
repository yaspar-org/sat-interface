// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust bindings for the IPASIR-UP fork of MiniSat at <https://github.com/amarshah1/minisat>.
//!
//! Provides a MiniSat handle over `i32` literals that can connect an [ExternalPropagator] with
//! IPASIR-UP callbacks and a [Terminator].

use crate::{ExternalPropagator, Terminator};
use cxx::UniquePtr;

#[cxx::bridge(namespace = "sat_interface::minisat")]
mod ffi {
    unsafe extern "C++" {
        include!("sat-interface/src/minisat/bridge.hpp");

        type Solver;
        type ExternalPropagator;
        type Terminator;

        fn new_solver() -> UniquePtr<Solver>;
        fn add_clause(self: Pin<&mut Solver>, lits: &[i32]);
        fn solve(self: Pin<&mut Solver>) -> i32;
        fn val(self: Pin<&mut Solver>, lit: i32) -> i32;
        fn add_observed_var(self: Pin<&mut Solver>, var: i32);
        fn connect_external_propagator(
            self: Pin<&mut Solver>,
            propagator: UniquePtr<ExternalPropagator>,
        );
        fn disconnect_external_propagator(self: Pin<&mut Solver>);
        fn connect_terminator(self: Pin<&mut Solver>, terminator: UniquePtr<Terminator>);
        fn disconnect_terminator(self: Pin<&mut Solver>);
        fn interrupt(self: Pin<&mut Solver>);
        fn set_option(self: Pin<&mut Solver>, name: &str, val: i32) -> bool;

        unsafe fn new_terminator(
            state: *mut u8,
            terminate: unsafe fn(*mut u8) -> bool,
        ) -> UniquePtr<Terminator>;

        #[allow(clippy::too_many_arguments)]
        unsafe fn new_external_propagator(
            state: *mut u8,
            is_lazy: bool,
            are_reasons_forgettable: bool,
            notify_assignment: unsafe fn(*mut u8, &[i32]),
            notify_new_decision_level: unsafe fn(*mut u8),
            notify_backtrack: unsafe fn(*mut u8, usize),
            cb_check_found_model: unsafe fn(*mut u8, &[i32]) -> bool,
            cb_decide: unsafe fn(*mut u8) -> i32,
            cb_propagate: unsafe fn(*mut u8) -> i32,
            cb_add_reason_clause_lit: unsafe fn(*mut u8, i32) -> i32,
            cb_has_external_clause: unsafe fn(*mut u8, *mut bool) -> bool,
            cb_add_external_clause_lit: unsafe fn(*mut u8) -> i32,
        ) -> UniquePtr<ExternalPropagator>;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Satisfiable,
    Unsatisfiable,
    Unknown,
}

pub struct MiniSat {
    inner: UniquePtr<ffi::Solver>,
}

impl Default for MiniSat {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniSat {
    pub fn new() -> Self {
        Self {
            inner: ffi::new_solver(),
        }
    }

    pub fn add_clause(&mut self, lits: &[i32]) {
        self.inner.pin_mut().add_clause(lits);
    }

    pub fn solve(&mut self) -> Status {
        match self.inner.pin_mut().solve() {
            10 => Status::Satisfiable,
            20 => Status::Unsatisfiable,
            _ => Status::Unknown,
        }
    }

    pub fn val(&mut self, lit: i32) -> i32 {
        self.inner.pin_mut().val(lit)
    }

    pub fn add_observed_var(&mut self, var: i32) {
        self.inner.pin_mut().add_observed_var(var);
    }

    pub fn interrupt(&mut self) {
        self.inner.pin_mut().interrupt();
    }

    pub fn set(&mut self, name: &str, val: i32) -> bool {
        self.inner.pin_mut().set_option(name, val)
    }

    pub fn connect_external_propagator<'a, 'b: 'a, T: ExternalPropagator>(
        &'a mut self,
        propagator: &'b mut T,
    ) {
        fn notify_assignment<T: ExternalPropagator>(state: *mut u8, lits: &[i32]) {
            unsafe { &mut *state.cast::<T>() }.notify_assignment(lits);
        }
        fn notify_new_decision_level<T: ExternalPropagator>(state: *mut u8) {
            unsafe { &mut *state.cast::<T>() }.notify_new_decision_level();
        }
        fn notify_backtrack<T: ExternalPropagator>(state: *mut u8, level: usize) {
            unsafe { &mut *state.cast::<T>() }.notify_backtrack(level);
        }
        fn cb_check_found_model<T: ExternalPropagator>(state: *mut u8, model: &[i32]) -> bool {
            unsafe { &mut *state.cast::<T>() }.cb_check_found_model(model)
        }
        fn cb_decide<T: ExternalPropagator>(state: *mut u8) -> i32 {
            unsafe { &mut *state.cast::<T>() }.cb_decide()
        }
        fn cb_propagate<T: ExternalPropagator>(state: *mut u8) -> i32 {
            unsafe { &mut *state.cast::<T>() }.cb_propagate()
        }
        fn cb_add_reason_clause_lit<T: ExternalPropagator>(state: *mut u8, propagated: i32) -> i32 {
            unsafe { &mut *state.cast::<T>() }.cb_add_reason_clause_lit(propagated)
        }
        fn cb_has_external_clause<T: ExternalPropagator>(
            state: *mut u8,
            is_forgettable: *mut bool,
        ) -> bool {
            unsafe { &mut *state.cast::<T>() }
                .cb_has_external_clause(unsafe { &mut *is_forgettable })
        }
        fn cb_add_external_clause_lit<T: ExternalPropagator>(state: *mut u8) -> i32 {
            unsafe { &mut *state.cast::<T>() }.cb_add_external_clause_lit()
        }

        let is_lazy = propagator.is_lazy();
        let are_reasons_forgettable = propagator.are_reasons_forgettable();
        let cpp = unsafe {
            ffi::new_external_propagator(
                std::ptr::from_mut(propagator).cast::<u8>(),
                is_lazy,
                are_reasons_forgettable,
                notify_assignment::<T>,
                notify_new_decision_level::<T>,
                notify_backtrack::<T>,
                cb_check_found_model::<T>,
                cb_decide::<T>,
                cb_propagate::<T>,
                cb_add_reason_clause_lit::<T>,
                cb_has_external_clause::<T>,
                cb_add_external_clause_lit::<T>,
            )
        };
        self.inner.pin_mut().connect_external_propagator(cpp);
    }

    pub fn disconnect_external_propagator(&mut self) {
        self.inner.pin_mut().disconnect_external_propagator();
    }

    pub fn connect_terminator<'a, 'b: 'a, T: Terminator>(&'a mut self, terminator: &'b mut T) {
        fn terminate<T: Terminator>(state: *mut u8) -> bool {
            unsafe { &mut *state.cast::<T>() }.terminated()
        }
        let cpp = unsafe {
            ffi::new_terminator(std::ptr::from_mut(terminator).cast::<u8>(), terminate::<T>)
        };
        self.inner.pin_mut().connect_terminator(cpp);
    }

    pub fn disconnect_terminator(&mut self) {
        self.inner.pin_mut().disconnect_terminator();
    }
}
