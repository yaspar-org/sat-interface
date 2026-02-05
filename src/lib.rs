//! This crate provides an abstraction layer to describe all SAT solvers. It provides a common
//! interface for different implementations of SAT solvers.
//!
//! c.f. [SatSolver], [SolverState]

#[cfg(feature = "cadical")]
pub mod cadical;
pub mod multisolver;

use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::vec::IntoIter;

/// This trait provides an abstraction for a solver state.
///
/// Invariant: exactly one of [Self::is_sat], [Self::is_unsat], and [Self::is_unknown] should be true.
pub trait SolverState {
    fn is_sat(&self) -> bool;
    fn is_unsat(&self) -> bool;
    fn is_unknown(&self) -> bool;
}

/// This trait captures the ability to add constraints to a SAT solver.
pub trait AddConstraints<C> {
    fn insert(&mut self, constraints: &C);
}

impl<T, C> AddConstraints<C> for [T]
where
    T: AddConstraints<C>,
{
    fn insert(&mut self, constraints: &C) {
        for t in self {
            t.insert(constraints);
        }
    }
}

/// This trait captures the interface of a SAT solver.
///
/// Its trait constraints require that its implementation is able to add clauses and formulas as constraints.
pub trait SatSolver: AddConstraints<Clause> + AddConstraints<Formula> {
    type Status: SolverState;

    /// Create a new SAT solver
    fn new() -> Self;
    /// Decide whether current constraints are satisfiable or not
    fn solve(&mut self) -> Self::Status;
    /// After [Self::solve] returns sat, query the solver for the boolean value of the SAT variable `lit`.
    ///
    /// Requirement: `lit` must be a SAT variable present in the constraints.
    fn val(&mut self, lit: i32) -> bool;
    /// After [Self::solve] returns sat, obtain the full model.
    fn load_model(&mut self) -> Model;

    /// After [Self::solve] returns sat, block the current model. It is useful for model enumeration.
    fn block_model(&mut self);
}

/// A clause; it's a list of disjoined literals
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Clause(pub Vec<i32>);

/// A formula; it's a list of conjoined clauses
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Formula(pub Vec<Clause>);

/// A model as a vector mapping literals to boolea values
///
/// Note that the indices are off by one as SAT variables start at 1.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Model(pub Vec<bool>);

impl Display for Clause {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for i in 0..self.0.len() {
            write!(f, "{}", self.0[i])?;
            if i < self.0.len() - 1 {
                f.write_str(", ")?;
            }
        }
        f.write_str("]")
    }
}

impl Display for Formula {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for i in 0..self.0.len() {
            write!(f, "{}", self.0[i])?;
            if i < self.0.len() - 1 {
                f.write_str(";\n")?;
            }
        }
        f.write_str("]")
    }
}

impl Clause {
    /// An empty clause
    pub fn empty() -> Self {
        Self::new(vec![])
    }

    /// A new clause with specified literals
    pub fn new(lits: Vec<i32>) -> Self {
        Clause(lits)
    }

    /// A singleton clause
    pub fn single(lit: i32) -> Self {
        Self::new(vec![lit])
    }

    /// Obtain a slice of the literals
    pub fn lits(&self) -> &[i32] {
        &self.0
    }

    fn filter_by_model<'a, 'b>(&'a self, model: &'b Model) -> impl Iterator<Item = &'a i32> + 'a
    where
        'b: 'a,
    {
        self.0.iter().filter(|&i| {
            let idx = (i.abs() - 1) as usize;
            if *i < 0 { !model.0[idx] } else { model.0[idx] }
        })
    }

    /// Given a model, check whether the current clause is true.
    pub fn evaluate(&self, model: &Model) -> bool {
        self.filter_by_model(model).any(|_| true)
    }

    /// Return a list of true literals
    pub fn find_true_vars(&self, model: &Model) -> Vec<i32> {
        self.filter_by_model(model).copied().collect()
    }

    /// Return a list of true literals that are not in the given `set`
    pub fn filter_true_vars(&self, model: &Model, set: &HashSet<i32>) -> Vec<i32> {
        self.filter_by_model(model)
            .filter(|&i| !set.contains(i))
            .copied()
            .collect()
    }

    /// Return a list of true literals as a clause
    ///
    /// c.f. [Self::find_true_vars]
    pub fn filter_vars(&self, model: &Model) -> Clause {
        Clause(self.find_true_vars(model))
    }

    /// Return an iterator for the literals
    pub fn iter(&self) -> impl Iterator<Item = &i32> {
        self.0.iter()
    }

    /// Concatenate two clauses
    pub fn concat(&self, clause: &Clause) -> Self {
        let mut c = self.clone();
        c.concat_mut(clause);
        c
    }

    /// Concatenate a given clause by modifying the self clause
    pub fn concat_mut(&mut self, clause: &Clause) -> &mut Self {
        self.0.extend(&clause.0);
        self
    }
}

impl IntoIterator for Clause {
    type Item = i32;
    type IntoIter = IntoIter<i32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Formula {
    /// An empty formula
    pub fn empty() -> Formula {
        Formula(vec![])
    }

    /// A new formula with the given vector of clauses
    pub fn new(clauses: Vec<Clause>) -> Formula {
        Formula(clauses)
    }

    /// Add a clause to the current formula
    pub fn add(&mut self, clause: Clause) {
        self.0.push(clause);
    }

    /// Return a formula only containing true literals in each clause
    pub fn filter_clauses(&self, model: &Model) -> Formula {
        Formula(self.0.iter().map(|c| c.filter_vars(model)).collect())
    }

    fn find_implicant_helper(self) -> HashSet<i32> {
        let mut vset = HashSet::new();
        let mut clauses: Vec<_> = self.0;

        loop {
            let begin_sz = vset.len();
            let mut kept_clauses = vec![];
            for clause in clauses.into_iter() {
                // 1. if clause has a variable in vset, then this clause is true and we drop it.
                // 2. if not, then does it have only one variable? if so, then we add the variable to vset.
                // 3. otherwise, we keep clause.

                if clause.0.iter().any(|&i| vset.contains(&i)) {
                    continue;
                }
                if clause.0.len() == 1 {
                    vset.insert(clause.0[0]);
                } else {
                    kept_clauses.push(clause);
                }
            }
            clauses = kept_clauses;
            if clauses.is_empty() {
                break;
            }
            if begin_sz == vset.len() {
                // we've known clauses is non-empty.
                let c = clauses.pop().unwrap();
                vset.insert(c.0[0]);
            }
        }

        vset
    }

    /// Obtain an implicant of the formula based on the model
    pub fn find_implicant(&self, model: &Model) -> HashSet<i32> {
        self.filter_clauses(model).find_implicant_helper()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Clause> {
        self.0.iter()
    }

    pub fn concat(&self, clause: &Clause) -> Self {
        let mut f = self.clone();
        f.concat_mut(clause);
        f
    }

    pub fn concat_mut(&mut self, clause: &Clause) -> &mut Self {
        for c in self.0.iter_mut() {
            c.concat_mut(clause);
        }
        self
    }

    pub fn distribute(&self, formula: &Formula) -> Self {
        Formula::new(formula.iter().flat_map(|c| self.concat(c)).collect())
    }

    pub fn distribute_mut(&mut self, formula: &Formula) -> &mut Self {
        self.0 = self.distribute(formula).0;
        self
    }
}

impl IntoIterator for Formula {
    type Item = Clause;
    type IntoIter = IntoIter<Clause>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Model {
    /// Obtain a model based on a given evaluation function `f`
    pub fn get_model<F>(vn: usize, f: F) -> Model
    where
        F: FnMut(usize) -> bool,
    {
        Model((1..vn + 1).map(f).collect())
    }

    /// Return a blocking clause for the model
    pub fn negate(&self) -> Clause {
        Clause(
            self.0
                .iter()
                .enumerate()
                .map(|(idx, v)| {
                    if *v {
                        -(idx as i32 + 1)
                    } else {
                        idx as i32 + 1
                    }
                })
                .collect(),
        )
    }
}
