# sat-interface

This crate provides an abstraction interface for SAT solvers. 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

This project includes a fork of MiniSat (https://github.com/amarshah1/minisat) in
the `deps/minisat` git submodule, which is compiled when the `minisat` feature is
enabled. The fork is based on the MiniSatUP work
(https://github.com/hchenqide/minisat), which added an IPASIR-UP interface to
MiniSat. MiniSat is licensed under the MIT License; see [NOTICE](NOTICE) and
`deps/minisat/LICENSE`.
