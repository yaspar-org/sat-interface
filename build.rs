// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "minisat")]
    build_minisat();
}

/// Builds the minisat gitsubmodule together + the bridge in src/minisat
#[cfg(feature = "minisat")]
fn build_minisat() {
    if !std::path::Path::new("deps/minisat/minisat/minisatup.h").exists() {
        panic!(
            "deps/minisat is empty; run `git submodule update --init` to fetch \
             https://github.com/amarshah1/minisat"
        );
    }

    cxx_build::bridge("src/minisat/mod.rs")
        .file("src/minisat/bridge.cc")
        .file("deps/minisat/minisat/minisatup.cc")
        .file("deps/minisat/minisat/core/Solver.cc")
        .file("deps/minisat/minisat/utils/Options.cc")
        .file("deps/minisat/minisat/utils/System.cc")
        .include("deps/minisat")
        .std("c++17")
        .define("__STDC_LIMIT_MACROS", None)
        .define("__STDC_FORMAT_MACROS", None)
        .define("NDEBUG", None)
        .flag_if_supported("-Wno-literal-suffix")
        .flag_if_supported("-Wno-reserved-user-defined-literal")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-class-memaccess")
        .flag_if_supported("-Wno-parentheses")
        .flag_if_supported("-Wno-sign-compare")
        .opt_level(3)
        .compile("minisat");

    println!("cargo:rerun-if-changed=src/minisat");
    println!("cargo:rerun-if-changed=deps/minisat");
}
