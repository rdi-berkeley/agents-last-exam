#!/usr/bin/env bash
# Verify — computing_math/cp_test_gen_1 : g++ compiles + runs a C++17 program
set -uo pipefail
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
g++ --version >/dev/null 2>&1 || fail "g++ missing"
W="$(mktemp -d)"; cat > "$W/t.cpp" <<'CPP'
#include <bits/stdc++.h>
using namespace std;
int main(){ vector<int> v{3,1,2}; sort(v.begin(),v.end()); for(auto x:v)cout<<x; auto f=[](auto a){return a*a;}; cout<<f(4); return 0; }
CPP
g++ -std=c++17 -O2 "$W/t.cpp" -o "$W/t" 2>/dev/null || fail "g++ -std=c++17 compile failed"
OUT="$("$W/t")"; [ "$OUT" = "12316" ] || fail "unexpected program output: $OUT"
echo "[verify] g++ C++17 compile+run OK (out=$OUT)"
echo "[verify] PASS"
