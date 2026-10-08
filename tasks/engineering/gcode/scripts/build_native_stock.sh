set -euo pipefail

source_root=$(realpath "$1")
build_root=$(realpath -m "$2")
script_root=$(cd "$(dirname "$0")" && pwd)
parent_dir=$(dirname "$build_root")
test -d "$parent_dir"
case "$(findmnt -n -o FSTYPE -T "$parent_dir")" in
  tmpfs|ramfs) echo 'Native build requires disk-backed storage' >&2; exit 1 ;;
esac
test ! -e "$build_root"
test "$(df --output=avail -B1 "$parent_dir" | tail -1)" -gt 1073741824
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -gt 4194304

cd "$source_root"
printf '%s\n' \
  '39354ffb8cc4a1e42d380bf975d8470a4c89a35c51731d784ecb1e22ee4ecb44  cbang-source.tar.gz' \
  '422a2dbcca41d8294ae5acbe7ad2e331b14e2b9748ee21c1a0ba1c265fc0808d  camotics-source.tar.gz' \
  | sha256sum --check --strict

mkdir "$build_root"
mkdir "$build_root/compat"
for name in ToolSweep.cpp CompositeSweep.cpp ConicSweep.cpp SpheroidSweep.cpp SpheroidSweep.h; do
  tar -xOf camotics-source.tar.gz \
    "CAMotics-e84665f2fa9d1151f03282ac7e01320bc65e015b/src/camotics/sim/$name" \
    > "$build_root/compat/$name"
done
patch --batch --fuzz=0 -d "$build_root/compat" -p1 < "$script_root/camotics-compat.patch"

cd "$source_root/CAMotics"
ulimit -v 3145728
ulimit -c 0
g++ -O2 -std=c++17 -DUSING_CBANG \
  -I "$build_root/compat" -I src -I src/camotics/sim -I ../cbang/src \
  "$script_root/native_stock_driver.cpp" "$build_root/compat/ToolSweep.cpp" \
  "$build_root/compat/CompositeSweep.cpp" "$build_root/compat/ConicSweep.cpp" \
  "$build_root/compat/SpheroidSweep.cpp" -o "$build_root/native-stock" \
  -L ../cbang/lib -Wl,--start-group build/libCAMotics.a build/libGCode.a \
  build/libSTL.a build/libDXF.a build/dxflib/libdxflib.a \
  -lcbang -lcbang-boost -lssl -lcrypto -lzstd -lre2 -levent -lmysqlclient \
  -lexpat -llz4 -lbz2 -lz -lpthread -lcrypt -ldl -lm -lpython3.10 -Wl,--end-group

sha256sum "$build_root/native-stock" "$script_root/native_stock_driver.cpp" \
  "$script_root/camotics-compat.patch" "$build_root"/compat/* \
  build/libCAMotics.a build/libGCode.a build/libSTL.a build/libDXF.a \
  build/dxflib/libdxflib.a ../cbang/lib/libcbang.a ../cbang/lib/libcbang-boost.a \
  src/camotics/contour/MarchingCubes.cpp src/camotics/contour/GridTreeNode.cpp \
  > "$build_root/build-sha256.txt"
g++ --version > "$build_root/compiler.txt"
