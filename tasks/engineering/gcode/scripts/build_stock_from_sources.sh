set -euo pipefail

archive_root=$(realpath "$1")
build_root=$(realpath -m "$2")
script_root=$(cd "$(dirname "$0")" && pwd)
test -d "$(dirname "$build_root")"
test ! -e "$build_root"
case "$(findmnt -n -o FSTYPE -T "$(dirname "$build_root")")" in
  tmpfs|ramfs) echo 'Native build requires disk-backed storage' >&2; exit 1 ;;
esac
test "$(df --output=avail -B1 "$(dirname "$build_root")" | tail -1)" -gt 1073741824
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -gt 4194304
cd "$archive_root"
printf '%s\n' \
  '39354ffb8cc4a1e42d380bf975d8470a4c89a35c51731d784ecb1e22ee4ecb44  cbang-source.tar.gz' \
  '422a2dbcca41d8294ae5acbe7ad2e331b14e2b9748ee21c1a0ba1c265fc0808d  camotics-source.tar.gz' \
  | sha256sum --check --strict
mkdir "$build_root"
mkdir "$build_root/cbang" "$build_root/CAMotics" "$build_root/tmp"
export TMPDIR="$build_root/tmp"
cp --reflink=auto cbang-source.tar.gz camotics-source.tar.gz "$build_root/"
tar -xzf cbang-source.tar.gz -C "$build_root/cbang" --strip-components=1
tar -xzf camotics-source.tar.gz -C "$build_root/CAMotics" --strip-components=1
patch --batch --fuzz=0 -d "$build_root/CAMotics" -p1 < "$script_root/camotics-sparse-empty.patch"
scons -C "$build_root/cbang" -j1 strict=0 > "$build_root/cbang-build.log" 2>&1
CBANG_HOME="$build_root/cbang" scons -C "$build_root/CAMotics" -j1 \
  strict=0 with_gui=0 with_tpl=0 > "$build_root/camotics-build.log" 2>&1
bash "$script_root/build_native_stock.sh" "$build_root" "$build_root/driver"
