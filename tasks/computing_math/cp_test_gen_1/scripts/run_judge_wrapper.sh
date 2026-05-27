#!/bin/bash
set -e

OUTPUT_DIR="$1"
REFERENCE_DIR="$2"
EVAL_DIR="$3"

if [ -z "$OUTPUT_DIR" ] || [ -z "$REFERENCE_DIR" ] || [ -z "$EVAL_DIR" ]; then
    echo "Usage: run_judge_wrapper.sh <output_dir> <reference_dir> <eval_dir>" >&2
    exit 1
fi

if [ ! -f "$OUTPUT_DIR/gen.cpp" ]; then
    echo "gen.cpp not found in $OUTPUT_DIR" >&2
    exit 1
fi

rm -rf "$EVAL_DIR/submissions" "$EVAL_DIR/tests" "$EVAL_DIR/verdicts.txt" \
       "$EVAL_DIR/gen.cpp" "$EVAL_DIR/gen_bin" "$EVAL_DIR/reference" \
       "$EVAL_DIR/judge.sh"
mkdir -p "$EVAL_DIR/submissions"

cp "$OUTPUT_DIR/gen.cpp" "$EVAL_DIR/gen.cpp"
cp "$REFERENCE_DIR"/*.cpp "$EVAL_DIR/submissions/"
cp "$REFERENCE_DIR/judge.sh" "$EVAL_DIR/judge.sh"
chmod +x "$EVAL_DIR/judge.sh"

# judge.sh uses set -e which aborts on non-zero exit codes from timeout (TLE
# submissions). Insert set +e before the judging loop so compilation still
# fails fast but judging handles non-zero exits from timeout/crashes.
sed -i '/^echo "Judging submissions/i set +e' "$EVAL_DIR/judge.sh"

cd "$EVAL_DIR"
timeout 1200 bash judge.sh

echo ""
echo "=== VERDICTS ==="
cat "$EVAL_DIR/verdicts.txt"
