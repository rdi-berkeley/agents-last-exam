#!/usr/bin/env bash
# Set up the S3 run-results bucket + the IAM instance profile that lets each
# sandbox push its output there. This is the DEFAULT output path for full
# benchmark runs (the AWS analogue of gcloud's ale-run-results bucket).
#
# Creates (idempotent):
#   • S3 bucket  ale-run-results-<account>           (results land here)
#   • IAM role + instance profile  ale-sandbox       (trust ec2; s3 write results
#     + s3 read on ale-* data buckets)
# Appends ALE_RESULTS_BUCKET + ALE_INSTANCE_PROFILE to tools/aws/.aws-env.
#
# After running: in configs/environments/environment_aws.yaml set
#   output_path: s3://${env:ALE_RESULTS_BUCKET}
# and add to each snapshot's aws: block
#   iam_instance_profile: ${env:ALE_INSTANCE_PROFILE}
# The in-box aws CLI then pushes output via the instance role (no key injected).
# NB: the AMI must contain the aws CLI for the VM-side push to work.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
R="${AWS_REGION:?set AWS_REGION}"
ACCT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${ALE_RESULTS_BUCKET:-ale-run-results-${ACCT}}"
PROFILE=ale-sandbox
OUT="$(dirname "$0")/.aws-env"
say() { printf '\n=== %s ===\n' "$*"; }

say "results bucket $BUCKET"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$R" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$R"
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$R" \
      --create-bucket-configuration "LocationConstraint=$R"
  fi
fi

say "IAM role + instance profile $PROFILE"
if ! aws iam get-role --role-name "$PROFILE" >/dev/null 2>&1; then
  TRUST=$(mktemp)
  cat > "$TRUST" <<'J'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
J
  aws iam create-role --role-name "$PROFILE" \
    --assume-role-policy-document "file://$TRUST" \
    --description "ALE sandbox: S3 results write + data read"
  rm -f "$TRUST"
fi
POL=$(mktemp)
cat > "$POL" <<J
{"Version":"2012-10-17","Statement":[
 {"Sid":"Results","Effect":"Allow",
  "Action":["s3:PutObject","s3:AbortMultipartUpload","s3:ListBucket"],
  "Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"]},
 {"Sid":"DataRead","Effect":"Allow",
  "Action":["s3:GetObject","s3:ListBucket"],
  "Resource":["arn:aws:s3:::ale-*","arn:aws:s3:::ale-*/*"]}]}
J
aws iam put-role-policy --role-name "$PROFILE" --policy-name "${PROFILE}-s3" \
  --policy-document "file://$POL"
rm -f "$POL"
aws iam create-instance-profile --instance-profile-name "$PROFILE" 2>/dev/null || true
aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" \
  --role-name "$PROFILE" 2>/dev/null || true

# record for the env yaml
grep -q ALE_RESULTS_BUCKET "$OUT" 2>/dev/null || cat >> "$OUT" <<ENV
export ALE_RESULTS_BUCKET=${BUCKET}
export ALE_INSTANCE_PROFILE=${PROFILE}
ENV

say "DONE — results bucket s3://$BUCKET, instance profile $PROFILE"
echo "set output_path: s3://\${env:ALE_RESULTS_BUCKET} and"
echo "iam_instance_profile: \${env:ALE_INSTANCE_PROFILE} in environment_aws.yaml"
