# Google Cloud VMs: command-first quickstart

This is the condensed setup for ALE's supported Google Cloud provider. For
explanations, current limitations, and networking context, see the
[Google Cloud VMs website guide](https://agents-last-exam.org/docs?p=pages/google-cloud.html).

Google Cloud products, credits, and Free Trial restrictions can change. Review
the current [Free Cloud features](https://docs.cloud.google.com/free/docs/free-cloud-features)
before relying on trial credits. Google documents Windows Server VMs as outside
the Free Trial, so plan on paid billing for Windows benchmark runs.

## 1. Install and authenticate

Install the
[Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk), then:

```bash
gcloud auth login
git clone git@github.com:rdi-berkeley/agents-last-exam.git
cd agents-last-exam
uv sync --all-packages
```

The host's `gcloud` login creates and deletes VMs. A separate service-account
key is injected into each sandbox for Cloud Storage.

## 2. Create the project

Choose a globally unique project ID and the public CIDR of the machine that
runs ALE:

```bash
export GCP_PROJECT="ale-$(whoami)"
export GCP_REGION="us-central1"
export GCP_SA_NAME="ale-runner"
export GCP_SA_EMAIL="${GCP_SA_NAME}@${GCP_PROJECT}.iam.gserviceaccount.com"
export GCP_BUCKET="${GCP_PROJECT}-ale-results"
export ALE_CLIENT_CIDR="203.0.113.10/32"  # replace with the ALE host's public IP/CIDR
```

Create the project, attach billing, and enable the APIs:

```bash
gcloud projects create "$GCP_PROJECT" --name="ALE"
gcloud config set project "$GCP_PROJECT"

gcloud billing accounts list
read -p "Billing account ID: " BILLING_ID
gcloud billing projects link "$GCP_PROJECT" --billing-account="$BILLING_ID"

gcloud services enable compute.googleapis.com storage.googleapis.com
```

`gcloud projects create` is not idempotent. If the project already exists,
skip that command and set it as the active project.

## 3. Create the guest storage identity

```bash
gcloud iam service-accounts create "$GCP_SA_NAME" \
  --display-name="ALE storage access"

for role in storage.objectViewer serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
    --member="serviceAccount:$GCP_SA_EMAIL" \
    --role="roles/$role" \
    --condition=None
done

mkdir -p secret
gcloud iam service-accounts keys create secret/gcp_key.json \
  --iam-account="$GCP_SA_EMAIL"
```

The repository ignores `secret/gcp_key.json`. Do not commit or share it.

## 4. Copy the published images

```bash
for image in ale-ubuntu22 ale-win10; do
  gcloud compute images create "$image" \
    --source-image="$image" \
    --source-image-project=agenthle-488519
done
```

## 5. Create restricted network access

ALE connects directly to the in-guest CUA server on TCP port 5000. Restrict the
firewall rule to the ALE host. Do not use `0.0.0.0/0`.

```bash
gcloud compute networks create ale-vpc --subnet-mode=auto

gcloud compute firewall-rules create ale-allow-cua \
  --network=ale-vpc \
  --direction=INGRESS \
  --allow=tcp:5000 \
  --source-ranges="$ALE_CLIENT_CIDR" \
  --target-tags=ale-run
```

If the ALE host's public IP changes, update the rule:

```bash
gcloud compute firewall-rules update ale-allow-cua \
  --source-ranges="$ALE_CLIENT_CIDR"
```

## 6. Create an optional results bucket

Skip this section if you will keep `output_path: local`.

```bash
gcloud storage buckets create "gs://$GCP_BUCKET" \
  --project="$GCP_PROJECT" \
  --location="$GCP_REGION" \
  --uniform-bucket-level-access

gcloud storage buckets add-iam-policy-binding "gs://$GCP_BUCKET" \
  --member="serviceAccount:$GCP_SA_EMAIL" \
  --role="roles/storage.objectAdmin"
```

To upload task output directly from each sandbox, set this in a copy of
`configs/environments/environment_gcloud.yaml`:

```yaml
output_path: gs://<your-results-bucket>
```

The default `output_path: local` copies output into each local run directory.

## 7. Configure secrets

```bash
cp secret/.env.example secret/.env
```

Set the agent API key required by your chosen preset and the Google Cloud
variables:

```dotenv
OPENROUTER_API_KEY=...
GCP_PROJECT=<your-project-id>
GCP_SA_KEY=secret/gcp_key.json
```

Judge-based tasks can require separate evaluator keys under
`secret/eval_time/`. The hello-world task does not.

## 8. Validate and run the Linux demo

The shipped `example_exp.yaml` uses the Google Cloud environment and
`selected_tasks/helloworld.txt`.

```bash
uv run python -m ale_run run example_exp.yaml --dry-run
uv run python -m ale_run run example_exp.yaml
```

Run records land under:

```text
.logs/ale/my_experiment/
```

Use `selected_tasks/hello_both.txt` after confirming that the project can run
Windows VMs.

## 9. Continue to a benchmark task list

Use `selected_tasks/unlicensed.txt` for the complete public set that does not
require licensed software. The environment profile maps CPU, GPU, Ubuntu, and
Windows snapshots to their corresponding images and zones.

```yaml
tasks: selected_tasks/unlicensed.txt
concurrency: 8
cleanup_mode: delete
```

Choose concurrency according to project quota, GPU availability, LLM rate
limits, and budget. Use `--resume` to skip units already recorded as
`completed` or `timeout`.

## Cleanup and recovery

The provider deletes each VM when `cleanup_mode: delete`. After an abrupt host
termination, inspect tagged instances:

```bash
gcloud compute instances list --filter="tags.items=ale-run"
```

Delete confirmed leftovers explicitly:

```bash
gcloud compute instances delete <instance-name> --zone=<zone>
```
