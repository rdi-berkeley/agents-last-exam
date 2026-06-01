# Google Cloud setup — from zero to your first ALE run

This guide assumes **you have never used Google Cloud before**. Total time
is about 20 minutes. Costs: new Google Cloud accounts get **$300 in free
trial credits** (90 days); one `demo/hello_win` run costs roughly $0.05 of
that.

The walkthrough has two kinds of steps:

- 🖱️  **Manual**: must be done in the browser (account creation, billing,
  credit-card entry). You'll spend ~5 minutes total here.
- ⌨️  **Automated**: copy-paste a single block of shell commands. You'll
  spend ~5 minutes here, mostly waiting for the image copy.

---

## Step 1 — 🖱️ Create a Google Cloud account

If you already have one, skip to Step 2.

1. Open **<https://cloud.google.com/free>** in a browser.
2. Click **Get started for free** (top right).
3. Sign in with a Google account (or create one).
4. Select your **country** and accept the Terms of Service.
5. Add a **payment method**. A credit card is required even for the free
   tier — Google verifies you're a real person but won't charge you until
   you exhaust the $300 credit or 90 days, whichever comes first. You can
   set a hard cap later if you're nervous.
6. Click **Start my free trial**.

You're now in the Google Cloud Console at <https://console.cloud.google.com>.
A starter project called *"My First Project"* is created automatically; you
can use it or create a fresh one in Step 3.

---

## Step 2 — 🖱️ Install the `gcloud` CLI on your laptop

The shell automation in Step 3 needs the `gcloud` command. Install it
once:

- **macOS:** `brew install --cask google-cloud-sdk`
  (or follow <https://cloud.google.com/sdk/docs/install#mac>)
- **Linux:** <https://cloud.google.com/sdk/docs/install#linux>
- **Windows:** <https://cloud.google.com/sdk/docs/install#windows>

Verify the install:

```bash
gcloud --version
```

Then sign in (opens a browser window):

```bash
gcloud auth login
```

You're done with manual setup.

> 💡 **Hand off the rest to your coding agent.** From here, everything is
> shell commands and a `.env` paste. You can drop into your editor with
> Claude Code, Codex, Cursor, etc. and tell it:
>
> > *"Open `docs/SETUP_GCP.md` and execute Steps 3–5 for me. Ask me before
> > you need the billing-account ID and my LLM API key, and stop after the
> > demo run completes."*
>
> The agent will run the bash block, prompt you for the two secrets, and
> verify `example_exp.yaml` runs end-to-end. The rest of this document is
> structured so an agent can follow it linearly.

---

## Step 3 — ⌨️ One-block automation

Copy this whole block into your terminal **after editing the two
variables at the top**. It creates a project, enables APIs, makes a
service account, copies the sandbox image, sets up networking, and
creates a results bucket for optional GCS output upload.

```bash
# ─── EDIT THESE TWO LINES ──────────────────────────────────────────────
export GCP_PROJECT="ale-$(whoami)-$(date +%s | tail -c 5)"   # any unique id, lowercase, ≤30 chars
export GCP_REGION="us-west1"                                  # change if you prefer another region
# ───────────────────────────────────────────────────────────────────────

export GCP_ZONE="${GCP_REGION}-b"
export GCP_SA_NAME="ale-runner"
export GCP_SA_EMAIL="${GCP_SA_NAME}@${GCP_PROJECT}.iam.gserviceaccount.com"
export GCP_RESULTS_BUCKET="${GCP_PROJECT}-ale-results"

cd <path-to>/agents-last-exam       # ← run from the repo root

# 3a. Create the project and make it default.
gcloud projects create "${GCP_PROJECT}" --name="ALE Workspace"
gcloud config set project "${GCP_PROJECT}"

# 3b. Link billing to the project (REQUIRED — even with free credits).
#     Lists your billing accounts; pick the one with credits attached.
gcloud billing accounts list
read -p "Paste billing-account ID from the list above: " BILLING_ID
gcloud billing projects link "${GCP_PROJECT}" --billing-account="${BILLING_ID}"

# 3c. Enable the APIs we need.
gcloud services enable compute.googleapis.com storage.googleapis.com

# 3d. Create the service account + download a JSON key.
gcloud iam service-accounts create "${GCP_SA_NAME}" \
  --display-name="ALE runner"

for role in \
    roles/compute.instanceAdmin.v1 \
    roles/compute.networkUser \
    roles/iam.serviceAccountUser \
    roles/storage.objectViewer \
    roles/serviceusage.serviceUsageConsumer ; do
  gcloud projects add-iam-policy-binding "${GCP_PROJECT}" \
    --member="serviceAccount:${GCP_SA_EMAIL}" \
    --role="${role}" --condition=None
done

mkdir -p secret
gcloud iam service-accounts keys create secret/gcp_key.json \
  --iam-account="${GCP_SA_EMAIL}"

# 3e. Copy the published sandbox image into your project (~3 min).
gcloud compute images create ale-unified-v1 \
  --source-image=ale-unified-v1 \
  --source-image-project=agenthle-488519

# 3f. Create a VPC and firewall rules.
gcloud compute networks create ale-vpc --subnet-mode=auto

gcloud compute firewall-rules create ale-allow-cua \
  --network=ale-vpc --direction=INGRESS \
  --allow=tcp:5000 --source-ranges=0.0.0.0/0

# (Optional) open RDP if you want to debug Windows guests:
gcloud compute firewall-rules create ale-allow-rdp \
  --network=ale-vpc --direction=INGRESS \
  --allow=tcp:3389 --source-ranges=0.0.0.0/0

# 3g. Results bucket for optional GCS output upload (artifacts_path.output_path).
if ! gcloud storage buckets describe "gs://${GCP_RESULTS_BUCKET}" \
    --project="${GCP_PROJECT}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${GCP_RESULTS_BUCKET}" \
    --project="${GCP_PROJECT}" \
    --location="${GCP_REGION}" \
    --uniform-bucket-level-access
fi
gcloud storage buckets add-iam-policy-binding "gs://${GCP_RESULTS_BUCKET}" \
  --member="serviceAccount:${GCP_SA_EMAIL}" \
  --role="roles/storage.objectAdmin"

echo
echo "✓ GCP project ready: ${GCP_PROJECT}"
echo "✓ Service account key: $(pwd)/secret/gcp_key.json"
echo "✓ Results bucket: gs://${GCP_RESULTS_BUCKET}"
```

If a step fails, fix it and re-run only that step — every command is
idempotent except `gcloud projects create` (which errors if the project
already exists).

---

## Step 4 — 🖱️ Fill in `secret/.env`

```bash
cp secret/.env.example secret/.env
```

Open `secret/.env` and paste your LLM API key and the GCP values from
Step 3:

```dotenv
# Pick one (claude-code auto-routes OPENROUTER through Anthropic):
ANTHROPIC_API_KEY=sk-ant-...
# OR
OPENROUTER_API_KEY=sk-or-...

GCP_PROJECT=<the value of $GCP_PROJECT from step 3>
GCP_SA_KEY=secret/gcp_key.json
```

You can leave `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `BRAVE_API_KEY` blank
— they're only used by other agents.

---

## Step 5 — ⌨️ Run the demo

```bash
uv sync --extra dev
uv run python -m ale_run run example_exp.yaml --dry-run   # validates config
uv run python -m ale_run run example_exp.yaml             # real run
```

Expect ~5 minutes: VM boot is 3–4 min, agent + eval is ~1 min, teardown
is ~30 s. A successful run prints:

```
agent                 task                                      var  status      score     dur
----------------------------------------------------------------------------------------------
claude_code           demo/hello_win                              0  completed    1.00   42.3s
```

Artifacts land in `.logs/my_experiment/<run_id>/`. The runner deletes the
VM on exit (success or failure). If the process is killed mid-run,
clean up leftovers:

```bash
gcloud compute instances list --filter="name~^ale-"
gcloud compute instances delete <name> --zone=<zone>
```

---

## Other Notes

### Region / zone

The gcloud env config lists the fallback zones under its `snapshots:`
block (see
[`configs/environments/gcloud.yaml`](../configs/environments/gcloud.yaml)).
If you set `GCP_REGION` to something else in Step 3, also edit the
`zones:` list in that env config (or copy it to a new
`configs/environments/*.yaml` and point your experiment's `environment:`
at it).

### Image / machine-type compatibility

- `ale-unified-v1` is a **Windows** image. The Linux demo `demo/hello`
  needs a Linux image (see the `cpu-free-ubuntu` snapshot tag in the env
  config). Use `demo/hello_win` for now.
- `c4-*` / `m4-*` / `x4-*` machine families require **all** disks to be
  `hyperdisk-balanced`. The default config uses `e2-standard-4` which
  is fine. If you switch to `c4-standard-4`, also set
  `data_disk_type: hyperdisk-balanced` in the env config.
- For CPU-only demo runs, `e2-*` and `n1-*` are cheapest. `g2-*` adds
  GPU but costs ~10× more.

### Per-task GCS-staged data (optional)

Tasks that declare `requires_task_data=True` rsync `input/`,
`software/`, `reference/` from `gs://ale-data-public`, the shared public
mirror we maintain. You do not configure this bucket. 

To upload run outputs to GCS (instead of pulling them locally), set
`output_path` in your **environment** yaml (`configs/environments/<env>.yaml`,
alongside `provider` + `task_data_source`). Step 3 already created the bucket
and granted the runner service account access:

```yaml
# in configs/environments/gcloud.yaml
output_path: gs://<GCP_PROJECT>-ale-results    # null = skip, "local" = pull to run dir
```

### Hard cost cap

If you want a budget alert (recommended):

1. Console → **Billing** → **Budgets & alerts** → **Create budget**.
2. Pick the ALE project, set a monthly limit (e.g. $50), enable email
   alerts at 50% / 90% / 100%.

GCP can't auto-shutdown on overrun, so this is alerts-only — but it'll
catch a runaway loop before damage.
