# Offline cloud task allocation demo

This folder contains an offline demonstration and the live folder-driven
allocation service. In the live workflow, a user uploads a `.py` script into
`cluster_task`; the hub writes an assignment JSON ticket beside the script,
the assigned worker executes it, and the output is stored in `cluster_result`.

The task's file size is measured by the hub and used as both `file_size` and
`disk_size`; Python scripts are typed as `text`, and baseline resource hints
default to zero. Tasks uploaded as scripts currently form independent jobs.
The offline path also supports explicit dependencies:

```text
TaskInput + InfrastructureContext
  -> TaskResourcePredictor (scikit-learn Gradient Boosting)
  -> TaskGraphEncoder (supervised resource refiner + task embeddings)
  -> build_joint_state (task/server candidate tensor and feasibility mask)
  -> MARLScheduler (shared actor, centralized critic, PPO update)
```

It also includes the live components adapted from the downloaded `hub.py`,
`agent.py`, `task_runner.py`, and `dashboard.html`: the hub accepts telemetry,
builds live task/server states, writes assigned tickets to a shared directory,
and learns from completed-task feedback.

## Run the live hub

Install the dependencies in `requirements.txt`. Train the predictor on measured
resource labels and save the fitted model; save the GNN encoder and scheduler
checkpoints with the persistence helpers. Then configure paths and start the
hub from this folder:

```powershell
$env:MARL_PREDICTOR_PATH = "C:\models\task_predictor.joblib"
$env:MARL_GNN_PATH = "C:\models\task_gnn.pt"
$env:MARL_SCHEDULER_PATH = "C:\models\marl_scheduler.pt"
$env:MARL_TASKS_DIR = "\\192.168.137.1\cluster_task"
$env:MARL_MAX_UPLOAD_BYTES = "67108864"
$env:MARL_NODE_IDS = "S1,S2,S3,S4"
$env:MARL_API_TOKEN = "<long-random-secret>"
$env:MARL_HUB_HOST = "0.0.0.0"
python hub.py
```

Open `http://<hub-ip>:8000/` for the dashboard. Upload `.py` scripts there, or
copy them directly into `cluster_task`; the hub watches for completed,
previously unassigned script files. Scripts must have a unique safe basename
ending in `.py`, and each must be no larger than the configured 64 MiB default
upload limit. For a shared-folder upload, wait until the file is fully copied;
the hub checks every 100 ms and waits for the size and modification time to
remain stable for 250 ms before scheduling it. Dashboard uploads are scheduled
as soon as the completed upload reaches the hub. Configure the shared directories on the hub and each worker to
point to the same locations. On each worker, set its node ID, then start both
processes in separate terminals:

```powershell
$env:MARL_TELEMETRY_URL = "http://<hub-ip>:8000/api/telemetry"
$env:MARL_NODE_ID = "S1"
$env:MARL_TASKS_DIR = "\\<hub-ip>\cluster_task"
$env:MARL_RESULTS_DIR = "\\<hub-ip>\cluster_result"
python agent.py
python task_runner.py
```

Use matching unique node IDs on the hub and workers, for example `S1`, `S2`,
`S3`, and `S4`. `agent.py` sends one heartbeat per machine; each
`task_runner.py` polls the shared JSON tickets. A GPU baseline hint greater
than zero makes CPU-only nodes infeasible. Folder-drop users can place an
optional `<script_stem>.hints.json` file next to the script with `doc_type`,
`baseline_cpu_req`, and `baseline_gpu_req`; uploaded scripts can set CPU/GPU
hints in the dashboard.

The hub creates a `<script_stem>.json` ticket containing the selected
`assigned_to` server, the predicted resources, and a `PENDING` status. Workers
poll the ticket share, execute only `.py` files directly inside that share,
then update the ticket and write `<script_stem>_result.txt` in
`cluster_result`. Poll `/api/tasks/{task_id}` or use the dashboard to view
ticket status and output-file location.
For live Python uploads, a GPU hint of zero means CPU-only; set a positive GPU
hint to require GPU capacity. The scheduler cannot reliably infer whether an
arbitrary Python script uses a GPU from its file size and file type alone.

The hub remains available for telemetry without models. Script uploads require
the API token; if model paths or eligible live nodes are not configured yet,
the uploaded script stays in `cluster_task` until it can be scheduled. Direct
JSON batch submission returns HTTP 503 without the model bundle; invalid
tokens receive HTTP 401. The agent reports CPU, RAM, network, RTT, and NVIDIA GPU utilization when
`nvidia-smi` is available. GPU tasks cannot be assigned to nodes that report no
GPUs. Configure `MARL_NODE_IDS` as a comma-separated ID list matching the
workers, and set `MARL_NODE_COST_PER_SECOND` on each agent to publish its
relative cost rate. Script upload and scheduling are disabled until
`MARL_API_TOKEN` is set; the dashboard asks for this token before uploading a
script. Same-origin dashboard access needs no CORS configuration. Set
`MARL_DASHBOARD_ORIGINS` to a comma-separated allowlist only if hosting the
dashboard on another origin.

For real predictor training, `TaskResourcePredictor.fit(tasks, contexts,
targets)` requires measured target dictionaries with exactly
`target_exec_time`, `target_cpu_req`, `target_gpu_req`, and `target_ram_req`
for every task. The GNN and MARL checkpoints also need meaningful training.
The demo trains the scheduler for 32 synthetic episodes by default; it is a
pipeline and learning-plumbing check, not evidence of production policy quality.
Use isolated, trusted workers for submitted scripts: the task runner executes
those scripts as Python programs with the worker's OS permissions.

## Run the synthetic demonstration

Requires Python 3.10+, PyTorch, NumPy, scikit-learn, and joblib:

```powershell
python main_engine.py --tasks 8 --seed 7 --checkpoint-dir models
```

Run the local four-agent integration test (uses simulated heartbeats and
temporary task/result folders, not real machines or network shares):

```powershell
python test_live_flow.py
python test_learning_quality.py
```

The demo generates synthetic training labels and task/server telemetry. These
rules are for exercising the pipeline only; train the predictor on measured
execution records before using its estimates for production allocations.
`torch_geometric` is optional. When present, the encoder uses `GCNConv`;
otherwise, it uses a small directed mean-neighbor convolution implemented in
PyTorch, so the offline demo can run without PyG.
The GNN is trained against synthetic resource labels, using predictor
estimates plus normalized file/infrastructure features. It reports held-out
resource MAE and only enables its resource-refinement head when that validation
MAE does not regress from the predictor baseline. Its graph embeddings still
feed the placement policy.

## Integrate with measured inputs

Create `TaskInput` objects with file sizes in KiB and a supported `doc_type`
(`text`, `image`, `video`, or `archive`). Supply one
`InfrastructureContext` per task with non-negative bandwidth, latency, and
available-RAM measurements. Train `TaskResourcePredictor` on your measured
labels, then construct a `TaskGraphEncoder`, `MARLScheduler`, and `Server`
records. `OfflineSchedulingEngine.run()` performs prediction, graph encoding,
feasibility filtering, sequential resource reservation, action selection, and
an episode PPO update.

The bridge layout is `[task_embedding, server_telemetry]` for each
task/server candidate. The simulation telemetry vector is ordered as normalized
remaining CPU, GPU and RAM capacity, bandwidth, latency, initial load, and cost
rate. Set the bridge's feasibility mask for current capacity and policy
constraints; every task must retain at least one feasible server.

`MARLScheduler.compute_reward()` requires actual or simulated execution time and
cost, pre-dispatch normalized system load in `[0, 1]`, and the number of queued
tasks. Its configurable reward weights favor cost at low pressure and speed at
high pressure. The demo trains over multiple synthetic PPO episodes and reports
deterministic pre-/post-training workload reward. This demonstrates learning
plumbing, not a guarantee of production policy quality.

## Checkpoints

The demo writes `task_predictor.joblib`, `task_gnn.pt`, and
`marl_scheduler.pt` into the selected checkpoint directory. These are
synthetic demonstration artifacts, not models for live deployment. Use
`load_predictor`, `load_gnn_encoder`, and `MARLScheduler.load` to restore
trusted checkpoints.

The synthetic offline demonstration and live hub use the same predictor, GNN,
bridge, and scheduler interfaces. The hub receives agent telemetry, accounts
for pending resource reservations, writes task-runner tickets, and uses
completed execution time/cost as PPO feedback. Production use still requires
measured training data and policy training, shared-directory configuration,
and workers isolated appropriately for submitted scripts.
