# Project Guide: Cloud Task Allocation and Scheduling

This guide explains the project architecture, data flow, model roles, and how
to run the offline demonstration and local live workflow. For the shorter
installation and operational reference, see [README.md](./README.md).

## 1. Project goal

The system accepts Python task scripts, estimates their resource needs, and
selects an available worker node to run each task. Its major stages are:

```text
Task script + optional hints + worker telemetry
                    |
                    v
         Gradient Boosting predictor
                    |
                    v
      GNN task features and embeddings
                    |
                    v
    Feasibility filter + MARL scheduler
                    |
                    v
      Assignment ticket in shared folder
                    |
                    v
        Worker executes and reports result
                    |
                    v
       Scheduler receives outcome feedback
```

The offline simulation and the live hub use the same general component
interfaces, but their data and outcomes are not the same: the offline demo
generates synthetic training labels and simulated execution outcomes, while
the live workflow obtains worker telemetry and real script completion results.

## 2. Main components

| Component | Responsibility |
|---|---|
| `hub.py` | Serves the dashboard and APIs, receives worker telemetry and task uploads, predicts resources, chooses nodes, writes task tickets, and collects completion feedback. |
| `agent.py` | Sends periodic CPU, RAM, network, latency, and optional NVIDIA GPU telemetry from a worker machine to the hub. |
| `task_runner.py` | Polls the shared task directory, claims tickets assigned to its node, waits for parent tasks, executes scripts, writes result files, and updates ticket status. |
| `task_predictors.py` | Defines task/context data structures and four Gradient Boosting regressors for time, CPU, GPU, and RAM estimates. |
| `gnn_encoder.py` | Builds GNN feature tensors and dependency edges, creates task embeddings, and optionally refines predicted resources. |
| `ast_dependency_detector.py` | Parses stable Python task scripts without running them and detects imports that match other task script filenames. |
| `marl_scheduler_bridge.py` | Combines each task embedding with each candidate server's telemetry and carries the feasibility mask and IDs. |
| `marl_scheduler.py` | Selects feasible task-to-server actions and updates a shared actor/centralized critic policy with PPO. |
| `mappo_scheduler.py` | Provides a parameter-shared task actor, centralized critic, masked actions, GAE, and clipped PPO; its checkpoint can be loaded by the live hub. |
| `train_reference_mappo.py` | Trains and evaluates adapted MAPPO while keeping the existing predictor and GNN checkpoints fixed. |
| `main_engine.py` | Runs the offline end-to-end simulation and can save model checkpoints. |
| `model_persistence.py` | Saves and loads the task predictor, GNN encoder, and scheduler. |

## 3. Task ingestion and feature inputs

The live hub accepts `.py` files through the dashboard or detects them in the
configured shared task directory. For each script it records:

- `file_size`: file size in KiB.
- `disk_size`: currently set to the same value as `file_size`; it is not a
  separately measured storage requirement.
- `doc_type`: one of `text`, `image`, `video`, or `archive`; uploaded Python
  files default to `text`.
- `baseline_cpu_req` and `baseline_gpu_req`: optional user hints; both default
  to zero.
- Infrastructure context: average bandwidth, round-trip latency, and free RAM
  reported by active workers.

The predictor feature vector has eight values:

```text
[file_size, disk_size, doc_type_code, available_bandwidth, latency,
 available_ram, baseline_cpu_req, baseline_gpu_req]
```

Document type codes are text `0`, image `1`, video `2`, and archive `3`.
Features are passed to the predictor as raw numeric values, so their units and
ranges need to be consistent between training and inference.

## 4. Gradient Boosting resource predictor

The project fits four separate `GradientBoostingRegressor` models to estimate:

1. `target_exec_time`
2. `target_cpu_req`
3. `target_gpu_req`
4. `target_ram_req`

Each model uses 80 shallow trees (`max_depth=2`) and Huber loss. Conceptually,
boosting adds a sequence of tree corrections:

```text
prediction(x) = initial_estimate + learning_rate * sum(tree_m(x))
```

For the offline demo, target values are generated from deterministic formulas.
Let `s = file_size_KiB / 1024`, `d = disk_size_KiB / 1024`, and
`b = max(available_bandwidth, 1)`. The type multiplier is `f = 0.75` for text,
`1.25` for image, `2.4` for video, and `1.1` for archive:

```text
execution_time = 0.5 + f*(0.15 + 0.08*s + 0.025*d)
                 + 0.015*latency + 0.2*s/b

cpu = max(cpu_hint, 0.15 + f*(0.12 + 0.018*s))
gpu = max(gpu_hint, f*(0.025 + 0.006*s))

ram_raw = max(0.1 + 0.012*d + s*r, 0.35*cpu_hint)
ram = min(ram_raw, max(available_ram, 0.1))
```

Here `r = 0.08` for video and `0.035` for other types. These generated labels
are synthetic examples, not observed execution data. At inference, negative
predictions are clamped to zero, and CPU/GPU predictions are raised to at
least their corresponding user hints. The live hub additionally treats a zero
GPU hint for an uploaded script as CPU-only.

## 5. AST import dependency prototype

The folder watcher uses Python's built-in AST parser on stable task scripts.
It looks for direct `import provider` or `from provider import value` forms
where `provider.py` is another script in the task directory. It does not
execute the script. A match is recorded as a parent relationship:

```text
provider.py -> consumer.py
```

The hub submits connected, stable scripts together so it can create both
tickets and their dependency links. The worker checks parent ticket states and
waits until they are `COMPLETED`; a failed parent causes a dependent ticket to
fail rather than run.

This prototype only detects direct, exact module-name matches. It does not
infer file/data dependencies, dynamic imports, package-relative imports, or
whether importing a module really means a task must execute first. A filename
collision with a standard-library or third-party module can also be ambiguous.
Treat detected links as a simple heuristic, not general workflow understanding.
Use explicit dependency declarations for workflows needing reliable semantics.

## 6. GNN task encoder

For each task, the GNN gets 12 features:

```text
four predictor estimates
file_size / 16384
disk_size / 32768
normalized doc type (0, 1/3, 2/3, 1)
available_bandwidth / 1000
latency / 120
available_ram / 64
baseline_cpu_req / 4
baseline_gpu_req / 2
```

The input matrix is `X` with shape `[num_tasks, 12]`. Dependencies are
represented separately by `edge_index` in parent-to-child direction. A graph
convolution combines each node's feature vector with information from its
incoming neighbors. The fallback layer used when PyTorch Geometric is
unavailable performs directed mean aggregation followed by a learned linear
transform:

```text
h_i' = W * (x_i + sum(x_parent)) / (1 + number_of_parents) + b
```

The encoder uses two graph-convolution layers with ReLU after the first. The
default offline demo dimensions are input `12`, hidden `32`, and embedding
`32`. The output embedding describes a task in its supplied graph context and
is consumed by the scheduler. If no edges exist, there is no dependency
message passing between tasks.

A separate four-value head predicts a residual correction to the predictor's
time/CPU/GPU/RAM estimates:

```text
refined_resources = max(predictor_estimates + GNN_residual, 0)
```

It is trained on synthetic demo residuals with AdamW and Smooth L1 loss. The
refinement is retained only when held-out aggregate MAE is no worse than the
predictor baseline; otherwise the correction head is reset to zero. Embeddings
are still produced either way.

## 7. MARL scheduler and state bridge

The bridge forms a candidate state for every task/server pair:

```text
candidate_state(task, server) = [task_embedding, server_telemetry]
```

In the simulation, server telemetry contains normalized remaining CPU, GPU,
and RAM capacity, bandwidth, latency, load, and relative cost. A feasibility
mask rejects candidate servers that cannot satisfy the predicted CPU, GPU,
and RAM requirements. If no candidate is feasible, allocation fails
explicitly rather than assigning an undersized node.

The scheduler uses one shared actor to score candidate servers and a
centralized critic to estimate state value. Training samples from a
categorical distribution over feasible choices; deterministic evaluation uses
the highest-scoring feasible choice. The policy is trained with Proximal
Policy Optimization (PPO), which limits policy changes while using reward
feedback.

The adaptive reward measures normalized execution time and cost. Queue and
system pressure control their relative weights:

```text
queue_pressure = min(tasks_waiting / queue_pressure_threshold, 1)
pressure = max(system_load, queue_pressure)

cost_weight = idle_cost_weight
              + pressure * (max_pressure_cost_weight - idle_cost_weight)
time_weight = 1 - cost_weight

reward = 1 - (time_weight * execution_time / execution_time_scale
              + cost_weight * cost / cost_scale)
```

At low pressure the reward gives more weight to cost; at high pressure it
increases the weight on time. In the offline engine, time and cost are
simulated. In the live system, completion time and configured node cost can
provide scheduler feedback. Each completion does not automatically retrain
the Gradient Boosting predictor or GNN.

## 8. Offline demonstration versus live execution

### Offline demo

`main_engine.py` generates synthetic training examples, fits the predictor and
GNN refiner, creates simulated servers and tasks, and runs the scheduler. It
simulates execution time/cost and runs PPO episodes. This is for exercising
the pipeline, not proof of real production performance.

### Live local workflow

The live workflow consists of three continuously running processes:

1. `hub.py`: dashboard/API, telemetry intake, scheduling, and tickets.
2. `agent.py`: telemetry heartbeat from each worker node.
3. `task_runner.py`: ticket polling, dependency waiting, script execution, and
   result/ticket updates.

The hub and workers must share the same task directory. The hub also needs the
three model checkpoints configured through `MARL_PREDICTOR_PATH`,
`MARL_GNN_PATH`, and `MARL_SCHEDULER_PATH`. Each worker's node ID must appear
in `MARL_NODE_IDS`.

## 9. Run commands

From Command Prompt on Windows, enter the project folder and activate the
virtual environment:

```cmd
cd /d "C:\Users\priyanshu solanki\OneDrive\Desktop\marl"
call .venv\Scripts\activate.bat
```

Run tests:

```cmd
python -m unittest discover -s . -p "test*.py"
```

Run the offline simulation and save model checkpoints once:

```cmd
python main_engine.py --tasks 8 --seed 7 --checkpoint-dir models
```

To start the live local setup without retraining, start the hub in one window
with the saved model paths configured:

```cmd
set "MARL_PREDICTOR_PATH=C:\Users\priyanshu solanki\OneDrive\Desktop\marl\models\task_predictor.joblib"
set "MARL_GNN_PATH=C:\Users\priyanshu solanki\OneDrive\Desktop\marl\models\task_gnn.pt"
set "MARL_SCHEDULER_PATH=C:\Users\priyanshu solanki\OneDrive\Desktop\marl\models\marl_scheduler.pt"
set "MARL_TASKS_DIR=C:\marl-test\cluster_task"
set "MARL_NODE_IDS=S1"
set "MARL_HUB_HOST=127.0.0.1"
set "MARL_API_TOKEN=choose-a-long-private-token"
python hub.py
```

In another window, start the telemetry agent:

```cmd
python agent.py --node-id S1 --hub-url http://127.0.0.1:8000/api/telemetry --task-directory C:\marl-test\cluster_task
```

In a third window, start the worker:

```cmd
python task_runner.py --node-id S1 --task-directory C:\marl-test\cluster_task --results-directory C:\marl-test\cluster_result
```

Open <http://127.0.0.1:8000/> and provide the same API token when uploading.
Only submit trusted scripts: the task runner executes them with the operating
system permissions of its worker account. For real separate machines, replace
the loopback address with the hub's reachable LAN address and configure
network-shared task/result directories.

## 10. Model persistence and dataset experiment

The offline command writes:

```text
models/task_predictor.joblib
models/task_gnn.pt
models/marl_scheduler.pt
```

These are demo checkpoints trained on generated labels and simulated
experience. Saving avoids retraining at each hub start, but does not make the
models production-calibrated.

The prepared cloud-task CSV and its separate Gradient Boosting experiment are
documented in [README.md](./README.md). That experiment uses different feature
and target schemas and includes a GPU heuristic pseudo-label; its artifact is
not compatible with the live predictor. Held-out regression results for that
dataset were generally close to a mean baseline, and the pseudo-label GPU
classifier did not identify positive cases. Do not present these results as
validated production model accuracy.

## 11. Train the adapted reference MAPPO scheduler

The linked
[Capstone_MARL](https://github.com/YashSigchi/Capstone_MARL) project uses a
shared MAPPO actor, centralized critic, masked discrete actions, GAE, PPO
clipping, and a weighted local/global reward. `train_reference_mappo.py`
adapts those mechanics to the existing predictor/GNN outputs without changing
their inputs or the seven-value server telemetry in the bridge.

Create or retain the fixed predictor and GNN checkpoints, then train:

```powershell
python main_engine.py --tasks 8 --seed 7 --checkpoint-dir models
python train_reference_mappo.py --episodes 300 --tasks 8 --eval-seeds 20 --seed 42
```

For three servers, the policy action set is feasible server placement, defer
once, and reject. The reward follows the reference's nine fixed components
(latency, energy, utilization, load balance, SLA, queue, throughput, action,
and admission) and local/global blend. Deadlines are synthetic reward-only
values based on the predicted execution time; energy is proxied from simulated
cost because the existing simulator has no power telemetry.

The run saves `results/mappo_reference/mappo_scheduler.pt`,
`training_history.csv`, `training.log`, `config.json`, `evaluation.json`,
`training_curves.png`, and `evaluation_comparison.png`. Evaluation uses the
same held-out seeds for untrained MAPPO, trained MAPPO, random-feasible,
cheapest, and fastest policies. These are simulated results, not measured
production performance. The predictor and GNN are loaded from `models/` and
are not retrained.

When all default model files are present, the live hub loads
`models/task_predictor.joblib`, `models/task_gnn.pt`, and the trained
`results/mappo_reference_1000/mappo_scheduler.pt` automatically. Set
`MARL_PREDICTOR_PATH`, `MARL_GNN_PATH`, and `MARL_SCHEDULER_PATH` together to
override those paths. The MAPPO checkpoint is trained for three servers, so
the bridge candidate count must also be three. The hub supports both this
checkpoint and the original `marl_scheduler.pt` checkpoint format.

The completed seed-42 run trained for 300 episodes and evaluated 20 held-out
seeds. Mean reward was `0.4868` for trained MAPPO, `-0.1905` for untrained
MAPPO, `0.0085` for random-feasible, `0.3930` for cheapest, and `0.4884` for
fastest. Trained MAPPO's completion/rejection rates were `1.0000`/`0.0000`;
its reward was `0.0015` below fastest. Full per-seed metrics, training curves,
and episode history are in `results/mappo_reference/`.

## 12. Limitations and responsible interpretation

- Demo predictor/GNN training labels, graph edges, server capacities, and
  offline execution outcomes are synthetic or simulated.
- Import-based AST links are a narrow heuristic, not general dependency
  discovery. Different users' tasks are not automatically related unless a
  matching import or explicit dependency exists.
- Resource prediction needs representative measured labels, consistent units,
  and inputs available before execution. Post-execution measurements must not
  leak into pre-execution features.
- The scheduler's feedback improves only the scheduler policy; predictor and
  GNN retraining require a separate measured-data training workflow.
- The task runner executes uploaded code. Use trusted scripts and isolate
  workers appropriately before exposing the system beyond a local demo.
