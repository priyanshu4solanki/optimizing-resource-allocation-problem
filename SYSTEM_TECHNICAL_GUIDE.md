# Cloud Resource Allocation and Task Scheduling

## Complete Technical Guide

This document explains the project from its inputs and model training through
offline simulation, live scheduling, worker execution, feedback, saved model
artifacts, and known limitations. It is intended to be a detailed guide to the
current source code, not a claim that the system has been validated for
production workloads.

For installation and a short command reference, see [README.md](./README.md).
For the existing component and architecture overview, see
[PROJECT_GUIDE.md](./PROJECT_GUIDE.md).

---

## Contents

1. [Project purpose and system boundaries](#1-project-purpose-and-system-boundaries)
2. [Architecture and data flow](#2-architecture-and-data-flow)
3. [Core concepts and terms](#3-core-concepts-and-terms)
4. [Task and infrastructure inputs](#4-task-and-infrastructure-inputs)
5. [Gradient Boosting resource predictor](#5-gradient-boosting-resource-predictor)
6. [Dependency graphs and the GNN](#6-dependency-graphs-and-the-gnn)
7. [Task/server scheduler bridge](#7-taskserver-scheduler-bridge)
8. [MARL and PPO scheduling](#8-marl-and-ppo-scheduling)
9. [MAPPO adaptation and training](#9-mappo-adaptation-and-training)
10. [Offline end-to-end simulation](#10-offline-end-to-end-simulation)
11. [Live hub and worker workflow](#11-live-hub-and-worker-workflow)
12. [Persistence and checkpoint compatibility](#12-persistence-and-checkpoint-compatibility)
13. [The separate cloud CSV experiment](#13-the-separate-cloud-csv-experiment)
14. [Installation and running the project](#14-installation-and-running-the-project)
15. [Tests and validation](#15-tests-and-validation)
16. [Configuration and operational reference](#16-configuration-and-operational-reference)
17. [Limitations, safety, and interpretation](#17-limitations-safety-and-interpretation)
18. [Source-file guide](#18-source-file-guide)
19. [Suggested production development path](#19-suggested-production-development-path)

---

## 1. Project purpose and system boundaries

The project explores allocating incoming tasks to worker machines with
different resource capacities. It contains two related but distinct systems:

1. **Offline modeling and simulation** exercises the predictor, task graph
   encoder, scheduler, and model persistence using synthetic or simulated
   data.
2. **Live task dispatch** accepts Python scripts, receives current worker
   telemetry, schedules scripts, writes task tickets to a shared directory,
   and monitors worker results.

The intended model pipeline is:

```text
Task metadata + infrastructure context
                 |
                 v
       Gradient Boosting regressors
                 |
                 v
    Predicted execution/resource needs
                 |
                 v
 GNN graph context, resource refinement, embeddings
                 |
                 v
   Task/server candidate state + mask
                 |
                 v
        MARL scheduler policy
                 |
                 v
         Selected worker node
```

In the live mode, the selected node is written into a JSON ticket. The worker
executes the script and reports its outcome in the ticket. The hub can then
use measured completion time and configured node cost to produce scheduler
feedback.

These components do not all learn online in the same way:

- The Gradient Boosting predictor is not automatically refit from live task
  outcomes.
- The GNN is not automatically retrained from live task outcomes.
- The scheduler can receive terminal task feedback and update its policy.
- Current MAPPO training is an offline simulation. The live MAPPO checkpoint
  loader and training environment use related interfaces, but online learning
  is not the same training process as the offline 1,000-episode run.

---

## 2. Architecture and data flow

### 2.1 Offline learning and simulation

`main_engine.py` provides a demonstration pipeline:

1. Generate synthetic examples for resource prediction.
2. Fit four Gradient Boosting regressors.
3. Predict execution time, CPU, GPU, and RAM needs.
4. Build a task graph and 12-dimensional node feature rows.
5. Run a graph encoder to obtain task embeddings and refined estimates.
6. Build a candidate vector for each task/server pair.
7. Mask servers that cannot meet the predicted resource requirements.
8. Ask a policy to choose a feasible server.
9. Simulate execution time, cost, and reward.
10. Optionally update the scheduler and save checkpoints.

The separate `train_reference_mappo.py` program loads the saved predictor and
GNN, keeps them fixed, and trains only the MAPPO policy in its own simulator.

### 2.2 Live task flow

```text
Browser upload or .py script in shared task directory
                         |
                         v
               hub.py task ingestion
                         |
      worker heartbeats -> live node state
                         |
                         v
  predictor -> graph encoder -> bridge -> scheduler
                         |
                         v
             atomic JSON ticket write
                         |
                         v
       task_runner.py claims ticket on a worker
                         |
                         v
      parent dependencies checked, script executed
                         |
                         v
     result file and terminal ticket status written
                         |
                         v
             hub records scheduler feedback
```

The live code schedules scripts, not arbitrary task containers. Worker
processes execute Python files using the worker's Python interpreter and OS
permissions.

### 2.3 The resource and action contracts

There are two important tensor boundaries:

- **GNN input:** one 12-feature row per task, with graph edges kept separately.
- **Scheduler candidate input:** one vector per task/server candidate,
  formed by concatenating a 32-value task embedding and 7 telemetry values.

For the default 32-dimensional embedding:

```text
bridge candidate vector = 32 embedding values + 7 telemetry values = 39
```

The scheduler feasibility mask accompanies the candidate tensor. It does not
remove candidates; it states which candidate actions are legal.

---

## 3. Core concepts and terms

| Term | Meaning here |
|---|---|
| Task | A submitted job or Python script and its metadata. |
| Resource demand | Predicted execution time, CPU, GPU, and RAM requirements. |
| Infrastructure context | Network bandwidth, latency, and available RAM used by the predictor. |
| Node/server/worker | A machine to which the scheduler may assign a task. The source uses “server” in the simulator and “node” in the live hub. |
| DAG | Directed acyclic graph of task dependencies. An edge `u -> v` means `v` depends on `u`. |
| Node embedding | A learned numeric vector representing a task in its graph context. “Node” here means graph node, not worker machine. |
| Feasibility mask | Boolean matrix marking candidate servers that satisfy predicted resource constraints. |
| Actor | Neural network that scores actions or produces an action distribution. |
| Critic | Neural network that estimates the value of a state. |
| PPO | Proximal Policy Optimization, a policy-gradient algorithm that clips updates to limit policy changes. |
| MAPPO | Multi-Agent PPO, commonly using parameter-shared actors with a centralized critic during training. |
| GAE | Generalized Advantage Estimation, a method for estimating how much better an action was than the value baseline. |
| CTDE | Centralized Training, Decentralized Execution: train using global information, then select actions using local policy inputs. |
| Checkpoint | A persisted model state and configuration used to resume or serve a trained model. |
| Synthetic target | A generated label from a formula, not a measurement from a real machine. |

---

## 4. Task and infrastructure inputs

### 4.1 Task model

`TaskInput` in `task_predictors.py` represents a pre-execution task. It
contains:

| Field | Description |
|---|---|
| `task_id` | Unique task identifier. |
| `file_size` | Numeric file size supplied to the predictor. |
| `disk_size` | Numeric disk-size estimate or task metadata. |
| `doc_type` | One of `text`, `image`, `video`, or `archive`. |
| `parents` | Optional parent task IDs. |
| `baseline_cpu_req` | Optional CPU resource hint; defaults to 0. |
| `baseline_gpu_req` | Optional GPU resource hint; defaults to 0. |

Values must be finite and non-negative. The code does not infer a distinct
disk requirement from file contents: for live Python uploads, the hub measures
the file size and uses it for both file and disk size.

Document type is represented as a numeric code for prediction:

| Type | Code |
|---|---:|
| `text` | 0 |
| `image` | 1 |
| `video` | 2 |
| `archive` | 3 |

### 4.2 Infrastructure context for the predictor

`InfrastructureContext` includes:

- `available_bandwidth`
- `latency`
- `available_ram`

These are task-level model inputs. In the live hub, they are aggregated from
the active worker telemetry; they are not a detailed path-specific network
measurement for every task/node pair.

### 4.3 The eight predictor features

The four regressors use the same eight ordered numeric inputs:

```text
[
  file_size,
  disk_size,
  doc_type_code,
  available_bandwidth,
  latency,
  available_ram,
  baseline_cpu_req,
  baseline_gpu_req
]
```

Consistency of units and feature meanings between training and inference is
critical. The scikit-learn model does not automatically discover or convert
units.

### 4.4 Predicted outputs

Each task prediction contains four values:

```text
[
  target_exec_time,
  target_cpu_req,
  target_gpu_req,
  target_ram_req
]
```

Predictions are clamped to be non-negative. CPU and GPU predictions are also
raised to at least the corresponding user hints.

---

## 5. Gradient Boosting resource predictor

### 5.1 What Gradient Boosting does

Gradient Boosting builds an ensemble of decision trees sequentially. Each
tree is trained to improve the current model's errors. In simplified form:

```text
prediction(x) = initial_prediction + learning_rate * sum(tree_m(x))
```

Trees can represent nonlinear relationships and interactions between numeric
features without requiring the user to specify a linear formula. This is why
they are a reasonable baseline for mixed task metadata and infrastructure
inputs.

### 5.2 Models used in this project

`TaskResourcePredictor` maintains **four independent
`GradientBoostingRegressor` models**, one per target:

1. `target_exec_time`
2. `target_cpu_req`
3. `target_gpu_req`
4. `target_ram_req`

Each is configured with 80 estimators, maximum tree depth 2, Huber loss, and a
deterministic random state derived from the supplied seed. All four models
receive the same eight feature columns.

The `fit(tasks, contexts, targets)` API expects a target mapping for every
task with exactly these four keys. Each target must be finite and
non-negative. The model refuses to predict until all four regressors are
fitted.

### 5.3 Synthetic demonstration training labels

The offline demo creates labels with deterministic rules. These formulas
exist to exercise the end-to-end pipeline; they are not measurements of
production systems.

Let:

```text
s = file_size / 1024
d = disk_size / 1024
b = max(available_bandwidth, 1)
```

The type factor `f` is `0.75` for text, `1.25` for image, `2.4` for video,
and `1.1` for archive. The synthetic labels are:

```text
execution_time = 0.5
                 + f * (0.15 + 0.08*s + 0.025*d)
                 + 0.015*latency
                 + 0.2*s/b

cpu = max(baseline_cpu_req, 0.15 + f*(0.12 + 0.018*s))

gpu = max(baseline_gpu_req, f*(0.025 + 0.006*s))

ram_raw = max(
    0.1 + 0.012*d + s*r,
    0.35*baseline_cpu_req
)

ram = min(ram_raw, max(available_ram, 0.1))
```

Here `r` is `0.08` for video and `0.035` otherwise. Because the labels are
generated from the same inputs, high prediction quality on this synthetic
distribution does not establish accuracy on real task workloads.

### 5.4 Other predictor-training entry points

- `main_engine.py` uses `generate_synthetic_training_data()` and
  `fit_synthetic()` for the demo.
- Real measured training can call `TaskResourcePredictor.fit()` with measured
  resource targets.
- `train_cloud_dataset.py` is a separate CSV experiment with a different
  schema; its models are not interchangeable with the task predictor used by
  the hub.

### 5.5 Predictor quality and evaluation

For a real deployment, use a holdout split that avoids leakage, appropriate
metrics per target, and records from representative tasks and machines.
Useful regression metrics include:

- MAE: average absolute prediction error, in target units.
- RMSE: gives larger errors more influence.
- R-squared: compares the fitted predictions with the variation in the
  evaluation labels; it can be negative on poor holdout predictions.

Metrics must be reported in the unit of each target. Do not combine the four
targets into a single accuracy percentage without a defined, meaningful
normalization.

---

## 6. Dependency graphs and the GNN

### 6.1 Why tasks are represented as a graph

Tasks can depend on other tasks. For example, if `consumer.py` imports a
module implemented by `provider.py`, the workflow may need the provider to
complete first:

```text
provider -> consumer
```

The graph gives the encoder context about connected tasks. It is a separate
input from the task's feature row.

### 6.2 Graph construction and validation

`build_dependency_edges(task_ids, parents_by_task)`:

1. Maps task IDs to integer row positions.
2. Converts each parent relationship into a directed edge from parent to
   child.
3. Rejects duplicate task IDs, unknown parents, self-dependencies, and cycles.
4. Returns PyTorch Geometric-compatible `edge_index` with shape
   `[2, number_of_edges]`.

For no dependencies, `edge_index` has shape `[2, 0]`. Edge direction matters:
the implementation aggregates source-node features into the target node.

### 6.3 Optional AST dependency detection

`ast_dependency_detector.py` parses stable Python files without executing
them. It looks for direct imports:

```python
import provider
from provider import function
```

An import is treated as a task dependency only if a matching `provider.py`
exists in the same set of task scripts. The detector does not interpret
dynamic imports, relative/package imports, arbitrary files, or whether
Python's import operation actually requires the imported task to finish.

Detected dependency edges are a workflow heuristic. The worker's ticket
dependency check provides the runtime ordering: wait for a parent ticket to
be `COMPLETED`; fail the child if a parent is `FAILED`.

### 6.4 The 12 GNN feature columns

`task_gnn_feature_tensor()` creates one row per task. Columns are:

| Index | Feature | Scaling |
|---:|---|---|
| 0 | Predicted execution time | Predictor output |
| 1 | Predicted CPU requirement | Predictor output |
| 2 | Predicted GPU requirement | Predictor output |
| 3 | Predicted RAM requirement | Predictor output |
| 4 | File size | `file_size / 16384` |
| 5 | Disk size | `disk_size / 32768` |
| 6 | Document type | 0, 1/3, 2/3, or 1 |
| 7 | Available bandwidth | `available_bandwidth / 1000` |
| 8 | Latency | `latency / 120` |
| 9 | Available RAM | `available_ram / 64` |
| 10 | Baseline CPU hint | `baseline_cpu_req / 4` |
| 11 | Baseline GPU hint | `baseline_gpu_req / 2` |

With `N` tasks, the feature matrix has shape `[N, 12]`. The graph edge list
has shape `[2, E]` where `E` is the number of directed dependencies.

### 6.5 Graph convolution in plain terms

A graph convolution updates each task's representation using its own features
and information from its incoming neighbors. With a simple mean-neighbor
view:

```text
new_task_features =
    learned_transform(mean(self features and parent features))
```

Thus a task can have an embedding that reflects both its own predicted
resources and the properties of its dependency context. Without edges, graph
message passing between tasks does not occur.

### 6.6 Encoder architecture and optional backends

`TaskGraphEncoder` uses two graph-convolution layers:

```text
12 feature values
       |
first graph convolution
       |
      ReLU
       |
second graph convolution
       |
32-value task embedding
```

The default offline dimensions are input 12, hidden 32, embedding 32. If
`torch_geometric` is installed, the encoder can use its `GCNConv`. Otherwise,
the project uses `_FallbackGCNLayer`, a PyTorch implementation that aggregates
the directed mean of the task and its incoming neighbors, then applies a
learned linear layer.

The fallback permits a demo without PyG, but it is not guaranteed to produce
the exact same numerics as `GCNConv`. The selected backend is stored in the
GNN checkpoint; load the compatible dependency/backend when restoring it.

### 6.7 Resource refinement head

Besides the embedding, a four-value linear head predicts a residual correction
to the predictor's four resource estimates:

```text
refined_resources = clamp_min(baseline_resources + residual, 0)
```

The output order is execution time, CPU, GPU, RAM. The residual head is
initialized to zero, so before learning it leaves the baseline unchanged.
`fit_resource_refiner()`:

1. Validates feature, target, and edge shapes.
2. Computes residual targets from measured or supplied resources minus
   baseline predictions.
3. Scales residual targets.
4. Trains with AdamW and Smooth L1 loss.
5. Monitors validation MSE and uses early stopping/patience.
6. Compares validation MAE to the original predictor.
7. Keeps the residual head only when validation MAE is not worse; otherwise
   resets it to zero.

This safeguard is only as meaningful as the validation data. Synthetic
validation labels can test training mechanics but cannot validate production
resource estimates.

---

## 7. Task/server scheduler bridge

### 7.1 Why the bridge exists

The GNN produces one embedding per task. The scheduler must compare that task
against current server conditions. `build_joint_state()` joins each task
embedding with every candidate server's telemetry.

For task `i` and server `j`:

```text
joint_state[i, j] = concat(task_embedding[i], server_telemetry[j])
```

The result has shape:

```text
[number_of_tasks, number_of_servers, embedding_dim + telemetry_dim]
```

It also carries:

- A boolean feasibility mask of shape `[tasks, servers]`.
- `task_ids` in task row order.
- `server_ids` in candidate column order.

The IDs keep the chosen action index tied to the correct task and server.

### 7.2 Seven normalized server telemetry features

The offline engine and live hub each form seven server features representing:

1. Remaining CPU capacity, normalized by CPU capacity.
2. Remaining GPU capacity, normalized by GPU capacity.
3. Remaining RAM capacity, normalized by RAM capacity.
4. Available bandwidth, scaled.
5. Latency, scaled.
6. Server load, normalized.
7. Relative cost rate.

The exact measured values differ between the offline simulator and live
heartbeat inputs, even though both produce seven numeric telemetry values.
Training and deployment need consistent feature order and appropriate
normalization.

### 7.3 Resource feasibility

For each task/server candidate the mask is true only when the predicted
requirements fit the server's remaining CPU, GPU, and RAM:

```text
feasible =
    cpu_req <= remaining_cpu
    and gpu_req <= remaining_gpu
    and ram_req <= remaining_ram
```

The policy distribution is masked before sampling or choosing the maximum
logit. Infeasible choices should never be selected. If the live hub has no
eligible candidate, scheduling returns an explicit error/waiting condition
rather than intentionally placing the task on an undersized node.

---

## 8. MARL and PPO scheduling

### 8.1 Basic reinforcement-learning view

The scheduler observes task/server states, chooses a server, and receives
reward based on the simulated or measured outcome. The policy parameters are
updated to make higher-reward decisions more likely.

The original project `MARLScheduler` in `marl_scheduler.py` uses:

- One shared actor that scores each task/server candidate.
- A centralized critic that pools feasible candidates to estimate value.
- A categorical action distribution over feasible servers.
- PPO's clipped policy objective.
- Discounted returns and a value baseline.
- Entropy regularization to discourage premature certainty.
- A reward that blends execution time and cost according to workload
  pressure.

The original scheduler's action dimension is the number of candidate servers.
It does not have the MAPPO trainer's defer/reject actions.

### 8.2 PPO clipped objective

For an action with old probability and new probability, define the ratio:

```text
ratio = new_action_probability / old_action_probability
```

PPO compares the unclipped advantage objective with a version clipped to a
small interval around 1:

```text
actor_loss = -mean(min(
    ratio * advantage,
    clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
))
```

This reduces the chance that a single update changes the policy too far.

The critic is trained to estimate return/value with a mean-squared error
objective. Entropy is subtracted from total loss to preserve exploration:

```text
total_loss = actor_loss
             + value_coefficient * critic_loss
             - entropy_coefficient * entropy
```

Gradients are clipped to a maximum norm to limit very large updates.

### 8.3 Original scheduler reward

`MARLScheduler.compute_reward()` uses actual execution time, actual cost,
system load, and waiting task count. Queue pressure is normalized and combined
with system load:

```text
queue_pressure = min(tasks_waiting / queue_pressure_threshold, 1)
pressure = max(system_load, queue_pressure)

cost_weight = idle_cost_weight
              + pressure * (max_pressure_cost_weight - idle_cost_weight)
time_weight = 1 - cost_weight

reward = 1 - (
    time_weight * execution_time / execution_time_scale
    + cost_weight * actual_cost / cost_scale
)
```

The default reward configuration gives cost more influence at idle and shifts
weight toward execution time as pressure rises. In the offline demo,
execution time and cost are simulated. In the live hub, execution time comes
from the worker ticket and cost uses the node's configured cost rate.

### 8.4 Returns and scheduler experience

An experience holds the bridge state, selected action indices, reward, and
episode-boundary marker. The original scheduler computes discounted returns
backward from the end of an episode, evaluates old action probabilities and
values, then applies PPO updates.

For multiple tasks in one bridge state, each task has one selected action. The
critic's state representation is formed by pooling candidate features under
the feasibility mask.

---

## 9. MAPPO adaptation and training

### 9.1 Why there is a second scheduler

`mappo_scheduler.py` implements MAPPO mechanics adapted to the existing
task/server bridge and hub interface. It does not load the external
repository's trained weights or replace this project's predictor/GNN. The
external and local projects use different state and action schemas.

The MAPPO actor sees a flattened vector for one task:

```text
observation_dim = number_of_servers * (embedding_dim + 7)
```

With the 32-dimensional GNN embedding and three servers:

```text
candidate features = 39 per server
actor observation = 3 * 39 = 117 values
```

The actor outputs `number_of_servers + 2` logits:

```text
server 0, server 1, server 2, defer once, reject
```

Actions that are not valid in the current environment are masked. The live
hub-facing inference adapter masks defer and reject, so the scheduler always
returns a feasible worker placement.

### 9.2 Shared actor and centralized critic

The actor is a two-hidden-layer MLP (128 units per layer by default) mapping
the flattened local task observation to action logits. The centralized critic
is a two-hidden-layer MLP (256 units per layer by default) mapping a global
state vector to a scalar value.

The offline MAPPO global state concatenates:

- Each task's flattened server-candidate observations.
- Four one-hot task status indicators.
- Eight global summaries: server utilization mean/std/max/min, rejection
  fraction, completion fraction, pending fraction, and episode progress.

The critic receives that global state during training. The live inference path
builds an approximate critic context from the current candidate observation
and telemetry summaries because live scheduling makes decisions in a
different request lifecycle than the offline simulator.

### 9.3 Action masks and defer/reject behavior

In MAPPO simulation:

- An infeasible server is masked out.
- A task may defer once.
- Rejection is allowed until a target rejection threshold is reached, or when
  no server is feasible.
- A defer action is penalized if a feasible server existed.
- The live scheduler adapter disables defer and reject; it performs
  assignment-only inference.

The simulator processes a fixed topological task queue. A defer returns a
task to the queue once. If no feasible server becomes available after its
single defer, rejection remains the terminal option.

### 9.4 MAPPO update

For each action, the trainer records:

- Flattened actor observation.
- Global critic state.
- Action mask.
- Sampled action.
- Old log-probability.
- Value estimate.
- Reward.
- Done marker.

The scheduler computes Generalized Advantage Estimation backward through the
rollout:

```text
delta_t = reward_t + gamma * value_(t+1) * (1 - done_t) - value_t
gae_t = delta_t + gamma * gae_lambda * (1 - done_t) * gae_(t+1)
return_t = gae_t + value_t
```

It normalizes advantages for multi-transition batches, shuffles transitions,
then performs the configured PPO epochs and minibatches. The policy and value
losses use the same clipped actor, critic MSE, entropy bonus, and gradient
clipping principles as described above.

### 9.5 MAPPO reward design

The local adaptation uses the same nine named components, fixed weights,
0.5 local/global blend, and broad local/global scopes as the linked
Capstone_MARL reference. The components are:

| Component | Scope | Local adaptation |
|---|---|---|
| Latency | Local | Execution estimate relative to a synthetic deadline. |
| Energy | Local | Cost-derived proxy, since simulated power is not modeled. |
| SLA | Local | Deadline result adjusted by a synthetic task priority. |
| Queue | Local | Remaining/pending task fraction. |
| Action penalty | Local | Penalizes deferring when a feasible server exists; smaller defer/reject penalty otherwise. |
| Utilization | Global | Reward based on average server load and a preferred band. |
| Load balance | Global | Penalty based on variance of server loads. |
| Throughput | Global | Positive reward when a task is placed/completed. |
| Admission | Global | Penalty derived from rejection rate and target rejection. |

The fixed weights are:

| Component | Weight |
|---|---:|
| Action penalty | 0.10 |
| Admission | 0.10 |
| Energy | 0.15 |
| Latency | 0.15 |
| Load balance | 0.10 |
| Queue | 0.10 |
| SLA | 0.10 |
| Throughput | 0.05 |
| Utilization | 0.15 |

Reward blending is:

```text
total_reward = 0.5 * local_reward + 0.5 * global_reward
```

This is an adaptation, not a byte-for-byte copy of the reference formulas.
The reference environment computes energy and SLA from its own simulator;
this project uses proxies and task-specific synthetic deadlines.

### 9.6 MAPPO run commands and artifacts

Ensure `models/task_predictor.joblib` and `models/task_gnn.pt` exist, then run
from the project directory:

```powershell
python train_reference_mappo.py `
  --episodes 1000 `
  --tasks 8 `
  --eval-seeds 20 `
  --seed 42 `
  --output-dir results\mappo_reference_1000
```

`--episodes 1000` means 1,000 episodes/rollouts, with one PPO update after
each rollout. Each PPO update uses four PPO epochs by default; therefore, the
run performs 1,000 policy updates and 4,000 PPO epoch passes over rollouts.
This is not 1,000 optimizer steps total: each epoch is split into configured
minibatches.

Each result directory contains:

| File | Contents |
|---|---|
| `mappo_scheduler.pt` | MAPPO weights, dimensions, and optimizer/config state needed to load the policy. |
| `training_history.csv` | Episode reward, simulated metrics, reward components, and update diagnostics for each training rollout. |
| `training.log` | Human-readable summary of training and evaluation. |
| `config.json` | Training parameters, action set, reward weights, model paths, seed, and evaluation seed list. |
| `evaluation.json` | Per-seed metrics, means/stds, baselines, and reward deltas. |
| `training_curves.png` | Reward during training. |
| `evaluation_comparison.png` | Held-out policy comparison plots. |

### 9.7 Recorded 300- and 1,000-episode results

These are the results from the local synthetic simulator, seed 42, eight tasks
per episode, three servers, and 20 held-out evaluation seeds:

| Policy | Mean reward | Reward std | Completion | Rejection | SLA violation |
|---|---:|---:|---:|---:|---:|
| Trained MAPPO, 300 episodes | 0.4868 | 0.0287 | 100% | 0% | 0% |
| Trained MAPPO, 1,000 episodes | 0.4868 | 0.0287 | 100% | 0% | 0% |
| Untrained MAPPO | -0.1905 | 0.0766 | 100% | 0% | 2.5% |
| Random feasible | 0.0085 | 0.1978 | 88.8% | 11.3% | 2.8% |
| Cheapest | 0.3930 | 0.0464 | 100% | 0% | 3.1% |
| Fastest | 0.4884 | 0.0282 | 100% | 0% | 0% |

The trained policy improves substantially over untrained MAPPO and the
cheapest/random baselines, but does not beat the fastest baseline. The
1,000-episode result is the same at the reported evaluation precision as the
300-episode result. Episode rewards had already settled around 0.49, so more
episodes alone did not produce measurable held-out improvement.

The exact full per-seed outputs are under `results/mappo_reference/` and
`results/mappo_reference_1000/`. Results are simulation evidence only.

### 9.8 Differences from the reference repository

The local MAPPO implementation follows core mechanics and many hyperparameter
values, but it is not the same complete project:

- The reference acts as three cluster agents; this local trainer selects one
  server for one task at a time.
- State features and global critic state are local to this project.
- The task embedding and seven telemetry inputs are retained.
- The action meanings are adapted to this project's server placement.
- Reward names/weights/scopes are aligned, but formulas and environment
  measurements differ.
- Training episodes and evaluation are run in this project's synthetic
  simulator.
- External repository checkpoints are not compatible or loaded.

---

## 10. Offline end-to-end simulation

### 10.1 `OfflineSchedulingEngine`

`OfflineSchedulingEngine` composes the predictor, GNN, bridge, scheduler, and
server descriptions. For each workflow it:

1. Predicts resources for the submitted tasks.
2. Builds dependency edges and task feature tensors.
3. Refines resource estimates and obtains embeddings from the encoder.
4. Computes a topological order so parent tasks precede children.
5. Initializes remaining server resources from capacities and initial load.
6. Builds the per-task bridge state and feasibility mask.
7. Selects a server using the policy.
8. Reserves predicted CPU/GPU/RAM on the selected server.
9. Simulates execution time and cost.
10. Computes reward and accumulates scheduler experiences.
11. Optionally updates the scheduler at episode end.

The simulator rejects a task explicitly if there is no feasible server; it
does not silently assign a task to an undersized server.

### 10.2 Demo workload and server assumptions

The offline demo creates three illustrative servers:

| Server | CPU | GPU | RAM | Bandwidth | Latency | Cost/sec |
|---|---:|---:|---:|---:|---:|---:|
| `edge-small` | 4 | 1 | 16 | 100 | 18 | 0.015 |
| `cloud-balanced` | 8 | 4 | 32 | 500 | 35 | 0.035 |
| `cloud-fast` | 16 | 8 | 64 | 900 | 48 | 0.08 |

These are simulation parameters, not discovered facts about live machines.
Execution time is estimated by a helper from predicted task requirements and
server parameters; cost is simulated execution time multiplied by the
server's configured cost rate.

### 10.3 `main_engine.py` command

```powershell
python main_engine.py --tasks 8 --seed 7 --checkpoint-dir models
```

The command runs a synthetic demonstration, prints schedule and training
summaries, and can save:

```text
models/
  task_predictor.joblib
  task_gnn.pt
  marl_scheduler.pt
```

The demo scheduler uses its own reward configuration and has a default short
training loop; it is distinct from the longer reference-style MAPPO training
script.

---

## 11. Live hub and worker workflow

### 11.1 Worker telemetry agent

`agent.py` sends periodic heartbeats to `/api/telemetry`. It gathers:

- CPU percentage.
- RAM percentage, free RAM, and total RAM.
- Network transmit/receive rates.
- Round-trip latency to the hub.
- Logical CPU count.
- GPU count/utilization when NVIDIA `nvidia-smi` is available.
- Active task count inferred from shared-directory tickets.
- Configured cost rate.

Worker status is reported as `FREE`, `OCCUPIED`, or `HIGH_LOAD` based on the
CPU percentage. The hub considers a node live only when its heartbeat is
recent enough for the configured timeout.

The agent reports telemetry, but its GPU collector only recognizes NVIDIA's
`nvidia-smi`; other GPU vendors require additional integration. Missing
`nvidia-smi` is represented as zero GPUs.

### 11.2 Hub ingestion and task validation

`hub.py` is a FastAPI application. It exposes:

| Route | Purpose |
|---|---|
| `POST /api/telemetry` | Ingest a validated worker heartbeat. |
| `GET /` | Serve the dashboard HTML. |
| `POST /api/tasks/upload` | Receive a script upload. |
| `GET /api/cluster/status` | Read current worker status. |
| `GET /api/network/graph` | Return worker network graph data. |
| `POST /api/tasks/submit` | Submit a task batch with metadata/dependencies. |
| `GET /api/tasks/{task_id}` | Read a task ticket/status. |

Task and telemetry request models validate fields, numeric ranges, task IDs,
and document types. Script names must be safe `.py` basenames; arbitrary
directory traversal names are rejected. Upload sizes are capped by the
configured maximum.

Dashboard script uploads require `MARL_API_TOKEN`. The dashboard sends the
token in the expected request header. CORS is disabled unless an explicit
origin allow-list is configured.

### 11.3 Scheduling a live task batch

The hub:

1. Confirms the model bundle is loaded.
2. Validates unique task IDs, dependencies, and script files.
3. Requires recent live nodes.
4. Aggregates current network/RAM information into an
   `InfrastructureContext`.
5. Predicts resources for every task.
6. Builds graph edges and GNN inputs; obtains refined estimates and
   embeddings.
7. Orders tasks topologically.
8. Estimates remaining node resources and subtracts reservations from
   already pending/running tasks.
9. Builds telemetry rows and feasibility masks.
10. Calls the configured scheduler, then reserves the selected node's
    predicted resources for subsequent tasks in the batch.
11. Writes JSON tickets atomically.

A ticket includes task metadata, assignment, prediction, GNN embedding,
feasible nodes, scheduler value, dependencies, and status. Atomic writes use
a temporary file and replacement to reduce the chance that a worker reads a
partially written JSON file.

### 11.4 Folder-drop scanning and dependency execution

The hub scans the shared task directory for stable `.py` files that do not
yet have an assignment ticket. It waits for file size and modification time
to stabilize, then groups detected dependency-connected scripts and submits
them together.

The worker:

1. Polls JSON tickets.
2. Uses an exclusive `.claim` file to reduce duplicate execution across
   worker processes.
3. Runs only tickets assigned to its node and still marked `PENDING`.
4. Waits for dependency tickets to become `COMPLETED`.
5. Marks the child failed if a dependency has failed.
6. Changes the ticket to `RUNNING`.
7. Runs the script with the worker's Python interpreter and a timeout.
8. Writes standard output/error and exit status to a result file.
9. Writes `COMPLETED` or `FAILED` and timing/error details back to the ticket.

Claims can become stale; the worker uses a timeout-based stale claim rule.
Network shares may have different file replacement and locking behavior than a
local disk, so validate on the target filesystem.

### 11.5 Live scheduler feedback

The hub records state, selected action, load, waiting-task count, and cost
rate with pending tasks. When a ticket is terminal:

- Successful task feedback uses measured `execution_time_sec` and configured
  node cost to compute an adaptive reward.
- Failed task feedback receives `-1`.
- Ready task experiences are batched into a scheduler update.
- If `MARL_SCHEDULER_PATH` is configured, the updated scheduler checkpoint is
  saved back to that path.
- Ticket metadata notes whether feedback was recorded and whether persistence
  succeeded.

This is scheduler-only adaptation. It does not refit the predictor or GNN.
The MAPPO scheduler's offline training reward is multi-component; the current
live feedback helper maintains the compatible project live reward contract
based on time, cost, load, and waiting count rather than replaying the full
offline synthetic reward calculation.

---

## 12. Persistence and checkpoint compatibility

### 12.1 Predictor

`model_persistence.save_predictor()` persists a fitted predictor with
`joblib`; loading validates that it is a fitted `TaskResourcePredictor`.
Only load joblib files from trusted sources because Python object
deserialization can execute code.

### 12.2 GNN

`save_gnn_encoder()` writes a PyTorch checkpoint containing encoder
dimensions, backend, and `state_dict`. `load_gnn_encoder()` reconstructs the
module, validates that the resource head outputs four values, loads weights,
and moves the model to the requested device.

### 12.3 Scheduler

The original `MARLScheduler` checkpoint includes its dimensions, PPO/reward
configuration, model weights, and optimizer state.

`mappo_scheduler.py` provides a scheduler-checkpoint loader that detects the
MAPPO checkpoint type and also loads the original scheduler checkpoint.
This keeps the hub compatible with both policies, subject to their expected
input dimensions.

Set all model path variables together:

```text
MARL_PREDICTOR_PATH
MARL_GNN_PATH
MARL_SCHEDULER_PATH
```

The hub rejects a partial set of paths. It also verifies that scheduler input
dimension equals `GNN embedding_dim + 7 telemetry values`.

The three-server MAPPO checkpoint was trained with three candidates. The
number of live candidate nodes and the feature ordering/units should match
the trained model's contract.

---

## 13. The separate cloud CSV experiment

`prepare_cloud_model_dataset.py` and `train_cloud_dataset.py` implement a
separate dataset-oriented experiment. It should not be confused with the
live eight-feature resource predictor.

### 13.1 Dataset model schema

The experiment uses categorical inputs such as data source, task type,
priority, and predicted load condition, plus active-user count. A
`DictVectorizer` converts feature dictionaries for scikit-learn pipelines.
The regressors predict dataset fields such as execution time, CPU utilization,
CPU allocation, RAM allocated, and RAM used.

A separate GPU classifier uses a **heuristic pseudo-label** indicating GPU
required. It is not a direct measured truth label unless the dataset
provenance explicitly establishes that.

### 13.2 Training/evaluation process

The code sorts by `task_start_time`, uses an 80/20 chronological split for
holdout evaluation, and reports regression MAE, RMSE, R-squared, and mean
baseline MAE. It also reports classification metrics for the GPU pseudo-label.
Final deployable experiment models are then fitted on the full available
dataset and serialized with joblib.

The chronological holdout is preferable to a random split for time-ordered
records, but it does not eliminate all confounding or distribution shift.
Inspect the detailed metrics and target provenance before drawing conclusions.

Past project notes report that some regression results were close to a mean
baseline and that the pseudo-label GPU classifier did not identify positive
cases. Do not present this separate experiment as a validated model for the
live `TaskResourcePredictor`; its schema and artifacts differ.

---

## 14. Installation and running the project

### 14.1 Python and dependencies

Use Python 3.10 or newer. Core dependencies are listed in `requirements.txt`:

- FastAPI and Uvicorn for the hub.
- Requests and HTTPX for network clients/tests.
- NumPy and scikit-learn for the predictor.
- PyTorch for the GNN and scheduler.
- Joblib for predictor persistence.
- psutil for worker telemetry.

PyTorch Geometric is optional; if absent, the project GNN uses its fallback
graph convolution.

From Windows PowerShell:

```powershell
cd "C:\Users\priyanshu solanki\OneDrive\Desktop\marl"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the environment already exists and dependencies are installed, activate it
and continue without recreating it.

### 14.2 Run the offline model-building demo

```powershell
python main_engine.py --tasks 8 --seed 7 --checkpoint-dir models
```

This trains the synthetic predictor/GNN demo models and the original
scheduler's short demo learning loop. It saves the three corresponding model
files into `models`.

### 14.3 Train reference-style MAPPO

If predictor and GNN checkpoints already exist and you only intend to train
MAPPO, do not regenerate them. Run:

```powershell
python train_reference_mappo.py `
  --episodes 1000 `
  --tasks 8 `
  --eval-seeds 20 `
  --seed 42 `
  --model-dir models `
  --output-dir results\mappo_reference_1000
```

This reads but does not retrain/overwrite the predictor and GNN. It writes the
MAPPO checkpoint and training/evaluation outputs into the specified results
directory.

### 14.4 Start a local live hub

Open a PowerShell window, configure the saved model paths and token, then run
the hub:

```powershell
$env:MARL_PREDICTOR_PATH = "$PWD\models\task_predictor.joblib"
$env:MARL_GNN_PATH = "$PWD\models\task_gnn.pt"
$env:MARL_SCHEDULER_PATH = "$PWD\results\mappo_reference_1000\mappo_scheduler.pt"
$env:MARL_TASKS_DIR = "C:\marl-test\cluster_task"
$env:MARL_NODE_IDS = "S1"
$env:MARL_HUB_HOST = "127.0.0.1"
$env:MARL_API_TOKEN = "<replace-with-a-long-private-token>"
New-Item -ItemType Directory -Force -Path $env:MARL_TASKS_DIR
python hub.py
```

The token above is a placeholder; never commit an actual token. The hub serves
its dashboard at `http://127.0.0.1:8000/`.

In a second PowerShell window, start the worker telemetry agent:

```powershell
$env:MARL_TASKS_DIR = "C:\marl-test\cluster_task"
python agent.py --node-id S1 --hub-url http://127.0.0.1:8000/api/telemetry --task-directory $env:MARL_TASKS_DIR
```

In a third window, start the task runner:

```powershell
python task_runner.py --node-id S1 --task-directory C:\marl-test\cluster_task --results-directory C:\marl-test\cluster_result
```

For multiple machines, use a reachable hub IP instead of loopback, configure
the same shared directories on hub and workers, and provide a unique node ID
for every worker. Configure `MARL_NODE_IDS` to exactly those IDs.

### 14.5 Use the original scheduler checkpoint instead

To use the original demo scheduler, point `MARL_SCHEDULER_PATH` at:

```powershell
$env:MARL_SCHEDULER_PATH = "$PWD\models\marl_scheduler.pt"
```

To use MAPPO, point it at the MAPPO checkpoint:

```powershell
$env:MARL_SCHEDULER_PATH = "$PWD\results\mappo_reference_1000\mappo_scheduler.pt"
```

Keep the predictor and GNN environment variables set either way. The scheduler
checkpoint must have an input dimension matching the GNN embedding plus seven
telemetry values.

---

## 15. Tests and validation

### 15.1 Run all unit tests

```powershell
python -m unittest discover -s . -p "test*.py"
```

The suite covers, among other things:

- AST dependency parsing, cycle/unknown dependency handling.
- Learning-quality and pipeline behavior.
- Live API/task flow with temporary directories and simulated telemetry.
- MAPPO action masks, PPO update, checkpoint round-trip, and loading of
  legacy/original scheduler checkpoints.

### 15.2 Additional checks

```powershell
python -m py_compile `
  task_predictors.py `
  gnn_encoder.py `
  marl_scheduler_bridge.py `
  marl_scheduler.py `
  mappo_scheduler.py `
  main_engine.py `
  train_reference_mappo.py `
  hub.py `
  agent.py `
  task_runner.py
```

The tests should be run after changes to scheduler state dimensions, action
index semantics, checkpoint formats, task validation, or ticket lifecycle.

### 15.3 Integration verification

A live-style smoke check should:

1. Load all three checkpoints through `load_model_bundle_from_environment()`.
2. Verify the scheduler and GNN dimensions agree.
3. Build one task/server bridge state with only one feasible server.
4. Confirm the scheduler selects that server.
5. Confirm the hub writes a valid ticket.
6. Run a worker against a temporary task folder and verify terminal status
   and result output.

Full production network shares, real cluster capacity, real GPU assignments,
and concurrent failure recovery still require environment-specific tests.

---

## 16. Configuration and operational reference

### 16.1 Hub environment variables

| Variable | Purpose |
|---|---|
| `MARL_PREDICTOR_PATH` | Joblib predictor checkpoint. |
| `MARL_GNN_PATH` | PyTorch GNN checkpoint. |
| `MARL_SCHEDULER_PATH` | Original PPO or MAPPO scheduler checkpoint; online updates can persist here. |
| `MARL_DEVICE` | PyTorch device, commonly `cpu` or `cuda`. |
| `MARL_TASKS_DIR` | Shared task scripts and JSON tickets directory. |
| `MARL_NODE_IDS` | Comma-separated allowed worker IDs. |
| `MARL_API_TOKEN` | Token required for script upload/task submission. |
| `MARL_MAX_UPLOAD_BYTES` | Maximum script upload size; default is 64 MiB. |
| `MARL_UPLOAD_STABILITY_SEC` | Stability period for folder-dropped uploads. |
| `MARL_HUB_HOST` | Uvicorn bind host; default is `0.0.0.0`. |
| `MARL_HUB_PORT` | Uvicorn port; default is 8000. |
| `MARL_DASHBOARD_ORIGINS` | Optional comma-separated CORS origin allow-list. |

### 16.2 Worker environment variables

| Variable | Purpose |
|---|---|
| `MARL_NODE_ID` | Unique worker identifier. |
| `MARL_TELEMETRY_URL` | Hub telemetry endpoint. |
| `MARL_TASKS_DIR` | Shared input/ticket directory. |
| `MARL_RESULTS_DIR` | Worker result output directory. |
| `MARL_NODE_COST_PER_SECOND` | Relative cost rate reported to the hub. |

### 16.3 Common task lifecycle statuses

```text
PENDING -> RUNNING -> COMPLETED
                    -> FAILED
```

A task can remain in the task directory waiting for an eligible scheduler or
node. A dependent task remains pending until its parent ticket is completed.
If a parent fails, the dependent task is marked failed.

### 16.4 Common troubleshooting points

**No assignment appears**

- Confirm the hub has all three model paths set and files exist.
- Confirm the API token is configured for uploads.
- Confirm recent telemetry is arriving and `MARL_NODE_IDS` matches workers.
- Confirm at least one active node can satisfy predicted CPU, GPU, and RAM.
- Check hub logs for waiting eligibility or malformed task/dependency data.

**Hub rejects the model bundle**

- Confirm all three path variables are set together.
- Check that the scheduler input dimension equals embedding dimension plus
  seven.
- Confirm the checkpoint is a supported scheduler checkpoint and the GNN
  encoder architecture/backend can be reconstructed.

**GPU tasks stay unassigned**

- Check the worker reports a positive GPU count and current utilization.
- `agent.py` detects NVIDIA devices only through `nvidia-smi`.
- A positive GPU requirement cannot fit on a reported zero-GPU node.
- The model cannot infer GPU needs reliably from arbitrary script source.
  Supply a correct baseline GPU hint.

**Dependency task does not run**

- Check each parent ticket ID and status.
- Confirm import matching found the intended script by exact module/file stem.
- Confirm there is no cycle.
- Remember that AST import matching is deliberately narrow.

**Worker output is missing**

- Check the result directory is writable and visible to the worker.
- Inspect the ticket's exit code, error, timeout, and `result_file`.
- The task runner captures stdout and stderr into the result text file.

---

## 17. Limitations, safety, and interpretation

### 17.1 Data and model limitations

- Demo predictor labels are synthetic, not measured.
- Demo GNN dependency edges and residual targets are generated.
- Demo server capacities, costs, and simulated execution outcomes are
  illustrative.
- The separate cloud CSV experiment has a different schema and includes a GPU
  pseudo-label.
- The 300/1,000-episode MAPPO runs are simulation results, not live workload
  benchmarks.
- Synthetic deadlines and cost-derived energy proxies differ from measured
  SLA and power/energy data.
- A 20-seed evaluation is useful as a smoke/evaluation check but is not a
  broad statistical validation across production workload distributions.

### 17.2 Scheduler limitations

- Action masks are only as correct as telemetry, capacity conversion, and
  predicted demand.
- Cost rates are configured inputs and may not represent billing cost.
- The live MAPPO critic context is approximate relative to the full offline
  training global state.
- Inference does not expose defer/reject in the hub path; it is expected to
  return a feasible placement.
- Online feedback updates the scheduler, not the predictor/GNN.
- A learned policy may simply approximate a heuristic (the current trained
  policy scores almost the same as fastest).
- Always compare against simple baselines such as cheapest, fastest, and
  random-feasible.

### 17.3 Infrastructure and security limitations

- The task runner executes submitted Python scripts with its OS account
  permissions. Use only trusted scripts or isolated workers/containers with
  least privilege.
- Protect the API token and do not write it into source control.
- Restrict hub network access and CORS origins to required clients.
- A shared folder is part of the control plane. Protect its permissions and
  verify behavior on the actual network filesystem.
- Joblib checkpoints must be trusted before loading.
- Do not expose worker shares, credentials, or production telemetry in logs
  or public evaluation artifacts.

### 17.4 What counts as a valid production claim

Before claiming resource prediction quality, collect measured task features
and measured execution/resource targets, define units, split by time/workload
or machine to avoid leakage, compare to simple baselines, and report per-target
metrics.

Before claiming scheduling quality, evaluate the same workload and server
conditions across the learned policy and baselines, use held-out tasks and
seeds, report completion/latency/cost/SLA/failure outcomes, and describe any
simulation assumptions. For live results, retain reproducible telemetry and
execution records and avoid changing the evaluation protocol after seeing
results.

---

## 18. Source-file guide

| File | Responsibility |
|---|---|
| `task_predictors.py` | Task/context/prediction data types, feature schema, four regressors, synthetic labels. |
| `gnn_encoder.py` | Dependency DAG validation, GNN inputs, graph encoder, resource residual head, training helper. |
| `marl_scheduler_bridge.py` | Pairwise task/server candidate states, IDs, and feasibility mask. |
| `marl_scheduler.py` | Original PPO scheduler, project adaptive reward, experience structures, checkpoint support. |
| `mappo_scheduler.py` | MAPPO shared actor/centralized critic, masks, GAE, PPO update, checkpoint factory. |
| `main_engine.py` | Offline pipeline, simulator, sample servers/workload, demo training and checkpoint output. |
| `train_reference_mappo.py` | Reference-style MAPPO simulator, rollout training, evaluation baselines, CSV/JSON/log/plot artifacts. |
| `model_persistence.py` | Predictor and GNN checkpoint save/load helpers. |
| `ast_dependency_detector.py` | Static import-to-task dependency heuristic. |
| `hub.py` | FastAPI service, telemetry, upload scanning, inference scheduling, ticket writing, live feedback updates. |
| `agent.py` | Worker machine telemetry heartbeat client. |
| `task_runner.py` | Ticket claiming, dependency checks, script execution, timeout and result recording. |
| `prepare_cloud_model_dataset.py` | Dataset cleaning/preparation for the separate CSV experiment. |
| `train_cloud_dataset.py` | Dataset-specific Gradient Boosting experiment and metrics. |
| `export_gnn_synthetic_dataset.py` | Exports synthetic graph training rows for inspection. |
| `test_ast_dependency_detector.py` | AST dependency detector tests. |
| `test_mappo_scheduler.py` | MAPPO masks/update/checkpoint compatibility tests. |
| `test_live_flow.py` | Live hub/worker flow integration test. |
| `test_learning_quality.py` | Offline learning/pipeline behavior tests. |
| `README.md` | Short setup, operating, and command guide. |
| `PROJECT_GUIDE.md` | Architecture and pipeline guide. |

---

## 19. Suggested production development path

1. **Collect measured training data.** Capture task inputs before execution
   and actual execution time, CPU, GPU, and RAM outcomes afterward. Maintain
   consistent units and provenance.
2. **Validate the predictor.** Use time-aware and machine-aware holdouts.
   Compare each target against useful simple baselines.
3. **Validate the graph semantics.** Use explicit dependency metadata when
   imports do not faithfully represent workflow order.
4. **Validate GNN value.** Compare the GNN-refined model against the predictor
   alone on untouched workflows and measured resource labels.
5. **Make live state match training state.** Ensure candidate feature
   normalization, telemetry availability, graph context, task batching, and
   server-count assumptions are consistent.
6. **Improve the simulator.** Calibrate execution, energy, queue, and SLA
   components against real measured behavior before relying on simulated
   rewards.
7. **Evaluate policies rigorously.** Compare MAPPO with fastest, cheapest,
   capacity-aware greedy, and random feasible strategies on the same held-out
   workloads and configurations.
8. **Add staged deployment.** Start with shadow decisions, then controlled
   canary workloads and rollback criteria. Do not let a new policy dispatch
   arbitrary untrusted code.
9. **Monitor drift and failures.** Track prediction error, infeasible-task
   rate, completion/failure rate, SLA misses, cost, latency, and policy
   feedback across machines and workload classes.
10. **Version artifacts and schemas.** Record feature order, units, model
    versions, training data fingerprints, and environment configuration with
    every deployed checkpoint.

The central lesson is that a working software pipeline and a trained
checkpoint are not, by themselves, proof of production allocation quality.
Reliable deployment depends on representative measurements, correct
state/action contracts, meaningful simulation or real feedback, and honest
comparison against strong operational baselines.
