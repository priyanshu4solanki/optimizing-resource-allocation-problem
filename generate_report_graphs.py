"""Generate report-ready evaluation plots for the MARL scheduling project."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gnn_encoder import TaskGraphEncoder, task_gnn_feature_tensor
from marl_scheduler import AdaptiveRewardConfig, MARLScheduler
from task_predictors import (
    TaskResourcePredictor,
    generate_synthetic_training_data,
    synthetic_resource_targets,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "report_graphs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def _as_targets_array(task_rows, contexts):
    rows = synthetic_resource_targets(task_rows, contexts)
    return np.array(
        [[row[name] for name in ("target_exec_time", "target_cpu_req", "target_gpu_req", "target_ram_req")] for row in rows],
        dtype=float,
    )


def generate_validation_scatter():
    tasks, contexts = generate_synthetic_training_data(240, seed=113)
    split = 180

    predictor = TaskResourcePredictor(random_state=113)
    predictor.fit_synthetic(tasks[:split], contexts[:split])

    baseline_predictions = predictor.predict(tasks[split:], contexts[split:])
    baseline_array = np.array(
        [[
            item.target_exec_time,
            item.target_cpu_req,
            item.target_gpu_req,
            item.target_ram_req,
        ] for item in baseline_predictions],
        dtype=float,
    )

    train_features = task_gnn_feature_tensor(
        tasks[:split], contexts[:split], predictor.predict(tasks[:split], contexts[:split])
    )
    validation_features = task_gnn_feature_tensor(
        tasks[split:], contexts[split:], baseline_predictions
    )
    train_targets = _as_targets_array(tasks[:split], contexts[:split])
    validation_targets = _as_targets_array(tasks[split:], contexts[split:])

    encoder = TaskGraphEncoder(input_dim=12, hidden_dim=16, embedding_dim=8)
    encoder.fit_resource_refiner(
        train_features,
        torch.empty((2, 0), dtype=torch.long),
        torch.tensor(train_targets, dtype=torch.float32),
        validation_features,
        torch.empty((2, 0), dtype=torch.long),
        torch.tensor(validation_targets, dtype=torch.float32),
        epochs=30,
        patience=8,
    )
    refined, _ = encoder.predict_resources(
        validation_features, torch.empty((2, 0), dtype=torch.long)
    )
    refined_array = refined.detach().cpu().numpy()

    labels = [
        "Execution time",
        "CPU requirement",
        "GPU requirement",
        "RAM requirement",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("GNN Demo: Synthetic Validation Targets vs Predictions", fontsize=20, y=0.98)

    for idx, ax in enumerate(axes.flat):
        target_col = validation_targets[:, idx]
        baseline_col = baseline_array[:, idx]
        refined_col = refined_array[:, idx]

        ax.scatter(target_col, baseline_col, s=15, alpha=0.7, label="Predictor baseline", color="#7bb7d8")
        ax.scatter(target_col, refined_col, s=15, alpha=0.7, label="GNN-refined", color="#5fc39a")

        min_val = float(min(target_col.min(), baseline_col.min(), refined_col.min()))
        max_val = float(max(target_col.max(), baseline_col.max(), refined_col.max()))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.2, alpha=0.8)

        ax.set_title(labels[idx])
        ax.set_xlabel("Synthetic hold-out target")
        ax.set_ylabel("Prediction")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "gnn_validation_targets_vs_predictions.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved validation scatter plot to {out_path}")


def generate_reward_tradeoff_plot():
    scheduler = MARLScheduler(
        input_dim=3,
        hidden_dim=8,
        reward_config=AdaptiveRewardConfig(
            execution_time_scale=10,
            cost_scale=10,
            cost_weight_at_idle=0.7,
            cost_weight_at_max_pressure=0.2,
            queue_pressure_threshold=10,
        ),
    )

    cases = {
        "Idle-fast-expensive": scheduler.compute_reward(
            actual_execution_time=2,
            actual_cost=8,
            system_load=0.0,
            tasks_waiting=0,
        ).reward,
        "Idle-slow-cheap": scheduler.compute_reward(
            actual_execution_time=8,
            actual_cost=2,
            system_load=0.0,
            tasks_waiting=0,
        ).reward,
        "Busy-fast-expensive": scheduler.compute_reward(
            actual_execution_time=2,
            actual_cost=8,
            system_load=0.9,
            tasks_waiting=0,
        ).reward,
        "Busy-slow-cheap": scheduler.compute_reward(
            actual_execution_time=8,
            actual_cost=2,
            system_load=0.9,
            tasks_waiting=0,
        ).reward,
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b"]
    bars = ax.bar(cases.keys(), cases.values(), color=colors)
    ax.set_title("Adaptive Reward Trade-off under Different Workload Conditions")
    ax.set_ylabel("Reward")
    ax.set_ylim(min(cases.values()) - 0.1, max(cases.values()) + 0.1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, cases.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "adaptive_reward_tradeoff.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved reward plot to {out_path}")


if __name__ == "__main__":
    generate_validation_scatter()
    generate_reward_tradeoff_plot()
    print(f"All plots stored in: {OUTPUT_DIR}")
