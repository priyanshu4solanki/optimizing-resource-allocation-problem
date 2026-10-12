"""Checks for GNN resource refinement and adaptive scheduler rewards."""

import unittest
import tempfile
from pathlib import Path

import torch

from gnn_encoder import (
    TaskGraphEncoder,
    task_gnn_feature_tensor,
)
from marl_scheduler import AdaptiveRewardConfig, MARLScheduler
from model_persistence import load_gnn_encoder, save_gnn_encoder
from task_predictors import (
    TaskResourcePredictor,
    generate_synthetic_training_data,
    synthetic_resource_targets,
)


class LearningQualityTest(unittest.TestCase):
    def test_gnn_supervised_resource_refinement_never_worsens_held_out_mae(self) -> None:
        tasks, contexts = generate_synthetic_training_data(240, seed=113)
        split = 180
        predictor = TaskResourcePredictor(random_state=113)
        predictor.fit_synthetic(tasks[:split], contexts[:split])
        train_predictions = predictor.predict(tasks[:split], contexts[:split])
        validation_predictions = predictor.predict(
            tasks[split:], contexts[split:]
        )
        train_features = task_gnn_feature_tensor(
            tasks[:split], contexts[:split], train_predictions
        )
        validation_features = task_gnn_feature_tensor(
            tasks[split:], contexts[split:], validation_predictions
        )
        train_targets = self._targets(tasks[:split], contexts[:split])
        validation_targets = self._targets(tasks[split:], contexts[split:])
        encoder = TaskGraphEncoder(input_dim=12, hidden_dim=16, embedding_dim=8)
        metrics = encoder.fit_resource_refiner(
            train_features,
            torch.empty((2, 0), dtype=torch.long),
            train_targets,
            validation_features,
            torch.empty((2, 0), dtype=torch.long),
            validation_targets,
            epochs=30,
            patience=8,
        )
        refined, embeddings = encoder.predict_resources(
            validation_features, torch.empty((2, 0), dtype=torch.long)
        )
        baseline_mae = (
            validation_features[:, :4] - validation_targets
        ).abs().mean()
        refined_mae = (refined - validation_targets).abs().mean()
        self.assertEqual(tuple(embeddings.shape), (60, 8))
        self.assertLessEqual(float(refined_mae), float(baseline_mae) + 1e-6)
        self.assertLessEqual(metrics["refined_mae"], metrics["baseline_mae"] + 1e-6)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "task_gnn.pt"
            save_gnn_encoder(encoder, checkpoint)
            restored = load_gnn_encoder(checkpoint)
            restored_predictions, restored_embeddings = restored.predict_resources(
                validation_features, torch.empty((2, 0), dtype=torch.long)
            )
            self.assertTrue(torch.allclose(refined, restored_predictions))
            self.assertTrue(torch.allclose(embeddings, restored_embeddings))

    def test_adaptive_reward_reverses_tradeoff_when_workload_is_busy(self) -> None:
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
        fast_expensive_idle = scheduler.compute_reward(
            actual_execution_time=2,
            actual_cost=8,
            system_load=0.0,
            tasks_waiting=0,
        )
        slow_cheap_idle = scheduler.compute_reward(
            actual_execution_time=8,
            actual_cost=2,
            system_load=0.0,
            tasks_waiting=0,
        )
        fast_expensive_busy = scheduler.compute_reward(
            actual_execution_time=2,
            actual_cost=8,
            system_load=0.9,
            tasks_waiting=0,
        )
        slow_cheap_busy = scheduler.compute_reward(
            actual_execution_time=8,
            actual_cost=2,
            system_load=0.9,
            tasks_waiting=0,
        )
        queue_pressure = scheduler.compute_reward(
            actual_execution_time=2,
            actual_cost=8,
            system_load=0.0,
            tasks_waiting=10,
        )

        self.assertGreater(slow_cheap_idle.reward, fast_expensive_idle.reward)
        self.assertGreater(fast_expensive_busy.reward, slow_cheap_busy.reward)
        self.assertGreater(
            fast_expensive_busy.execution_time_weight,
            fast_expensive_idle.execution_time_weight,
        )
        self.assertEqual(queue_pressure.workload_pressure, 1.0)
        self.assertEqual(
            queue_pressure.cost_weight,
            scheduler.reward_config.cost_weight_at_max_pressure,
        )
        self.assertAlmostEqual(fast_expensive_idle.reward, 0.38)
        self.assertAlmostEqual(fast_expensive_busy.reward, 0.65)

    @staticmethod
    def _targets(tasks, contexts):
        return torch.tensor(
            [
                [
                    row[name]
                    for name in (
                        "target_exec_time",
                        "target_cpu_req",
                        "target_gpu_req",
                        "target_ram_req",
                    )
                ]
                for row in synthetic_resource_targets(tasks, contexts)
            ],
            dtype=torch.float32,
        )


if __name__ == "__main__":
    unittest.main()
