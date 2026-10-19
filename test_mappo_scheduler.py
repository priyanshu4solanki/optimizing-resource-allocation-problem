import tempfile
import unittest
from pathlib import Path

import torch

from marl_scheduler import MARLScheduler
from marl_scheduler_bridge import build_joint_state
from mappo_scheduler import (
    MAPPORolloutTransition,
    MAPPOScheduler,
    load_scheduler_checkpoint,
)
from train_reference_mappo import (
    REWARD_BLEND_ALPHA,
    REWARD_WEIGHTS,
    _SchedulingEpisode,
)


class MAPPOSchedulerTests(unittest.TestCase):
    def make_scheduler(self):
        torch.manual_seed(4)
        return MAPPOScheduler(
            input_dim=8,
            num_servers=2,
            observation_dim=16,
            global_state_dim=32,
            actor_hidden_dim=16,
            critic_hidden_dim=24,
            update_epochs=2,
            minibatches=2,
        )

    def make_bridge_state(self):
        return build_joint_state(
            torch.tensor([[0.25]]),
            torch.tensor(
                [
                    [0.8, 0.7, 0.6, 0.5, 0.2, 0.3, 0.5],
                    [0.4, 0.6, 0.9, 0.7, 0.3, 0.5, 0.9],
                ]
            ),
            task_ids=("task-1",),
            server_ids=("node-a", "node-b"),
            feasibility_mask=torch.tensor([[True, False]]),
        )

    def test_bridge_inference_masks_defer_and_reject(self):
        scheduler = self.make_scheduler()
        decision = scheduler.select_actions(
            self.make_bridge_state(), deterministic=True
        )

        self.assertEqual(decision.action_indices, (0,))
        self.assertEqual(decision.assignments, (("task-1", "node-a"),))

    def test_mappo_update_uses_masked_gae_rollout(self):
        scheduler = self.make_scheduler()
        observation = torch.randn(16)
        global_state = torch.randn(32)
        mask = torch.tensor([True, False, True, True])
        action = scheduler.act(observation, global_state, mask)
        transition = MAPPORolloutTransition(
            observation=action.observation,
            global_state=action.global_state,
            action_mask=action.action_mask,
            action=action.action,
            log_probability=action.log_probability,
            value=action.value,
            reward=0.75,
            done=True,
        )

        metrics = scheduler.update([transition])

        self.assertEqual(metrics["transitions"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["actor_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["critic_loss"])))

    def test_checkpoint_round_trip_and_factory(self):
        scheduler = self.make_scheduler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mappo.pt"
            scheduler.save(str(path))

            loaded = MAPPOScheduler.load(str(path))
            factory_loaded = load_scheduler_checkpoint(str(path))

        self.assertIsInstance(loaded, MAPPOScheduler)
        self.assertIsInstance(factory_loaded, MAPPOScheduler)
        expected = scheduler.select_actions(self.make_bridge_state(), deterministic=True)
        actual = loaded.select_actions(self.make_bridge_state(), deterministic=True)
        self.assertEqual(expected.action_indices, actual.action_indices)

    def test_factory_preserves_original_scheduler_checkpoint_loading(self):
        scheduler = MARLScheduler(input_dim=8, hidden_dim=16)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ppo.pt"
            scheduler.save(str(path))
            loaded = load_scheduler_checkpoint(str(path))

        self.assertIsInstance(loaded, MARLScheduler)

    def test_reward_blends_action_penalty_locally_and_admission_globally(self):
        components = {name: 0.0 for name in REWARD_WEIGHTS}
        components["action_penalty"] = 1.0
        components["admission"] = 1.0

        reward = _SchedulingEpisode._combine_reward(components)

        expected = REWARD_WEIGHTS["action_penalty"] * REWARD_BLEND_ALPHA
        expected += REWARD_WEIGHTS["admission"] * (1 - REWARD_BLEND_ALPHA)
        self.assertAlmostEqual(reward, expected)


if __name__ == "__main__":
    unittest.main()
