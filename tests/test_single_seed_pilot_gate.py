"""A one-seed pilot cannot silently masquerade as formal three-seed P1."""

from argparse import Namespace
import unittest

from scripts.train_fixed_probe import _validate_seed_plan


def args(*, seeds=(42,), pilot=False, only_seed=None, smoke=False,
         initialize_only=False, resume=False):
    return Namespace(seeds=seeds, single_seed_pilot=pilot,
                     only_seed=only_seed, smoke=smoke,
                     initialize_only=initialize_only, resume=resume)


class SingleSeedPilotGateTests(unittest.TestCase):
    def test_one_seed_requires_explicit_pilot_flag(self):
        with self.assertRaisesRegex(ValueError, "three-seed"):
            _validate_seed_plan(args())

    def test_pilot_runs_full_BCD_for_one_seed(self):
        _validate_seed_plan(args(pilot=True))
        with self.assertRaisesRegex(ValueError, "full B/C/D"):
            _validate_seed_plan(args(pilot=True, only_seed=42))

    def test_pilot_rejects_three_seed_claim(self):
        with self.assertRaisesRegex(ValueError, "exactly one seed"):
            _validate_seed_plan(args(seeds=(42, 43, 44), pilot=True))

    def test_old_formal_protocol_still_requires_three_seeds(self):
        _validate_seed_plan(args(seeds=(42, 43, 44)))
        with self.assertRaisesRegex(ValueError, "present"):
            _validate_seed_plan(args(seeds=(42, 43, 44), only_seed=99))


if __name__ == "__main__":
    unittest.main()
