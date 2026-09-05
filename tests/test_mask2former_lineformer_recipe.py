from types import SimpleNamespace

from training.train_mask2former_lineformer import training_arguments


def test_recipe_does_not_finetune_released_lineformer_or_enable_early_stopping():
    args = SimpleNamespace(lineformer_root="upstream", dataset="mixed", output="run",
                           max_iters=100000, workers_per_gpu=4, num_gpus=2,
                           seed=20260905, resume=False, dry_run=False, smoke_test=False)
    command = training_arguments(args)
    values = dict(zip(command[::2], command[1::2]))
    assert "--weights" not in command
    assert values["--samples-per-gpu"] == "4"
    assert values["--num-gpus"] == "2"
    assert values["--max-iters"] == "100000"
    assert values["--base-lr"] == "1e-4"
    assert values["--early-stopping-patience"] == "0"
    assert values["--eval-interval"] == "250"
    assert values["--checkpoint-interval"] == "500"
    assert values["--log-interval"] == "100"
