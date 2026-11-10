from scripts.run_p0_when_free import available_gpu


def test_admission_needs_memory_utilization_and_no_compute_process(monkeypatch):
    responses = iter([
        "0, GPU-a, 24199, 0\n1, GPU-b, 23000, 5\n2, GPU-c, 21999, 0\n3, GPU-d, 24000, 10\n",
        "GPU-b, 4217\n",
    ])
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: next(responses))
    assert available_gpu() == [0]


def test_admission_refuses_unparseable_gpu_snapshot(monkeypatch):
    responses = iter(["unexpected\n", ""])
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: next(responses))
    try:
        available_gpu()
    except RuntimeError as exc:
        assert "unexpected nvidia-smi" in str(exc)
    else:
        raise AssertionError("unparseable GPU state was accepted")
