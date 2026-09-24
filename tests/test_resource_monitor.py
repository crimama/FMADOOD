from scripts.resource_monitor import parse_sample


def test_parse_sample() -> None:
    sample = "CPU\t12.5\nMEM\t1048576\t2097152\nGPU\t0, RTX 4090, 21, 512, 24564, 48\n"

    actual = parse_sample(sample)

    assert actual["cpu"] == 12.5
    assert actual["memory"] == {"used_gib": 1.0, "total_gib": 2.0}
    assert actual["gpus"] == [{"index": "0", "name": "RTX 4090", "utilization": 21.0, "used_mib": 512.0, "total_mib": 24564.0, "temperature": 48.0}]
