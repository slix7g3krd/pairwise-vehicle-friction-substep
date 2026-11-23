import pytest
import gameplay.vehicle_friction_substep as mod

CASES = [
    ("friction_state", (1)),
    ("shared_substep", (1)),
    ("surface_reset", (1)),
    ("torque_damp", (1)),
    ("integrate_order", (1)),
    ("collision_response", (1)),
    ("slip_ratio", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("suspension", (1)),
    ("metrics", (1)),
    ("replay_trace", (1)),
    ("tire_curve", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
