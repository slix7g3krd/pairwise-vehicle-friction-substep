"""载具摩擦与悬挂子步核心模块。

修复的六类问题：
1. 摩擦状态机：静摩擦/滑动摩擦带迟滞，低帧率下不再从静摩擦直接跳到滑动。
2. 悬挂与车身同子步：悬挂力与车身积分在同一个固定子步内推进，消除抖动。
3. 路面材质切换重置：切换材质时重置摩擦状态、滑移历史与缓存摩擦系数。
4. 四驱扭矩分配阻尼：打滑时对分配比例做低通阻尼，避免振荡。
5. 确定性积分：固定子步长 + 固定阶段顺序 + 无随机/时间依赖，相同输入轨迹一致。
6. 碰撞响应：基于冲量的法向/切向响应，含恢复系数与摩擦钳制。

所有函数纯 Python、无随机数、无墙钟依赖，保证确定性。
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 常量与默认配置
# ---------------------------------------------------------------------------

FIXED_DT = 1.0 / 120.0          # 固定子步长（秒）
MAX_SUBSTEPS = 16               # 单帧最大子步数，防止低帧率螺旋
GRAVITY = 9.81

SLIP_ENTER_KINETIC = 0.12       # 进入滑动摩擦的滑移率阈值
SLIP_EXIT_KINETIC = 0.06        # 退出滑动摩擦的滑移率阈值（迟滞）
STATIC_EPSILON = 1e-4           # 视为静止的速度阈值

TORQUE_SPLIT_DAMPING = 0.25     # 扭矩分配比例低通系数（0~1，越小越平滑）

DEFAULT_SURFACE = {
    "name": "asphalt",
    "mu_static": 1.0,
    "mu_kinetic": 0.7,
    "restitution": 0.2,
}

SURFACES = {
    "asphalt": DEFAULT_SURFACE,
    "gravel": {"name": "gravel", "mu_static": 0.6, "mu_kinetic": 0.45, "restitution": 0.1},
    "ice": {"name": "ice", "mu_static": 0.25, "mu_kinetic": 0.1, "restitution": 0.05},
}

DEFAULT_CONFIG = {
    "mass": 1200.0,             # 车身质量 kg
    "wheel_radius": 0.33,       # 轮胎半径 m
    "wheel_inertia": 1.2,       # 单轮转动惯量
    "spring_k": 35000.0,        # 悬挂刚度 N/m
    "damper_c": 3500.0,         # 悬挂阻尼 N·s/m
    "rest_length": 0.45,        # 悬挂自然长度 m
    "max_travel": 0.15,         # 悬挂最大行程 m
    "surface": "asphalt",
    "wheels": ["fl", "fr", "rl", "rr"],  # 固定顺序，保证确定性
    "drive": "awd",
}


def _as_config(cfg) -> dict:
    """把任意输入规整为完整配置 dict（合并默认值，键序固定）。"""
    merged = dict(DEFAULT_CONFIG)
    if isinstance(cfg, dict):
        for key in DEFAULT_CONFIG:
            if key in cfg and cfg[key] is not None:
                merged[key] = cfg[key]
    return merged


def _as_state(state, cfg: dict) -> dict:
    """把任意输入规整为完整车辆状态 dict。"""
    base = {
        "velocity": 0.0,            # 车身纵向速度 m/s
        "position": 0.0,
        "suspension": {w: 0.0 for w in cfg["wheels"]},   # 压缩量
        "suspension_vel": {w: 0.0 for w in cfg["wheels"]},
        "wheel_speed": {w: 0.0 for w in cfg["wheels"]},  # 轮心线速度 m/s
        "friction_state": {w: "static" for w in cfg["wheels"]},
        "slip_history": {w: 0.0 for w in cfg["wheels"]},
        "surface": cfg["surface"],
        "torque_split": {"front": 0.5, "rear": 0.5},
    }
    if isinstance(state, dict):
        for key, value in state.items():
            if key in ("suspension", "suspension_vel", "wheel_speed",
                       "friction_state", "slip_history") and isinstance(value, dict):
                merged = dict(base[key])
                merged.update(value)
                base[key] = merged
            elif key == "torque_split" and isinstance(value, dict):
                merged = dict(base["torque_split"])
                merged.update(value)
                base[key] = merged
            elif key in base:
                base[key] = value
    return base


# ---------------------------------------------------------------------------
# 轮胎与滑移
# ---------------------------------------------------------------------------

def slip_ratio(state=None, wheel: str = "fl", cfg=None):
    """滑移率 s = (v_wheel - v_body) / max(|v_body|, v_min)，钳制到 [-1, 1]。"""
    config = _as_config(cfg)
    st = _as_state(state, config)
    v_body = st["velocity"]
    v_wheel = st["wheel_speed"].get(wheel, 0.0)
    denom = max(abs(v_body), 0.5)   # 低速时用最小速度避免除零爆炸
    slip = (v_wheel - v_body) / denom
    return max(-1.0, min(1.0, slip))


def tire_curve(slip=None, mu_peak: float = 1.0):
    """简化 Pacejka 轮胎曲线：mu(slip) = mu_peak * sin(C * atan(B * slip))。

    确定性纯函数：相同 slip 必得相同 mu。
    """
    if not isinstance(slip, (int, float)) or isinstance(slip, bool):
        slip = 0.0
    B, C = 10.0, 1.9
    return mu_peak * math.sin(C * math.atan(B * float(slip)))


# ---------------------------------------------------------------------------
# 1. 摩擦状态机（带迟滞，低帧率下不跳变）
# ---------------------------------------------------------------------------

def friction_state(state=None, wheel: str = "fl", cfg=None):
    """静/滑摩擦状态机。

    迟滞：slip > SLIP_ENTER_KINETIC 才进入滑动；slip < SLIP_EXIT_KINETIC 才回到静摩擦。
    低帧率单帧滑移率再大，也只推进一个状态转移，不直接从 static 跳到完全滑动
    失控——中间经 transition 态按子步逐步放开。
    """
    config = _as_config(cfg)
    st = _as_state(state, config)
    s = abs(slip_ratio(st, wheel, config))
    current = st["friction_state"].get(wheel, "static")

    if current == "static":
        nxt = "transition" if s > SLIP_ENTER_KINETIC else "static"
    elif current == "transition":
        if s < SLIP_EXIT_KINETIC:
            nxt = "static"
        elif s > SLIP_ENTER_KINETIC * 2.0:
            nxt = "kinetic"
        else:
            nxt = "transition"
    else:  # kinetic
        nxt = "static" if s < SLIP_EXIT_KINETIC else "kinetic"

    surface = SURFACES.get(st["surface"], DEFAULT_SURFACE)
    mu_cap = surface["mu_static"] if nxt == "static" else surface["mu_kinetic"]
    if nxt == "transition":
        mu_cap = 0.5 * (surface["mu_static"] + surface["mu_kinetic"])

    return {
        "wheel": wheel,
        "state": nxt,
        "slip": s,
        "mu": tire_curve(s, mu_cap),
        "mu_cap": mu_cap,
    }


# ---------------------------------------------------------------------------
# 悬挂模型
# ---------------------------------------------------------------------------

def suspension(state=None, wheel: str = "fl", cfg=None):
    """弹簧-阻尼悬挂力，行程钳制到 [-max_travel, max_travel]。"""
    config = _as_config(cfg)
    st = _as_state(state, config)
    compression = st["suspension"].get(wheel, 0.0)
    travel = max(-config["max_travel"], min(config["max_travel"], compression))
    vel = st["suspension_vel"].get(wheel, 0.0)
    force = config["spring_k"] * travel + config["damper_c"] * vel
    return {
        "wheel": wheel,
        "compression": travel,
        "force": force,
        "clamped": travel != compression,
    }


# ---------------------------------------------------------------------------
# 2. 悬挂与车身同子步积分
# ---------------------------------------------------------------------------

def _substep_once(st: dict, config: dict, dt: float, throttle: float, brake: float):
    """一个固定子步：悬挂力 -> 轮速 -> 摩擦 -> 车身，全部在同一 dt 内推进。"""
    total_susp_force = 0.0
    traction = 0.0
    for wheel in config["wheels"]:  # 固定顺序，确定性
        # 悬挂（与车身同子步）
        sus = suspension(st, wheel, config)
        st["suspension"][wheel] = sus["compression"]
        total_susp_force += sus["force"]

        # 轮速：驱动/制动力矩
        drive_torque = throttle * 400.0
        brake_torque = brake * 800.0
        omega_acc = (drive_torque - brake_torque) / config["wheel_inertia"]
        st["wheel_speed"][wheel] += omega_acc * config["wheel_radius"] * dt * 0.01

        # 摩擦状态机 -> 牵引力
        fr = friction_state(st, wheel, config)
        st["friction_state"][wheel] = fr["state"]
        st["slip_history"][wheel] = fr["slip"]
        normal = config["mass"] * GRAVITY / len(config["wheels"])
        traction += fr["mu"] * normal * (1.0 if st["wheel_speed"][wheel] >= st["velocity"] else -1.0)

    # 车身积分（半隐式欧拉，确定性）
    accel = traction / config["mass"] - 0.02 * st["velocity"]
    st["velocity"] += accel * dt
    st["position"] += st["velocity"] * dt

    # 悬挂几何随车身姿态更新（简化：载荷转移回压缩量）
    for wheel in config["wheels"]:
        target = total_susp_force / len(config["wheels"]) / config["spring_k"] * 0.001
        st["suspension"][wheel] += (target - st["suspension"][wheel]) * 0.1
    return st


def shared_substep(state=None, dt_frame: float = None, throttle: float = 0.0,
                   brake: float = 0.0, cfg=None):
    """把一帧切成固定 dt 的子步，悬挂与车身在每个子步内同步积分。

    低帧率（大 dt_frame）时增加子步数而不是放大步长，消除悬挂/车身失步抖动。
    """
    config = _as_config(cfg)
    st = _as_state(state, config)
    if not isinstance(dt_frame, (int, float)) or isinstance(dt_frame, bool) or dt_frame <= 0:
        dt_frame = FIXED_DT
    dt_frame = float(dt_frame)

    n = max(1, min(MAX_SUBSTEPS, math.ceil(dt_frame / FIXED_DT)))
    dt = dt_frame / n  # 均分，保证整帧时间精确覆盖且每步 <= FIXED_DT
    for _ in range(n):
        st = _substep_once(st, config, dt, throttle, brake)
    st["substeps"] = n
    st["substep_dt"] = dt
    return st


# ---------------------------------------------------------------------------
# 3. 路面材质切换重置
# ---------------------------------------------------------------------------

def surface_reset(state=None, new_surface: str = None, cfg=None):
    """切换路面材质：重置摩擦状态机、滑移历史与扭矩分配缓存。

    不重置会沿用旧材质的滑动状态，导致新材质上摩擦系数错误。
    """
    config = _as_config(cfg)
    st = _as_state(state, config)
    if not isinstance(new_surface, str) or new_surface not in SURFACES:
        new_surface = config["surface"]
    st["surface"] = new_surface
    for wheel in config["wheels"]:
        st["friction_state"][wheel] = "static"
        st["slip_history"][wheel] = 0.0
    st["torque_split"] = {"front": 0.5, "rear": 0.5}
    st["surface_mu"] = {
        "mu_static": SURFACES[new_surface]["mu_static"],
        "mu_kinetic": SURFACES[new_surface]["mu_kinetic"],
    }
    return st


# ---------------------------------------------------------------------------
# 4. 四驱扭矩分配阻尼
# ---------------------------------------------------------------------------

def torque_damp(state=None, cfg=None):
    """四驱扭矩分配：按轴打滑差计算目标分配，再低通阻尼防止振荡。

    打滑时目标分配会剧烈变化，直接应用会在前后轴间来回跳；
    对分配比例做指数低通（TORQUE_SPLIT_DAMPING）后输出。
    """
    config = _as_config(cfg)
    st = _as_state(state, config)
    wheels = config["wheels"]
    front = [w for w in wheels if w.startswith("f")]
    rear = [w for w in wheels if w.startswith("r")]

    def axle_slip(axle):
        if not axle:
            return 0.0
        return sum(st["slip_history"].get(w, 0.0) for w in axle) / len(axle)

    slip_f = axle_slip(front)
    slip_r = axle_slip(rear)
    # 目标：把扭矩往打滑小的轴转移
    diff = slip_f - slip_r
    target_front = max(0.2, min(0.8, 0.5 - 0.5 * diff))

    prev = st["torque_split"].get("front", 0.5)
    smoothed = prev + TORQUE_SPLIT_DAMPING * (target_front - prev)
    smoothed = max(0.2, min(0.8, smoothed))
    return {
        "front": smoothed,
        "rear": 1.0 - smoothed,
        "target_front": target_front,
        "damped": abs(smoothed - target_front) > 1e-12,
    }


# ---------------------------------------------------------------------------
# 5. 确定性积分顺序
# ---------------------------------------------------------------------------

INTEGRATION_STAGES = (
    "input",        # 采样输入
    "suspension",   # 悬挂力
    "wheel",        # 轮速/扭矩
    "friction",     # 摩擦状态机
    "chassis",      # 车身积分
    "collision",    # 碰撞响应
    "bookkeeping",  # 历史与缓存
)


def integrate_order(state=None, cfg=None):
    """返回固定的积分阶段顺序；若给定状态，按该顺序执行一个子步。

    阶段顺序固定 + 轮序固定 + 固定 dt => 相同输入两次轨迹完全一致。
    """
    config = _as_config(cfg)
    if not isinstance(state, dict):
        return list(INTEGRATION_STAGES)
    st = _substep_once(_as_state(state, config), config, FIXED_DT, 0.0, 0.0)
    st["order"] = list(INTEGRATION_STAGES)
    return st


def determinism(state=None, cfg=None):
    """确定性校验：相同初始状态与输入跑两遍，轨迹必须逐比特一致。"""
    config = _as_config(cfg)
    st = _as_state(state, config)

    def run():
        s = _as_state(st, config)
        trace = []
        for _ in range(10):
            s = shared_substep(s, FIXED_DT, throttle=0.5, cfg=config)
            trace.append((s["position"], s["velocity"],
                          tuple(s["friction_state"][w] for w in config["wheels"])))
        return trace

    t1 = run()
    t2 = run()
    return {"deterministic": t1 == t2, "steps": len(t1), "trace": t1}


# ---------------------------------------------------------------------------
# 6. 碰撞响应
# ---------------------------------------------------------------------------

def collision_response(state=None, normal=(0.0, 1.0), penetration: float = 0.0, cfg=None):
    """冲量式碰撞响应：法向恢复 + 切向摩擦钳制 + 穿透位置修正。"""
    config = _as_config(cfg)
    st = _as_state(state, config)
    surface = SURFACES.get(st["surface"], DEFAULT_SURFACE)
    e = surface["restitution"]
    mu = surface["mu_kinetic"]

    nx, ny = normal
    norm = math.hypot(nx, ny)
    if norm < STATIC_EPSILON:
        nx, ny, norm = 0.0, 1.0, 1.0
    nx, ny = nx / norm, ny / norm

    v = st["velocity"]
    vn = v * ny  # 简化：一维速度投影到法向
    vt = v * nx
    jn = -(1.0 + e) * vn * config["mass"] if vn < 0 else 0.0
    jt = max(-mu * abs(jn), min(mu * abs(jn), -vt * config["mass"]))

    new_v = v + (jn * ny + jt * nx) / config["mass"]
    correction = max(0.0, penetration) * 0.8  # Baumgarte 式位置修正
    st["velocity"] = new_v
    st["position"] += correction * ny
    return {
        "state": st,
        "impulse_normal": jn,
        "impulse_tangent": jt,
        "restitution": e,
        "penetration_corrected": correction,
    }


# ---------------------------------------------------------------------------
# 辅助：校验 / 审计 / 回退 / 指标 / 回放
# ---------------------------------------------------------------------------

def validation(cfg=None):
    """配置校验：返回问题列表，空列表表示合法。"""
    config = _as_config(cfg)
    problems = []
    if config["mass"] <= 0:
        problems.append("mass must be positive")
    if config["wheel_radius"] <= 0:
        problems.append("wheel_radius must be positive")
    if config["spring_k"] <= 0 or config["damper_c"] < 0:
        problems.append("invalid suspension constants")
    if config["surface"] not in SURFACES:
        problems.append("unknown surface: %s" % config["surface"])
    if not config["wheels"]:
        problems.append("no wheels configured")
    return {"ok": not problems, "problems": problems, "config": config}


def fallback(cfg=None):
    """校验失败时的安全回退配置。"""
    report = validation(cfg)
    if report["ok"]:
        return report["config"]
    safe = dict(DEFAULT_CONFIG)
    safe["fallback"] = True
    safe["problems"] = report["problems"]
    return safe


def audit(state=None, cfg=None):
    """状态审计：检查 NaN/Inf、摩擦状态合法性、扭矩分配守恒。"""
    config = _as_config(cfg)
    st = _as_state(state, config)
    issues = []
    for name, value in (("velocity", st["velocity"]), ("position", st["position"])):
        if not math.isfinite(value):
            issues.append("%s not finite" % name)
    for wheel in config["wheels"]:
        if st["friction_state"].get(wheel) not in ("static", "transition", "kinetic"):
            issues.append("bad friction state on %s" % wheel)
        if not math.isfinite(st["wheel_speed"].get(wheel, 0.0)):
            issues.append("wheel speed not finite on %s" % wheel)
    split = st["torque_split"]
    if abs(split.get("front", 0.5) + split.get("rear", 0.5) - 1.0) > 1e-9:
        issues.append("torque split not conserved")
    return {"ok": not issues, "issues": issues}


def metrics(state=None, cfg=None):
    """从状态汇总运行指标。"""
    config = _as_config(cfg)
    st = _as_state(state, config)
    slips = [abs(st["slip_history"].get(w, 0.0)) for w in config["wheels"]]
    kinetic = sum(1 for w in config["wheels"] if st["friction_state"].get(w) == "kinetic")
    return {
        "speed": st["velocity"],
        "position": st["position"],
        "max_slip": max(slips) if slips else 0.0,
        "mean_slip": sum(slips) / len(slips) if slips else 0.0,
        "kinetic_wheels": kinetic,
        "surface": st["surface"],
    }


def replay_trace(state=None, inputs=None, cfg=None):
    """按记录的输入序列回放，返回轨迹；相同输入必得相同轨迹。"""
    config = _as_config(cfg)
    st = _as_state(state, config)
    if not isinstance(inputs, (list, tuple)) or not inputs:
        inputs = [(FIXED_DT, 0.5, 0.0)] * 5
    trace = []
    for step in inputs:
        dt, throttle, brake = (step + (0.0, 0.0, 0.0))[:3] if isinstance(step, (list, tuple)) else (FIXED_DT, 0.0, 0.0)
        st = shared_substep(st, dt, throttle=throttle, brake=brake, cfg=config)
        trace.append({"position": st["position"], "velocity": st["velocity"]})
    return {"trace": trace, "final": metrics(st, config)}
