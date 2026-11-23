# pairwise-vehicle-friction-substep

## 领域

游戏、图形与多媒体

## 主题

载具摩擦与悬挂子步

## 说明

摩擦状态机、同子步积分与扭矩阻尼。

初始状态下 `gameplay/vehicle_friction_substep.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
