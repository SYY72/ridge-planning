# 露地蔬菜垄线规划模型

通用露地蔬菜垄线规划与评价软件，支持四边形地块输入、地块规整化、南北/东西方向方案比较、垄线生成、作业路径组织、障碍物避让、指标评价和结果导出。

## 本次修复

应用内置案例后会立即重新创建规划引擎并刷新模型结果，避免界面继续显示上一个案例的旧方案。

## 运行

```powershell
python -m pip install numpy matplotlib
python ridge_planning.py
```

列出内置案例：

```powershell
python ridge_planning.py --list-cases
```

## 构建 Windows 应用

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name "露地蔬菜垄线规划模型" ridge_planning.py
```

构建后的可执行文件位于 `dist/`。

## 内置案例

- `standard_rectangle`
- `trapezoid_demo`
- `wide_quad_demo`
- `bayannur_demo`
