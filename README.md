# 测杆球头端点标定

从三维点云中生成测杆两端的原始扫描点候选，并通过交互窗口确认球头接触点。程序直接保存原始扫描点坐标，不对接触点做投影或坐标修正。

## 环境要求

- Python 3.10 或更高版本
- Windows、Linux 或 macOS（需要可用的 Tk 图形界面）
- NumPy 与 Pillow，具体版本范围见 `pyproject.toml`

## 安装

在项目目录执行：

```powershell
py -m pip install -e .
```

如需运行代码检查工具：

```powershell
py -m pip install -e ".[dev]"
```

## 使用方法

提供输入文件时：

```powershell
probe-calibrate "D:\data\测杆.extract.txt" --output "D:\data\标定结果.json"
```

也可以双击 `运行标定.cmd`。启动脚本会在当前目录及其上一级目录查找以下文件：

- `测杆.extract`
- `测杆.extract.txt`

输入文件使用空白字符分列，每行至少包含 X、Y、Z 三个数值。额外列会被忽略，包含 `NaN` 或无穷值的行会被过滤。

## 交互操作

- 鼠标右键拖动：旋转点云
- 鼠标中键拖动：平移点云
- 鼠标滚轮：缩放
- 鼠标左键：选择距离点击位置最近的端点候选
- “杆身侧视 1”和“杆身侧视 2”：切换观察方向

选择候选点后，程序会显示精确坐标并要求确认。确认结果保存为 UTF-8 JSON。

## 计算方法

1. 过滤无效数据并保留前三列 XYZ 坐标。
2. 从点云中心找到最远原始点。
3. 连续进行两次最远点搜索，得到点云两端候选。
4. 使用 PCA 生成便于观察的初始坐标系。PCA 只影响显示方向，不修改候选坐标。
5. 用户确认球头端点后，直接保存对应的原始扫描点。

端点搜索的时间复杂度为 O(n)，不会构造占用大量内存的两两距离矩阵。

## 输出字段

- `mode`：当前标定模式
- `input_file`：输入文件的绝对路径
- `point_count`：参与计算的有效点数
- `contact_point`：人工确认的原始扫描点 XYZ 坐标
- `contact_point_is_original_scan_point`：固定为 `true`
- `other_endpoint_candidate`：另一端候选点坐标
- `distance_between_endpoint_candidates`：两个候选点的欧氏距离

示例见 `examples/calibration_result.example.json`。

## 测试与代码检查

```powershell
$env:PYTHONPATH = "src"
py -m unittest discover -s tests -v
py -m ruff check src tests
```

测试覆盖输入校验、无效数据过滤、端点搜索、结果构造和 JSON 输出。图形界面需要人工确认交互行为。

## 项目结构

```text
src/probe_calibration/
  core.py       点云读取、端点搜索和结果模型
  gui.py        Tkinter 点云显示和端点选择
  cli.py        命令行参数与执行流程
tests/          核心逻辑自动化测试
examples/       输出格式示例
```

点云文件和实际标定结果可能包含设备数据或本机路径，默认不会提交到 Git。
