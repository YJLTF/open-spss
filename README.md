# OpenSPSS — 开源统计分析平台

> 参考 [SPSSPRO](https://www.spsspro.com/) 复现的开源自用数据分析工具:**108 个分析模型**,覆盖描述统计、假设检验、回归建模、降维聚类、机器学习、综合评价、时间序列预测、医学统计、质量管理、问卷研究与统计绘图。纯 Python 本地运行,现代 Web 界面,一键打包分发。

![界面预览](docs/screenshot.png)

## 功能特性

- **108 个分析模型 / 11 大类** —— 与主流在线统计平台的方法体系对齐(见下方完整清单)
- **智能解读** —— 每个模型自动生成结论文本:统计量、p 值、显著性判断、效应量一目了然
- **可视化图表** —— 内置 ECharts:热力图、箱线图、散点图、岭迹图、ROC 曲线、控制图、树状图等,支持导出 PNG
- **数据管理** —— 上传 csv / xlsx / xls / sav(SPSS,自动读取变量/值标签) / tsv / json / parquet;内置 8 套示例数据即点即用,支持删除、同名去重与一键恢复初始
- **变量类型系统** —— 定量 / 定类 / 定级三级类型,分析时自动校验,定类变量自动哑变量化
- **数据工作台** —— 单元格直接编辑、撤销/重做(Ctrl+Z)、增删行列;缺失值填充、异常值缩尾、标准化、哑变量、分箱重编码、计算变量、筛选排序;多数据集纵向追加 / 横向关联合并;变量标签与值标签贯通展示
- **分析即代码** —— 高频分析一键生成等效 Python 代码(scipy / statsmodels / scikit-learn),结果页查看复制,报告随附,研究全流程可复现
- **结果导出** —— 一键导出 Excel(全部表格),单表/单图亦可导出
- **加权个案与暗色主题** —— 数据页指定权重列,描述/频数/卡方自动加权;明暗主题一键切换并记忆,图表与报告同步换肤
- **综合分析报告** —— 会话内每次分析自动汇总成报告页,一键导出独立 HTML / **Word / PDF**;每项分析可「重新运行」回放参数
- **纯本地运行** —— 无需注册、无网络依赖(ECharts 已本地化),数据不出本机
- **分析方案** —— 报告页「💾 存为方案」把本次会话的分析(模型+数据集+参数)存成命名方案,跨重启管理;「▶ 运行全部」一键批量重放,数据集失效可改绑
- **数据集导出 / 重命名** —— 当前数据(含编辑与模型写回的派生列)一键下载 csv / xlsx;数据集可随时重命名
- **数据落盘** —— 上传/示例的数据集自动保存到本地数据目录(exe:%APPDATA%\OpenSPSS\datasets;源码运行:data/datasets,可用 OPENSPSS_DATA_DIR 覆盖),重启自动恢复;分析方案存于数据目录的 plans 子目录;删除数据集同步删文件

## 快速开始

### 方式零:直接下载(免安装)

从 [GitHub Releases](https://github.com/YJLTF/open-spss/releases) 下载 `OpenSPSS.exe`,双击即用,无需 Python 环境。

### 方式一:从源码运行(Python ≥ 3.10)

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. 安装
pip install -e .

# 4. 启动(自动打开浏览器)
open-spss
# 或:python -m open_spss
# 自定义端口:open-spss --port 9000 --host 127.0.0.1 --no-browser
```

访问 http://127.0.0.1:8420 ,点击任意示例数据即可开始分析。

### 方式二:安装分发包(wheel)

```bash
pip install dist/open_spss-1.1.0-py3-none-any.whl
open-spss
```

### 方式三:打包为独立可执行文件(免 Python 环境)

```bash
pip install -e ".[exe]"
pyinstaller openspss.spec
# 产物为 dist/OpenSPSS.exe,双击即用
```

- **双击启动**:约 1 秒内显示品牌过渡图(覆盖解压等待期),服务就绪后自动打开浏览器;
- **端口说明**:默认使用 8420;若端口被其他程序占用会自动向后换空闲端口,若已有 OpenSPSS 在运行则直接打开其页面(可多开不冲突);
- **无命令行窗口**:后台静默运行;异常信息写入 `%TEMP%/OpenSPSS.log`;
- **退出方式**:应用左下角「退出」按钮(停止本地服务),或任务管理器结束 OpenSPSS.exe 进程。

Linux/macOS 用户可用附带的一体化启动脚本:

```bash
bash scripts/start.sh     # 自动创建 venv、安装依赖并启动
```

<!-- MODELS:BEGIN -->
## 模型清单(108 个)

| 大类 | 模型 |
|---|---|
| **通用统计** | 频数分析、描述性统计、分类汇总、正态性检验、方差齐性检验、P-P图 / Q-Q图、相关分析、偏相关分析、RFM 客户价值模型 |
| **降维聚类** | 主成分分析（PCA）、探索性因子分析、K-means 聚类、层次聚类、DBSCAN 密度聚类、判别分析、典型相关分析、对应分析 |
| **综合评价** | 层次分析法（AHP）、熵值法（熵权法）、TOPSIS 优劣解距离法、灰色关联分析、模糊综合评价、秩和比（RSR）综合评价、耦合协调度模型、CRITIC 权重法、数据包络分析（DEA-CCR）、组合权重（AHP+熵权）、变异系数权重法、主成分权重法、因子分析权重法、DEMATEL 决策实验室法、ISM 解释结构模型、德尔菲法（专家权重） |
| **时间预测** | ARIMA 时间序列预测、灰色预测 GM（1,1）、指数平滑预测、移动平均预测、马尔科夫预测、灰色 Verhulst 预测 |
| **医学统计** | ROC 曲线、Kappa 一致性检验、Cox 比例风险回归、Kaplan-Meier 生存曲线、Log-rank 生存曲线比较、诊断试验评价、广义估计方程（GEE） |
| **机器学习** | 决策树、随机森林、支持向量机（SVM）、K近邻（KNN）、朴素贝叶斯、神经网络（MLP）、梯度提升树（GBDT）、异常检测（孤立森林） |
| **统计绘图** | 直方图、箱线图、散点图、折线图、条形图 / 饼图、饼图 / 环形图、雷达图、帕累托图、漏斗图、小提琴图 |
| **质量管理** | X̄-R 控制图、p 控制图（不合格品率）、过程能力分析、X-MR 控制图（单值-移动极差）、验收抽样（OC 曲线） |
| **回归建模** | 多元线性回归、分层（层次）回归、二元 Logistic 回归、多分类 Logistic 回归、有序 Logistic 回归、泊松回归、岭回归、Lasso 回归、曲线估计（曲线回归）、Probit 回归、负二项 / 泊松回归（计数）、分位数回归、稳健回归（RLM）、曲面响应回归（二次RSM）、中介效应分析、调节效应分析、线性混合模型 |
| **问卷研究** | 信度分析、效度分析、项目分析（区分度）、多选题分析、McDonald's ω 信度、IPA 分析（重要性-满意度） |
| **差异检验** | 单样本 T 检验、独立样本 T 检验、配对样本 T 检验、单因素方差分析、多因素方差分析、协方差分析、卡方检验（交叉分析）、配对卡方检验（McNemar）、单样本 Wilcoxon 检验、Mann-Whitney U 检验、Kruskal-Wallis 检验、Friedman 检验、符号检验、事后多重比较、卡方拟合优度检验、重复测量方差分析 |
<!-- MODELS:END -->

## 使用指南

1. **载入数据** —— 首页拖入文件,或点击示例数据卡片;
2. **检查变量类型** —— 数据页点击列头/变量芯片可切换定量 / 定类 / 定级,亦可双击单元格改数、编辑变量/值标签、撤销重做;
3. **选择模型** —— 左侧模型树(支持搜索),按提示把变量加入槽位、调整选项;
4. **运行分析** —— 得到智能解读、结果表格与图表,可切换标签页查看;
5. **导出结果** —— 「导出 Excel」保存全部表格,图表支持 PNG 下载;
6. **综合分析报告** —— 本次会话的每次分析自动汇总到「综合分析报告」页(侧边栏),带目录与图表,一键导出为**独立 HTML 报告**(内嵌图表 PNG,浏览器打印即可转 PDF);
7. **复用研究** —— 报告页「💾 存为方案」保存本次分析链,之后在方案页「▶ 运行全部」对新数据一键重放;数据页可随时重命名、导出数据集。

分析中产生的派生数据(主成分得分、聚类类别、预测值等)会自动写回数据集,可继续用于后续分析。

## 项目结构

```
open-spss/
├── open_spss/
│   ├── core/            # 注册表、数据集管理(含落盘)、结果结构、ECharts 构建器
│   ├── models/          # 12 个模块文件,100 个模型实现
│   ├── web/             # 前端 SPA(原生 JS + ECharts,无构建步骤;core/charts/render/app 四个模块)
│   ├── app.py           # FastAPI 服务
│   ├── processing.py    # 数据处理操作
│   ├── samples.py       # 内置示例数据集
│   └── cli.py           # 命令行入口
├── tests/               # pytest:全模型冒烟测试(108/108)与单元测试
├── scripts/             # start.sh 启动脚本、make_splash/make_logo 品牌资产生成
├── docs/                # 截图等文档资源
└── openspss.spec        # PyInstaller 打包配置(构建前自动重生成过渡图)
```

## 添加新模型

只需三步:在 `open_spss/models/` 写一个 runner 函数 → 用 `ModelSpec` 声明槽位与选项 → `register_model` 注册。前端表单、变量类型校验、结果渲染、Excel 导出全部自动生效:

```python
@register_model  # 或 register_model(ModelSpec(...))
def _run_my_model(ds, sl, op) -> Result: ...

register_model(ModelSpec(
    id="my_model", name="我的模型", category="通用统计",
    desc="一句话说明",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,))],
    options=[Option("method", "方法", "select", default="a",
                    choices=[{"value": "a", "label": "方法A"}])],
    runner=_run_my_model,
))
```

## 测试

```bash
pytest tests/ -q      # 冒烟测试(108 个模型逐个跑通)+ 标点/持久化/边界单元测试
```

## 技术栈

FastAPI · Uvicorn · pandas · NumPy · SciPy · statsmodels · scikit-learn · openpyxl · pyreadstat · Apache ECharts(本地化)· 原生 JavaScript

## License

MIT — 见 [LICENSE](LICENSE)。

> 本项目为学习研究性质的开源复现,与 SPSSPRO 官方无关;方法实现以 scipy / statsmodels / scikit-learn 等开源库为准,结果解读仅供科研参考。
