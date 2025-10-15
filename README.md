# cpdata_analyzer
用于分析cp测试结果

## 1. 文件扫描与解析  
- 自动扫描指定/当前文件夹内所有 `*CP*.csv` 文件  
- 正则提取 lot 批号 & 晶圆编号（支持 `_CP` / `_CP1` / `_CP2` …）  
- 跳过无法解析的文件并给出提示  

## 2. 数据合并  
- 同一晶圆编号的多份 CSV 纵向拼接为单一大表  
- 自动转换数据类型（数值列 → float，无效值 → NaN）  
- 保留 Limit/Unit 行，供后续规格判定与图表标注  

## 3. 逐晶圆分析（每晶圆独立文件夹）  
| 输出文件名 | 内容 |
|------------|------|
| `<wafer>_yield.csv` | 每颗芯片 PASSFG 及良率统计 |
| `<wafer>_stats.csv` | 各参数 Mean/Std/Min/Max/FailCount/Cpk |
| `<wafer>_histograms.pdf` | 3×3 每页直方图，自动标规格线 |

## 4. Lot 级汇总（结果根目录）  
| 输出文件名 | 内容 |
|------------|------|
| `<lot>_lot_stats.csv` | Lot 内所有晶圆参数的均值、波动、Cpk 汇总 |
| `<lot>_wafer_cpk.csv` | 各晶圆 Cpk 对比矩阵 |
| `<lot>_lot_histograms.pdf` | Lot 合并数据 3×3 直方图，统一看分布 |

## 5. 目录结构  
所有结果**仅一级目录**：
```
CP_Results/
├─ <wafer>/               ← 晶圆级结果
├─ <lot>_lot_stats.csv
├─ <lot>_wafer_cpk.csv
└─ <lot>_lot_histograms.pdf
```

## 6. 运行方式  
```bash
python wafer_cp_full.py  [csv_folder]
```
无参数时默认扫描当前目录。

## 7. 依赖列表  
```
pandas >= 1.0
matplotlib >= 3.0
numpy >= 1.18
```
一键安装：
```bash
pip install pandas matplotlib numpy
```
