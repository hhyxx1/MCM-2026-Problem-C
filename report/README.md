# Phase 6: Report Deliverables

## 文件清单

### 必交文档 (Required Deliverables)

| 文件 | 说明 | 页数 |
|------|------|------|
| `summary_sheet.pdf` | 单页摘要 (Summary Sheet) | 1 |
| `main_paper.pdf` | 主论文 (含目录、正文、参考文献) | ~12 |
| `producer_memo.pdf` | 制片人备忘录 (1-2页) | 2 |
| `ai_use_report.pdf` | AI使用声明 | 1 |

### LaTeX源文件

- `summary_sheet.tex` - 摘要页源文件
- `main_paper.tex` - 主论文源文件  
- `producer_memo.tex` - 备忘录源文件
- `ai_use_report.tex` - AI声明源文件

### 图片 (来自 cleaned_outputs/)

- `bayesian_inference_summary.png` - 贝叶斯推断结果
- `pareto_optimization.png` - Pareto前沿图
- `global_scan_heatmap.png` - 全局扫描热力图
- `simulator_comparison.png` - 模拟对比图
- `final_recommendation.png` - 最终推荐图
- ...等11张图

## 编译命令

```bash
cd report
latexmk -pdf main_paper.tex
latexmk -pdf summary_sheet.tex
latexmk -pdf producer_memo.tex
latexmk -pdf ai_use_report.tex
```

## 提交前检查清单

- [ ] Summary Sheet 不超过1页
- [ ] 主论文不超过25页 (含图表)
- [ ] Producer Memo 1-2页
- [ ] AI Use Report 单独1页
- [ ] 所有量化陈述有数据支撑
- [ ] 团队编号已填写 (替换 XXXXXX)
- [ ] 参考文献格式正确

## 关键统计数据 (用于论文引用)

```
预测准确率: 95.6%
后验一致性 P̄: 0.649
JFI (Rank): 0.727
FFI (Pct): 0.788
Top-3 重叠: 2.76/3
冠军改变: 3/34 季
极端事件: 2 次
Pro Dancer 方差解释: 37.9%
Celebrity 方差解释: 74.0%
```
