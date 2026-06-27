# Top-K Analysis for Lip Reading Models

## 📖 Overview

This directory contains tools for performing **Top-K analysis** on trained lip reading models. Top-K analysis evaluates how often the correct prediction appears within the top K candidates, providing insights into:

- **Model confidence** - How certain is the model about its predictions?
- **Potential improvements** - Could ensemble or reranking methods help?
- **Error analysis** - What are the common confusions?

---

## 🚀 Quick Start

### Test on Best Models (Recommended First Step)

```bash
cd /home/thesis/Thesis/AVEC
bash scripts/run_topk_quick_test.sh
```

This runs Top-K analysis on:
- **LRW-AR T3_lr_1_0** (80.14% accuracy)
- **LipBengal T3_lr_1_0** (35.77% accuracy)

Results will be in: `topk_analysis_results/`

### Run on All Completed Experiments

```bash
bash scripts/run_topk_analysis_all.sh
```

This analyzes all 16 completed experiments across both datasets.
⏱️ Estimated time: 2-4 hours (depending on dataset sizes)

### Run on a Specific Experiment

```bash
python scripts/topk_analysis.py \
    -c configs/LRW-AR/AV/ablations/T3_lr_1_0.py \
    --load_last \
    --k_values 1 3 5 10 20 \
    --output_dir topk_analysis_results/custom_run
```

---

## 📊 What You'll Get

For each experiment, the analysis generates:

### 1. **JSON Results** (`topk_results_*.json`)
```json
{
  "k_values": [1, 3, 5, 10, 20],
  "accuracies": {
    "top1": 80.14,
    "top3": 92.45,
    "top5": 95.23,
    "top10": 97.88,
    "top20": 99.12
  },
  "improvements": {
    "top3_improvement": 12.31,
    "top5_improvement": 15.09,
    ...
  },
  "example_predictions": [...]
}
```

### 2. **Visualization** (`topk_plot_*.png`)
- High-resolution plot showing Top-K accuracy curve
- Clear visualization of improvement over Top-1

### 3. **Summary Report** (`TOPK_SUMMARY_REPORT.txt`)
- Comparison across all experiments
- Detailed breakdown by dataset

### 4. **Log Files** (`*_log.txt`)
- Detailed execution logs
- Example predictions with confidence scores

---

## 📈 Interpreting Results

### Top-K Accuracy Metrics

- **Top-1 Accuracy**: Traditional accuracy (best prediction only)
- **Top-3 Accuracy**: Correct label is in top 3 predictions
- **Top-5 Accuracy**: Correct label is in top 5 predictions
- **Top-10 Accuracy**: Correct label is in top 10 predictions

### What Good Results Look Like

#### High-Resource Dataset (LRW-AR)
- **Top-1**: ~80%
- **Top-3**: ~92-95% (+12-15%)
- **Top-5**: ~95-97% (+15-17%)
- **Top-10**: ~98-99% (+18-19%)

#### Low-Resource Dataset (LipBengal)
- **Top-1**: ~35%
- **Top-3**: ~55-60% (+20-25%)
- **Top-5**: ~65-70% (+30-35%)
- **Top-10**: ~75-80% (+40-45%)

### Key Insights

1. **Large Top-1 to Top-3 jump** → Model often has correct answer but lacks confidence
2. **Flat curve after Top-5** → Most confusions are within top 5
3. **Low Top-10** → Fundamental recognition issues, not just ranking

---

## 🛠️ Advanced Usage

### Custom K Values

```bash
python scripts/topk_analysis.py \
    -c configs/LRW-AR/AV/ablations/T3_lr_1_0.py \
    --load_last \
    --k_values 1 2 3 5 10 15 20 50 100
```

### Use CPU Instead of GPU

```bash
python scripts/topk_analysis.py \
    -c configs/LipBengal/AV/ablations/T3_lr_1_0.py \
    --load_last \
    --cpu
```

### Specify Custom Checkpoint

```bash
python scripts/topk_analysis.py \
    -c configs/LRW-AR/AV/ablations/T3_lr_1_0.py \
    -i checkpoints_epoch_95_step_130000.ckpt
```

---

## 📁 Output Structure

```
topk_analysis_results/
├── LRW-AR_T3_lr_1_0/
│   ├── topk_results_T3_lr_1_0.json
│   └── topk_plot_T3_lr_1_0.png
├── LipBengal_T3_lr_1_0/
│   ├── topk_results_T3_lr_1_0.json
│   └── topk_plot_T3_lr_1_0.png
├── TOPK_SUMMARY_REPORT.txt
├── topk_comparison_all.png
└── *_log.txt
```

---

## 💡 Use Cases for Top-K Analysis

### 1. **Model Selection**
Compare different models not just on Top-1, but also Top-K performance

### 2. **Ensemble Methods**
If Top-3 is significantly better than Top-1, ensemble voting could help

### 3. **Reranking Strategies**
High Top-K suggests a language model or reranker could improve results

### 4. **Error Analysis**
Examine example predictions to understand common confusions

### 5. **User Interface Design**
For interactive systems, showing Top-3 predictions may improve UX

---

## 🎓 For Your Thesis/Paper

### Recommended Figures

1. **Top-K Curves** - Show for best model from each dataset
2. **Comparison Bar Chart** - Top-1 vs Top-5 across all experiments
3. **Improvement Table** - Quantify Top-K improvement percentages

### Key Statistics to Report

- Top-1, Top-3, Top-5 accuracies for best models
- Average improvement from Top-1 to Top-5
- Dataset comparison (how does low-resource benefit more from Top-K?)

### Example Text

> "Our best LRW-AR model achieves 80.14% Top-1 accuracy, which increases 
> to 95.23% at Top-5, indicating that the correct prediction is often 
> ranked highly even when not the top choice. This 15.09% improvement 
> suggests potential for ensemble or reranking methods."

---

## ⚡ Performance Notes

- **Memory**: ~4GB GPU memory per batch
- **Speed**: ~5-10 minutes per experiment (depends on eval set size)
- **Parallelization**: You can run multiple experiments in parallel on different GPUs

### Running in Parallel

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python scripts/topk_analysis.py -c configs/LRW-AR/AV/ablations/T3_lr_1_0.py --load_last &

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python scripts/topk_analysis.py -c configs/LipBengal/AV/ablations/T3_lr_1_0.py --load_last &
```

---

## 🐛 Troubleshooting

### Issue: "No evaluation dataset found"
**Solution**: Check that your config file has `evaluation_dataset` defined

### Issue: "CUDA out of memory"
**Solution**: Use `--cpu` flag or reduce batch size in config

### Issue: "Checkpoint not found"
**Solution**: Verify the experiment has completed (epoch 99 checkpoint exists)

---

## 📚 References

- **Top-K Accuracy**: Standard metric in computer vision and NLP
- **Related Work**: Many papers report Top-5 for ImageNet, Top-10 for speech recognition

---

**Created**: November 23, 2025  
**Author**: Thesis Project - AVEC  
**Purpose**: Comprehensive Top-K analysis for lip reading ablation studies



