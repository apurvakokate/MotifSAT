# How to Use the Tuning Framework

## ✅ Fixed: Proper File Paths

The scripts now correctly reference `src/run_gsat.py` and use relative config paths!

---

## Complete Workflow

### 1. Generate Configs (from project root)

```bash
# Make sure you're in the project root directory
cd /Users/apurvakokate/Documents/MotifSAT

# Generate configs
python src/generate_tuning_configs.py \
  --experiment baseline \
  --output_dir configs/tuning \
  --datasets Mutagenicity \
  --models GIN \
  --folds 0 1 \
  --seeds 0 1
```

**This creates:**
```
configs/tuning/baseline/
├── config_0000.yaml          # motif_loss_coef: 0.0
├── config_0001.yaml          # motif_loss_coef: 0.5
├── config_0002.yaml          # motif_loss_coef: 1.0
├── config_0003.yaml          # motif_loss_coef: 2.0
├── manifest.json
├── summary.csv
├── run_Mutagenicity_GIN_fold0.sh  ← Run script
├── run_Mutagenicity_GIN_fold1.sh
└── run_all.sh                      ← Master script
```

---

### 2. Check Generated Scripts

Look at one of the generated scripts:

```bash
cat configs/tuning/baseline/run_Mutagenicity_GIN_fold0.sh
```

**You'll see:**
```bash
#!/bin/bash
# Run Mutagenicity with GIN on fold 0

# Config: config_0000
# Params: {'tuning_id': 'config_0000', 'pred_loss_coef': 1.0, ...}
# Seed 0
python src/run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 0 \
  --cuda 0 \
  --config configs/tuning/baseline/config_0000.yaml

# Seed 1
python src/run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 0 \
  --cuda 0 \
  --config configs/tuning/baseline/config_0000.yaml

# ... more configs and seeds ...
```

**Key points:**
- ✅ Uses `python src/run_gsat.py` (correct path)
- ✅ Config path is relative: `configs/tuning/baseline/config_0000.yaml`
- ✅ All paths work from project root

---

### 3. Run Experiments (from project root)

**Option A: Run all experiments**
```bash
bash configs/tuning/baseline/run_all.sh
```

**Option B: Run specific dataset/model/fold**
```bash
bash configs/tuning/baseline/run_Mutagenicity_GIN_fold0.sh
```

**Option C: Run manually with specific config**
```bash
python src/run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 0 \
  --cuda 0 \
  --config configs/tuning/baseline/config_0001.yaml
```

---

### 4. What Happens When You Run

```bash
$ bash configs/tuning/baseline/run_all.sh

# The script will execute:
python src/run_gsat.py --dataset Mutagenicity --backbone GIN --fold 0 --cuda 0 --config configs/tuning/baseline/config_0000.yaml
python src/run_gsat.py --dataset Mutagenicity --backbone GIN --fold 0 --cuda 0 --config configs/tuning/baseline/config_0001.yaml
python src/run_gsat.py --dataset Mutagenicity --backbone GIN --fold 0 --cuda 0 --config configs/tuning/baseline/config_0002.yaml
python src/run_gsat.py --dataset Mutagenicity --backbone GIN --fold 0 --cuda 0 --config configs/tuning/baseline/config_0003.yaml
# ... for each seed and fold ...

# Outputs will be saved to:
tuning_results/Mutagenicity/model_GIN/...
```

---

### 5. Check Output Structure

```bash
ls -la tuning_results/Mutagenicity/model_GIN/

# You'll see directories like:
# tuning_config_0000/pred1.0_info1.0_motif0.0/...
# tuning_config_0001/pred1.0_info1.0_motif0.5/...
# tuning_config_0002/pred1.0_info1.0_motif1.0/...
# tuning_config_0003/pred1.0_info1.0_motif2.0/...
```

---

### 6. Analyze Results

```bash
python src/analyze_tuning_results.py \
  --results_dir tuning_results \
  --output_dir analysis_results
```

---

## Important: Always Run from Project Root

All commands should be run from the **project root directory** where these exist:
- `src/` directory
- `configs/` directory
- `tuning_results/` directory (created automatically)

```bash
# Good ✅
cd /Users/apurvakokate/Documents/MotifSAT
bash configs/tuning/baseline/run_all.sh

# Bad ❌
cd configs/tuning/baseline
bash run_all.sh  # Won't find src/run_gsat.py!
```

---

## File Path Summary

| File/Directory | Path | Relative To |
|----------------|------|-------------|
| Training script | `src/run_gsat.py` | Project root |
| Config files | `configs/tuning/baseline/config_*.yaml` | Project root |
| Run scripts | `configs/tuning/baseline/run_*.sh` | Project root |
| Results | `tuning_results/...` | Project root |
| Analysis script | `src/analyze_tuning_results.py` | Project root |

---

## Testing the Setup

### Quick Test

```bash
# 1. Go to project root
cd /Users/apurvakokate/Documents/MotifSAT

# 2. Generate test configs
python src/generate_tuning_configs.py \
  --experiment baseline \
  --output_dir configs/tuning \
  --datasets Mutagenicity \
  --models GIN \
  --folds 0 \
  --seeds 0

# 3. Check the generated script
cat configs/tuning/baseline/run_Mutagenicity_GIN_fold0.sh

# 4. Verify paths in the script
# Should see: python src/run_gsat.py ...
# Should see: --config configs/tuning/baseline/config_0000.yaml

# 5. Test run one config (dry run)
head -20 configs/tuning/baseline/run_Mutagenicity_GIN_fold0.sh
```

---

## Portability

With relative paths, you can:

1. **Share configs with others**
   ```bash
   # Zip and share
   zip -r tuning_configs.zip configs/tuning/baseline/
   
   # Others can extract and run
   unzip tuning_configs.zip
   bash configs/tuning/baseline/run_all.sh
   ```

2. **Move project directory**
   ```bash
   # Move entire project
   mv /path/to/MotifSAT /new/path/MotifSAT
   cd /new/path/MotifSAT
   
   # Scripts still work!
   bash configs/tuning/baseline/run_all.sh
   ```

3. **Version control**
   ```bash
   # Commit configs to git
   git add configs/tuning/baseline/
   git commit -m "Add baseline experiment configs"
   
   # Others can clone and run
   git clone <repo>
   cd MotifSAT
   bash configs/tuning/baseline/run_all.sh
   ```

---

## Troubleshooting

### Issue: "No such file or directory: src/run_gsat.py"

**Cause**: Running from wrong directory

**Solution**: Make sure you're in project root
```bash
pwd  # Should show: /Users/.../MotifSAT
ls src/run_gsat.py  # Should exist
```

### Issue: "Config file not found"

**Cause**: Config path is wrong

**Solution**: Verify config files exist
```bash
ls configs/tuning/baseline/*.yaml  # Should show config files
```

### Issue: Scripts not executable

**Solution**: Make them executable
```bash
chmod +x configs/tuning/baseline/*.sh
```

---

## Summary

✅ **Fixed Issues:**
1. Scripts now correctly call `src/run_gsat.py` (not `run_gsat.py`)
2. Config paths are relative to project root (portable)
3. All scripts work when run from project root

✅ **How to Use:**
1. Generate configs: `python src/generate_tuning_configs.py ...`
2. Run from project root: `bash configs/tuning/baseline/run_all.sh`
3. Analyze results: `python src/analyze_tuning_results.py ...`

✅ **Best Practice:**
- Always work from project root directory
- Paths are relative and portable
- Easy to share and version control

---

## Next Steps

1. ✅ Add config loading to `src/run_gsat.py` (see CONFIG_INTEGRATION_COMPLETE.md)
2. ✅ Apply other changes from CHANGES_TO_RUN_GSAT.md
3. ✅ Test with single experiment
4. ✅ Run full experiments
5. ✅ Analyze results

You're all set! The file path issues are fixed. 🎉

