# Commands

```bash
# Extract training clips from rally CSV
python scripts/extract_training_clips.py \
  --video path/to/game.mp4 \
  --csv path/to/rallies.csv \
  --output data/train

python scripts/extract_training_clips.py \
  --video /Users/arashoutadi/volleyball-videos/balltime_versions/Sun_June_22,_2025_Vollocity_vs_Setters_of_Catan_Game_2_Set_2.mp4 \
  --csv /Users/arashoutadi/volleyball-videos/balltime_versions/Sun_June_22,_2025_Vollocity_vs_Setters_of_Catan_Game_2_Set_2.csv \
  --output data/train/

# Train
MODEL_PATH=output/best_model DATA_DIR=game_state_incrediballs uv run python train.py

# Extract screenshots from video
python3 scripts/extract_screenshots.py "/Users/arashoutadi/volleyball-videos/balltime_BobaLeagueFall2025Tournament_Block_Pink_vs_ThirsTea_Ducks.mp4" output/action_locations.csv screenshots 15

# Label images with Claude Vision
# EXPORT ANTHROPIC_API_KEY=""
python -m action_detector.label_images './screenshots/0_24*.jpg' -o labels/
python -m action_detector.label_images './screenshots/3_33*.jpg' -o labels/ --skip-empty

# View labels
uv run python -m action_detector.view_labels './screenshots/3_05*.jpg' -l ./labels
```
