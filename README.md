# Commands

```bash
# Train
MODEL_PATH=output/best_model DATA_DIR=game_state_incrediballs uv run python train.py

# Extract screenshots from video
python3 scripts/extract_screenshots.py "/Users/arashoutadi/volleyball-videos/balltime_BobaLeagueFall2025Tournament_Block_Pink_vs_ThirsTea_Ducks.mp4" output/action_locations.csv screenshots 15

# Label images with Claude Vision
# EXPORT ANTHROPIC_API_KEY=""
python -m action_detector.label_images './screenshots/0_24*.jpg' -o labels/

# View labels
uv run python -m action_detector.view_labels './screenshots/3_05*.jpg' -l ./labels
```
