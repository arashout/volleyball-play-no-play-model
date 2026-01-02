# Commands

## Extraction

### Balltime Scripts

Requires Chrome with remote debugging:
```bash
# Quit Chrome, then start with:
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
# Log into balltime.com in that browser
# Go to chrome://settings/downloads and:
#   - Turn OFF "Ask where to save each file before downloading"
#   - Set location to /Users/arashoutadi/volleyball-videos/balltime_versions/
```

#### List available videos
```bash
python scripts/balltime_crawler.py --folder-id <folder-id>
```

#### Process all available videos (download + extract rallies)
```bash
python scripts/balltime_crawler.py --folder-id <folder-id> --process
```

### Extract Training Clips
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

# Extract no-play clips from timestamp ranges
python scripts/extract_noplay_clips.py \
  --video path/to/game.mp4 \
  --timestamps "0:00-1:30,45:00-46:00" \
  --output data/train/no-play \
  --segments 3

# Extract screenshots from video
python3 scripts/extract_screenshots.py "/Users/arashoutadi/volleyball-videos/balltime_BobaLeagueFall2025Tournament_Block_Pink_vs_ThirsTea_Ducks.mp4" output/action_locations.csv screenshots 15

## Labelling
# Label images with Claude Vision
# EXPORT ANTHROPIC_API_KEY=""
python -m action_detector.label_images './screenshots/0_24*.jpg' -o labels/
python -m action_detector.label_images './screenshots/3_33*.jpg' -o labels/ --skip-empty

# View labels
uv run python -m action_detector.view_labels './screenshots/3_05*.jpg' -l ./labels
```

## Augmentation
`uv run python augmentations.py --video data/train/play/play_0002.mp4 --output preview.mp4`

## Train
MODEL_PATH=output/best_model DATA_DIR=game_state_incrediballs uv run python train.py


# Pipeline
- Put video into ball time
- Download video from balltime
- Extract clips using CSV of rallies
- Use data augment to generate more clips
- Train it on all data
- Infer using ONNX

