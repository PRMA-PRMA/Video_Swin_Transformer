# Video Swin Transformer Evaluation on Kinetics 600

This repository contains scripts to evaluate the Video Swin Transformer model on the Kinetics 600 dataset. The implementation is a pure PyTorch version without mmcv or other dependencies.

## Requirements

1. Install the required dependencies:

```bash
pip install torch torchvision tqdm numpy opencv-python decord requests
```

2. Optional dependencies:
   - FFmpeg (for video validation)
   - CUDA-compatible GPU (for faster evaluation)

## Setup Instructions

### 1. Prepare your Video Swin Transformer Implementation

Make sure your Video Swin Transformer implementation (`vst.py`) is in your Python path or in the current directory. The script expects the following functions to be available:

- `video_swin_b()`, `video_swin_s()`, `video_swin_t()`, `video_swin_l()`
- `get_device()`

### 2. Prepare the Dataset (S3 Method)

The Kinetics 600 dataset is available on Amazon S3. The `prepare_kinetics_s3.py` script provides easy access to this data:

```bash
# Step 1: Download the official class list and create class mapping
python prepare_kinetics_s3.py --output_dir data/kinetics600 --create_classnames --use_official_indices

# Step 2: Download val set annotations with official class indices
python prepare_kinetics_s3.py --output_dir data/kinetics600 --download_annotations --subset val --use_official_indices

# Step 3: Download a few test video parts (each part is about 1.5GB)
python prepare_kinetics_s3.py --output_dir data/kinetics600 --download_videos --parts 0 1 2 --extract

# Step 4: Filter annotations to match downloaded videos
python prepare_kinetics_s3.py --output_dir data/kinetics600 --filter_annotations --video_dir data/kinetics600/videos/extracted

# Step 5: Update the filtered annotations to use official class indices (if needed)
python prepare_kinetics_s3.py --output_dir data/kinetics600 --update_existing data/kinetics600/annotations/val_filtered.json --use_official_indices
```

#### Space-Saving Options:

1. **Download only annotations**: Just use `--download_annotations` without downloading videos.
2. **Download only a few parts**: Use `--parts 0 1` to download only parts 0 and 1.
3. **Limit extracted videos**: Use `--max_videos_per_part 50` to extract just 50 videos per archive.

### 3. Download Pretrained Weights

Download the pretrained weights for the Video Swin Transformer:

- For Swin-B, use the ImageNet-22K pretrained weights: `swin_base_patch244_window877_kinetics600_22k.pth`

Place the downloaded weights in a directory, e.g., `checkpoints/`.

## Running the Evaluation

Use the `evaluate_kinetics.py` script to evaluate the model on the Kinetics 600 val set:

```bash
python evaluate_kinetics.py \
  --video_root data/kinetics600/videos/extracted \
  --annotation_file data/kinetics600/annotations/val_filtered.json \
  --pretrained_path checkpoints/swin_base_patch244_window877_kinetics600_22k.pth \
  --model_type base \
  --window_size 8,7,7 \
  --num_clips 10 \
  --num_crops 3 \
  --batch_size 4 \
  --num_workers 8 \
  --amp \
  --output_file results.json \
  --classnames_file data/kinetics600/annotations/class_names.json
```

or

```bash
python evaluate_kinetics.py --video_root data/kinetics600/videos/extracted --annotation_file data/kinetics600/annotations/val_filtered.json --pretrained_path PATH/TO/WEIGHTS.pth --model_type base --classnames_file data/kinetics600/annotations/class_names.json
```

### Key Parameters:

- `--video_root`: Path to the directory containing extracted Kinetics 600 videos
- `--annotation_file`: Path to the val set annotation file (filtered to match available videos)
- `--pretrained_path`: Path to pretrained model weights
- `--model_type`: Model size (`tiny`, `small`, `base`, or `large`)
- `--window_size`: Window size in the format "temporal,height,width" (e.g., "8,7,7")
- `--num_clips`: Number of temporal clips to sample from each video
- `--num_crops`: Number of spatial crops (1 or 3)
- `--amp`: Use Automatic Mixed Precision for faster evaluation

## Understanding the Results

The evaluation script will output results in a JSON file with the following structure:

```json
{
  "config": {
    // Command line arguments used
  },
  "metrics": {
    "top1_accuracy": 0.75,
    "top5_accuracy": 0.92,
    "total_videos": 1000,
    "eval_time": 3600
  },
  "predictions": {
    // Individual video predictions
  },
  "class_names": {
    // Mapping from class indices to names
  }
}
```

The top-1 and top-5 accuracy metrics can be compared with the original implementation to validate your PyTorch version.

## Troubleshooting

### Common Issues:

1. **S3 Download Errors**: Some parts might be unavailable. Try different part numbers.
2. **CUDA Out of Memory**: Reduce the batch size or number of clips/crops.
3. **Slow Evaluation**: Enable AMP (`--amp`) for faster evaluation.
4. **Incorrect Results**: Verify window size matches pretrained weights.

### Testing with Limited Data

If you want to verify your implementation works correctly without downloading a lot of data:

```bash
# Create dummy annotations (for pipeline testing)
python prepare_kinetics_s3.py --output_dir data/kinetics600 --create_dummy --num_dummy_videos 50

# Download just 1-2 parts and extract a few videos
python prepare_kinetics_s3.py --output_dir data/kinetics600 --download_videos --parts 0 --extract --max_videos_per_part 20

# To download all parts (omit the --parts parameter)
# This will download all 60 parts (around 90GB total)
python prepare_kinetics_s3.py --output_dir data/kinetics600 --download_videos --extract --max_videos_per_part 50
```

## Citations

If you use this code for research, please cite the original Video Swin Transformer paper:

```
@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}
```