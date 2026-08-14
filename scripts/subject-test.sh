for subject in S1 S2 S3 S4; do
  python3 -m scripts.train_frame_baseline \
    --features data/features \
    --held-out-subject "$subject" \
    --num-classes 73 \
    --epochs 20 \
    --batch-size 64 \
    --seed 0 \
    --output outputs/frame_baseline
done