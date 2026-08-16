for features in data/features_paired_full data/features_restricted; do
  name=$(basename "$features")

  for subject in S1 S2 S3 S4; do
    python -m scripts.train_frame_baseline \
      --features "$features" \
      --held-out-subject "$subject" \
      --num-classes 73 \
      --epochs 20 \
      --seed 0 \
      --output "outputs/$name"
  done
done
