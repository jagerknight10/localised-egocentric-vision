for subject in S1 S2 S3 S4; do
  python3 -m scripts.train_causal_transformer \
    --features data/features \
    --held-out-subject "$subject" \
    --num-classes 73 \
    --context 128 \
    --epochs 20 \
    --seed 0
done