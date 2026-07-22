1. Generate data — all worlds into ONE folder

!python -m action.generate_data --world object    --episodes 2500 --out data/all --workers 8
!python -m action.generate_data --world ball      --episodes 1200 --out data/all --workers 8
!python -m action.generate_data --world leaf      --episodes 1200 --out data/all --workers 8
!python -m action.generate_data --world pendulum2 --episodes 1000 --out data/all --workers 8
!python -m action.generate_data --world pendulum3 --episodes 1000 --out data/all --workers 8
!python -m action.generate_data --world nbody3    --episodes 1000 --out data/all --workers 8
Filenames are tagged per world, so they share one folder safely. Every file is the same (T, 13) universal format.

2. Train ONE general model

python -m action.train_memory --data data/all --out runs/general.pt \
    --epochs 30 --batch 3072 --fut-cap 120 --hist-cap 160 --ckpt-chunk 24 \
    --device cuda --lr 2e-3
(On Kaggle: %env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True first; drop --batch if it OOMs.)

3. Use it

# live window - watch it predict fresh objects it has never seen
python -m action.live --memory runs/general.pt --world object --drops 5 --show

# any world, to video
python -m action.live --memory runs/general.pt --world ball      --drops 5 --out runs/live_ball.mp4
python -m action.live --memory runs/general.pt --world pendulum3 --drops 3 --out runs/live_pend.mp4

# numbers (window + sharpen curve)
python -m action.train_memory --measure runs/general.pt --data data/all