parent_dir=$(dirname "$(pwd)")

echo "Workspace directory: $parent_dir"

cd "$parent_dir"

python dev/train.py --data_dir "$parent_dir/data/retinal-tiff" \
                    --img_dir "$parent_dir/data/retinal-tiff/images" \
                    --train_json "$parent_dir/splits/train_fold_1.json" \
                    --val_json "$parent_dir/splits/val_fold_1.json" \
                    --test_json "$parent_dir/splits/test_fold_1`.json" \
                    --model_name "frcnn" \
                    --save_dir "$parent_dir/checkpoints" \
                    --batch_size 4 \
                    --num_epochs 10 \
                    --learning_rate 0.001 \
                    --num_workers 4 \