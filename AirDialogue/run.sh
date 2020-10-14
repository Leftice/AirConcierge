set -e # stop script when there is an error
# air
############################################################################################################################
cd parser
bash parser_run.sh
cd ..
# step 1. train a model 
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 320 --mask --sigmkb --shuffle --sql

# step 2. Generate SQL query and check its quality
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 500 --mask --sigmkb --sql --resume --dev --eval

# step 3. inference (without parallel)
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu_prior --mode 'bleu_t1t2'
python3 utils/self_play_simulate_DB.py --air --infer_dev
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu --mode 'bleu_t1t2'
airdialogue score --pred_data results/airdialogue/Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data data/airdialogue/tokenized/infer/dev.infer.tar.data \
                  --task infer \
                  --output results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json
python evaluator/evaluator_main.py --pred_data ./results/airdialogue/Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data ./data/airdialogue/tokenized/infer/dev.infer.tar.data \
                  --task infer \
                  --output ./results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json
python evaluator/old_evaluator/evaluator_main.py --pred_data ./results/airdialogue/Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data ./data/airdialogue/tokenized/infer/dev.infer.tar.data \
                  --task infer \
                  --output ./results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json
cp ./results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json ./results/airdialogue/

# step 4. Self play (without parallel)
CUDA_VISIBLE_DEVICES=2 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval_prior
python3 utils/self_play_simulate_DB.py --air --self_play_dev
CUDA_VISIBLE_DEVICES=2 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval

# syn
############################################################################################################################
cd parser
bash parser_run.sh
cd ..
# step 1. train a model 
CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 320 --mask --sigmkb --shuffle --sql

# step 2. Generate SQL query and check its quality
CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 500 --mask --sigmkb --sql --resume --dev --eval

# step 3. inference (without parallel)
CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu_prior --mode 'bleu_t1t2'
python3 utils/self_play_simulate_DB.py --syn --infer_dev
CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu --mode 'bleu_t1t2'
airdialogue score --pred_data results/synthesized/Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data results/synthesized/Inference_Bleu/t1t2/dev.infer.tar.data \
                  --task infer \
                  --output results/synthesized/Inference_Bleu/t1t2/dev_bleu.json
python evaluator/evaluator_main.py --pred_data ./results/synthesized/Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data ./data/synthesized/tokenized/infer/dev.infer.tar.data \
                  --task infer \
                  --output ./results/synthesized/Inference_Bleu/t1t2/dev_bleu.json
python evaluator/old_evaluator/evaluator_main.py --pred_data ./results/synthesized/Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data ./data/synthesized/tokenized/infer/dev.infer.tar.data \
                  --task infer \
                  --output ./results/synthesized/Inference_Bleu/t1t2/dev_bleu.json
cp ./results/synthesized/Inference_Bleu/t1t2/dev_bleu.json ./results/synthesized/

# step 4. Self play (without parallel)
CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval_prior
python3 utils/self_play_simulate_DB.py --syn --self_play_dev
CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval

# toy
############################################################################################################################
cd parser
bash parser_run.sh
cd ..
# step 1. train a model 
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 320 --mask --sigmkb --shuffle --sql --toy

# step 2. Generate SQL query and check its quality
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 500 --mask --sigmkb --sql --resume --dev --eval --toy
python3 utils/lf_ex_acc.py --dev --air --toy

# step 3. inference (without parallel)
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu_prior --mode 'bleu_t1t2' --toy
python3 utils/self_play_simulate_DB.py --air --infer_dev --toy
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu --mode 'bleu_t1t2' --toy
airdialogue score --pred_data results/airdialogue/Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data results/airdialogue/Inference_Bleu/t1t2/dev.infer.tar.data \
                  --task infer \
                  --output results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json
python evaluator/evaluator_main.py --pred_data ./results/airdialogue/Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data ./data/airdialogue/tokenized/sub_air/toy_dev.infer.tar.data \
                  --task infer \
                  --output ./results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json
python evaluator/old_evaluator/evaluator_main.py --pred_data ./results/airdialogue/Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data ./data/airdialogue/tokenized/sub_air/toy_dev.infer.tar.data \
                  --task infer \
                  --output ./results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json
cp ./results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json ./results/airdialogue/

# step 4. Self play (without parallel)
CUDA_VISIBLE_DEVICES=2 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval_prior --toy
python3 utils/self_play_simulate_DB.py --air --self_play_dev --toy
CUDA_VISIBLE_DEVICES=2 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval --toy
