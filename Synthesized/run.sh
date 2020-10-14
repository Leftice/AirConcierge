
############################################################################################################################
cd parser
bash parser_run.sh
cd ..
# step 1. train a model ex : new gp | supervised_trainer.py 
UDA_VISIBLE_DEVICES=0 python air.py --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --shuffle --init mos --sql

# step 2. Generate SQL query | supervised_trainer.py 
CUDA_VISIBLE_DEVICES=0 python air.py --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --decay_steps 20000 --N_h 100 --max_len 500 --mask --sigmkb --init mos --sql --resume --dev --eval

# step 3. inference
CUDA_VISIBLE_DEVICES=0 python air.py --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --init mos --sql --resume --dev --infer_bleu_prior --mode 'bleu_t1t2'

python3 self_play_simulate_DB.py --infer_dev
CUDA_VISIBLE_DEVICES=0 python air.py --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --init mos --sql --resume --dev --infer_bleu --mode 'bleu_t1t2'
airdialogue score --pred_data Inference_Bleu/t1t2/dev_inference_out.txt \
                  --true_data Inference_Bleu/t1t2/dev.infer.tar.data \
                  --task infer \
                  --output Inference_Bleu/t1t2/dev_bleu.json
# step 4. Self play
CUDA_VISIBLE_DEVICES=2 python air.py --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --init mos --sql --resume --dev --self_play_eval_prior
python3 self_play_simulate_DB.py --self_play_dev
CUDA_VISIBLE_DEVICES=2 python air.py --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --init mos --sql --resume --dev --self_play_eval

############################################################################################################################
cd parser
bash parser_run.sh
cd ..
# step 1. train a model ex : new gp | supervised_trainer.py 
CUDA_VISIBLE_DEVICES=1 python air.py --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --shuffle --init mos --sql

# step 2. Generate SQL query | supervised_trainer.py 
CUDA_VISIBLE_DEVICES=1 python air.py --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --decay_steps 20000 --N_h 100 --max_len 500 --mask --sigmkb --init mos --sql --resume --dev --eval

# step 3. inference
CUDA_VISIBLE_DEVICES=1 python air.py --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --init mos --sql --resume --dev --infer_bleu_prior --mode 'bleu_t1t2'
python3 self_play_simulate_DB.py --infer_dev
CUDA_VISIBLE_DEVICES=1 python air.py --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --init mos --sql --resume --dev --infer_bleu --mode 'bleu_t1t2'

# step 4. Self play
CUDA_VISIBLE_DEVICES=1 python air.py --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --init mos --sql --resume --dev --self_play_eval_prior
python3 self_play_simulate_DB.py --self_play_dev
CUDA_VISIBLE_DEVICES=1 python air.py --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --decay_steps 20000 --N_h 100 --max_len 320 --mask --sigmkb --init mos --sql --resume --dev --self_play_eval
