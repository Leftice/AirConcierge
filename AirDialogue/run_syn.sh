set -e # stop script when there is an error

cd parser
bash parser_run_syn.sh
cd ..

if [ "$1" = "toy" ]
then
    echo "Use syn toy"
    # toy syn
    ############################################################################################################################
    # step 1. train a model 
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 320 --mask --sigmkb --shuffle --sql --toy

    # step 2. Generate SQL query and check its quality
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 500 --mask --sigmkb --sql --resume --dev --eval --toy
    python3 utils/lf_ex_acc.py --dev --syn --toy

    # step 3. inference (without parallel)
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu_prior --mode 'bleu_t1t2' --toy
    python3 utils/self_play_simulate_DB.py --syn --infer_dev --toy
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu --mode 'bleu_t1t2' --toy
    # airdialogue score --pred_data results/synthesized/Inference_Bleu/t1t2/dev_inference_out.txt \
    #                   --true_data results/synthesized/Inference_Bleu/t1t2/dev.infer.tar.data \
    #                   --task infer \
    #                   --output results/synthesized/Inference_Bleu/t1t2/dev_bleu.json
    python evaluator/evaluator_main.py --pred_data ./results/synthesized/Inference_Bleu/t1t2/dev_inference_out.txt \
                      --true_data ./data/synthesized/tokenized/infer/dev.infer.tar.data \
                      --task infer \
                      --output ./results/synthesized/Inference_Bleu/t1t2/dev_bleu.json
    python evaluator/old_evaluator/evaluator_main.py --pred_data ./results/synthesized/Inference_Bleu/t1t2/dev_inference_out.txt \
                      --true_data ./data/synthesized/tokenized/infer/dev.infer.tar.data \
                      --task infer \
                      --output ./results/synthesized/Inference_Bleu/t1t2/old_dev_bleu.json
    cp ./results/synthesized/Inference_Bleu/t1t2/dev_bleu.json ./results/synthesized/
    cp ./results/synthesized/Inference_Bleu/t1t2/old_dev_bleu.json ./results/synthesized/

    # step 4. Self play (without parallel)
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval_prior --toy
    python3 utils/self_play_simulate_DB.py --syn --self_play_dev --toy
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval --toy

else
    # syn
    ############################################################################################################################
    # step 1. train a model 
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 320 --mask --sigmkb --shuffle --sql

    # step 2. Generate SQL query and check its quality
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 500 --mask --sigmkb --sql --resume --dev --eval
    python3 utils/lf_ex_acc.py --dev --syn

    # step 3. inference (without parallel)
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu_prior --mode 'bleu_t1t2'
    python3 utils/self_play_simulate_DB.py --syn --infer_dev
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu --mode 'bleu_t1t2'
    # airdialogue score --pred_data results/synthesized/Inference_Bleu/t1t2/dev_inference_out.txt \
    #                   --true_data results/synthesized/Inference_Bleu/t1t2/dev.infer.tar.data \
    #                   --task infer \
    #                   --output results/synthesized/Inference_Bleu/t1t2/dev_bleu.json
    python evaluator/evaluator_main.py --pred_data ./results/synthesized/Inference_Bleu/t1t2/dev_inference_out.txt \
                      --true_data ./data/synthesized/tokenized/infer/dev.infer.tar.data \
                      --task infer \
                      --output ./results/synthesized/Inference_Bleu/t1t2/dev_bleu.json
    python evaluator/old_evaluator/evaluator_main.py --pred_data ./results/synthesized/Inference_Bleu/t1t2/dev_inference_out.txt \
                      --true_data ./data/synthesized/tokenized/infer/dev.infer.tar.data \
                      --task infer \
                      --output ./results/synthesized/Inference_Bleu/t1t2/old_dev_bleu.json
    cp ./results/synthesized/Inference_Bleu/t1t2/dev_bleu.json ./results/synthesized/
    cp ./results/synthesized/Inference_Bleu/t1t2/old_dev_bleu.json ./results/synthesized/

    # step 4. Self play (without parallel)
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval_prior
    python3 utils/self_play_simulate_DB.py --syn --self_play_dev
    CUDA_VISIBLE_DEVICES=1 python air.py --syn --save_dir 'runs/AirConciergeSQL-syn-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-syn-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval

fi
