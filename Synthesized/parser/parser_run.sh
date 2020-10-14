python3 parser_SQL2.py --train # train.data train.kb train.sql.txt
python3 process_sql.py --train # train.sql.txt train_tok.jsonl 
# python3 parser_State_Tracking2.py --train # train.data train.kb State_Tracking.txt
python parser_syn_state_tracking.py --train
python3 process_filtered_kb.py --train

python3 parser_SQL2.py --dev # train.data train.kb train.sql.txt
python3 process_sql.py --dev # train.sql.txt train_tok.jsonl 
# python3 parser_State_Tracking2.py --dev # train.data train.kb State_Tracking.txt
python parser_syn_state_tracking.py --dev
python3 process_filtered_kb.py --dev

python3 parser_SQL2.py --self_play_eval # train.data train.kb train.sql.txt
python3 process_sql.py --self_play_eval # train.sql.txt train_tok.jsonl 