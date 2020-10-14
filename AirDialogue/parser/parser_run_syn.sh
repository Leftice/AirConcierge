set -e # stop script when there is an error
python3 parser_SQL.py --train --syn           # train.data     -> train.sql.txt                    : Use train.data to generate SQL queue                       
python3 process_sql.py --train --syn          # train.sql.txt  -> train_tok.jsonl                  : Generate jsonl like with {queue, cond}, similar to SQLNet 
python3 parser_syn_state_tracking.py --train  # train.data     -> State_Tracking.txt               : Use train.data to generate each token state                
python3 process_filtered_kb.py --train --syn  # train.data train.kb train.sql.txt ->  filtered_kb  : Use queue and kb to get filtered result                    

python3 parser_SQL.py --dev --syn
python3 process_sql.py --dev --syn
python3 parser_syn_state_tracking.py --dev
python3 process_filtered_kb.py --dev --syn

python3 parser_SQL.py --selfplay_eval --syn 
python3 process_sql.py --selfplay_eval --syn