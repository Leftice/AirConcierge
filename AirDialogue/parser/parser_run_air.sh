set -e # stop script when there is an error
python3 parser_SQL.py --train --air            # train.data     -> train.sql.txt                    : Use train.data to generate SQL queue                       
python3 process_sql.py --train --air           # train.sql.txt  -> train_tok.jsonl                  : Generate jsonl like with {queue, cond}, similar to SQLNet
python3 parser_State_Tracking.py --train --air # train.data     -> State_Tracking.txt               : Use train.data to generate each token state                
python3 process_filtered_kb.py --train --air   # train.data train.kb train.sql.txt ->  filtered_kb  : Use queue and kb to get filtered result                    

python3 parser_SQL.py --dev --air
python3 process_sql.py --dev --air
python3 parser_State_Tracking.py --dev --air
python3 process_filtered_kb.py --dev --air

python3 parser_SQL.py --selfplay_eval --air 
python3 process_sql.py --selfplay_eval --air