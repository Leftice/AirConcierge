python3 parser_SQL.py --train --air            # train.data     -> train.sql.txt                    : Use train.data to generate SQL queue                       | ex : SELECT * FROM Airdialogue_Table WHERE departure_airport = '19' AND return_airport = '16' AND departure_month = '2' AND return_month = '2' AND departure_day = 15 AND return_day = 17 AND departure_time_num = '2' AND price <= 1 AND num_connections <= 1
python3 process_sql.py --train --air           # train.sql.txt  -> train_tok.jsonl                  : Generate jsonl like with {queue, cond}, similar to SQLNet  | ex : {"query": "SELECT * FROM Airdialogue_Table WHERE departure_airport = 19 AND return_airport = 16 AND departure_month = 2 AND return_month = 2 AND departure_day = 15 AND return_day = 17 AND departure_time_num = 2 AND price <= 1 AND num_connections <= 1", "sql": {"conds": [[0, 0, 19], [1, 0, 16], [2, 0, 2], [3, 0, 2], [4, 0, 15], [5, 0, 17], [6, 0, 2], [9, 1, 1], [10, 1, 1]]}, "table_id": "0-0000"}
python3 parser_State_Tracking.py --train --air # train.data     -> State_Tracking.txt               : Use train.data to generate each token state                | ex : '32'  | '3', '15', '32' | N_SQL 
python3 process_filtered_kb.py --train --air   # train.data train.kb train.sql.txt ->  filtered_kb  : Use queue and kb to get filtered result                    | ex : 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 | 1 | 4 

python3 parser_SQL.py --dev --air
python3 process_sql.py --dev --air
python3 parser_State_Tracking.py --dev --air
python3 process_filtered_kb.py --dev --air

python3 parser_SQL.py --selfplay_eval --air 
python3 process_sql.py --selfplay_eval --air