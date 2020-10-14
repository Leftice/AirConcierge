data_file = '../data/airdialogue/tokenized/train.data'
def tokenize(path):
    sents = []
    sents_len = [] 
    count = 0
    with open(path, 'r') as f:
        for line in f:
            items = line.split("|")
            sent = []

            intent_item = items[0].split()
            intent_goal = intent_item[14].split('_', 1)[1].split('>', 1)[0]
            ground_truth = items[1].split()
            if len(ground_truth) != 4:
                final_state = ground_truth[1].split('_', 1)[1].split('>', 1)[0]
            else:
                final_state = ground_truth[3].split('_', 1)[1].split('>', 1)[0]

            for i in range(4):
                words = []
                for word in items[i].split(" "):
                    if i < 3: # tokenize intent, action, dialogue
                        words.append(word)
                    else: # tokenize boundaries
                        words.append(int(word))
                    if i == 2 and ((intent_goal == 'book' and final_state == 'book') or (intent_goal == 'change' and final_state == 'change')):
                        if word.isdigit() and (int(word) >= 1001 and int(word) <= 1029):
                            word = '<mask_flight>'
                            count += 1
                            print(words[-5:])
                        else:
                            for f in range(1001, 1030):
                                str_f = str(f)
                                if str_f in word:
                                    word = '<mask_flight>'
                                    count += 1
                                    print(words[-5:])
                        if '1000' in word and 'ight' in words[-4:]:
                            word = '<mask_flight>'
                            count += 1
                            print(words[-5:])
                sent.append(words)
            # a, b, c, d = sent[0], sent[1], sent[2], sent[3]
            sents.append(sent)
            sents_len.append(len(sent[2]))
    print('Count : ', 100. * count/284547)
    return sents, sents_len
sents, sents_len = tokenize(data_file)
