import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true', help='Use dev')
parser.add_argument('--train', action='store_true', help='Use train')
parser.add_argument('--syn', action='store_true', help='Use syn')
parser.add_argument('--air', action='store_true', help='Use air')
args = parser.parse_args()

if args.syn:
    data_path = '.data//synthesized/'
elif args.air:
    data_path = '.data/airdialogue/'
else:
    print('Pleae use --syn or --air !')
    raise

if args.dev:
    kb_file = data_path + 'tokenized/dev/dev.eval.kb'
elif args.train:
    kb_file = data_path + 'tokenized/train/train.kb'
else:
    print('Please use --dev or --train !')
    raise


def tokenize_kb(path):
    # <res_no_res> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_10> <tn2_21> <cl_business> <pr_800> <cn_1> <al_AA> <fl_1000> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_21> <tn2_0> <cl_economy> <pr_200> <cn_0> <al_UA> <fl_1001> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_14> <tn1_21> <tn2_6> <cl_economy> <pr_100> <cn_1> <al_Delta> <fl_1002> <a1_MSP> <a2_IAD> <m1_Sept> <m2_Sept> <d1_10> <d2_14> <tn1_21> <tn2_2> <cl_economy> <pr_100> <cn_1> <al_UA> <fl_1003> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_13> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_Southwest> <fl_1004> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_1> <tn2_15> <cl_economy> <pr_100> <cn_0> <al_Frontier> <fl_1005> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_12> <tn1_8> <tn2_21> <cl_economy> <pr_200> <cn_1> <al_Delta> <fl_1006> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_6> <tn2_5> <cl_economy> <pr_200> <cn_1> <al_AA> <fl_1007> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_10> <d2_14> <tn1_23> <tn2_12> <cl_economy> <pr_100> <cn_1> <al_Southwest> <fl_1008> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_21> <tn2_14> <cl_economy> <pr_200> <cn_1> <al_UA> <fl_1009> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_14> <tn2_12> <cl_business> <pr_500> <cn_1> <al_Southwest> <fl_1010> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_6> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_Spirit> <fl_1011> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_13> <d2_12> <tn1_0> <tn2_21> <cl_economy> <pr_200> <cn_0> <al_UA> <fl_1012> <a1_ATL> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_7> <tn2_5> <cl_economy> <pr_200> <cn_1> <al_JetBlue> <fl_1013> <a1_ATL> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_7> <tn2_0> <cl_economy> <pr_200> <cn_1> <al_AA> <fl_1014> <a1_MSP> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_6> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_UA> <fl_1015> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_10> <d2_13> <tn1_23> <tn2_18> <cl_economy> <pr_200> <cn_1> <al_Hawaiian> <fl_1016> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_3> <tn2_17> <cl_economy> <pr_200> <cn_1> <al_Spirit> <fl_1017> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_10> <tn2_8> <cl_economy> <pr_200> <cn_1> <al_JetBlue> <fl_1018> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_17> <tn2_14> <cl_economy> <pr_100> <cn_1> <al_Southwest> <fl_1019> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_4> <tn2_20> <cl_economy> <pr_100> <cn_1> <al_Delta> <fl_1020> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_5> <tn2_15> <cl_economy> <pr_200> <cn_1> <al_Southwest> <fl_1021> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_12> <d2_12> <tn1_12> <tn2_5> <cl_economy> <pr_100> <cn_1> <al_UA> <fl_1022> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_14> <tn2_16> <cl_economy> <pr_100> <cn_1> <al_Southwest> <fl_1023> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_4> <tn2_7> <cl_economy> <pr_100> <cn_1> <al_Spirit> <fl_1024> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_11> <tn2_16> <cl_economy> <pr_200> <cn_1> <al_Frontier> <fl_1025> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_12> <d2_14> <tn1_8> <tn2_1> <cl_economy> <pr_100> <cn_1> <al_Hawaiian> <fl_1026> <a1_MSP> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_2> <tn2_5> <cl_economy> <pr_200> <cn_1> <al_UA> <fl_1027> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_17> <tn2_23> <cl_economy> <pr_100> <cn_1> <al_UA> <fl_1028> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_2> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_Frontier> <fl_1029>
    kb_sents = []
    state_combination = [[], [], [], [], [], [], [], []]
    with open(path, 'r') as f:
        for line in f:
            words = []
            for word in line.split(" "):
                if ('<a1' in word or '<a2' in word) and word not in state_combination[0]:
                    state_combination[0].append(word)
                elif('<m1' in word or '<m2' in word) and word not in state_combination[1]:
                    state_combination[1].append(word)
                elif('<d1' in word or '<d2' in word) and word not in state_combination[2]:
                    state_combination[2].append(word)
                elif('<tn1' in word or '<tn2' in word) and word not in state_combination[3]:
                    state_combination[3].append(word)
                elif('<cl' in word ) and word not in state_combination[4]:
                    state_combination[4].append(word)
                elif('<pr' in word ) and word not in state_combination[5]:
                    state_combination[5].append(word)
                elif('<cn' in word ) and word not in state_combination[6]:
                    state_combination[6].append(word)
                elif('<al' in word) and word not in state_combination[7]:
                    state_combination[7].append(word)
    print('state_combination : ', state_combination)
    return state_combination
dic = tokenize_kb(kb_file)

 # [['<a1_IAD>', '<a2_ATL>', '<a1_MSP>', '<a2_IAD>', '<a2_MSP>', '<a1_ATL>', '<a1_HOU>', '<a2_HOU>', '<a2_LAS>', '<a1_LAS>', '<a2_ORD>', '<a1_DEN>', '<a2_DEN>', '<a1_ORD>', '<a1_OAK>', '<a2_JFK>', '<a1_PHL>', '<a2_PHL>', '<a1_JFK>', '<a2_OAK>', '<a1_PHX>', '<a2_AUS>', '<a1_AUS>', '<a2_PHX>', '<a1_BOS>', '<a2_DTW>', '<a1_SEA>', '<a2_BOS>', '<a1_DTW>', '<a2_SEA>', '<a1_IAH>', '<a2_IAH>', '<a1_LGA>', '<a2_LGA>', '<a2_CLT>', '<a1_CLT>', '<a1_SFO>', '<a2_SFO>', '<a2_DCA>', '<a1_DCA>', '<a2_DFW>', '<a1_DFW>', '<a2_MCO>', '<a1_MCO>', '<a2_EWR>', '<a1_EWR>', '<a2_LAX>', '<a1_LAX>'], \
 # ['<m1_Sept>', '<m2_Sept>', '<m1_Apr>', '<m2_Apr>', '<m1_Feb>', '<m2_Feb>', '<m1_Jan>', '<m1_Aug>', '<m2_Aug>', '<m1_June>', '<m2_June>', '<m2_Jan>', '<m1_Nov>', '<m2_Nov>', '<m1_Oct>', '<m2_Oct>', '<m1_May>', '<m2_May>', '<m1_July>', '<m2_July>', '<m2_Mar>', '<m1_Mar>', '<m1_Dec>', '<m2_Dec>'], \
 # ['<d1_11>', '<d2_13>', '<d2_14>', '<d1_12>', '<d1_10>', '<d2_12>', '<d1_13>', '<d1_3>', '<d2_6>', '<d2_5>', '<d1_4>', '<d2_4>', '<d1_5>', '<d1_2>', '<d1_1>', '<d2_3>', '<d1_31>', '<d1_15>', '<d2_16>', '<d2_17>', '<d1_14>', '<d2_18>', '<d1_21>', '<d2_23>', '<d1_22>', '<d2_24>', '<d1_23>', '<d2_25>', '<d2_22>', '<d1_16>', '<d1_17>', '<d2_20>', '<d2_19>', '<d1_18>', '<d1_25>', '<d2_26>', '<d1_24>', '<d2_28>', '<d2_27>', '<d1_26>', '<d1_20>', '<d2_21>', '<d2_7>', '<d1_6>', '<d1_7>', '<d2_8>', '<d1_19>', '<d1_9>', '<d2_10>', '<d2_11>', '<d1_8>', '<d1_30>', '<d2_2>', '<d2_1>', '<d2_31>', '<d1_29>', '<d2_9>', '<d1_27>', '<d2_29>', '<d2_30>', '<d1_28>', '<d2_15>'], \
 # ['<tn1_10>', '<tn2_21>', '<tn1_21>', '<tn2_0>', '<tn2_6>', '<tn2_2>', '<tn1_13>', '<tn2_20>', '<tn1_1>', '<tn2_15>', '<tn1_8>', '<tn1_6>', '<tn2_5>', '<tn1_23>', '<tn2_12>', '<tn2_14>', '<tn1_14>', '<tn1_0>', '<tn1_7>', '<tn2_18>', '<tn1_3>', '<tn2_17>', '<tn2_8>', '<tn1_17>', '<tn1_4>', '<tn1_5>', '<tn1_12>', '<tn2_16>', '<tn2_7>', '<tn1_11>', '<tn2_1>', '<tn1_2>', '<tn2_23>', '<tn1_9>', '<tn2_11>', '<tn2_4>', '<tn1_20>', '<tn1_16>', '<tn2_22>', '<tn2_13>', '<tn2_3>', '<tn2_10>', '<tn1_18>', '<tn2_9>', '<tn1_19>', '<tn2_19>', '<tn1_22>', '<tn1_15>'], \
 # ['<cl_business>', '<cl_economy>'], \
 # ['<pr_800>', '<pr_200>', '<pr_100>', '<pr_500>', '<pr_600>', '<pr_400>', '<pr_300>', '<pr_1100>', '<pr_900>', '<pr_700>', '<pr_1000>', '<pr_1300>', '<pr_1200>', '<pr_1600>', '<pr_1400>', '<pr_1500>', '<pr_1700>', '<pr_1800>', '<pr_1900>', '<pr_2000>', '<pr_2100>', '<pr_2200>'], \
 # ['<cn_1>', '<cn_0>', '<cn_2>'], \
 # ['<al_AA>', '<al_UA>', '<al_Delta>', '<al_Southwest>', '<al_Frontier>', '<al_Spirit>', '<al_JetBlue>', '<al_Hawaiian>']]