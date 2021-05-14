# -*- coding:utf-8 -*-

# Deep Streaming Label Learning
# Pepijn Sibbes adapted

from helpers import split_label, split_label_pandas
from sklearn.model_selection import train_test_split
import arff
import scipy.io
import numpy as np
import pandas as pd
import re

def load_dataset(dataset, split, hyper_params):
    if dataset == "yeast":
        data_dir = 'datasets/'
        train_data = arff.load(open(data_dir + 'yeast-train.arff', 'rt'))
        train_data = np.array(train_data['data']).astype(np.float)

        test_data = arff.load(open(data_dir + 'yeast-test.arff', 'rt'))
        test_data = np.array(test_data['data']).astype(np.float)

        train_X = train_data[:, :103]
        train_Y_full = train_data[:, 103:]
        train_Y, train_Y_rest = split_label(train_Y_full, split)

        test_X = test_data[:, :103]
        test_Y_full = test_data[:, 103:]
        test_Y, test_Y_rest = split_label(test_Y_full, split)

    elif dataset == "nus":
        data_dir = 'datasets/'
        # uncomment if you need to make the npy file
        # train_data = arff.load(open(data_dir + 'nus-wide-full_BoW_l2-train.arff', 'rt'))
        # train_data = np.array(train_data['data'])
        # np.save('nus_train', train_data)
        train_data = np.load('datasets/nus_train.npy')
       

        # test_data = arff.load(open(data_dir + 'nus-wide-full_BoW_l2-test.arff', 'rt'))
        # test_data = np.array(test_data['data'])
        # np.save('nus_test', test_data)
        test_data = np.load('datasets/nus_test.npy')
        train_data, test_data = train_data[:50_000], test_data[:10_000]

         # remove name of image
        train_data = train_data[:,1:].astype(np.float)
        test_data = test_data[:,1:].astype(np.float)

        train_X = train_data[:, :-81]
        train_Y_full = train_data[:, -81:]
        train_Y, train_Y_rest = split_label(train_Y_full, split)

        test_X = test_data[:, :-81]
        test_Y_full = test_data[:, -81:]
        test_Y, test_Y_rest = split_label(test_Y_full, split)
        # exit()
    elif dataset == "mirfl":
        mat = scipy.io.loadmat('datasets/mirflickr.mat')

        train_X = mat['X1'].astype(np.float)
        train_Y_full = mat['X2'].astype(np.float)
    
        test_X = mat['XV1'].astype(np.float)
        test_Y_full = mat['XV2'].astype(np.float)

        train_Y, train_Y_rest = split_label(train_Y_full, split)
        test_Y, test_Y_rest = split_label(test_Y_full, split)
    elif dataset == "leda":
        
        # obtain the leda dataset with ingredientekst-regex function
        try:
            df_1 = pd.read_pickle("./df_1.pkl")
            df_0_big = pd.read_pickle("./df_0_big.pkl")
        except:
            # print("nah")
            # exit()
            df_1, df_0_big = get_leda(hyper_params)

        df_1.to_pickle("./df_1.pkl")
        df_0_big.to_pickle("./df_0_big.pkl")

        

        df_1_train, df_1_test = train_test_split(df_1, test_size=0.95)
        print(f"start columns {list(df_1_test)}, same as [imp_colb6, imp_colfol, imp_colij, imp_colca, imp_colb1]??")
        df_0_indices = np.arange(0, len(df_0_big))
        test_indices = np.random.choice(df_0_indices, len(df_1_test))
        df_0_test = df_0_big.iloc[test_indices]

        df_0_new_indices = list(set(df_0_indices)-set(test_indices))
        train_indices = np.random.choice(df_0_new_indices, len(df_1_train)*int(hyper_params.zero_multiplier))
        df_0_train = df_0_big.iloc[train_indices]

        # downsample the dataframe with 0's to x times the other
        # df_0 = df_0_big.sample(n=len(df_1)*int(hyper_params.zero_multiplier))
        # del df_0_old
        # df = pd.concat([df_0,df_1]).fillna(0)
        train_df = pd.concat([df_0_train,df_1_train]).fillna(0)
        test_df = pd.concat([df_0_test,df_1_test]).fillna(0)
        
        

        # until 3 because we have only 3 labels, put more when we have more labels
        # TODO if there are new labels this needs  to be adjusted in code
        
        train_X, train_Y_full = train_df[list(train_df)[5:]],train_df[list(train_df)[:5]]
        test_X, test_Y_full = test_df[list(test_df)[5:]],test_df[list(test_df)[:5]]
                
        # predict_xgboost(train_X, train_Y_full, test_X, test_Y_full)
        # exit()

        # Y = full_df[list(full_df)[:5]]
        # X = full_df[list(full_df)[5:]]
        # 'calcium toegevoegd als nutrient','VITB1 als toegevoegd nutrient','ijzer als toegevoegd nutrient','VITB6 als ingredient','FOLAC als ingredient SW'
        print(f"Label sequence: {list(test_Y_full)}")
        # train_X, test_X, train_Y_full, test_Y_full = train_test_split(X, Y, test_size=0.2)
        train_Y, train_Y_rest = split_label_pandas(train_Y_full, split, hyper_params)
        test_Y, test_Y_rest = split_label_pandas(test_Y_full, split, hyper_params)
        # print(f"OLD LABELS: {list(test_Y)}\nOLD Shape: {test_Y.shape}")
        # print(f"NEW LABELS: {list(test_Y_rest)}\nNEW Shape: {test_Y_rest.shape}")
        define_old_new(list(test_Y), list(test_Y_rest), hyper_params)
        # print(train_Y.shape, train_Y_rest.shape)
        # print(test_Y.shape, test_Y_rest.shape)
        # exit()
        train_Y, train_Y_rest, test_Y, test_Y_rest = train_Y.to_numpy(), train_Y_rest.to_numpy(), test_Y.to_numpy(), test_Y_rest.to_numpy()
        train_X, test_X = train_X.to_numpy(), test_X.to_numpy()

    return train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest

def predict_xgboost(train_X, train_Y_full, test_X, test_Y_full):
    import torch
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import hamming_loss
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_score
    from sklearn.metrics import jaccard_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import label_ranking_loss
    from sklearn.metrics import coverage_error
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import log_loss
    import sklearn.metrics as metrics
    rounded = 4
    # xgboost tryout
    import xgboost as xgb
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
            max_depth = 5, alpha = 10, n_estimators = 10)
    
    xg_reg.fit(train_X,train_Y_full['calcium toegevoegd als nutrient'])


    prediction = xg_reg.predict(test_X)
    thresh = 0.5
    print(f"Threshold at {thresh}")
    prediction = np.where(prediction > thresh, 1, 0) # same as round but with 0.4
    ground_truth = test_Y_full['calcium toegevoegd als nutrient']
    # print(ground_truth)
    # print(prediction)
    print(len(test_X))
    print(len(train_X))
    F1_Micro = round(f1_score(ground_truth, prediction, average='micro'), rounded)
    F1_Macro = round(f1_score(ground_truth, prediction, average='macro'), rounded)
    Hamming_loss = round(hamming_loss(ground_truth, prediction), rounded)
    Accuracy = round(accuracy_score(ground_truth, prediction), rounded)
    Recall_score_macro = round(recall_score(ground_truth, prediction, average='macro'), rounded)
    Recall_score_micro = round(recall_score(ground_truth, prediction, average='micro'), rounded)
    Precision_score_macro = round(precision_score(ground_truth, prediction, average='macro'), rounded)
    Precision_score_micro = round(precision_score(ground_truth, prediction, average='micro'), rounded)
    print('Recall_score_macro:   ', Recall_score_macro)
    print('Recall_score_micro:   ', Recall_score_micro)
    print('Precision_score_macro:   ', Precision_score_macro)
    print('Precision_score_micro:   ', Precision_score_micro)
    print('F1_Micro ', F1_Micro)
    print('F1_Macro ', F1_Macro)
    print('Hamming_loss: ', Hamming_loss)
    print("Accuracy = ", Accuracy)

def get_full_X(hyper_params):
    try:
        df_1 = pd.read_pickle("./df_1.pkl")
        df_0_big = pd.read_pickle("./df_0_big.pkl")
    except:
        df_1, df_0_big = get_leda(hyper_params)
    full_df = pd.concat([df_1,df_0_big]).fillna(0)
    full_df_X = full_df[list(full_df)[5:]]
    return full_df_X


def define_old_new(OLD, NEW, hyper_params):
    # 'calcium toegevoegd als nutrient','VITB1 als toegevoegd nutrient','ijzer als toegevoegd nutrient','VITB6 als ingredient','FOLAC als ingredient SW'
    old_list = ""
    new_list = ""
    for e, vitmin in enumerate(['calcium', 'ijzer', 'FOLAC', 'VITB1', 'VITB6']):
        # print("start")
        # print(vitmin)
        for i in OLD:
            # print(i)
            if vitmin in i:
                if e <= 1:
                    vitmin = vitmin[:2]
                if e >= 3:
                    vitmin = vitmin[-2:]
                old_list += vitmin
                # print("old")
        for j in NEW:
            # print(j)
            if vitmin in j:
                if e <= 1:
                    vitmin = vitmin[:2]
                if e >= 3:
                    vitmin = vitmin[-2:]
                new_list += vitmin
        #         print("new")
        # print("end")
    hyper_params.oldlist = old_list
    hyper_params.newlist = new_list
    # exit()

def get_leda(hyper_params):
    xls = pd.ExcelFile('LEDAExport_20200224_verrijkt_2_voorPepijn.xlsx')
    df = pd.read_excel(xls, 'LEDA_20200224')
    # TODO right column name
    # 'VITB6 als ingredient','ijzer als toegevoegd nutrient',
    # df = df[['ijzer als toegevoegd nutrient','VITB6 als ingredient','FOLAC als ingredient SW','CPV_INGREDIENTTEKST','ENERGIEWAARDEINKCAL_KCAL','EIWITTOTAAL_G','KOOLHYDRATEN_G','POLYOLEN_G','VOEDINGSVEZEL_G','ALCOHOL_G','WATER_G','ORGANISCHEZURENTOTAAL_G','VETZURENVERZADIGD_G','VETZURENENKELVONVERZDCIS_G','VETZURENMEERVONVERZ_G','VETZURENTRANS_G','CHOLESTEROL_MG', 'NATRIUM_MG','KALIUM_MG','CALCIUM_MG','FOSFOR_MG','MAGNESIUM_MG','IJZER_MG','KOPER_MG','SELENIUM_MUG','ZINK_MG','JODIUM_MUG','AS_G','RETINOLACTEQUIVALENT_MUG','RETINOL_MUG','BETACAROTEEN_MUG','ALFACAROTEEN_MUG','LUTEÏNE_MUG','ZEAXANTHINE_MUG','BETACRYPTOXANTHINE_MUG', 'LYCOPEEN_MUG','VITAMINED_MUG', 'VITAMINEE_MG','VITAMINEKTOTAAL_MUG','VITAMINEB1_MG','VITAMINEB2_MG','VITAMINEB6_MG', 'VITAMINEB12_MUG', 'NICOTINEZUUR_MG','VITAMINEC_MG','FOLIUMZUURTOEGEVOEGD_MUG LEDA_orig','FOLAATEQUIVALENTEN_MUG LEDA orig']] #, 'FOLAATEQUIVALENTEN_MUG', 'FOLIUMZUURTOEGEVOEGD_MUG'
    df = df[['calcium toegevoegd als nutrient','VITB1 als toegevoegd nutrient','ijzer als toegevoegd nutrient','VITB6 als ingredient','FOLAC als ingredient SW','CPV_INGREDIENTTEKST','ENERGIEWAARDEINKCAL_KCAL','EIWITTOTAAL_G','KOOLHYDRATEN_G','POLYOLEN_G','VOEDINGSVEZEL_G','ALCOHOL_G','WATER_G','ORGANISCHEZURENTOTAAL_G','VETZURENVERZADIGD_G','VETZURENENKELVONVERZDCIS_G','VETZURENMEERVONVERZ_G','VETZURENTRANS_G','CHOLESTEROL_MG', 'NATRIUM_MG','KALIUM_MG','CALCIUM_MG','FOSFOR_MG','MAGNESIUM_MG','IJZER_MG','KOPER_MG','SELENIUM_MUG','ZINK_MG','JODIUM_MUG','AS_G','RETINOLACTEQUIVALENT_MUG','RETINOL_MUG','BETACAROTEEN_MUG','ALFACAROTEEN_MUG','LUTEÏNE_MUG','ZEAXANTHINE_MUG','BETACRYPTOXANTHINE_MUG', 'LYCOPEEN_MUG','VITAMINED_MUG', 'VITAMINEE_MG','VITAMINEKTOTAAL_MUG','VITAMINEB1_MG','VITAMINEB2_MG','VITAMINEB6_MG', 'VITAMINEB12_MUG', 'NICOTINEZUUR_MG','VITAMINEC_MG','FOLIUMZUURTOEGEVOEGD_MUG LEDA_orig','FOLAATEQUIVALENTEN_MUG LEDA orig']] #, 'FOLAATEQUIVALENTEN_MUG', 'FOLIUMZUURTOEGEVOEGD_MUG'


    # 82 is 84 in excel (excel-2 voor hier gebruik) 27 true
    df = df.dropna(subset=['CPV_INGREDIENTTEKST'])
    bool_list1 = df['CPV_INGREDIENTTEKST'].str.contains("vitamine")
    bool_list2 = df['CPV_INGREDIENTTEKST'].str.contains("mineraal")
    bool_list3 = df['CPV_INGREDIENTTEKST'].str.contains("mineralen")
    # print(len(bool_list1),len(bool_list2),len(bool_list3))
    bool_list = np.array(bool_list1) | np.array(bool_list2) | np.array(bool_list3)
    # bool_list = list(bool_list1) or list(bool_list2) or list(bool_list3)
    j_indices = np.where(bool_list)[0]
    # print(len(j_indices))

    # lists of all vitamines the first being the vitamin
    vit_c = ['C','ascorbinezuur', 'l-ascorbinezuur', 'l-dehydroascorbinezuur']
    vit_b3 = ['B3','niacine']
    vit_b8 = ['B8', 'biotine']
    vit_b11 = ['B11', 'folaat', 'folacine', 'foline','foliumzuur']



    # combine all vitamin lists and write down mineral list
    vitamins = [vit_c, vit_b3, vit_b8, vit_b11]
    all_mineral_list = ['ijzer', 'calcium']

    j_indices_vitaminelist = []

    # put new vitamins in a list
    for i in range(len(j_indices)):
        new_vitamins = []
        # split ingredienttekst line in strings of words
        inglist = df['CPV_INGREDIENTTEKST'].iloc[j_indices[i]].replace('.',',').split(',')
        # print(inglist)
        # the previous strings can contain minerals or vitamins, put them in the respective lists
        mineralenlist = [item for item in inglist if 'mineraa' in item.lower() or 'mineralen' in item.lower()]
        # print(mineralenlist)
        vitaminelist = [item for item in inglist if 'vitamine' in item.lower() or 'vitamines' in item.lower()]
        # print('in')
        # print(inglist)
        vit_start_index = 0
        min_start_index = 0
        # mineralen 
    #     if ''
        # go through the string of the ingredienttekst and put all the vits/mins which are between brackets 
        # in the respective list
        if 'mineralen' in df['CPV_INGREDIENTTEKST'].iloc[j_indices[i]]:
            min_start_index = [inglist.index(i) for i in inglist if 'mineralen' in i]
            for j in range(min_start_index[0],len(inglist)-1):
                mineralenlist.append(inglist[j])
                if inglist[j] == '':
                    continue
                try:
                    if inglist[j][-1] == ")":
                        break
                except:
                    print("mineralenexcept")
                    print(f"inglist: {inglist}. j {j} ")
                    continue
                    
        if 'mineraa' in df['CPV_INGREDIENTTEKST'].iloc[j_indices[i]]:
            min_start_index = [inglist.index(i) for i in inglist if 'mineraa' in i]
            for j in range(min_start_index[0],len(inglist)-1):
                mineralenlist.append(inglist[j])
                if inglist[j] == '':
                    continue
                try:
                    if inglist[j][-1] == ")":
                        break
                except:
                    print("mineralenexcept")
                    print(f"inglist: {inglist}. j {j} ")
                    continue
                    
        for onemin in all_mineral_list:
            for min_ingr in inglist:
                if onemin in min_ingr:
                    mineralenlist.append(min_ingr)
            
        # print(mineralenlist)
        

        # werkt niet voor vitaminen: A en D3)
        if 'vitamines' in df['CPV_INGREDIENTTEKST'].iloc[j_indices[i]]:
            vit_start_index = [inglist.index(i) for i in inglist if 'vitamines' in i]
            for j in range(vit_start_index[0],len(inglist)-1):
                # print("inglistj-1", inglist[j][-1])
                # print(inglist[j])
                vitaminelist.append(inglist[j])
                if inglist[j] == '':
                    continue
                try:
                    if inglist[j][-1] == ")":
                        break
                except:
                    print("vitaminesexcept")
                    print(f"inglist: {inglist}. j {j} ")
                    continue
        if 'vitaminen' in df['CPV_INGREDIENTTEKST'].iloc[j_indices[i]]:
            vit_start_index = [inglist.index(i) for i in inglist if 'vitaminen' in i]
            for j in range(vit_start_index[0],len(inglist)-1):
                # print("inglistj-1", inglist[j][-1])
                # print(inglist[j])
                vitaminelist.append(inglist[j])
                if inglist[j] == '':
                    continue
                try:
                    if inglist[j][-1] == ")":
                        break
                except:
                    print('vitaminenexcept')
                    print(f"inglist: {inglist}. j {j} ")
                    continue

        # print(vitaminelist)

        # make string out of list strings
        vitaminelist = ''.join(vitaminelist)
        # print("vitaminelist na string", vitaminelist)
        # make uppercase after "vitamine" and "vitamine en" 
        vitaminelist = re.sub(r'vitamine ([a-z]+)', lambda match: fr'vitamin {match.group(1).upper()}',vitaminelist)
        vitaminelist = re.sub(r'vitamine ([a-z]+) en ([a-z]+)', lambda match: fr'vitamin {match.group(1).upper()} en {match.group(2).upper()}', vitaminelist)

        # loop over vitamine replacement names
        for vitamin in vitamins:
            for vitaminname in vitamin[1:]:
                # if the new vitamin name is in the list of vitamins add that vitamin
                if vitaminname in vitaminelist:
                    new_vitamins.append(vitamin[0])
                    vitaminelist = vitaminelist.replace(vitaminname,'')

        # loop over vitamin B
        for i in range(13,0, -1):
            if f'B{i}' in vitaminelist:
                new_vitamins.append(f'B{i}') 
                vitaminelist = vitaminelist.replace(f'B{i}','')

        # loop over vitamin A, C, D, en B11
        # print("vitaminelist", vitaminelist)
        # print(new_vitamins)
        # print("="* 40)
        for i in ['A','C','D,', 'D)','B11', 'K']:
            if f'{i}' in vitaminelist:
                if i == 'D)' or i == 'D,':
                    new_vitamins.append(f'{i[0]}') 
                else:
                    new_vitamins.append(f'{i}') 
                vitaminelist = vitaminelist.replace(f'{i}','')

        for mineraal in mineralenlist:
            if "ijzer" in mineraal:
                new_vitamins.append('ijzer')
            if 'calcium' in mineraal:
                new_vitamins.append('calcium')

        # add vitamins to the list
        j_indices_vitaminelist.append(list(set(new_vitamins)))

    # print(j_indices_vitaminelist)
    # print(j_indices)
    for i,vitamins in zip(j_indices, j_indices_vitaminelist):
        # print(df['CPV_INGREDIENTTEKST'].iloc[i])
        for vitamin in vitamins:
            if vitamin not in list(df.columns):
                df[str(vitamin)] = 0
            df[str(vitamin)].iloc[i] = 1
    
    df = df.drop(columns=['CPV_INGREDIENTTEKST'])
    
        # replace J and N with 1 and 0 resp.
    imp_colb6 = 'VITB6 als ingredient'
    imp_colfol = 'FOLAC als ingredient SW'
    imp_colij = 'ijzer als toegevoegd nutrient'
    imp_colca = 'calcium toegevoegd als nutrient'
    imp_colb1 = 'VITB1 als toegevoegd nutrient'

    all_cols = [imp_colb6, imp_colfol, imp_colij, imp_colca, imp_colb1]
    # transform yes and no to numbers
    for col in all_cols:
        df[[col]] = df[[col]].replace('nan', np.nan).fillna(0)
        df[[col]] = df[[col]].replace("J",1)
        df[[col]] = df[[col]].replace("N",0)
        df[[col]] = df[[col]].replace("j",1)
        df[[col]] = df[[col]].replace("n",0)
        df[[col]] = df[[col]].replace("J ",1)
        df[[col]] = df[[col]].replace("N ",0)
        df[[col]] = df[[col]].replace("?",1)
    
    # drop rows if they dont have any values in the to be predicted vitamin column
    df = df.drop(columns=[col for col in all_cols if col not in all_cols])

    # concatenate all the to-be-predicted vitamin columns with 1's and 0's in a separate dataframe
    df_0_all = []
    df_1_all = []
    for col in all_cols:
        df_0_all.append(df[df[col] == 0])
        df_1_all.append(df[df[col] == 1])

    df_0_big = pd.concat(df_0_all)
    df_1 = pd.concat(df_1_all)

    
    
    return df_1, df_0_big

# def get_leda_old():
#     xls = pd.ExcelFile('LEDAexport_20200224_verrijkt_voorPepijn.xlsx')
#     df = pd.read_excel(xls, 'LEDA_20200224')
#     df = df.sample(frac=1).reset_index(drop=True)
#     # TODO right column name
#     # 'VITB6 als ingredient','ijzer als toegevoegd nutrient',
#     df = df[['ijzer als toegevoegd nutrient','VITB6 als ingredient','FOLAC als ingredient SW','CPV_INGREDIENTTEKST','ENERGIEWAARDEINKCAL_KCAL','EIWITTOTAAL_G','KOOLHYDRATEN_G','POLYOLEN_G','VOEDINGSVEZEL_G','ALCOHOL_G','WATER_G','ORGANISCHEZURENTOTAAL_G','VETZURENVERZADIGD_G','VETZURENENKELVONVERZDCIS_G','VETZURENMEERVONVERZ_G','VETZURENTRANS_G','CHOLESTEROL_MG', 'NATRIUM_MG','KALIUM_MG','CALCIUM_MG','FOSFOR_MG','MAGNESIUM_MG','IJZER_MG','KOPER_MG','SELENIUM_MUG','ZINK_MG','JODIUM_MUG','AS_G','RETINOLACTEQUIVALENT_MUG','RETINOL_MUG','BETACAROTEEN_MUG','ALFACAROTEEN_MUG','LUTEÏNE_MUG','ZEAXANTHINE_MUG','BETACRYPTOXANTHINE_MUG', 'LYCOPEEN_MUG','VITAMINED_MUG', 'VITAMINEE_MG','VITAMINEKTOTAAL_MUG','VITAMINEB1_MG','VITAMINEB2_MG','VITAMINEB6_MG', 'VITAMINEB12_MUG', 'NICOTINEZUUR_MG','VITAMINEC_MG','FOLIUMZUURTOEGEVOEGD_MUG LEDA_orig','FOLAATEQUIVALENTEN_MUG LEDA orig']] #, 'FOLAATEQUIVALENTEN_MUG', 'FOLIUMZUURTOEGEVOEGD_MUG'
#     # 82 is 84 in excel (excel-2 voor hier gebruik) 27 true

#     # delete rows which have nothing in ingredienttekst
#     df = df.dropna(subset=['CPV_INGREDIENTTEKST'])
#     bool_list1 = df['CPV_INGREDIENTTEKST'].str.contains("vitamine")
#     bool_list2 = df['CPV_INGREDIENTTEKST'].str.contains("mineraal")
#     bool_list3 = df['CPV_INGREDIENTTEKST'].str.contains("mineralen")
#     # print(len(bool_list1),len(bool_list2),len(bool_list3))
#     # make  a boolean list of 1/0 if the items contain any of the above strings
#     bool_list = np.array(bool_list1) | np.array(bool_list2) | np.array(bool_list3)
#     # bool_list = list(bool_list1) or list(bool_list2) or list(bool_list3)
#     # the yes indices (ja --> j)
#     j_indices = np.where(bool_list)[0]
#     # print(len(j_indices))

#     # lists of all vitamines the first being the vitamin   the rest are other names for the vitamin
#     vit_c = ['C','ascorbinezuur', 'l-ascorbinezuur', 'l-dehydroascorbinezuur']
#     vit_b3 = ['B3','niacine']
#     vit_b8 = ['B8', 'biotine']

#     # combine all vitamin lists
#     vitamins = [vit_c, vit_b3, vit_b8]

#     j_indices_vitaminelist = []
#     print(f"original df shape: {df.shape}")
#     # put new vitamins in a list
#     for i in range(len(j_indices)):
#         new_vitamins = []
#         inglist = df['CPV_INGREDIENTTEKST'].iloc[j_indices[i]].replace('.',',').split(',')
#         # print(inglist)
#         mineralenlist = [item for item in inglist if 'mineraa' in item.lower() or 'mineralen' in item.lower()]
#         # print(mineralenlist)
#         vitaminelist = [item for item in inglist if 'vitamine' in item.lower() or 'vitamines' in item.lower()]
#         # print('in')
#         # print(inglist)
#         vit_start_index = 0
#         min_start_index = 0
#         # mineralen
#         if 'mineralen' in df['CPV_INGREDIENTTEKST'].iloc[j_indices[i]]:
#             min_start_index = [inglist.index(i) for i in inglist if 'mineralen' in i]
#             for j in range(min_start_index[0],len(inglist)-1):
#                 mineralenlist.append(inglist[j])
#                 if inglist[j] == '':
#                     continue
#                 try:
#                     if inglist[j][-1] == ")":
#                         break
#                 except:
#                     print("mineralenexcept")
#                     print(f"inglist: {inglist}. j {j} ")
#                     continue
#         # print(mineralenlist)
        

#         # werkt niet voor vitaminen: A en D3)
#         if 'vitamines' in df['CPV_INGREDIENTTEKST'].iloc[j_indices[i]]:
#             vit_start_index = [inglist.index(i) for i in inglist if 'vitamines' in i]
#             for j in range(vit_start_index[0],len(inglist)-1):
#                 # print("inglistj-1", inglist[j][-1])
#                 # print(inglist[j])
#                 vitaminelist.append(inglist[j])
#                 if inglist[j] == '':
#                     continue
#                 try:
#                     if inglist[j][-1] == ")":
#                         break
#                 except:
#                     print("vitaminesexcept")
#                     print(f"inglist: {inglist}. j {j} ")
#                     continue
#         if 'vitaminen' in df['CPV_INGREDIENTTEKST'].iloc[j_indices[i]]:
#             vit_start_index = [inglist.index(i) for i in inglist if 'vitaminen' in i]
#             for j in range(vit_start_index[0],len(inglist)-1):
#                 # print("inglistj-1", inglist[j][-1])
#                 # print(inglist[j])
#                 vitaminelist.append(inglist[j])
#                 if inglist[j] == '':
#                     continue
#                 try:
#                     if inglist[j][-1] == ")":
#                         break
#                 except:
#                     print('vitaminenexcept')
#                     print(f"inglist: {inglist}. j {j} ")
#                     continue

#         # print(vitaminelist)

#         # make string out of list strings
#         vitaminelist = ''.join(vitaminelist)
#         # print("vitaminelist na string", vitaminelist)
#         # make uppercase after "vitamine" and "vitamine en" 
#         vitaminelist = re.sub(r'vitamine ([a-z]+)', lambda match: fr'vitamin {match.group(1).upper()}',vitaminelist)
#         vitaminelist = re.sub(r'vitamine ([a-z]+) en ([a-z]+)', lambda match: fr'vitamin {match.group(1).upper()} en {match.group(2).upper()}', vitaminelist)

#         # loop over vitamine replacement names
#         for vitamin in vitamins:
#             for vitaminname in vitamin[1:]:
#                 # if the new vitamin name is in the list of vitamins add that vitamin
#                 if vitaminname in vitaminelist:
#                     new_vitamins.append(vitamin[0])
#                     vitaminelist = vitaminelist.replace(vitaminname,'')

#         # loop over vitamin B
#         for i in range(13,0, -1):
#             if f'B{i}' in vitaminelist:
#                 new_vitamins.append(f'B{i}') 
#                 vitaminelist = vitaminelist.replace(f'B{i}','')

#         # loop over vitamin A, C, D, en B11
#         # print("vitaminelist", vitaminelist)
#         # print(new_vitamins)
#         # print("="* 40)
#         for i in ['A','C','D,', 'D)','foliumzuur', 'K']:
#             if f'{i}' in vitaminelist:
#                 if i == 'D)' or i == 'D,':
#                     new_vitamins.append(f'{i[0]}') 
#                 else:
#                     new_vitamins.append(f'{i}') 
#                 vitaminelist = vitaminelist.replace(f'{i}','')

#         for mineraal in mineralenlist:
#             if "ijzer" in mineraal:
#                 new_vitamins.append('ijzer')

#         # add vitamins to the list
#         j_indices_vitaminelist.append(list(set(new_vitamins)))

#     # print(j_indices_vitaminelist)
#     # print(j_indices)
#     for i, vitamins in zip(j_indices, j_indices_vitaminelist):
#         # print(df['CPV_INGREDIENTTEKST'].iloc[i])
#         for vitamin in vitamins:
#             if vitamin not in list(df.columns):
#                 df[str(vitamin)] = 0
#             df[str(vitamin)].iloc[i] = 1
    
#     df = df.drop(columns=['CPV_INGREDIENTTEKST'])
    
#     # replace J and N with 1 and 0 resp.
#     imp_colb6 = 'VITB6 als ingredient'
#     imp_colfol = 'FOLAC als ingredient SW'
#     imp_colij = 'ijzer als toegevoegd nutrient'

#     df[[imp_colfol]] = df[[imp_colfol]].replace("J",1)
#     df[[imp_colfol]] = df[[imp_colfol]].replace("N",0)

#     df[[imp_colb6]] = df[[imp_colb6]].replace("J",1)
#     df[[imp_colb6]] = df[[imp_colb6]].replace("N",0)

#     df[[imp_colij]] = df[[imp_colij]].replace("J",1)
#     df[[imp_colij]] = df[[imp_colij]].replace("N",0)

#     all_cols = [imp_colb6, imp_colfol, imp_colij]

#     # imp_col = imp_colij # TODO CHANGE HERE FOR NEW PREDS
#     # imp_cols = ['VITB6 als ingredient', 'FOLAC als ingredient SW', 'ijzer als toegevoegd nutrient']

#     # df = df.drop(columns=[col for col in all_cols if col != imp_col])
#     # drop rows if they dont have any values in the to be predicted vitamin column
#     df = df.drop(columns=[col for col in all_cols if col not in all_cols])

#     # concatenate all the to-be-predicted vitamin columns with 1's and 0's in a separate dataframe
#     df_0_all = []
#     df_1_all = []
#     for col in all_cols:
#         df_0_all.append(df[df[col] == 0])
#         df_1_all.append(df[df[col] == 1])

#     df_0_big = pd.concat(df_0_all)
#     df_1 = pd.concat(df_1_all)

#     # downsample the dataframe with 0's to x times the other
#     df_0 = df_0_big.sample(n=len(df_1)*2)
#     # del df_0_old

#     df = pd.concat([df_0,df_1]).fillna(0)
    
#     return df



