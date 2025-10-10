import pandas as pd

for fold in ['F1','F2','F3','F4','F5']:

    labels = pd.read_csv(f'data/{fold}_labels.txt')
    train = pd.read_csv(f'data/{fold}_train.txt')
    val = pd.read_csv(f'data/{fold}_val.txt')
    test = pd.read_csv(f'data/{fold}_test.txt')

    C18_RT = []
    Cyn_RT = []
    Phen_RT = []
    AQ_RT = []

    C18_SMILES = []
    Cyn_SMILES = []
    Phen_SMILES = []
    AQ_SMILES = []

    for i,j in zip(val['smiles'].values,val['RT'].values):
        ind = labels.values[j][0].split('_')[-1]
        ind = int(ind)
        if ind in [0,1]:
            C18_SMILES.append(i)
            C18_RT.append(j)
        elif ind in [2,3]:
            Cyn_SMILES.append(i)
            Cyn_RT.append(j)
        elif ind in [4,5]:
            Phen_SMILES.append(i)
            Phen_RT.append(j)
        else:
            AQ_SMILES.append(i)
            AQ_RT.append(j)


    for i,j in zip(test['smiles'].values,test['RT'].values):
        ind = labels.values[j][0].split('_')[-1]
        ind = int(ind)
        if ind in [0,1]:
            C18_SMILES.append(i)
            C18_RT.append(j)
        elif ind in [2,3]:
            Cyn_SMILES.append(i)
            Cyn_RT.append(j)
        elif ind in [4,5]:
            Phen_SMILES.append(i)
            Phen_RT.append(j)
        else:
            AQ_SMILES.append(i)
            AQ_RT.append(j)


    C18 = pd.DataFrame({'smiles':C18_SMILES, 'RT':C18_RT})

    CYN = pd.DataFrame({'smiles':Cyn_SMILES, 'RT':Cyn_RT})

    PHEN = pd.DataFrame({'smiles':Phen_SMILES, 'RT':Phen_RT})

    AQ = pd.DataFrame({'smiles':AQ_SMILES, 'RT':AQ_RT})

    pd.concat([train,C18]).to_csv(f'data/C18_{fold}_train.txt',index=False)
    pd.concat([train,CYN]).to_csv(f'data/CYN_{fold}_train.txt',index=False)
    pd.concat([train,PHEN]).to_csv(f'data/PHEN_{fold}_train.txt',index=False)
    pd.concat([train,AQ]).to_csv(f'data/AQ_{fold}_train.txt',index=False)


    test[~test.RT.isin(C18.RT)].to_csv(f'data/C18_{fold}_test.txt',index=False)
    test[~test.RT.isin(CYN.RT)].to_csv(f'data/CYN_{fold}_test.txt',index=False)
    test[~test.RT.isin(PHEN.RT)].to_csv(f'data/PHEN_{fold}_test.txt',index=False)
    test[~test.RT.isin(AQ.RT)].to_csv(f'data/AQ_{fold}_test.txt',index=False)

    val[~val.RT.isin(C18.RT)].to_csv(f'data/C18_{fold}_val.txt',index=False)
    val[~val.RT.isin(CYN.RT)].to_csv(f'data/CYN_{fold}_val.txt',index=False)
    val[~val.RT.isin(PHEN.RT)].to_csv(f'data/PHEN_{fold}_val.txt',index=False)
    val[~val.RT.isin(AQ.RT)].to_csv(f'data/AQ_{fold}_val.txt',index=False)



