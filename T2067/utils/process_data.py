import pandas as pd
import re
import os
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

def create_class_labels(mask_types, gender, age_groups):
    class_labels = {}
    label = 0
    for m_type in mask_types:
        for gend in gender:
            for age in age_groups:
                tmp = m_type+gend+age
                class_labels[tmp] = class_labels
                label += 1
    return class_labels


def create_train_dataframe(train_path, cropped_path):
    dataframe = pd.read_csv(os.path.join(train_path,"train.csv"))

    mask_types = ['mask', 'incorrect_mask', 'normal']
    gender = ['male', 'female']
    age_groups = ['x<30', '30 <= x < 60', '60<=x']

    class_labels = create_class_labels(mask_types, gender, age_groups)
    train_data = []
    train_labels = []

    images_path = os.path.join(train_path, 'images')
    for i in tqdm(range(len(dataframe)), desc="Processing Input DataFrame"):
        file_name = dataframe.iloc[i]['path']
        gn = dataframe.iloc[i]['gender']
        age = dataframe.iloc[i]['age']
        age_group = []
        for grp in age_groups:
            if eval(re.sub("x",str(age),grp)):
                age_group.append(grp)
            age_group.append(grp)
        age_group = age_group[0]
        tmp = os.path.join(images_path, file_name)
        for imfile in os.listdir(tmp):
            if not imfile.startswith("._"):
                mask_label= re.sub("[0-9]","",imfile.split(".")[0])
                classif = class_labels[mask_label+gn+age_group]
                train_labels.append(classif)
                original_im_path = os.path.join(tmp, imfile)
                cropped_im_path = os.path.join(cropped_path, "".join(original_im_path.split("/")[-2:]))
                train_data.append(cropped_im_path)

    df = pd.DataFrame(list(zip(train_data, train_labels)), columns = ['path', 'class'])
    return df


def split_df(df, kfold_n):
    kfold = StratifiedKFold(n_splits = kfold_n)
    X = df['path'].values
    y = df['class'].values
    datas = []
    for i, (train_index, valid_index) in enumerate(kfold.split(X,y)):
        train_df = df.iloc[train_index].copy().reset_index(drop=True)
        valid_df = df.iloc[valid_index].copy().reset_index(drop=True)

        datas.append((train_df, valid_df))
    return datas
