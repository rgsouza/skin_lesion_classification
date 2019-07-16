# -*- coding: utf-8 -*-
#
# Last modification: 1 July. 2019
# Author: Rayanne Souza

import os
import pandas as pd
from glob import glob
from PIL import Image
from PIL import ImageFilter
from sklearn.model_selection import train_test_split


df = pd.read_csv('metadata/HAM10000_metadata_MODIFIED2.csv')
all_image_path = glob(os.path.join('images/', '*', '*.jpg'))
print(len(all_image_path))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
df['path'] = df['new_image_id'].map(imageid_path_dict.get)
df['categorical_label'] = pd.Categorical(df['dx']).codes

# Creates new metadata 
new_metadata = df[['new_image_id','categorical_label','path']].copy()


v = new_metadata.categorical_label.value_counts()
print(v)

# Path for news images
direc = 'images/copy_120_90/'

for i in range(7):
    if v[i] < 600:
        
        print("Creating image for categorical_label", i)
        
        df_aux = df.loc[df['categorical_label']==i, :]
        tdf = df_aux[['new_image_id','categorical_label','path']].copy()

        
        for index, row in tdf.iterrows():

            basename = row.new_image_id
            
            im=Image.open(row.path)
            im=im.convert("RGB")
            r,g,b=im.split()
            r=r.convert("RGB")
            g=g.convert("RGB")
            b=b.convert("RGB")
            im_blur=im.filter(ImageFilter.GaussianBlur)
 
            r_name = basename + '_r_copy'
            g_name = basename + '_g_copy'
            b_name = basename + '_b_copy'
            bl_name = basename + '_bl_copy'
 
 
            r_path = direc + r_name + ".jpg"
            g_path = direc + g_name + ".jpg"
            b_path = direc + b_name + ".jpg"
            bl_path = direc + bl_name + ".jpg"
       
            
            r.save(r_path)
            g.save(g_path)
            b.save(b_path)
            im_blur.save(bl_path)
            
            df1 = pd.DataFrame({"new_image_id":[r_name, g_name, b_name, bl_name], 
                         "categorical_label":[i, i, i, i],
                         'path': [r_path, g_path, b_path, bl_path]})
 
            if v[i] < 300:
                im_unsharp=im.filter(ImageFilter.UnsharpMask)
                im_max=im.filter(ImageFilter.MaxFilter)
                im_median=im.filter(ImageFilter.MedianFilter)
                im_rot180=im.transpose(Image.ROTATE_180)
                im_trans=im.transpose(Image.FLIP_LEFT_RIGHT)
    
                un_name = basename + '_un_copy'
                rot_name = basename + '_rot_copy'
                trans_name = basename + '_trans_copy'
                max_name = basename + '_max_copy'
                median_name = basename + '_median_copy'
    
                un_path = direc + un_name + ".jpg"
                rot_path = direc + rot_name + ".jpg"
                trans_path = direc + trans_name + ".jpg"
                max_path = direc + max_name + ".jpg"
                median_path = direc + median_name + ".jpg"
    
                im_unsharp.save(un_path)
                im_rot180.save(rot_path)
                im_trans.save(trans_path)
                im_max.save(max_path)
                im_median.save(median_path)
    
            
                df1 = pd.DataFrame({"new_image_id":[r_name, g_name, b_name, bl_name, un_name, rot_name,trans_name, max_name, median_name], 
                         "categorical_label":[i, i, i, i, i, i, i, i, i],
                         'path': [r_path, g_path, b_path, bl_path, un_path, rot_path,trans_path, max_path, median_path]}) 
                
           
            new_metadata = new_metadata.append(df1, ignore_index=True)
            

print("-------Size after augmentation-------")
print(new_metadata.categorical_label.value_counts())

# Splits 10% for testing
train, test = train_test_split(new_metadata, test_size=0.1,random_state = 60)
  
# Splits 10% for validation
train, validation = train_test_split(train, test_size=0.1, random_state=300)

# Exports data to csv file 
train.to_csv('train_df.csv')
validation.to_csv('validation_df.csv')
test.to_csv('test_df.csv')
new_metadata.to_csv('new_metadata.csv')

print(train.categorical_label.value_counts())
print(validation.categorical_label.value_counts())
print(test.categorical_label.value_counts())
