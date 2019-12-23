import os
import base64
import csv
import sys
import zlib
import json
import argparse

import numpy as np


parser = argparse.ArgumentParser()
# parser.add_argument('--imgid_list', default='data/coco_precomp/train_ids.txt',
#                     help='Path to list of image id')
parser.add_argument('--input_dir', default='/media/data/kualee/coco_bottom_up_feature/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv',
                    help='tsv of all image data (output of bottom-up-attention/tools/generate_tsv.py), \
                    where each columns are: [image_id, image_w, image_h, num_boxes, boxes, features].')
parser.add_argument('--output_dir', default='data/coco_precomp/',
                    help='Output directory.')
parser.add_argument('--split', default='train',
                    help='train|dev|test')
opt = parser.parse_args()
print(opt)


# meta = []
# feature = {}
# for line in open(opt.imgid_list):
#     sid = int(line.strip())
#     meta.append(sid)
#     feature[sid] = None

# csv.field_size_limit(sys.maxsize)
# FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    # input_file = os.path.join(opt.input_dir, 'merged_out.tsv')

    tsv_list = os.listdir(opt.input_dir)
    tsv_list.sort()
    tsv_list = tsv_list[7:8]
    # print('tsv_list', tsv_list)

    for tsv_file in tsv_list:
        input_file = os.path.join(opt.input_dir, tsv_file)
        print('input_file', input_file)
        print('Check Numbering', tsv_file.split('_')[1][0])
        print("Processing %s" % (tsv_file))
        numbering = tsv_file.split('_')[1][0]
    # input_file = opt.input_dir

        with open(input_file, "r+b") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            count = 0
            img_id_dic = {}
            feature = {}
            for item in reader:
                # item['image_id'] = item['image_id']
                # print('a', item['image_id'])
                img_id_dic[count] = item['image_id']
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                # print('aa', np.array(item['features'], dtype=np.float32).shape)
                # print('aaa', type(item['boxes']))
                for field in ['boxes', 'features']:
                    data = item[field]
                    # print('data',data.shape)
                    buf = base64.decodestring(data)
                    # print('buf',buf.shape)
                    temp = np.frombuffer(buf, dtype=np.float32)
                    # print('temp',type(temp))
                    item[field] = temp.reshape((item['num_boxes'],-1))
                    # print('item[num_boxes]',item['num_boxes'])
                    # print('item[boxes]',item['boxes'].shape)
                    # print('item[features]',item['features'].shape)
                # if item['image_id'] in feature:
                #     feature[item['image_id']] = item['features']
                feature[count] = item['features']
                # print('hi', item['features'].shape)

                if count % 100 == 0:
                    print("Process %d images" %(count+1))
                    # print('b',item['image_id'])
                    # print('c',item['image_h'])
                    # print('d',item['image_w'])
                    # print('e',item['num_boxes'])
                    # print('f',type(item['features']))
                    # print('ff',item['features'].shape)
                count += 1

        # data_out = np.stack([feature[sid] for sid in meta], axis=0)
        data_out = np.stack([feature[num] for num in range(count)], axis=0)
        print("Final numpy array shape:", data_out.shape)
        np.save(os.path.join(opt.output_dir, '{}_ims_{}.npy'.format(opt.split, numbering)), data_out)

        with open(os.path.join(opt.output_dir, 'img_id_dic_{}.tsv'.format(numbering)), "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['image_id', 'real_id'])

            for key in img_id_dic.keys():
                writer.writerow([key, img_id_dic[key]])
            
