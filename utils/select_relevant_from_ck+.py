
f = open("/Users/aryaman/research/all_imgs_ck+", "r")
g = open("/Users/aryaman/research/all_code/utils/emotion_gt.txt")
image_paths = f.readlines()
gt_paths = g.readlines()

for path in gt_paths:
    path = path[:-1]
    #print(path)
    emotion_file_name = path.rsplit('/', 1)[-1]
    emotion_file_name_parts = emotion_file_name.split("_")
    cnt_expr_taken = 4
    seq_num = int(emotion_file_name_parts[2])
    img_base_path = image_paths[0][:-1].rsplit('/',1)[0]
    new_seq_num = [seq_num-i for i in range(0, cnt_expr_taken)]
    new_seq_num.append(1)
    for seq in new_seq_num:
        new_image_name = emotion_file_name_parts[0] + '_' + emotion_file_name_parts[1] + '_'  +\
                         str(seq).zfill(8) + ".png"
        new_image_path = img_base_path + '/' + new_image_name
        print(new_image_path)


