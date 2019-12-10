
f = open("/Users/aryaman/research/all_imgs_ck+", "r")
image_paths = f.readlines()


d = {}
for path in image_paths:
    path = path[:-1]
    d[path] = 1

relevant_frames = []

for path in image_paths:
    path = path[:-1]
    print(path)

    person = path.split("/")[-3]
    seq = path.split("/")[-2]
    image_name = path.split("/")[-1]

    cnt_expr_taken = 3

    def check_if_exist_in_dict(name):
        if seq_num == 1:
            return False
        if d.get(name) != None:
            return True
        return False

    seq_num = int(image_name.split("_")[-1].split(".")[0])
    seq_should_exist = seq_num + cnt_expr_taken
    new_image_name = image_name.split("_")[0] + '_' + \
                     image_name.split("_")[1] + '_' + \
                     str(seq_should_exist).zfill(8) + '.' + image_name.split(".")[-1]

    new_path = path.rsplit('/', 1)[0] + "/" + new_image_name

    relevant = not check_if_exist_in_dict(new_path, seq_num)
    if relevant:
        print

    break