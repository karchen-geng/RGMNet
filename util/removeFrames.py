import glob
import os

# #davis17
# name_list_path = '/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Dataset/DAVIS-2017-trainval-480p/davis_max_gap/ImageSets/2017/val.txt'
# name_list = [i.rstrip('\n') for i in open(name_list_path, 'r').readlines()]
# # print(name_list)
# jpg_path = "/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Dataset/DAVIS-2017-trainval-480p/davis_max_gap/JPEGImages/480p"
# anno_path = "/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Dataset/DAVIS-2017-trainval-480p/davis_max_gap/Annotations/480p"  # anno
# files = glob.glob(anno_path + "/*")


# divis17test
name_list_path = '/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Dataset/DAVIS-2017-test-dev-480p/davis_max_gap/ImageSets/2017/test-dev.txt'
name_list = [i.rstrip('\n') for i in open(name_list_path, 'r').readlines()]
# print(name_list)
jpg_path = "/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Dataset/DAVIS-2017-test-dev-480p/davis_max_gap/JPEGImages/480p"
files = glob.glob(jpg_path + "/*")

for file in files:
    file_name = file.split('.')[0].split('/')[-1]  # parkour
    if file_name in name_list:  # 在测试集中
        this_files = glob.glob(file + "/*")
        for this_file in this_files:
            file_index = int(this_file.split('.')[0].split('/')[-1])  # 下标
            # 除了第一帧，后续帧每隔3帧删除一帧
            if file_index != 0 and file_index % 3 != 0:
                os.remove(this_file)
                # print(file)
