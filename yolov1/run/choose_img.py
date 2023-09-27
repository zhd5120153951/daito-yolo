# ----------------------------------------------------------------------------------------------------------------------
# 筛选单类别标签，可不看
# ----------------------------------------------------------------------------------------------------------------------

image_list = []  # image path list
f = open('./data/combine_doc/voc2007test.txt')
lines = f.readlines()
file_list = []
for line in lines:
    splited = line.strip().split()

    print(splited)
    # if len(splited)<12:
    if len(splited) < 7:
        if splited[5] == '0' or splited[5] == '1' or splited[5] == '11':
            print(splited)
            file_list.append(splited)

f.close()

# filename = r'A:\Learning_doc\detection_learning\yolov1\data_dir\VOC0712\one_obj.txt'
filename = r'A:\Learning_doc\detection_learning\yolov1\data_dir\VOC0712\0_1_11.txt'
with open(filename, 'a') as f2:
    for line1 in file_list:
        for word in line1:
            f2.write(str(word))
            f2.write(' ')
        f2.write('\n')

f2.close()
