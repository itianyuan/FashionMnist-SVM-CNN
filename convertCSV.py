# 将数据集转换为csv文件
def convert(imgf, labelf, outf, n):
    # rb，以二进制只读方式从文件开头打开
    # w，从文件开头开始写入
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    # 读入指定字节数
    f.read(16)
    l.read(8)

    # 创建一个列表
    images = []

    for i in range(n):
        # ord()返回字符对应的Asc码
        image = [ord(l.read(1))]
        # print("***************")
        # print(len(image))
        # print(image)
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        # print(len(image))
        images.append(image)
        # print("------------")
        # print(images)
    # 写入输出文件
    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()

if __name__ == '__main__':
    convert(r'data/fashion/train-images-idx3-ubyte', r"data/fashion/train-labels-idx1-ubyte",
            r"data/fashion/fashionmnist_train.csv", 60000)

    convert(r"data/fashion/t10k-images-idx3-ubyte", r"data/fashion/t10k-labels-idx1-ubyte",
            r"data/fashion/fashionmnist_test.csv", 10000)
print("Convert Finished!")
