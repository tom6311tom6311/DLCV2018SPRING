import os, sys
from skimage import color, io
import numpy as np

np.random.seed(226)

def read_train(path, base=True, sample=5):
    '''
    if m == True: return sat_name, sat, mask
    else: return sat_name, sat, []
    '''

    base_path = os.path.join(path, 'base')
    novel_path = os.path.join(path, 'novel')
    novel_file = sorted([i for i in os.listdir(novel_path)])
    
    novel_imgs_name = []
    novel_imgs = []
    novel_label = []
    for f in novel_file:
        imgs_name = sorted([i for i in os.listdir(os.path.join(novel_path, f, 'train'))])
        imgs_name = np.random.choice(imgs_name, size=sample, replace=False)
        for i in imgs_name:
            img = io.imread(os.path.join(novel_path, f, 'train', i))
            img = np.expand_dims(np.array(img), axis = 0)
            novel_imgs_name.append(os.path.join(f, 'train', i))
            novel_imgs.append(img)
            novel_label.append(f[-2:])

    with open('novel_imgs_name' + str(sample) + '.txt', 'w') as output:
        output.write("\n".join(novel_imgs_name))

    base_imgs = []
    base_label = []
    if base:
        base_path = os.path.join(path, 'base')
        base_file = sorted([i for i in os.listdir(base_path)])
    
        for f in base_file:
            imgs_name = sorted([i for i in os.listdir(os.path.join(base_path, f, 'train'))])
            for i in imgs_name:
                img = io.imread(os.path.join(base_path, f, 'train', i))
                img = np.expand_dims(np.array(img), axis = 0)
                base_imgs.append(img)
                base_label.append(f[-2:])


    novel_imgs = np.vstack(novel_imgs).astype(np.float32) / 255
    novel_label = np.array(novel_label)
    base_imgs = np.vstack(base_imgs).astype(np.float32) / 255
    base_label = np.array(base_label)
    

    return novel_imgs, novel_label, base_imgs, base_label


def read_test(train_path, test_path, sample=5):
    ## Read real test data #
    with open('novel_imgs_name' + str(sample) + '.txt', 'r') as ifile:
        novel_path = ifile.read().splitlines()

    train_imgs = []
    test_imgs = []
    label = []

    for f in novel_path:
        img = io.imread(os.path.join(train_path, 'novel', f))
        img = np.expand_dims(np.array(img), axis = 0)
        train_imgs.append(img)
        l = f.find('_')+1
        label.append(f[l:l+2])


    test_file = sorted([i for i in os.listdir(test_path)])
    test_idx = [int(i[:i.find('.')]) for i in test_file]
    test_idx, test_file = zip(*sorted(zip(test_idx, test_file)))
    for f in test_file:
        img = io.imread(os.path.join(test_path, f))
        img = np.expand_dims(np.array(img), axis = 0)
        test_imgs.append(img)

    train_imgs = np.vstack(train_imgs).astype(np.float32) / 255 
    test_imgs = np.vstack(test_imgs).astype(np.float32) / 255
    label = np.array(label)

    print(train_imgs.shape)
    print(test_imgs.shape)
    print(label.shape)

    return train_imgs, label, test_imgs

def read_test2(train_path, test_path, sample=5):
    ## read novel training data 
    with open('novel_imgs_name' + str(sample) + '.txt', 'r') as ifile:
        novel_path = ifile.read().splitlines()

    train_imgs = []
    test_imgs = []
    label = []
    test_label = []

    for f in novel_path:
        img = io.imread(os.path.join(train_path, 'novel', f))
        img = np.expand_dims(np.array(img), axis = 0)
        train_imgs.append(img)
        l = f.find('_')+1
        label.append(f[l:l+2])


    test_file = sorted([i for i in os.listdir(test_path)])
    for f in test_file:
        imgs_file = sorted([i for i in os.listdir(os.path.join(test_path, f, 'train'))])[:100]
        for i in imgs_file:
            img = io.imread(os.path.join(test_path, f, 'train', i))
            img = np.expand_dims(np.array(img), axis = 0)
            test_imgs.append(img)
            test_label.append(f[-2:])

    train_imgs = np.vstack(train_imgs).astype(np.float32) / 255 
    test_imgs = np.vstack(test_imgs).astype(np.float32) / 255
    label = np.array(label)
    test_label = np.array(test_label)

    print(train_imgs.shape)
    print(test_imgs.shape)
    print(label.shape)
    print(test_label.shape)

    return train_imgs, label, test_imgs, test_label

def save_predict(predict, name):
    ## save predict 
    idx = np.arange(predict.shape[0])
    df = pd.DataFrame({'image_id':idx, 'predicted_label':predict})
    df.to_csv(name, index=False)


if __name__ == '__main__':
    print('read')
    sample = 5
    print('train data')
    #imgs, label = read_train(sys.argv[1], sample=sample)
    print('test data')
    train_imgs, label, test_imgs = read_test(sys.argv[1], sys.argv[2], sample=sample)
    #train_imgs, label, test_imgs, test_label = read_test2(sys.argv[1], sys.argv[2], sample=sample)
