'''
决策树：
构建决策树：
step1:选择最佳特征 选择最佳特征依据：信息增益
step2：根据最佳特征将数据集分成若干子数据集 在每个子数据集上递归的创建决策树 选择信息增益为依据选择最佳特征
step3：结束条件 子数所有的分类都数据同一个分类或者所有特征遍历结束
'''

def formatDataset(filename):
    f = open(filename)
    linses = [inst.strip().split("\t") for inst in f.readlines()]
    linesLabels = ['age','prescript','astigmatic','tearRate']
    return linses,linesLabels

def majority(classlist):
    label_count = {}
    total_count = len(classlist)
    for item in classlist:
        if item in label_count:
            label_count[item] += 1
        else:
            label_count[item] = 1
    max_num = 0
    for key in label_count:
        if label_count[key] > key:
            max_num = key
    return max_num

def calShannonEntropy(dataset):
    import math
    from math import log
    numTraining = len(dataset)
    label_count = {}
    for item in dataset:
        label = item[-1]
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    shannonEntropy = 0
    for key in label_count:
        prob = float(label_count[key]) / numTraining
        shannonEntropy -= prob * log(prob,2)
    return shannonEntropy

def chooseBestFeature(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calShannonEntropy(dataset)
    bestInfoGain = 0
    bestFeatureIndex = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataset]
        uniqueValue = set(featureList)
        tiaojianEntropy = 0
        for value in uniqueValue:
            subDataset = splitDataset(dataset,i,value)
            prob = len(subDataset) / float(len(dataset))
            tiaojianEntropy += prob * calShannonEntropy(subDataset)
        infoGain = baseEntropy - tiaojianEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatureIndex = i
    return bestFeatureIndex

def splitDataset(dataset,index,value):
    retDataset = []
    for item in dataset:
        if item[index] == value:
            reducedFeatureVector = item[:index]
            reducedFeatureVector.extend(item[index + 1:])
            retDataset.append(reducedFeatureVector)
    return retDataset


def createTree(dataset,labels):
    classlist = [item[-1] for item in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majority(classlist)
    bestFeature = chooseBestFeature(dataset)
    bestFeatureLabel = labels[bestFeature]
    tree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featureValues = [example[bestFeature] for example in dataset]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subLabels = labels[:]
        tree[bestFeatureLabel][value] = createTree(splitDataset(dataset,bestFeature,value),subLabels)
    return tree

if __name__ == "__main__":
    dataset,labels = formatDataset("lenses.txt")
    a = createTree(dataset,labels)
    print a
