from sklearn.ensemble import RandomForestClassifier
import numpy as np


def entropy(s):
    _, counts = np.unique(list(s), return_counts=True)
    total = sum(counts)
    percent = list(map(lambda x: x / total, counts))
    return sum(-n * np.log(n) for n in percent)

def processDomain(s):
    return len(s), sum(c.isdigit() for c in s), entropy(s)

domainlist = []
class Domain:
    def __init__(self,_domain,_label):
        self.domain = _domain
        self.label = _label
        self.domainNameLength, self.domainNumberCount, self.letterEntropy = processDomain(self.domain)
    def returnData(self):
        return [self.domainNameLength, self.domainNumberCount, self.letterEntropy]
    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

    def initData(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                tokens = line.split(",")
                domain = tokens[0]
                label = tokens[1]
                domainlist.append(Domain(domain, label))

    def initTest(filename):
        with open(filename) as f:
            testlist=[]
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                testlist.append(line)
            return testlist

    if __name__ == '__main__':
        initData("train.txt")
        featureMatrix = list(map(lambda domain: domain.returnData(), domainlist))
        labelList = list(map(lambda domain: domain.returnLabel(), domainlist))
        testDomains = initTest("test.txt")
        testDomainFeatures = []
        for domain in testDomains:
            domainNameLength, domainNumberCount, letterEntropy = processDomain(domain)
            testDomainFeatures.append([domainNameLength, domainNumberCount, letterEntropy])
        clf = RandomForestClassifier(random_state=0)
        #train
        clf.fit(featureMatrix, labelList)
        #predict
        testLabels = clf.predict(testDomainFeatures)
        #output
        output = list(zip(testDomains, testLabels))
        with open("result.txt", "w+") as f:
            for domain, label in output:
                line = domain + ","
                if label == 0:
                    line = line + "notdga\n"
                else:
                    line = line + "dga\n"
                f.write(line)
