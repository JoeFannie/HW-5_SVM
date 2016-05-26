# !/usr/bin/env python

from numpy import *
import matplotlib.pyplot as plt


def Alpha_adjust(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj
	
def Jrand_selection(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def QProgam(dataMatIn, 
	classLabels, 
	C, 
	tolerance, 
	maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)

    b = 0
    alphas = mat(zeros((m,1)))
    iter = 0

    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])

            if ((labelMat[i]*Ei < -tolerance) and (alphas[i] < C)) or ((labelMat[i]*Ei > tolerance) and (alphas[i] > 0)):
                j = Jrand_selection(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H: print "L==H"; continue

                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T

                if eta >= 0: print "eta>=0"; continue

                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = Alpha_adjust(alphas[j],H,L)

                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0

                alphaPairsChanged += 1
                print "Iter: %d i:%d, Pairs Changed %d" % (iter,i,alphaPairsChanged)

        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "Iter#%d" % iter

    return b,alphas
	
def Plot(data, label, w, b, text = 'SVM'):
    fig = plt.figure()
    nsample = label.size
    for i in range(nsample):
        if label[i] == -1:
            c = 'red'
        elif label[i] == 1:
            c = 'blue'
        else:
            print 'label error!'
        plt.plot(data[i][0], data[i][1], marker = 'o', color = c)
    sortedInd = data.argsort(axis = 0)
    x1min, x1max = data[sortedInd[0][0]][0], data[sortedInd[-1][0]][0]
    x1 = linspace(x1min, x1max, 1000)
    x2 = -(w[0] * x1 + b[0]) / w[1]
    plt.plot(x1, x2, linewidth = 2)
    plt.title(text)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.text(3, -2, 'wx+b=0')
    plt.show()
    pass

def My_SVM(data, label, c = 1, epsilon = 1e-6, iterator = 256):
    nsample = label.size
    dim = data.shape[1]
    b, alphas =  QProgam(data, label, c, epsilon, iterator)
    alphas = array(alphas)
    b = array(b)
    w = zeros(dim)
    for i in range(nsample):
        w += alphas[i][0] * label[i] * data[i]
    print "w=",w
    print "b=",b
    Plot(data, label, w, b)

def Test():
    nsample = 64
    random.seed(0xdeadbeef)
    x1 = random.multivariate_normal([0,0], [[1,0],[0,1]], nsample)
    y1 = -ones(nsample)
    x2 = random.multivariate_normal([2,2], [[1,0.5],[0.5,1]], nsample)
    y2 = ones(nsample)
    data = concatenate((x1, x2))
    label = concatenate((y1, y2))
    My_SVM(data, label)

def main():
    Test()
    pass

if __name__ == '__main__':
    main()
