import numpy as np
#X_Edge = np.load('ei-transfer_learning-mfcc-X.npy')
#Y_Edge = np.load('ei-transfer_learning-mfcc-y.npy')[:, 0]


X_My = np.load('./features/x_wav_feature_test.npy')
Y_My = np.load('./features/y_wav_feature_test.npy')

X_DSP = np.load('features/x_dsp.npy')
Y_DPS = np.load('features/y_dsp.npy')

xm=X_My[0]
xd=X_DSP[0]

sum=0
for i in range(len(xm)):
    sum=sum+(xm[i]-xd[i])
    print(xm[i]-xd[i])

print("Average Error:")
print(sum/len(xm))