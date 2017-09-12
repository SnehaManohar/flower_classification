# flower_classification
Classify flowers' images. Gives the name of the flower for the given image. Trained for 5 classes: Daisy, dandelion, rose, sunflower, tulip

Data: https://www.kaggle.com/alxmamaev/flowers-recognition

On running train.py you should see something like this:

0 model saved
Epoch {1}, Training Acc: {tensor(47.6662, device='cuda:0')} , Training loss: {0.6679016842784774} , Test Acc: {tensor(44.6488, device='cuda:0')}
1 model saved
Epoch {2}, Training Acc: {tensor(62.2584, device='cuda:0')} , Training loss: {0.5133580246493256} , Test Acc: {tensor(59.8067, device='cuda:0')}
2 model saved
Epoch {3}, Training Acc: {tensor(67.6332, device='cuda:0')} , Training loss: {0.45299297564156715} , Test Acc: {tensor(69.3776, device='cuda:0')}
Epoch {4}, Training Acc: {tensor(72.4658, device='cuda:0')} , Training loss: {0.40258510955967247} , Test Acc: {tensor(67.2560, device='cuda:0')}
4 model saved
Epoch {5}, Training Acc: {tensor(75.0589, device='cuda:0')} , Training loss: {0.36899652713905357} , Test Acc: {tensor(77.4399, device='cuda:0')}
5 model saved
Epoch {6}, Training Acc: {tensor(78.5950, device='cuda:0')} , Training loss: {0.33003863273210426} , Test Acc: {tensor(79.6558, device='cuda:0')}
6 model saved
Epoch {7}, Training Acc: {tensor(81.8482, device='cuda:0')} , Training loss: {0.29242199912969374} , Test Acc: {tensor(85.7614, device='cuda:0')}
7 model saved
Epoch {8}, Training Acc: {tensor(83.1212, device='cuda:0')} , Training loss: {0.2735002170203149} , Test Acc: {tensor(86.5865, device='cuda:0')}
8 model saved
Epoch {9}, Training Acc: {tensor(86.6337, device='cuda:0')} , Training loss: {0.2290562414535651} , Test Acc: {tensor(88.0481, device='cuda:0')}
9 model saved
Epoch {10}, Training Acc: {tensor(90.6412, device='cuda:0')} , Training loss: {0.18583735646939065} , Test Acc: {tensor(91.8906, device='cuda:0')}
10 model saved
Epoch {11}, Training Acc: {tensor(92.2442, device='cuda:0')} , Training loss: {0.16102507873800553} , Test Acc: {tensor(95.4267, device='cuda:0')}
Epoch {12}, Training Acc: {tensor(94.6016, device='cuda:0')} , Training loss: {0.1310181355578952} , Test Acc: {tensor(84.3470, device='cuda:0')}
Epoch {13}, Training Acc: {tensor(95.8510, device='cuda:0')} , Training loss: {0.11496748440828261} , Test Acc: {tensor(95.3795, device='cuda:0')}
13 model saved
Epoch {14}, Training Acc: {tensor(96.5347, device='cuda:0')} , Training loss: {0.100312180623925} , Test Acc: {tensor(96.5111, device='cuda:0')}
14 model saved
Epoch {15}, Training Acc: {tensor(95.4267, device='cuda:0')} , Training loss: {0.10737005685127214} , Test Acc: {tensor(98.5148, device='cuda:0')}
Epoch {16}, Training Acc: {tensor(96.3461, device='cuda:0')} , Training loss: {0.09821061749241364} , Test Acc: {tensor(97.1711, device='cuda:0')}
Epoch {17}, Training Acc: {tensor(97.9255, device='cuda:0')} , Training loss: {0.07684969464523339} , Test Acc: {tensor(96.5111, device='cuda:0')}
17 model saved
Epoch {18}, Training Acc: {tensor(99.1278, device='cuda:0')} , Training loss: {0.0584697476160071} , Test Acc: {tensor(100.7308, device='cuda:0')}
Epoch {19}, Training Acc: {tensor(99.3871, device='cuda:0')} , Training loss: {0.05239027322132539} , Test Acc: {tensor(99.4578, device='cuda:0')}
Epoch {20}, Training Acc: {tensor(98.2320, device='cuda:0')} , Training loss: {0.06553301278225997} , Test Acc: {tensor(99.1749, device='cuda:0')}


We can then test our models by running:python3 classify.py <path to model> <path to image>
sample command: python3 classify.py ../model/15.model.epoch ../download.jpeg

For the picture of the dandelion(download.jpeg), we see this on the screen

tensor([[-1.8133,  4.9129, -1.0326, -1.7789, -1.7550]],
       grad_fn=<AddmmBackward>)
Predicted class: dandelion
