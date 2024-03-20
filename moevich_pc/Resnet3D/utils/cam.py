import numpy as np
import torch
import cv2

class GradCAM():
    def __init__(self, model, use_cuda):
        self.model = model.eval()
        self.use_cuda = use_cuda
        self.feature_map = 0
        self.grad = 0
        if self.use_cuda==True:
            self.model = self.model.cuda()
        #print(self.model)
        #module = self.model.conv6
        module = self.model.sum
        #print(module)

        module.register_forward_hook(self.save_feature_map)
        module.register_backward_hook(self.save_grad)

    
    def save_feature_map(self, module, input, output):
        self.feature_map =  output.detach()
        
    def save_grad(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def save_gradient(self, grad):
        self.gradients = grad
        
    def __call__(self, x, index=None):
        x = x.cuda()
        output, h1 = self.model(x)
        #print(output)

        if index == None:
            index = np.argmax(output.cpu().detach().numpy())

        # backword
        one_hot = np.zeros((1, 2), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad_()
        one_hot = torch.sum(one_hot.cuda() * output)
        self.model.zero_grad()
        one_hot.backward()

        # grad 
        self.feature_map = self.feature_map[0].cpu().numpy()
        self.grad = self.grad[0].cpu().numpy()
        feature_map_F = np.transpose(self.feature_map, (1,0,2,3))
        grad_F = np.transpose(self.grad, (1,0,2,3))

        #print(grad_F.shape)

        # image plot
        cv_tensor = np.zeros((grad_F.shape[0],grad_F.shape[2],grad_F.shape[3])).astype(np.float32)
        heatmap = np.zeros((grad_F.shape[0], 256, 256, 3)).astype(np.float32)
        for i in range(grad_F.shape[0]):
            weights = np.mean(grad_F[i], axis=(1,2))
            cam = np.tensordot(weights, feature_map_F[i], axes=(0,0))
            cv_tensor[i] = cam
        for j in range(grad_F.shape[0]):
            cam1 = (cv_tensor[j]>0)*cv_tensor[j] / cv_tensor.max()
            cam1 = cam1*255.0
            cam1 = cv2.resize(np.uint8(cam1), (256, 256), cv2.INTER_LANCZOS4)
            heatmap[j] = cv2.applyColorMap(cam1, cv2.COLORMAP_JET)

        return heatmap, output