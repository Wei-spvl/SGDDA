from copy import deepcopy

import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class EMATeacher(nn.Module):

    def __init__(self, model,classifier, alpha, pseudo_label_weight):
        super(EMATeacher, self).__init__()
        self.ema_model = deepcopy(model) # 创建学生模型的深拷贝作为EMA模型
        self.f=deepcopy(classifier) # 创建分类器的深拷贝
        self.alpha = alpha # EMA衰减率
        self.pseudo_label_weight =  pseudo_label_weight # 伪标签的权重
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None # 将字符串'None'转换为实际的None

    def _init_ema_weights(self, model,classifier):
        # ema model
        for param in self.ema_model.parameters():
            param.detach_()   ## 将EMA模型的参数从计算图中分离
        mp = list(model.parameters()) ## 将主模型的参数转换为列表
        mcp = list(self.ema_model.parameters())  ## 将EMA模型的参数转换为列表
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor    # 如果是标量张量
                mcp[i].data = mp[i].data.clone()          #对于标量参数，我们直接克隆主模型的参数到EMA模型。
            else:
                mcp[i].data[:] = mp[i].data[:].clone()   #对于非标量参数（即具有形状的张量），我们同样克隆主模型的参数到EMA模型，
        # f
        #初始化分类器的的EMA权重
        for param in self.f.parameters():
            param.detach_()
        mp = list(classifier.parameters())
        mcp = list(self.f.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model,classifier, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)   #计算当前的EMA衰减率
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            if not param.data.shape:  # scalar tensor 如果参数是标量（即没有形状的张量），则直接更新EMA模型的参数。
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:  # 如果参数是非标量（即具有形状的张量），则需要在原地（in-place）更新EMA模型的参数。
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
        for ema_param, param in zip(self.f.parameters(),classifier.parameters()): #更新分类器的参数
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, model,classifier, iter):
        # Init/update ema model
        if iter == 0:   #如果这是第一次迭代（iter == 0），则调用_init_ema_weights方法来初始化EMA模型的权重。
            self._init_ema_weights(model,classifier)
        if iter > 0:    #如果迭代次数大于0，则调用_update_ema方法来更新EMA模型的权重。
            self._update_ema(model,classifier, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False    #因为在推理模式下（即，当我们想要生成稳定的预测时），我们通常希望禁用Dropout和DropPath，因为这些技术主要用于在训练过程中防止过拟合。
        feature= self.ema_model(target_img) #将target_img（目标图像）通过EMA模型进行前向传播
        logits , _ = self.f(feature)   #分类器进行预测，并且返回logits

        ema_softmax = torch.softmax(logits.detach(), dim=1) #计算预测概率,detach()防止梯度更新
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1) #获得概率和对应的伪标签

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
        elif self.pseudo_label_weight == 'prob': #执行这一步，伪标签的权重就是概率
            pseudo_weight = pseudo_prob
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight
