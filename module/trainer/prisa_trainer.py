from module.trainer.trainer_base import trainer_base
from torch.optim import Adam,SGD
from module.metrics.mosei_metrics import mosei_metrics,sims_metrics,urfunny_metrics
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from runx.logx import logx
import math
from module.loss.loss_base import gca_uot,hs_rince,gca_uot_labelaware,NCELoss
import sys
import matplotlib
# [关键修改] 必须在 import pyplot 之前设置后端为 'Agg'
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
class prisa_trainer(trainer_base):
    def visualize_tsne(self, epoch, all_features, all_labels, phase='test'):
        """
        修改版：不再运行模型，直接接收已经提取好的特征和标签进行画图
        """
        print(f"正在生成 Epoch {epoch} 的 t-SNE 图 (基于已提取特征)...")
        
        # 1. 检查数据是否为空
        if len(all_features) == 0:
            print("⚠️ 警告: 特征列表为空，跳过 t-SNE。")
            return

        # 2. 拼接数据
        try:
            # 如果传入的是 list of numpy arrays
            if isinstance(all_features[0], np.ndarray):
                all_features = np.concatenate(all_features, axis=0)
                all_labels = np.concatenate(all_labels, axis=0).squeeze()
            else:
                # 兼容性处理
                all_features = np.array(all_features)
                all_labels = np.array(all_labels)
        except Exception as e:
            print(f"⚠️ 数据拼接失败: {e}")
            return

        # 3. NaN 检查
        if np.isnan(all_features).any() or np.isinf(all_features).any():
            print(f"⚠️ Epoch {epoch}: 特征包含 NaN/Inf，跳过绘图。")
            return

        # 4. 采样 (防止数据过多卡死)
        if all_features.shape[0] > 2000:
            rng = np.random.RandomState(42) 
            indices = rng.choice(all_features.shape[0], 2000, replace=False)
            all_features = all_features[indices]
            all_labels = all_labels[indices]
            
        print("正在运行 t-SNE 降维...")
        
        # 5. 绘图
        try:
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
            tsne_results = tsne.fit_transform(all_features)
            
            plt.figure(figsize=(10, 8))
            df_subset = pd.DataFrame()
            df_subset['tsne-2d-one'] = tsne_results[:,0]
            df_subset['tsne-2d-two'] = tsne_results[:,1]
            df_subset['Sentiment'] = all_labels
            
            scatter = plt.scatter(
                x=df_subset['tsne-2d-one'], 
                y=df_subset['tsne-2d-two'], 
                c=df_subset['Sentiment'], 
                cmap='coolwarm', 
                alpha=0.6, 
                s=20
            )
            
            plt.colorbar(scatter, label='Sentiment Intensity')
            plt.title(f't-SNE visualization of {self.cfg.dataset_name} (Epoch {epoch})')
            plt.xlabel('Dim 1')
            plt.ylabel('Dim 2')
            
            save_name = os.path.join(self.save_model_path, f'tsne_{phase}_epoch_{epoch}.png')
            plt.savefig(save_name, dpi=300)
            print(f"t-SNE 图已保存至: {save_name}")
            plt.close()
        except Exception as e:
            print(f"⚠️ 画图过程报错: {e}")
            plt.close()
    def __init__(self,cfg):
        super(prisa_trainer,self).__init__(cfg)
        self.target_class_idx = 0
        self.modality_weight = {'video':1,'audio':1,'text':1,'fusion':1}
    def init_metrics(self):
        if(self.cfg.dataset_name == 'sims'):
            self.metrics = sims_metrics
        elif(self.cfg.dataset_name == 'urfunny'):
            self.metrics = urfunny_metrics
        else:
            self.metrics = mosei_metrics
    def init_optimizer(self):
        self.optimizer_dict = {'adam':Adam,'sgd':SGD}
        if(self.cfg.using_bert):
            bert_params = list(map(id,self.model.t_model.parameters()))
            vt_params = list(map(id,self.model.vt_layers.parameters()))
            at_params = list(map(id,self.model.at_layers.parameters()))
            base_params = filter(lambda p: id(p) not in (bert_params+vt_params+at_params),self.model.parameters())
            params_group = [{'params': base_params},
                            {'params': self.model.t_model.parameters(),'lr': self.cfg.text_lr},
                            {'params': self.model.vt_layers.parameters(),'lr': self.cfg.video_lr},
                            {'params': self.model.at_layers.parameters(),'lr': self.cfg.audio_lr}]
            self.optimizer = self.optimizer_dict[self.cfg.optimizer](params=params_group,lr=self.cfg.base_lr)
        else:
            self.optimizer = self.optimizer_dict[self.cfg.optimizer](params=self.model.parameters(),lr=self.cfg.base_lr)
    def init_schedule(self):
        warm_up_epochs = 5
        max_num_epochs = self.cfg.epochs
        lr_milestones = [20,50]
        warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs else 0.1**len([m for m in lr_milestones if m <= epoch])
        warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
        else 0.5 * ( math.cos((epoch - warm_up_epochs) /(max_num_epochs - warm_up_epochs) * math.pi) + 1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=warm_up_with_multistep_lr)
    def auto_to_gpu(self,input_dict):
        output_dict = {}
        if(self.use_cuda):
            for key in input_dict.keys():
                if('cpu' not in key):
                    output_dict[key] = input_dict[key].cuda()
                else:
                    output_dict[key] = input_dict[key]
            return output_dict
        else:
            return input_dict
    def init_criterion(self):
        self.criterion_list = {'MSELoss':nn.MSELoss(),'CrossEntropyLoss':nn.CrossEntropyLoss(),'BCELoss':nn.BCELoss(),'L1Loss':nn.L1Loss(),'SmoothL1Loss':nn.SmoothL1Loss()}
        self.criterion = self.criterion_list[self.cfg.criterion]
        # self.cl_criterion = NCELoss(temperature=self.cfg.T,d=self.cfg.d)
        self.cl_criterion = gca_uot_labelaware(
            batch_size=self.cfg.batch_size,
            epsilon=self.cfg.epsilon,
            iterations=10,
            lam=0.01,
            q=0.6,
            relax_items1=1e-4,
            relax_items2=1e-4,
            r1=1.0,
            r2=0.5,
            alpha=1.0,
            beta=0.1,
            sigma=0.5,
            d=None,
            same_polarity=True,
        )
        # self.cl_criterion = gca_uot(batch_size=self.cfg.batch_size, epsilon=self.cfg.epsilon)
        # self.cl_criterion = hs_rince(
        #     batch_size=self.cfg.batch_size, 
        #     epsilon=self.cfg.epsilon
        # )
    def label_conversion(self,label):
        target = label>0
        target = target.long()
        return target
    def directly_save_model(self,epoch):
        self.model.eval()
        if(self.best_model_name is not None):
            os.remove(os.path.join(self.save_model_path,self.best_model_name)) 
        self.best_model_name = f'best_model.pt'
        self.save_model_path_epoch = os.path.join(self.save_model_path,self.best_model_name)
        torch.save(self.model.module.state_dict(),self.save_model_path_epoch)
    def pretrain(self):
        for epoch in range(self.cfg.pretrain_epochs):
            logx.msg('='*30+str(epoch)+'='*30)
            train_loss = self.pretrain_one_epoch(epoch)
            self.lr_scheduler.step()
        self.init_optimizer()
        self.init_schedule()
    def pretrain_one_epoch(self,epoch):
        phase = 'train'
        self.model.train()
        input_dict = {}
        y_true, y_pred = {}, {}
        losses = []
        c_loss =[]
        msg_string = ''
        for batch in tqdm(self.train_loader):
            self.model.zero_grad()
            input_dict = self.batch2dict(batch)
            input_dict = self.auto_to_gpu(input_dict)
            score_dict,state_dict = self.model(input_dict)
            loss  = state_dict['loss']
            loss.backward()
            self.optimizer.step()
            c_loss.append(state_dict['loss'].item())
        c_loss_mean = np.mean(c_loss)
        msg_string = f'c_loss={c_loss_mean:.3f} '
        logx.msg(msg_string)
        return c_loss_mean
    def train_one_epoch(self,epoch):
        phase = 'train'
        self.model.train()
        input_dict = {}
        y_true, y_pred = {}, {}
        losses = []
        a_at_losses =[]
        v_vt_losses = []
        at_vt_losses = []
        msg_string = ''
        for batch in self.train_loader:
            self.model.zero_grad()
            input_dict = self.batch2dict(batch)
            input_dict = self.auto_to_gpu(input_dict)
            score_dict,state_dict = self.model(input_dict)
            loss = None
            for key in self.cfg.modality:
                if(loss is None):
                    loss = self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                else:
                    loss += self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
            task_loss = loss.item()
            v_private_loss = self.criterion(score_dict['video_private'],input_dict['label'])
            a_private_loss = self.criterion(score_dict['audio_private'],input_dict['label'])
            loss += v_private_loss+a_private_loss
            if 'dec_loss' in state_dict:
                loss += self.cfg.dec_lambda * state_dict['dec_loss']

            target = input_dict['label'].view(-1)

            def prepare_interleaved_tensor(f1, f2):
                batch_size = f1.shape[0]
                feature_dim = f1.shape[1]
                z = torch.empty((2 * batch_size, feature_dim), dtype=f1.dtype, device=f1.device)
                z[0::2] = f1
                z[1::2] = f2
                return z

            # v_vt_z = prepare_interleaved_tensor(state_dict['v_private'], state_dict['vt_fusion'])
            # a_at_z = prepare_interleaved_tensor(state_dict['a_private'], state_dict['at_fusion'])
            # at_vt_z = prepare_interleaved_tensor(state_dict['vt_fusion'], state_dict['at_fusion'])

            # v_vt_loss = self.cfg.alpha * self.cl_criterion(v_vt_z)
            # a_at_loss = self.cfg.alpha * self.cl_criterion(a_at_z)
            # at_vt_loss = self.cfg.beta * self.cl_criterion(at_vt_z)
            v_vt_loss = self.cfg.alpha * (self.cl_criterion(state_dict['v_private'],state_dict['vt_fusion'],target) + self.cl_criterion(state_dict['vt_fusion'],state_dict['v_private'],target))
            a_at_loss = self.cfg.alpha * (self.cl_criterion(state_dict['a_private'],state_dict['at_fusion'],target) + self.cl_criterion(state_dict['at_fusion'],state_dict['a_private'],target))
            at_vt_loss = self.cfg.beta * (self.cl_criterion(state_dict['vt_fusion'],state_dict['at_fusion'],target) + self.cl_criterion(state_dict['at_fusion'],state_dict['vt_fusion'],target))

            if(self.cfg.contrastive_learning):
                loss += at_vt_loss
            for key in self.cfg.modality:
                if(key not in y_pred.keys()):
                    y_pred[key] = []
                    y_true[key] = []
                if(self.cfg.dataset_name == 'urfunny'):
                    y_pred[key].append(score_dict[key].argmax(dim=1).detach().cpu().numpy())
                else:
                    y_pred[key].append(score_dict[key].detach().cpu().numpy())
                y_true[key].append(input_dict['label'].detach().cpu().numpy())
            loss.backward()
            self.optimizer.step()
            losses.append(task_loss)
            a_at_losses.append(a_at_loss.item())
            v_vt_losses.append(v_vt_loss.item())
            at_vt_losses.append(at_vt_loss.item())
        loss_mean = np.mean(losses) 
        a_at_mean = np.mean(a_at_losses) 
        v_vt_mean = np.mean(v_vt_losses)
        at_vt_mean = np.mean(at_vt_losses)
        for key in y_true.keys():
            y_true[key] = np.concatenate(y_true[key], axis=0).squeeze()
            y_pred[key] = np.concatenate(y_pred[key], axis=0).squeeze()
        if(self.cfg.dataset_name == 'urfunny'):
            target_names = None
        else:
            target_names = list(self.valid_dataset.e_label2id.keys())
        metric_dict = {} 
        for key in y_true.keys():
            metric_dict.update(self.metrics(y_true[key],y_pred[key],target_names=target_names,key_head=phase+'/'+key)) 
        msg_string = f'train loss={loss_mean:.3f} a_at_loss={a_at_mean} v_vt_loss={v_vt_mean} at_vt_loss={at_vt_mean} '
        for key in y_pred.keys():
            m = metric_dict[phase+'/'+key+'/'+'mae']
            msg_string += f'{key}:{m:.3f} '
            m = metric_dict[phase+'/'+key+'/'+self.cfg.report_metric]
            msg_string += f'{m:.3f} '
        logx.msg(msg_string)
        
        return loss_mean,metric_dict
    def eval_one_epoch(self,epoch):
        phase = 'eval'
        self.model.eval()
        input_dict = {}
        y_true, y_pred = {}, {}
        losses = []
        msg_string = ''
        with torch.no_grad():
            for batch in self.valid_loader:
                input_dict = self.batch2dict(batch)
                input_dict = self.auto_to_gpu(input_dict)
                score_dict,state_dict = self.model(input_dict)
                loss = None
                for key in self.cfg.modality:
                    if(loss is None):
                        loss = self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                    else:
                        loss += self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                losses.append(loss.item())
                for key in self.cfg.modality:
                    if(key not in y_pred.keys()):
                        y_pred[key] = []
                        y_true[key] = []
                    if(self.cfg.dataset_name == 'urfunny'):
                        y_pred[key].append(score_dict[key].argmax(dim=1).detach().cpu().numpy())
                    else:
                        y_pred[key].append(score_dict[key].detach().cpu().numpy())
                    y_true[key].append(input_dict['label'].detach().cpu().numpy())
        eval_loss = np.mean(losses)
        for key in y_true.keys():
            y_true[key] = np.concatenate(y_true[key], axis=0).squeeze()
            y_pred[key] = np.concatenate(y_pred[key], axis=0).squeeze()
        if(self.cfg.dataset_name == 'urfunny'):
            target_names = None
        else:
            target_names = list(self.valid_dataset.e_label2id.keys())
        metric_dict = {}
        for key in y_true.keys():
            metric_dict.update(self.metrics(y_true[key],y_pred[key],target_names=target_names,key_head=phase+'/'+key)) 
        msg_string = f'eval_loss={eval_loss:.3f} '
        for key in y_pred.keys():
            m = metric_dict[phase+'/'+key+'/'+'mae']
            msg_string += f'{key}:{m:.3f} '
            m = metric_dict[phase+'/'+key+'/'+self.cfg.report_metric]
            msg_string += f'{m:.3f} '
        logx.msg(msg_string)
        return eval_loss,metric_dict
    def test_one_epoch(self, epoch):
        phase = 'test'
        self.model.eval()
        input_dict = {}
        y_true, y_pred = {}, {}
        losses = []
        msg_string = ''
        
        # ================= [新增] 1. 初始化特征收集列表 =================
        tsne_features_list = []
        tsne_labels_list = []
        # ================================================================

        with torch.no_grad():
            for batch in self.test_loader:
                input_dict = self.batch2dict(batch)
                input_dict = self.auto_to_gpu(input_dict)
                score_dict, state_dict = self.model(input_dict)
                
                # 计算 Loss
                loss = None
                for key in self.cfg.modality:
                    if(loss is None):
                        loss = self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                    else:
                        loss += self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                losses.append(loss.item())
                
                # 收集预测结果用于计算指标
                for key in self.cfg.modality:
                    if(key not in y_pred.keys()):
                        y_pred[key] = []
                        y_true[key] = []
                    if(self.cfg.dataset_name == 'urfunny'):
                        y_pred[key].append(score_dict[key].argmax(dim=1).detach().cpu().numpy())
                    else:
                        y_pred[key].append(score_dict[key].detach().cpu().numpy())
                    y_true[key].append(input_dict['label'].detach().cpu().numpy())

                # ================= [新增] 2. 收集 t-SNE 特征 (边跑边存) =================
                # 优先从 state_dict 获取融合特征
                if 'fusion' in state_dict:
                    feat = state_dict['fusion']
                elif 'fusion_output' in state_dict:
                    feat = state_dict['fusion_output']
                else:
                    # 如果 state_dict 没有，尝试从 score_dict 获取
                    feat = score_dict['fusion']
                
                # 转为 numpy 并存入列表 (不占用 GPU 显存)
                tsne_features_list.append(feat.detach().cpu().numpy())
                tsne_labels_list.append(input_dict['label'].detach().cpu().numpy())
                # ======================================================================

        # 计算平均 Loss
        test_loss = np.mean(losses)
        
        # 拼接预测结果
        for key in y_true.keys():
            y_true[key] = np.concatenate(y_true[key], axis=0).squeeze()
            y_pred[key] = np.concatenate(y_pred[key], axis=0).squeeze()
            
        if(self.cfg.dataset_name == 'urfunny'):
            target_names = None
        else:
            target_names = list(self.valid_dataset.e_label2id.keys())
            
        metric_dict = {}
        
        # 处理 Logx Metric (保留你原代码逻辑)
        # 注意：通常 best_eval_epoch 是在 eval 阶段更新的，test 阶段可能为 None 或旧值，加个 getattr 保护
        best_epoch = getattr(self, 'best_eval_epoch', -1)
        if(epoch-1 == best_epoch):
            if hasattr(self, 'logx_metric_dict'):
                logx.metric(phase='val', metrics=self.logx_metric_dict, epoch=best_epoch)
        
        self.logx_metric_dict = {}
        for key in y_true.keys():
            metric = self.metrics(y_true[key], y_pred[key], target_names=target_names, key_head=phase+'/'+key)
            metric_dict.update(metric)
            if('fusion' in key):
                for k in metric.keys():
                    self.logx_metric_dict[k.split('/')[-1]] = metric[k]
                    
        msg_string = f'test_loss={test_loss:.3f} '
        for key in y_pred.keys():
            m = metric_dict[phase+'/'+key+'/'+'mae']
            msg_string += f'{key}:{m:.3f} '
            m = metric_dict[phase+'/'+key+'/'+self.cfg.report_metric]
            msg_string += f'{m:.3f} '
        logx.msg(msg_string)

        # ================= [新增] 3. 调用 t-SNE 画图 =================
        # 设置画图条件：最后几轮，或者每隔 10 轮
        is_last_epoch = (epoch == self.cfg.epochs - 1)
        # 如果当前是最佳 epoch (根据 eval 结果)
        is_best_epoch = (epoch-1 == best_epoch)
        
        if is_last_epoch or is_best_epoch or (epoch % 10 == 0):
            # 直接传入收集好的列表，不需要再运行模型
            self.visualize_tsne(epoch, tsne_features_list, tsne_labels_list, phase='test')
        # ============================================================

        return test_loss, metric_dict
