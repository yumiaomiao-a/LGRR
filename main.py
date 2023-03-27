#_*_coding:utf-8 _*_
import numpy
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from torchvision import datasets, transforms
import os           
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from sklearn.metrics import roc_auc_score
from regionlabel import region_label,fakelabel_new,reallabel_new,fakelabel_new_30,reallabel_new_30,region_label
import matplotlib.pyplot as plt
from regionlabel import reallabel_new_120,reallabel_new_60,fakelabel_new_120,fakelabel_new_60,reallabel_new_15,fakelabel_new_15
from sklearn import metrics
from FeatureMap2Heatmap import featuremap2heatmap
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')


# read data
data_dir = './ffdfc23'


data_transform = {
    'train':transforms.Compose([
        transforms.Scale([240,240]),
        # transforms.ColorJitter(hue=.05,saturation= .05),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        # transforms.RandomErasing(p=0.5,scale=(0.02,0.33),ratio=(0.3,0.3),value=0,inplace=False)
    ]),
    'val':transforms.Compose([
        transforms.Scale([240,240]),
        transforms.ToTensor(),
        # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}


image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_dir,x),
                                         transform = data_transform[x]) for x in ['train', 'val']}

train_set = image_datasets['train']
val_set = image_datasets['val']

batch_size =16
dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                            batch_size = 16,
                                            shuffle = True,drop_last=True) for x in ['train','val'] } 


train_dataloader = dataloader['train']
val_dataloader = dataloader ['val']

dataset_size = {x:len(image_datasets[x]) for x in ['train','val']}


from Model import CDCN_my
model = CDCN_my


###load the pretrained model
# dic = torch.load('./save_model/df_0.1.pth')  #the best para
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k,v in dic.items():
#     name = k[7:]
#     new_state_dict[name] = v
# model.load_state_dict(new_state_dict)


def adjust_learning_rate(epoch):
    lr = 0.00002
    if epoch > 5:
        lr = lr / 10
    elif epoch > 10:
        lr = lr / 100
    elif epoch > 15:
        lr = lr / 1000

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


######## define loss function        
loss_f = torch.nn.CrossEntropyLoss()
l1loss = nn.L1Loss()
mseloss = nn.MSELoss()
smoothl1 = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)

#trained with two GPUs in this paper
model = torch.nn.DataParallel(model,device_ids=[0,1]) 
model = model.cuda()

epoch_n = 20


# save root of the trained model
def save_models(epoch):
    torch.save(model.state_dict(), "./save_model/model_{}.pth".format(epoch))
print("Chekcpoint saved")


class ConsistencyCos(nn.Module):
    def __init__(self):
        super(ConsistencyCos, self).__init__()
        self.mse_fn = nn.MSELoss()
    def forward(self, feat):
        feat = nn.functional.normalize(feat, dim=1)
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1)
        labels = torch.ones((cos.shape[0],1), dtype=torch.float, requires_grad=False)
        if torch.cuda.is_available():
            labels = labels.cuda()
        loss = self.mse_fn(cos, labels)
        return loss

ConsistencyCos = ConsistencyCos()



#########################evaluation
def val():
    model.eval()
    val_acc = 0.0
    prob_all = []
    label_all = []
    prob_all_soft = []
    for i, (images, labels) in enumerate(val_dataloader):
        
        images = images.cuda()
        labels1 = labels.cuda()
        labels = labels.numpy().astype(np.float)

        with torch.no_grad():
            x_consis,x_,w_res_c,w_f_c,w_res_c4,w_f_c4,x_res_4,x_f_4,x_construct_30,x_construct_120,x_construct_60,x_construct_15,y_construct_30,y_construct_120,y_construct_60,y_construct_15,x_res,x_Block1,x_Block2,x_Block3,x_Block4,x_org_f,x_Block11,x_Block22,x_Block33,x_Block44= model(images)

        pred = w_f_c4 + w_res_c4

        _, prediction = torch.max(pred.data, 1)

        val_acc += torch.sum(prediction == labels1.data)

        pred = torch.sigmoid(pred).cpu()

        w_res_c = torch.max(pred,1)[0].cpu().numpy()
        w_res_cc = torch.max(pred,1)[1].cpu().numpy()
       
        out_pred = w_res_cc
        
        prob_all.extend(out_pred)
        label_all.extend(labels)
        prob_all_soft.extend(pred)


    label_all = np.array(label_all).astype(np.int64)
    prob_all = np.array(prob_all).astype(np.int64)
    
    acc_valid = metrics.accuracy_score(label_all, prob_all)
    label_all = label_all.astype(np.float32)

    prob_all_soft = torch.tensor([item.cpu().detach().numpy() for item in prob_all_soft])

    auc_valid = metrics.roc_auc_score(label_all, 1-prob_all_soft[:,0])

    val_acc_org = val_acc / len(val_set)

    return auc_valid,val_acc_org


##### split tensor
def window_partition(x,window_size):
    '''
    Args:
        param x: (B,H,W,C)
        param window_size(int): window size
    return:
        windows:(num_windows*B,window_size,window_size,C)
    '''
    x = x.permute(0,2,3,1)
    # print('//+++++++++++',x.shape)
    B,H,W,C = x.shape
    x = x.view(B,H // window_size, window_size, W // window_size, window_size,C)
    # print('+++++++++++',x.shape)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(B,-1,1,(window_size*window_size*C))
    # print('&&&&&+++++++++++',windows.shape)
    return windows


#######################training
def train(num_epochs):
    best_acc = 0.0
    best_auc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        print(train_dataloader)
        for i, (images, labels) in enumerate(train_dataloader):

            i+=1
            
            images = images.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()

           
            ########################## create binary label 
            binary_label=[] 
            for j in range(len(images)):
                if labels[j]==1: 
                    l = reallabel_new()
                    binary_label.append(l) 
                else:
                    
                    l = fakelabel_new()
                    binary_label.append(l)

            binary_label = torch.tensor(binary_label)
           
            # model outputs
            x_consis,x_,w_res_c,w_f_c,w_res_c4,w_f_c4,x_res_4,x_f_4,x_construct_30,x_construct_120,x_construct_60,x_construct_15,y_construct_30,y_construct_120,y_construct_60,y_construct_15,x_res,x_Block1,x_Block2,x_Block3,x_Block4,x_org_f,x_Block11,x_Block22,x_Block33,x_Block44= model(images)

            w_res = window_partition(x_res_4,5) # window size=5
            w_f = window_partition(x_f_4,5) # window size=5
            w_binary_label = window_partition(binary_label,80) # window size=80

            w_res_patch = window_partition(x_res_4,15) # window size=15
            w_f_patch = window_partition(x_f_4,15) # window size=15
            

            w_res_60 = window_partition(x_Block2,20) # window size=20
            w_f_60 = window_partition(x_Block22,20) # window size=20
            # print('----------',w_res.shape,w_res_60.shape)
            


            ####### computer w_res inter cosine distance
            kkk=[]
            for m in range(len(w_res)):
                kk=[]
                # print('%%%',w_res.shape)
                for n in range(len(w_res[m])):
                    for v in range(len(w_res[m])):
                        # print('$$$$%%%',w_res[m][n].shape)
                        rr = cosine_similarity(w_res[m][n].cpu().detach().numpy(),w_res[m][v].cpu().detach().numpy())
                        # rr = (rr+1)/2
                        kk.append(rr)
                kkk.append(kk)
                res_cosdis = torch.tensor(kkk)
                res_cosdis = res_cosdis.view(-1,81)
                # print('^^^^^^^^^^',res_cosdis)
            res_cosdis = res_cosdis.cuda()


            ####### computer w_f inter cosine distance
            lll=[]
            for m in range(len(w_f)):
                ll=[]
                for n in range(len(w_f[m])):
                    for v in range(len(w_f[m])):
                        rr = cosine_similarity(w_f[m][n].cpu().detach().numpy(),w_f[m][v].cpu().detach().numpy())
                        # rr = (rr+1)/2
                        # print('___________',rr)
                        ll.append(rr)
                lll.append(ll)
                f_cosdis = torch.tensor(lll)
                f_cosdis = f_cosdis.view(-1,81)
            # print('$$$$$$$^^^^^^^^^^',f_cosdis)
            f_cosdis = f_cosdis.cuda()
            # f_cosdis = f_cosdis.to(device3)


            ########## Calculate the possible forgery rate for each patch in binary_label.
            # print('$$$$$$$^^^^^^^^^^',a.shape)
            ddd=[]
            for m in range(len(w_binary_label)):
                dd = []
                for n in range(len(w_binary_label[m])):   #n=[0-8]
                    w = w_binary_label[m][n].cpu().detach().numpy()
                    w = numpy.transpose(w)
                    n = int(len(w))
                    one1 = numpy.sum(w)
                    # print('_____________________________',one1)
                    # one1 = one1.item()
                    one1 = int(one1/255)
                    p1 = one1 / n

                    for v in range(len(w_binary_label[m])):
                        ww = w_binary_label[m][v].cpu().detach().numpy()
                        ww = numpy.transpose(ww)
                        nn = int(len(ww))
                        one2 = numpy.sum(ww)
                        # one2 = one2.item()
                        one2 = int(one2/255)
                        p2 = one2 / nn

                        s = 1-(p1-p2)*(p1-p2)
                        # print('&&&&&&&&&&&&&&&&',p1,p2,s)
                        dd.append(s)

                ddd.append(dd)
                # print('%%%%$$$$$%%%%%%%',ddd)

            dislabels = torch.tensor(ddd)
            dislabels = dislabels.view(-1,81).cuda()
           


            #Calculate the cosine distance of each patch of the two images, between [0,1], 9 numbers, [32,9], label==1
            sss = []
            for k in range(len(w_res)): #取一张输入图 [9,6400]patch
                # print('##########',w_res.shape,w_res[k].shape)
                ss = []
                for h in range(len(w_res[k])):  #取9个feature里的一个feature [1.6400]
                    # print('**********',w_res[k][h].shape)
                    s = cosine_similarity(w_res[k][h].cpu().detach().numpy(),w_f[k][h].cpu().detach().numpy())
                    # print(s)
                    # ss = (ss+1)/2
                    ss.append(s)
                sss.append(ss)
                cosdis1 = torch.tensor(sss)
                # print('-----------',cosdis1)
                cosdis2 = cosdis1.view(-1,9)

            dim = 9
            batchsize,z,zz,zzz=images.shape
            cos_lables = torch.ones(batchsize,dim)  #两个矩阵对应元素的余弦距离应该接近1，所以label应该都是1
            # print('--------------,cosdis2.shape, cos_lables.shape)
            cos_lables = cos_lables.cuda()




            ######################### define loss items
            # GSM loss
            los1 = ConsistencyCos(x_consis).cuda()
            
            # classification loss
            loss22 = loss_f(w_f_c4,labels).cuda()
            loss2 = loss_f(w_res_c4, labels).cuda()
            los2 = loss22+loss2


            #IPSM loss
            cosdis2 = cosdis2.cuda()
            #cosdis4 = cosdis4.cuda()
            loss3 = mseloss(cosdis2, cos_lables).cuda()
            #loss33 = mseloss(cosdis4, cos_lables).cuda()


            # total loss
            loss = los2 + 0.1*loss3 + 0.1+los1

            

            torch.backends.cudnn.enabled = False
            loss.backward()


            optimizer.step()
            # scheduler.step()

            train_loss += loss.cpu().item() * images.size(0)
            # _, prediction = torch.max(outputs.data, 1)

            # LGRR-P
            w_res_c = w_f_c4 + w_res_c4
            # print('------------------',w_f_c)
            # print('==================',w_res_c)

            _, prediction = torch.max(w_res_c.data, 1)
           

            train_acc += torch.sum(prediction == labels.data)

            
            batch_size,m,mm,mmm = images.shape

            if i% 5 == 0:
                batch_loss = train_loss / (batch_size*i)
                batch_acc = train_acc / (batch_size*i)

                print('Epoch[{}] batch[{}],Loss:{:.4f},Acc:{:.4f}'.format( epoch, i, batch_loss, batch_acc))

            # visualization every 500 step
            if i% 500 == 0:
                featuremap2heatmap(images, x_res,x_org_f,x_Block1, x_Block2, x_Block3, x_Block4,x_Block11, x_Block22, x_Block33, x_Block44)

            torch.cuda.empty_cache()

        # adjust learning rate
        adjust_learning_rate(epoch)

        # compute train acc and loss
        train_acc = train_acc / len(train_set)
        train_loss = train_loss / len(train_set)

        val_auc,val_acc= val()

        # save model
        if val_acc_org > best_acc:
            save_models(epoch)
            best_acc = val_acc_org

        if val_auc > best_auc:
            best_auc = val_auc

        # print results
        print(
            "Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {},Best Acc:{},Test AUC:{}, Best AUC:{}".format(
                epoch, train_acc, train_loss, val_acc_org, best_acc, val_auc, best_auc))
            
           



if __name__ == '__main__':
    train(20)
   
