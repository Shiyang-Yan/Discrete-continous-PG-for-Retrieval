import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
import numpy
from collections import OrderedDict
import torch.nn.functional as F
from GCN_lib.Rs_GCN import Rs_GCN
from pytorch_revgrad import RevGrad
from convcap2 import convcap
import opts
import misc.utils as utils
import torch.optim as optim
from conv_decoder import conv_decoder
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.distributions import Normal


def eval_func(qf, gf, q_pids, g_pids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """

    cosine = 0
    q_pids = torch.tensor(q_pids).int().cpu().numpy()
    g_pids = torch.tensor(g_pids).int().cpu().numpy()
    q_camids = np.ones(q_pids.shape[0])
    g_camids = np.zeros(q_pids.shape[0])
    m, n = qf.shape[0], gf.shape[0]
    if cosine == 0:
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
    else:
        distmat = qf.mm(gf.t())
    
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1


        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if num_rel !=0:
            AP = tmp_cmc.sum() / num_rel
        else:
            AP = 0
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    if not all_AP:
        mAP = 0
       # all_cmc = 0
    else:
     #   all_cmc = np.asarray(all_cmc).astype(np.float32)
     #   cmc_value = all_cmc[:, 0] + all_cmc[:, 4] + all_cmc[:, 9]
        mAP = np.asarray(all_AP).astype(np.float32)
        #mmAP = np.mean(mAP)
    return  mAP


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = (input).contiguous()
       # mask = (seq>-1).float()
        #mask = (torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).contiguous()
        output = - input * reward.unsqueeze(1)
        output = torch.mean(output)

        return output


def i2t(images, captions, images2, captions2, measure='cosine'):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = int(images.shape[0])
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[1 * index].reshape(1, images.shape[1])
        im_2 = images2[1 * index].reshape(1, images2.shape[1])
        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], (index + bs))
                im2 = images[index:mx]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
            d2 = numpy.dot(im_2, captions2.T).flatten()
            d = (d + d2) / 2

        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(index, index+1, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    return r1 + r5 + r10


def t2i(images, captions, images2, captions2, measure='cosine'):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = int(images.shape[0])
    ims = numpy.array([images[i] for i in range(0, len(images), 1)])

    ims2 = numpy.array([images2[i] for i in range(0, len(images2), 1)])

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query captions
        queries = captions[index:index + 1]
        queries2 = captions2[index:index + 1]
        # Compute scores
        if measure == 'order':
            bs = 100
            if 1 * index % bs == 0:
                mx = min(captions.shape[0], 1* index + bs)
                q2 = captions[1 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (1 * index) % bs:(1 * index) % bs +1].T
        else:
            d = numpy.dot(queries, ims.T)
            d2 = numpy.dot(queries2, ims2.T)
            d = (d + d2) / 2

        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[1 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[1 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    return r1 + r5 + r10

def get_grad_cos_sim(grad1, grad2):
    """Computes cos simillarity of gradients after flattening of tensors.
    """
    grad1 = torch.cat([x.data.view((-1,)).cpu() for x in grad1 if x is not None], 0).cpu()
    grad2 = torch.cat([x.data.view((-1,)).cpu() for x in grad2 if x is not None], 0).cpu()

   # grad1 = torch.tensor(grad1.data).view((-1,))
   # grad2 = torch.tensor(grad2.data).view((-1,))
    return F.cosine_similarity(grad1,grad2,dim = 0)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False, use_txt_emb=True):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        if use_txt_emb == True:
            img_enc = EncoderImagePrecompAttn(
                img_dim, embed_size, data_name, use_abs, no_imgnorm)
        else:
            img_enc = EncoderImagePrecomp(
                img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        # print(images)
        # images = images.view(images.size(0), 73728)
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)




import math
def callogprob(mu, std, action):
    p1 = -((mu - action) ** 2) /(2*std.clamp(min= 1e-3))
    p2 = - torch.log(torch.sqrt(2*math.pi * std))
    return p1 + p2


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)

    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, train, temperature = 0.8, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    b, h, d = logits.size()
    logits = logits.view(-1, d)
    if train == False:
        y = F.softmax(logits / temperature, dim = -1)
        #prob, ind = y.max(dim=-1)
    else:
        y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    if train == True:
        ind = torch.multinomial(y, 1)
        prob = y.gather(1, torch.tensor(ind))
    else:
        prob, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return ind.view(b,h,-1), prob.view(b,h,-1), y_hard.view(b,h,-1)

class EncoderImagePrecompAttn(nn.Module):

    def __init__(self, img_dim, embed_size, data_name, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecompAttn, self).__init__()
        self.head = 2
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.data_name = data_name

        self.fc = nn.Linear(2048, embed_size)
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)
        self.init_weights()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
        self.linear_rl_image_mus = nn.Linear(2048*2, 500*self.head).cuda()
        self.linear_logvars = nn.Linear(2048*2, self.head)
        # GSR

        self.rl_rnn = nn.GRUCell(embed_size, embed_size).cuda()
        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

        #if self.data_name == 'f30k_precomp':
        self.bn = nn.BatchNorm1d(embed_size)
        self.bn.momentum = 0.01

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, train):
        """Extract image feature vectors."""
        aspace = 500
        fc_img_emd = self.fc(images)
        if self.data_name != 'f30k_precomp':
            fc_img_emd = l2norm(fc_img_emd)

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        hx_image = torch.zeros(images.size(0), self.embed_size).cuda()

        action_img_all = []

        sampled_log_img = []

        image_all = []
        img_att_all = []
        log_all = []
        for i in range(GCN_img_emd.size(1)):
            if i <= GCN_img_emd.size(1) - 1:
                img = GCN_img_emd[:, i, :]
            hx_image = self.rl_rnn(img, hx_image)
            img_mu = self.linear_rl_image_mus(F.relu(torch.cat([img, hx_image], 1)))
            logvar = F.softplus(self.linear_logvars(F.relu(torch.cat([img, hx_image], 1))))
            img_mu = img_mu.view(-1, self.head, aspace)
            ind, prob, out = gumbel_softmax(img_mu, train)
            prob = prob.squeeze()
            action_space = torch.tensor([i for i in range(0, aspace)]).cuda()
            action_space = action_space.unsqueeze(0).unsqueeze(1).expand(GCN_img_emd.size(0), self.head, aspace).detach()

            action_img = (action_space * out.view(GCN_img_emd.size(0), self.head, aspace)).sum(2).view(GCN_img_emd.size(0), -1)


            action_img = action_img.float()/aspace
            action_img_all += [action_img]
            sampled_log_img += [torch.log(prob)]
            std = torch.exp(0.5*logvar).view(GCN_img_emd.size(0), self.head, -1)
            if  train == True:
                #img_att = eps * std + ((action_img).view(-1, 1).float())# continues
                action_img = action_img.view(action_img.size(0), self.head, -1)
                img_att = []
                logprob = []
                for h in range(self.head):
                    dist = Normal(action_img[:,h,:], std[:,h,:])
                    img_att.append(dist.sample())
                    logprob.append(dist.log_prob(dist.sample()))
                logprob = torch.stack(logprob, 1)
                log_all.append(logprob)
                img_att = torch.stack(img_att, 2)
                img_att = img_att.view(-1, self.head, 1)
            else:
                img_att = action_img.view(-1, self.head, 1)


            img_att = 20*F.sigmoid(img_att)
            img_att_all.append(img_att)

            img = img.view(img.size(0), self.head, -1) * img_att.view(img_att.size(0), self.head,
                                                                           1)  ####use the hidden neurons
            img = img.view(img.size(0), -1)
            image_all.append(img)

        if train == True:
            log_all = torch.stack(log_all, 2)
            log_all = log_all.view(GCN_img_emd.size(0), self.head, -1)
        else:
            log_all = 1.
        sampled_log_img_all = torch.stack(sampled_log_img, 1).squeeze()
        sampled_log_img_all = sampled_log_img_all.view(GCN_img_emd.size(0), self.head, -1)
        action_imgs_all = torch.stack(action_img_all, 1).view(GCN_img_emd.size(0), self.head, -1)
        img_emb = torch.stack(image_all, 2)
        rnn_img, hidden_state = self.img_rnn(img_emb.transpose(2,1))
        features = img_emb.mean(2) + hidden_state.squeeze()
        if train == False:
            self.bn.training = True
        features = self.bn(features)
        if self.use_abs:
            features = torch.abs(features)

        return features, GCN_img_emd, sampled_log_img_all, action_imgs_all, log_all

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompAttn, self).load_state_dict(new_state)

from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.word_to_vector import GloVe
# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.head = 2
        # word embedding

        self.embed = nn.Embedding(vocab_size, word_dim)
        self.fc_transform = nn.Linear(300, 2048)
        self.rl_rnn = nn.GRUCell(300, 300).cuda()
        self.linear_rl_image_mus =nn.Linear(300 + 300, 500 * self.head).cuda()
        self.linear_logvars = nn.Linear(300 + 300, self.head).cuda()
        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(embed_size)
        self.bn.momentum = 0.01
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths, train):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        aspace = 500

        x = x.detach()

        feature = x.cuda().float()

        hx_image = torch.zeros(x.size(0), (300)).cuda()

        action_img_all = []

        sampled_log_img = []

        image_all = []
        img_att_all = []
        log_all = []
        for i in range(feature.size(1)):
            if i <= feature.size(1) - 1:
                img = feature[:, i, :]
            hx_image = self.rl_rnn(img, hx_image)
            img_mu = self.linear_rl_image_mus(F.relu(torch.cat([img, hx_image], 1)))
            logvar = F.softplus(self.linear_logvars(F.relu(torch.cat([img, hx_image], 1))))
            img_mu = img_mu.view(-1, self.head, aspace)
            ind, prob, out = gumbel_softmax(img_mu, train)
            prob = prob.squeeze()
            action_space = torch.tensor([i for i in range(0, aspace)]).cuda()
            action_space = action_space.unsqueeze(0).unsqueeze(1).expand(img.size(0), self.head, aspace).detach()

            action_img = (action_space * out.view(img.size(0), self.head, aspace)).sum(2).view(img.size(0), -1)

            action_img = action_img.float() / aspace
            action_img_all += [action_img]

            sampled_log_img += [torch.log(prob)]    #### b, head, 1
            # print (action_img)

            std = torch.exp(0.5 * logvar).view(feature.size(0), self.head, -1)
            # eps = torch.randn_like(std)

            if  train == True:
                # img_att = eps * std + ((action_img).view(-1, 1).float())# continues
                action_img = action_img.view(action_img.size(0), self.head, -1)
                img_att = []
                logprob =[]
                for h in range(self.head):
                  dist = Normal(action_img[:,h,:], std[:,h,:])
                  img_att.append(dist.sample())
                  logprob.append(dist.log_prob(dist.sample()))
                logprob = torch.stack(logprob, 1)
                log_all.append(logprob)
                img_att = torch.stack(img_att, 2)
                img_att = img_att.view(-1, self.head, 1)
            else:
                img_att = action_img.view(-1,self.head, 1)
               # print (img_att[0])
            img_att = 20*F.sigmoid(img_att)
            img_att_all.append(img_att)

            text = img.view(img.size(0), self.head, -1) * img_att.view(img_att.size(0), self.head, 1) ####use the hidden neurons
            text = text.view(text.size(0), -1)
            image_all.append(text)

        if train == True:
            log_all = torch.stack(log_all, 2)
            log_all = log_all.view(feature.size(0), self.head, -1)
        else:
            log_all = 1.
        sampled_log_img_all = torch.stack(sampled_log_img, 1).squeeze()
        sampled_log_img_all = sampled_log_img_all.view(feature.size(0), self.head, -1)
        action_imgs_all = torch.stack(action_img_all, 2).view(feature.size(0), self.head, -1)
        img_emb = torch.stack(image_all, 2)

        img_emb = img_emb[:,:,:max(lengths)]
        # (img_emb.size(), max(lengths))
        packed = pack_padded_sequence(img_emb.transpose(2,1), lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.tensor(lengths).view(-1, 1, 1).cuda()
        I = Variable(I.expand(x.size(0), 1, self.embed_size) - 1)
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space

        feature_end = out

        feature_end = self.bn(feature_end)


        #feature_end = l2norm(feature_end)

        # take absolute value, used by order embeddings
        if self.use_abs:
            feature_end = torch.abs(feature_end)

        return feature_end, sampled_log_img_all, action_imgs_all, log_all


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        #im = l2norm(im)
        #s = l2norm(s)
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        scores1 = self.sim(im, im)
        scores2 = self.sim(s, s)
        diagonal = scores.diag().view(im.size(0), 1)
        diagonal1 = scores1.diag().view(im.size(0), 1)
        diagonal2 = scores2.diag().view(im.size(0), 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        d11 = diagonal1.expand_as(scores1)
        d21 = diagonal2.expand_as(scores2)
        cost11 =  (self.margin + scores1 - d11).clamp(min=0)
        cost21 = (self.margin + scores2 - d21).clamp(min=0)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)

        mask1 = torch.eye(scores1.size(0)) > .5
        I1 = Variable(mask1)

        mask2 = torch.eye(scores2.size(0)) > .5
        I2 = Variable(mask2)
        if torch.cuda.is_available():
            I = I.cuda()
            I1 = I1.cuda()
            I2 = I2.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        cost11 = cost11.masked_fill_(I1, 0)
        cost21 = cost21.masked_fill_(I2, 0)
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            cost11 = cost11.max(0)[0]
            cost21 = cost21.max(0)[0]
        return cost_s.sum() + cost_im.sum() + cost11.sum() + cost21.sum()




class VSRN(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models

        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True
        self.classifier1 = nn.Linear(2048, 512).cuda()
        self.classifier2 = nn.Linear(512, 113288).cuda()
        self.linear1 = nn.Linear(opt.embed_size, 256).cuda()
        self.linear2 = nn.Linear(256, 1).cuda()


        #####   captioning elements
        self.d_criterion = nn.BCELoss()
        self.rl_criterion = RewardCriterion()
   
        self.classify_criterion = CrossEntropyLabelSmooth(113288)
        self.caption_model = convcap(opt.vocab_size)
        self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.rev_grad = RevGrad()
        if torch.cuda.is_available():
            self.caption_model.cuda()
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        self.common_params = list(self.txt_enc.parameters())
        self.common_params += list(self.img_enc.parameters())
        #self.common_params += list(self.decoder.parameters())
        #self.common_params += list(self.encoder.parameters())



        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.classifier1.parameters())
        params += list(self.classifier2.parameters())
        params += list(self.linear1.parameters())
        params += list(self.linear2.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0


    def calculate_reward(self, image_emb, text_emb, qids, gids):
        image_emb1 = image_emb.detach().cpu()
        text_emb1 = text_emb.detach().cpu()
        rank1_i2t = i2t(image_emb1.cpu().numpy(), text_emb1.cpu().numpy(), image_emb1.cpu().numpy(), text_emb1.cpu().numpy())/100
        rank2_t2i = t2i(image_emb1.cpu().numpy(), text_emb1.cpu().numpy(), image_emb1.cpu().numpy(), text_emb1.cpu().numpy())/100

        map1 = eval_func(image_emb1, text_emb1, qids, gids)
        map2 = eval_func(text_emb1, image_emb1, gids, qids)

        reward = map1 + rank1_i2t + map2 + rank2_t2i


        rr1 = []
        for i in range(image_emb.size(0)):
            rrr1 = reward
            np.delete(rrr1, i, axis = 0)
            rr1.append(np.mean(rrr1))

        return reward - 0.5* np.array(rr1)


    def calcualte_caption_loss(self, fc_feats, fc_feat, labels, masks):
        max_tokens = 61
        batchsize_cap = fc_feats.size(0)
        torch.cuda.synchronize()
        labels = labels.cuda()
        seq_probs, _ = self.caption_model(fc_feats, fc_feat, labels, False)
        wordact = seq_probs[:, :, :-1]
        wordclass_v = labels[:, 1:]
        mask = masks[:, 1:].contiguous()
        wordact_t = wordact.permute(0, 2, 1).contiguous().view( \
            batchsize_cap * (max_tokens - 1), -1)
        wordclass_t = wordclass_v.contiguous().view( \
            batchsize_cap * (max_tokens - 1), 1)
        maskids = torch.nonzero(mask.view(-1)).cpu().numpy().reshape(-1)
        loss = F.cross_entropy(wordact_t[maskids, ...], \
                                   wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))
        return loss
    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
    
    def calculate_text_decoder_loss(self, text_embedding, captions, labels, masks):
        max_tokens = 61
        batchsize_cap = text_embedding.size(0)
        torch.cuda.synchronize()
        labels_all = torch.zeros([captions.size(0), max_tokens, 300]).cuda()
        labelss = labels.cuda()

        if labelss.size(1) > max_tokens:
            indd = max_tokens
        else:
            indd = labelss.size(1)
        labels_all[:,:indd,:] = labelss[:,:indd, :]
        seq_probs, _ = self.caption_model(text_embedding.cuda(), text_embedding.cuda(), labels_all, False)
        wordact = seq_probs[:, :, :-1]
        wordclass_v = captions[:, 1:].cuda()
        mask = masks[:, 1:].contiguous()
        wordact_t = wordact.permute(0, 2, 1).contiguous().view( \
            batchsize_cap * (max_tokens - 1), -1)
        wordclass_t = wordclass_v.contiguous().view( \
            batchsize_cap * (max_tokens - 1), 1)
        maskids = torch.nonzero(mask.view(-1)).cpu().numpy().reshape(-1)
        loss = F.cross_entropy(wordact_t[maskids, ...], \
                               wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))
        return loss




    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        a = 1
    def discriminator(self, emb):
        feat = F.relu(self.linear1(emb))
        result = F.sigmoid(self.linear2(feat))
        return result



    def forward_emb(self, images, captions, lengths, volatile=False, train = True):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        # images = Variable(images, volatile=volatile)
        # captions = Variable(captions, volatile=volatile)
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward

        cap_emb, sampled_log_t, actions_t, logprob_t = self.txt_enc(captions, lengths, train)
        img_emb, GCN_img_emd, sampled_log, actions, logprob = self.img_enc(images, train)

        if train == False:
            img_emb = l2norm(img_emb)
            cap_emb = l2norm(cap_emb)

        return img_emb, cap_emb, GCN_img_emd, sampled_log, actions, logprob, sampled_log_t, actions_t, logprob_t





    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        # self.logger.update('Le', loss.item(), img_emb.size(0))
        self.logger.update('Le_retrieval', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids, caption_labels, caption_masks, idx, boxes, image_id, language, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        self.optimizer.zero_grad()
        # compute the embeddings、
        img_emb, cap_emb, GCN_img_emd,  sampled_log_img, action_img, logprob, sampled_log_img_t, action_img_t, logprob_t = self.forward_emb(images, language, lengths)


        imgs = img_emb
        caps = cap_emb
        evalid = [i for i in range(imgs.size(0))]
        reward = self.calculate_reward(l2norm(imgs), l2norm(caps), evalid, evalid)
        head = 2
        self.optimizer.zero_grad()
        text_rl1 = 0
        img_rl1 = 0
        for h in range(head):
            text_rl1 += self.rl_criterion(sampled_log_img_t[:,h,:], action_img_t[:,h,:], torch.tensor(reward).cuda())
            img_rl1 += self.rl_criterion(sampled_log_img[:,h,:], action_img[:,h,:], torch.tensor(reward).cuda())

        rl_loss1 = img_rl1 + text_rl1

        self.optimizer.zero_grad()
        text_rl2 = 0
        img_rl2 = 0
        for h in range(head):
            text_rl2 += torch.mean(-logprob_t[:,h,:]*torch.tensor(reward).cuda().unsqueeze(1))
            img_rl2 = torch.mean(-logprob[:,h,:] * torch.tensor(reward).cuda().unsqueeze(1))
        rl_loss2 = img_rl2 + text_rl2

        self.optimizer.zero_grad()

        t_id = F.cross_entropy(self.classifier2(F.relu(self.classifier1(l2norm(img_emb)))), idx.cuda())
        i_id = F.cross_entropy(self.classifier2(F.relu(self.classifier1(l2norm(cap_emb)))), idx.cuda())


        classfication_loss = t_id + i_id

        text_decoder_loss = self.calculate_text_decoder_loss(l2norm(cap_emb),caption_labels, language, caption_masks)
        # measure accuracy and record loss
        retrieval_loss2 = self.forward_loss(l2norm(img_emb), l2norm(cap_emb))

        loss =  rl_loss1 + retrieval_loss2 + rl_loss2  + text_decoder_loss + classfication_loss
        loss.backward()
        #self.logger.update('Le_caption', caption_loss.item(), img_emb.size(0))
        self.logger.update('Le', loss.item(), img_emb.size(0))
        self.logger.update('class', classfication_loss.item(), img_emb.size(0))
        self.logger.update('Le_text_decode', text_decoder_loss.item(), img_emb.size(0))
        self.logger.update('reward', reward.mean(), img_emb.size(0))
        self.logger.update('rl_loss', rl_loss1, img_emb.size(0))
        self.logger.update('rl_loss2', rl_loss2, img_emb.size(0))

        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
