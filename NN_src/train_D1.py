import argparse
import math
import time
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from BG_loader import BG_Dataset_Labeled
from torch.utils.data import DataLoader
from transformer.Models import Transformer
# from pinching4_GCD8_Real_Eval import PC4_Cal_Insert


global_step = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def generate_para():
    global global_step
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    opt.root_path_T = "D:\XYJ\Para_Iden_Pingching4\Dataset_Cal_2"
    opt.root_path_V = opt.root_path_T
    opt.root_path_Te = opt.root_path_T
    opt.root_path_R = opt.root_path_T
    opt.mat_file_T = "Dataset_LHS_5000.mat"
    opt.mat_file_V = opt.mat_file_T
    opt.mat_file_Te = opt.mat_file_T
    opt.mat_file_R = opt.mat_file_T

    opt.para_weight_file = "You Weight File" # in a shape of (80,)
    opt.para_weight = np.loadtxt(opt.para_weight_file, delimiter=",", dtype="float")
    opt.para_weight_1 = opt.para_weight[0:40].reshape(40, 1)
    opt.para_weight_2 = opt.para_weight[40:].reshape(40, 1)
    opt.para_weight = np.concatenate((opt.para_weight_1, opt.para_weight_2), axis=1)
    opt.para_weight = torch.from_numpy(opt.para_weight).cuda().float()
    # opt.key_X_train = "X_train"
    # opt.key_y_train = "y_train"
    # opt.key_X_valid = "X_valid"
    # opt.key_y_valid = "y_valid"
    # opt.key_X_test = "X_test"
    # opt.key_y_test = "y_test"

    opt.random_seed = 1
    opt.src_dim = 25
    opt.trg_dim_list = [40]
    assert sum(opt.trg_dim_list) == 40
    opt.seq_length = 1000  # TODO: should be changed during training

    assert int(opt.src_dim * opt.seq_length) == 25000

    opt.epoch = 500000

    opt.n_head = 1
    opt.n_layers = 1
    opt.batch_size = 200
    opt.d_model = 128
    opt.d_word_vec = opt.d_model
    opt.d_inner_hid = opt.d_model * 2
    opt.d_k = int(opt.d_model / opt.n_head)
    opt.d_v = opt.d_k
    opt.dropout = 0.001
    opt.LR = 5e-4

    opt.series = "Trans_Norm_D1_%s_%s_%s_%s_%s_%.4f"\
                 % (opt.n_head, opt.n_layers, opt.batch_size, opt.d_model, opt.LR, rd.random())

    print("Training model with series:  %s" % opt.series)

    opt.log_name = "./log/log_%s" % opt.series
    opt.save_model_name = "./model/model_%s" % opt.series

    if not os.path.exists("./log"):
        os.mkdir("./log")
    if not os.path.exists("./model"):
        os.mkdir("./model")

    return opt


def generate_model(opt, teacher=False):

    model = Transformer(
        n_src_vocab=opt.src_dim,
        n_trg_vocab_list=opt.trg_dim_list,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        n_position=opt.seq_length,
        dropout=opt.dropout).cuda()

    return model


def cal_loss(pred, real_value):
    # Calculate cross entropy loss, apply label smoothing if needed.
    loss_func = torch.nn.MSELoss(reduction="mean")
    # print(pred.size(), real_value.size())
    loss = loss_func(pred, real_value)

    return loss


def get_grad_mask(grad):
    const = 1 / 99
    mask_1, mask_2 = grad[:, :, 0].squeeze(), grad[:, :, 1].squeeze()
    mask_1 = (mask_1 + const) / (1 + const)
    mask_2 = (mask_2 + const) / (1 + const)
    return mask_1.cuda(), mask_2.cuda()


def cal_loss_mask(pred, trg, mask):
    if len(mask.size()) == len(trg.size()) - 1:
        mask = mask.unsqueeze(0)
        mask = mask.repeat(trg.size(0), 1, 1)

    # print(pred.size(), trg.size(), mask.size())
    loss_func = torch.nn.MSELoss(reduction="mean")
    diff = pred - trg
    # average_const = float(1 / torch.mean(mask))
    # diff_mask = diff * mask * average_const
    diff_mask = diff * mask
    loss = loss_func(diff_mask, trg - trg)
    return loss


def train_epoch(model, training_data, optimizer, opt, epoch):
    ''' Epoch operation in training phase'''
    model_S, model_T = model
    optimizer_S, optimizer_T = optimizer
    model_S.train()
    # model_T.train()
    loss_pred_only = [0, 0]
    loss_pred_only_recons = [0, 0]

    step = 0
    for src_seq, trg_seq, _, _, _ in training_data:

        src_seq = src_seq.cuda().float()
        trg_seq = trg_seq.cuda().float()

        pred_labeled_para, re_src_seq = model_S(src_seq, opt.num_model)
        # pred_labeled_2 = model_T(src_seq)
        # mask_1, mask_2 = grad[:, :, 0].squeeze(), grad[:, :, 1].squeeze()
        # loss_1 = cal_loss(pred_labeled_para, trg_seq)
        # loss_2 = cal_loss(pred_labeled_2, trg_seq_2)
        loss_1 = cal_loss_mask(pred_labeled_para, trg_seq, opt.para_weight) * (1 / torch.mean(opt.para_weight))
        loss_1_no_grad = cal_loss(pred_labeled_para, trg_seq)
        loss_1_recons = cal_loss(re_src_seq, src_seq)
        loss = loss_1_no_grad * 0.5 + loss_1 * 0.5
        # loss = loss_1 + loss_1_no_grad
        # cal_loss_mask(pred_labeled_1, trg_seq_1, mask_1)
        # loss_2_mask = cal_loss_mask(pred_labeled_2, trg_seq_2, mask_2)

        optimizer_S.zero_grad()
        # optimizer_T.zero_grad()
        loss.backward()
        # loss_2.backward()
        optimizer_S.step()
        # optimizer_T.step()

        loss_pred_only[0] += loss_1.item() * src_seq.size(0)
        loss_pred_only[1] += loss_1_no_grad.item() * src_seq.size(0)
        loss_pred_only_recons[0] += loss_1_recons.item() * src_seq.size(0)
        loss_pred_only_recons[1] += loss.item() * src_seq.size(0)
        step += src_seq.size(0)

    loss_pred_average = [x / step for x in loss_pred_only]
    loss_pred_average_recons = [x / step for x in loss_pred_only_recons]

    return loss_pred_average, loss_pred_average_recons


def eval_epoch(model, validation_data, save_flag, opt, epoch):
    ''' Epoch operation in evaluation phase '''
    model_S, model_T = model
    model_S.train()

    pred_seq_v_all, trg_seq_v_all = np.zeros((1, 80)), np.zeros((1, 80))

    with torch.no_grad():
        total_loss = [0.0, 0.0]
        total_loss_recons = [0.0, 0.0]
        step_valid = 0

        for src_seq, trg_seq, _, _, _ in validation_data:

            src_seq = src_seq.cuda().float()
            trg_seq = trg_seq.cuda().float()

            pred_labeled_para, re_src_seq = model_S(src_seq, opt.num_model)
            # pred_labeled_2 = model_T(src_seq)
            # loss_1 = cal_loss(pred_labeled_para, trg_seq)
            loss_1 = cal_loss_mask(pred_labeled_para, trg_seq, opt.para_weight) * (1 / torch.mean(opt.para_weight))
            loss_1_no_grad = cal_loss(pred_labeled_para, trg_seq)
            loss = loss_1_no_grad * 0.5 + loss_1 * 0.5
            loss_1_recons = cal_loss(re_src_seq, src_seq)

            total_loss[0] += loss_1.item() * src_seq.size(0)
            total_loss[1] += loss_1_no_grad.item() * src_seq.size(0)
            total_loss_recons[0] += loss_1_recons.item() * src_seq.size(0)
            total_loss_recons[1] += loss.item() * src_seq.size(0)
            step_valid += src_seq.size(0)

            if save_flag:
                pred_labeled_para = pred_labeled_para.cpu().numpy().reshape(pred_labeled_para.shape[0], -1)
                pred_seq_v_all = np.concatenate((pred_seq_v_all, pred_labeled_para), axis=0)
                trg_seq = trg_seq.cpu().numpy().reshape(trg_seq.shape[0], -1)
                trg_seq_v_all = np.concatenate((trg_seq_v_all, trg_seq), axis=0)

    loss_pred_average = [x / step_valid for x in total_loss]
    loss_pred_average_recons = [x / step_valid for x in total_loss_recons]

    if save_flag:
        # pred_seq_v_all = pred_seq_v_all.reshape(-1, 1)
        # trg_seq_v_all = trg_seq_v_all.reshape(-1, 1)
        np.savetxt("./Log/Result_pred_%s.csv" % (opt.series), pred_seq_v_all, delimiter=",", fmt="%f")
        np.savetxt("./Log/Result_trg_%s.csv" % (opt.series), trg_seq_v_all, delimiter=",", fmt="%f")

    return loss_pred_average, loss_pred_average_recons


def eval_epoch_real(model_list, real_data, opt):
    ''' Epoch operation in evaluation phase '''
    model_S, _ = model_list
    model_S.eval()

    pred_seq_v_all = np.zeros((1, 80))

    with torch.no_grad():
        for src_seq, trg_seq, _, _, _ in real_data:

            src_seq = src_seq.cuda().float()
            pred_para_inte, _ = model_S(src_seq, opt.num_model)

            pred_para_inte = pred_para_inte.cpu().numpy().reshape(pred_para_inte.shape[0], -1)
            pred_seq_v_all = np.concatenate((pred_seq_v_all, pred_para_inte), axis=0)

    np.savetxt("./Log/Result_pred_real_%s.csv" % (opt.series), pred_seq_v_all[1:, :], delimiter=",", fmt="%f")
    # np.savetxt("./Log/Result_trg_%s.csv" % (opt.series), trg_seq_v_all, delimiter=",", fmt="%f")


def train(model_list, data_loaders, optimizer_list, decay_optim_list, opt):
    ''' Start training '''

    training_data, validation_data, testing_data, real_data = data_loaders
    decay_optim, decay_optim_T = decay_optim_list

    def print_performances(header, loss, start_time):
        print('  - {header:15} : {loss_1: 8.5f}, {loss_2: 8.5f},'' elapse: {elapse:3.3f} min'
              .format(header=f"({header})", loss_1=loss[0] * 1000, loss_2=loss[1] * 1000, elapse=(time.time() - start_time) / 60))

    valid_losses = 1e9

    for epoch_i in range(opt.epoch):

        print("\n[Epoch Number:", epoch_i, "]")

        start = time.time()
        train_loss, train_loss_recons = train_epoch(model_list, training_data, optimizer_list, opt, epoch_i)
        print_performances('Training Loss', train_loss, start)
        print_performances('Training Loss Recons', train_loss_recons, start)

        decay_optim.step()

        start = time.time()
        valid_loss, valid_loss_recons = eval_epoch(model_list, validation_data, False, opt, epoch_i)
        print_performances('Validation Loss', valid_loss, start)
        print_performances('Validation Loss Recons', valid_loss_recons, start)

        test_flag = False
        if valid_loss_recons[1] <= valid_losses:
            valid_losses = valid_loss_recons[1]
            checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model_list[0].state_dict()}
            model_name = opt.save_model_name + '_Model_0_best.chkpt'
            torch.save(checkpoint, model_name)
            test_flag = True

        if test_flag:
            start = time.time()
            test_loss, test_loss_recons = eval_epoch(model_list, testing_data, True, opt, epoch_i)
            print_performances('Testing Loss', test_loss, start)
            print_performances('Testing Loss Recons', test_loss_recons, start)
            print('  - [Info] The checkpoint file of has been updated.')

            eval_epoch_real(model_list, real_data, opt)

        f = open(opt.log_name, "a")
        f.write("%s" % epoch_i)
        for loss in [train_loss, train_loss_recons, valid_loss, valid_loss_recons, test_loss, test_loss_recons]:
            f.write(",%s,%s" % (loss[0], loss[1]))
        f.write("\n")
        f.close()


def prepare_dataloaders(opt):
    batch_size = opt.batch_size

    train_data = BG_Dataset_Labeled(opt.root_path_T, opt.mat_file_T, "train", opt.src_dim)
    validate_data = BG_Dataset_Labeled(opt.root_path_V, opt.mat_file_V, "valid", opt.src_dim)
    test_data = BG_Dataset_Labeled(opt.root_path_Te, opt.mat_file_Te, "test", opt.src_dim)
    real_data = BG_Dataset_Labeled(opt.root_path_R, opt.mat_file_R, "real", opt.src_dim)

    train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_iterator = DataLoader(validate_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_iterator = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    real_iterator = DataLoader(real_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print("*" * 100)
    print("training dataset size: %s samples" % train_data.len)
    print("validation dataset size: %s samples" % validate_data.len)
    print("testing dataset size: %s samples" % test_data.len)
    print("real dataset size: %s samples" % real_data.len)

    return train_iterator, val_iterator, test_iterator, real_iterator


def main():
    opt = generate_para()
    # device = torch.device('cuda')
    data_loaders = prepare_dataloaders(opt)
    opt.num_model = len(opt.trg_dim_list)

    transformer = generate_model(opt, teacher=False)
    transformer = torch.nn.DataParallel(transformer).cuda()

    # print("\nTotal number of paramerters in networks is {}  ".format(sum(x.numel() for x in transformer.parameters())))
    # a = input()
    # transformer_T = copy.deepcopy(transformer)

    optimizer = optim.Adam(transformer.parameters(), lr=opt.LR, betas=(0.9, 0.99), eps=1e-09)
    # optimizer_T = optim.Adam(transformer_T.parameters(), lr=opt.LR, betas=(0.9, 0.99), eps=1e-09)
    decay_optim = optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
    # decay_optim_T = optim.lr_scheduler.ExponentialLR(optimizer_T, 0.999)

    model_list = [transformer, None]
    optimizer_list = [optimizer, None]
    decay_optim_list = [decay_optim, None]

    train(model_list, data_loaders, optimizer_list, decay_optim_list, opt)


if __name__ == '__main__':
    main()
