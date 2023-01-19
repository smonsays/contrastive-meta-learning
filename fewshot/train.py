import os
import pickle
from os import path
import numpy as np
from datetime import datetime
from collections import OrderedDict

import argparse

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
from multiprocessing import Semaphore, Queue

import copy 
from time import sleep

import model
from utils import data_utils as utils
from tqdm import tqdm

    
def create_slow_optimizer(model, args):

    # all parameters are slow weights i.e. the init of the fast weights
    slow_weights = []
    for name, params in model.named_parameters():
        slow_weights.append(params)
    
    if args.optimizer_slow == "ADAM":
        optimizer_slow=torch.optim.Adam(list(slow_weights), 
                                        args.lr_slow,
                                        betas= (args.momentum_slow, 0.999),)

    elif args.optimizer_slow == "RMSprop":
        optimizer_slow = torch.optim.RMSprop(list(slow_weights), args.lr_slow,
                                            momentum=args.momentum_slow)

    else:
        optimizer_slow = torch.optim.SGD(list(slow_weights), args.lr_slow, 
                                    momentum=args.momentum_slow,
                                    nesterov=(args.momentum_slow > 0.0))
    return optimizer_slow

def create_fast_optimizer(model, args):
    
    # depending on the setup, get fast weights 
    fast_weights_names = []
    fast_weights_groups = ["classifier", "BatchNorm", "Conv2d", "Linear"]

    fast_weights = []
    for name, params in model.named_parameters():
        if args.hnet_model and not args.only_head:
            if "conv.coefficients" in name or "classifier"  in name:
                fast_weights_names.append(name)
                fast_weights.append(params)
        elif args.only_head:
            if "classifier" in name:
                fast_weights_names.append(name)
                fast_weights.append(params)
        else:
            fast_weights_names.append(name)
            fast_weights.append(params)

    if args.optimizer_fast == "ADAM":
        optimizer_fast = torch.optim.Adam(list(fast_weights), 
                                        args.lr_fast,
                                        betas = (args.momentum_fast, 0.999),)

    elif args.optimizer_fast == "RMSprop": 
        optimizer_fast = torch.optim.RMSprop(list(fast_weights), args.lr_fast,
                                            momentum=args.momentum_fast)
    else:
        optimizer_fast = torch.optim.SGD(list(fast_weights), args.lr_fast, 
                                    momentum=args.momentum_fast,
                                    nesterov=(args.momentum_fast > 0.0))

    return optimizer_fast, fast_weights_names

def first_phase(args, model, optimizer_fast, fast_weights_names, 
                loss_function, train_inputs, train_targets, 
                test_inputs, test_targets, 
                params_init_data, training_stats):
    
    # run inner loop
    loss_second_last = 1.
    for kk in range(args.steps_first_phase):
        train_logits = model(train_inputs)
        loss = loss_function(train_logits, train_targets)
        # weight decay
        if args.wd_fast > 0:
            wd_loss = 0 
            for (name, param) in model.named_parameters():
                if name in fast_weights_names:
                    wd_loss += 0.5*torch.sum((param-params_init_data[name])**2)
            loss += args.wd_fast*wd_loss
        model.zero_grad()
        loss.backward()
        optimizer_fast.step()

        # save second last loss for ratio computation
        if kk == args.steps_first_phase - 2:
            loss_second_last = loss.item()

    # after the inner loop we have to do a couple of things
    # 1. get the gradients wrt to theta
    # 2. save the params for later initialisation for the third phase 
    #    which starts where the 1. phase ended
    # 3. print sume stuff

    with torch.no_grad():
        if loss.item() != 0:
            ratio_loss = loss_second_last/loss.item()
        else:
            ratio_loss = 1.
        training_stats[0] = loss_second_last
        training_stats[1] = loss.item()
        
        if args.verbose:
            grad_norm = 0.
            for name, param in model.named_parameters():
                if name in fast_weights_names:
                    grad_norm += torch.sum(param.grad.data.detach()**2)
                    
            grad_norm = torch.sqrt(grad_norm)
            print("\nStats Phase 1")
            print("Grad norm free phase ", grad_norm.item())
            print("Loss", loss.item())
            print("Loss decay ratio", ratio_loss)
        
        # Save params
        grad_params_beta_zero = OrderedDict()
        for (name, param) in model.named_parameters():
            if name in fast_weights_names:
                # this is the derivative of the norm**2 
                grad_params_beta_zero[name] = -1*args.wd_fast*\
                                                 param.detach().clone()
            else:
                grad_params_beta_zero[name] = param.grad.data.detach().clone()
        
    # compute partial/partial_theta L_out(theta) at phi^*_0
    test_logits = model(test_inputs)
    loss = loss_function(test_logits, test_targets)
    # weight decay
    if args.wd_fast > 0:
        wd_loss = 0 
        for (name, param) in model.named_parameters():
            if name in fast_weights_names:
                wd_loss += 0.5*torch.sum((param -params_init_data[name])**2)
        loss += args.wd_fast*wd_loss
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        # Save params
        L_out_partial_theta = OrderedDict()
        for (name, param) in model.named_parameters():
            if name in fast_weights_names:
                # this is the derivative of the norm**2 
                L_out_partial_theta[name] = 0.0
            else:
                L_out_partial_theta[name] = param.grad.data.detach().clone()

        if args.symmetric_ep or args.forward_fd:
            # Save params for minus phase
            params_after_first_phase = OrderedDict()
            for (name, param) in model.named_parameters():
                if name in fast_weights_names:
                    params_after_first_phase[name] = param.data.detach().clone()

            return grad_params_beta_zero, params_after_first_phase, \
                                                            L_out_partial_theta
        else:
            return grad_params_beta_zero, None, L_out_partial_theta
        
def second_phase(args, model, optimizer_fast, fast_weights_names, 
                loss_function, train_inputs, train_targets, 
                test_inputs, test_targets, 
                params_init_data, training_stats):
    
    loss_second_last = 1.
    for kk in range(args.steps_sec_phase):
        
        train_logits = model(train_inputs)
        test_logits = model(test_inputs)
        loss = loss_function(train_logits, train_targets)
        nudge_loss = loss_function(test_logits, test_targets)

        # weight decay
        if args.wd_fast > 0:
            wd_loss = 0 
            for (name, param) in model.named_parameters():
                if name in fast_weights_names:
                    wd_loss += 0.5*torch.sum((param -params_init_data[name])**2)
            loss += args.wd_fast*wd_loss
        
        if args.beta_rescaling:
            loss = (loss + args.beta*nudge_loss)/(1+args.beta)
        else:
            loss = loss + args.beta*nudge_loss
            
        model.zero_grad()
        loss.backward()
        optimizer_fast.step()

        if kk <= args.steps_sec_phase - 2:
            loss_second_last = loss.item()
    
    # the partial / partial_theta is only computed over L_in = L_train + reg
    train_logits = model(train_inputs)
    loss=loss_function(train_logits, train_targets)
    if args.wd_fast > 0:
        wd_loss = 0 
        for (name, param) in model.named_parameters():
            if name in fast_weights_names:
                wd_loss += 0.5*torch.sum((param -params_init_data[name])**2)
        loss += args.wd_fast*wd_loss
    model.zero_grad()
    loss.backward()

    # after the inner loop we have to do a couple of things
    # 1. get the gradients wrt to theta
    # 2. print sume stuff
    with torch.no_grad():

        training_stats[2] = loss_second_last
        training_stats[3] = loss.item()

        if args.verbose:
            grad_norm = 0.
            for name, param in model.named_parameters():
                if name in fast_weights_names:
                    grad_norm += torch.sum(param.grad.data.detach()**2)
            grad_norm = torch.sqrt(grad_norm)
            if loss.item() != 0:
                ratio_loss = loss_second_last/loss.item()
            else:
                ratio_loss = 1.

            print("\nStats Phase 2")
            print("Grad norm free phase ", grad_norm.item())
            print("Loss", loss.item())
            print("Loss decay ratio", ratio_loss)

        # Save params
        grad_params_beta_plus = OrderedDict()
        for (name, param) in model.named_parameters():
            if name in fast_weights_names:
                grad_params_beta_plus[name] = -1*args.wd_fast*\
                                                 param.detach().clone()
            else:
                grad_params_beta_plus[name] = param.grad.data.detach().clone()

    return grad_params_beta_plus 

def third_phase(args, model, optimizer_fast, fast_weights_names, 
                loss_function, train_inputs, train_targets, 
                test_inputs, test_targets, 
                params_init_data, training_stats):
        
    loss_second_last = 1.
    for kk in range(args.steps_third_phase):

        train_logits = model(train_inputs)
        test_logits = model(test_inputs)
        loss = loss_function(train_logits, train_targets)
        nudge_loss = loss_function(test_logits, test_targets)
        
        # weight decay
        if args.wd_fast > 0:
            wd_loss = 0 
            for (name, param) in model.named_parameters():
                if name in fast_weights_names:
                    wd_loss += 0.5*torch.sum((param -params_init_data[name])**2)
            loss += args.wd_fast*wd_loss

        if args.forward_fd:
            loss = loss + 2.*args.beta*nudge_loss
        else:
            # NOTE this is the minus phase for the symmetric ep
            loss = loss - args.beta*nudge_loss

        model.zero_grad()
        loss.backward()
        optimizer_fast.step()

        if kk <= args.steps_third_phase - 2:
            loss_second_last = loss.item()
    
    # the partial / partial_theta is only computed over L_in = L_train + reg
    train_logits = model(train_inputs)
    loss=loss_function(train_logits, train_targets)
    if args.wd_fast > 0:
        wd_loss = 0 
        for (name, param) in model.named_parameters():
            if name in fast_weights_names:
                wd_loss += 0.5*torch.sum((param -params_init_data[name])**2)
        loss += args.wd_fast*wd_loss
    model.zero_grad()
    loss.backward()

    # after the inner loop we have to do a couple of things
    # 1. get the gradients wrt to theta
    # 2. print sume stuff
    with torch.no_grad():

        training_stats[4] = loss_second_last
        training_stats[5] = loss.item()

        if args.verbose:
            grad_norm = 0.
            for name, param in model.named_parameters():
                grad_norm += torch.sum(param.grad.data.detach()**2)
            grad_norm = torch.sqrt(grad_norm)
            if loss.item() != 0:
                ratio_loss = loss_second_last/loss.item()
            else:
                ratio_loss = 1.
                
            print("\nStats Phase 3")
            print("Grad norm free phase ", grad_norm.item())
            print("Loss", loss.item())
            print("Loss decay ratio", ratio_loss)

        # Save params
        if args.symmetric_ep:
            grad_params_beta_zero = OrderedDict()
            for (name, param) in model.named_parameters():
                if name in fast_weights_names:  
                    grad_params_beta_zero[name] = -1*args.wd_fast*\
                                                     param.detach().clone()
                else:
                    grad_params_beta_zero[name]=param.grad.data.detach().clone()
            return grad_params_beta_zero

        elif args.forward_fd:
            grad_params_2beta = OrderedDict()
            for (name, param) in model.named_parameters():
                if name in fast_weights_names:
                    grad_params_2beta[name] = -1*args.wd_fast*\
                                                 param.detach().clone()
                else:
                    grad_params_2beta[name] = param.grad.data.detach().clone()
            return grad_params_2beta

def compute_theta_grad(args, equi_prop_grads_slow, process_id,
                            grad_params_beta_zero, grad_params_beta_plus, 
                            grad_params_third_phase, L_out_partial_theta,
                            wait_flag, processed_tasks):
    
    # compute final theta gradients (total derivative)
    for name in grad_params_beta_plus:
        if args.forward_fd:
            # theta computation for forward differences
            equi_prop_grads_slow[name] += 1./args.beta*((-3./2.)\
                                    * grad_params_beta_zero[name] \
                                    + 2.*grad_params_beta_plus[name] \
                                    - (1./2.)*grad_params_third_phase[name])\
                                    + L_out_partial_theta[name]
        else:
            if args.beta_rescaling:
                # beta rescaling for classic EP
                # TODO check if beta_rescaling with the better finite diff
                # approx still makes sense
                equi_prop_grads_slow[name] += (1./(args.beta))\
                            *(grad_params_beta_plus[name]*(args.beta + 1) \
                            - grad_params_beta_zero[name]) + \
                            L_out_partial_theta[name]
            else:
                # classic EP or symmetric EP
                beta_int = 2. if args.symmetric_ep else 1.
                if args.symmetric_ep:
                    grad_params_beta_zero = grad_params_third_phase
                    
                equi_prop_grads_slow[name] += (1./(args.beta*beta_int))\
                                                *(grad_params_beta_plus[name] \
                                                - grad_params_beta_zero[name]) \
                                                + L_out_partial_theta[name]

    wait_flag[process_id] += 1
    processed_tasks += 1

def step(args, model, loss_function, task_list_shared, process_id, 
                                                    batch_acc, train_flag, 
                                                    wait_flag, processed_tasks,
                                                    equi_prop_grads_slow, 
                                                    params_init_data, 
                                                    training_stats, lock):

    while train_flag[process_id] == 1:
        if wait_flag[process_id] == 1 or wait_flag[process_id] == -1:
            sleep(0.1)
        else:
            with lock:
                task = task_list_shared.get()
            
            # get optimiser for inner loop
            optimizer_fast, fast_weights_names=create_fast_optimizer(model,args)
            
            # create train and test data
            train_inputs, train_targets = task[0].to(args.device),\
                                        task[1].to(args.device) #support set
            test_inputs, test_targets = task[2].to(args.device),\
                                        task[3].to(args.device) #querry set
            
            # run free phase
            grad_params_beta_zero, params_after_first_phase, \
                 L_out_partial_theta =  first_phase(args, model, optimizer_fast, 
                                    fast_weights_names, loss_function, 
                                    train_inputs, train_targets,    
                                    test_inputs, test_targets, 
                                    params_init_data, training_stats)

            with torch.no_grad():
                # track validation acc after first phase
                # as a proxy for meta-train test set acc
                test_logits = model(test_inputs)
                val_acc =  accuracy(test_logits, test_targets)
                batch_acc += val_acc

                # undo learning
                if args.undo_learning: 
                    for (name, param) in model.named_parameters():
                        if name in fast_weights_names:
                            param.copy_(params_init_data[name])

                if args.reset_optimiser:
                    optimizer_fast, _ = create_fast_optimizer(model, args)

            # run 2nd phase
            grad_params_beta_plus =  second_phase(args, model, 
                                        optimizer_fast, fast_weights_names, 
                                        loss_function, train_inputs, 
                                        train_targets, 
                                        test_inputs, test_targets, 
                                        params_init_data, training_stats)            

            # run 3rd phase
            if args.symmetric_ep or args.forward_fd:
    
                with torch.no_grad():
                    # undo learning if configured
                    if args.undo_learning_third_first: 
                        for (name, param) in model.named_parameters():
                            if name in fast_weights_names:
                                param.copy_(params_init_data[name])
                    else:
                        # Before the 3rd phase we need to reset the parameters  
                        # if we use symmetric ep.
                        if args.symmetric_ep:
                            # undo learning of second phase
                            for (name, param) in model.named_parameters():
                                if name in fast_weights_names:
                                    param.copy_(params_after_first_phase[name])

                    if args.reset_optimiser:
                        optimizer_fast, _ = create_fast_optimizer(model, args)

                grad_params_third_phase = third_phase(args, 
                                    model, optimizer_fast, fast_weights_names, 
                                    loss_function, train_inputs, train_targets, 
                                    test_inputs, test_targets, 
                                    params_init_data, training_stats)

            else:
                grad_params_third_phase = None

            # compute theta gradient
            with torch.no_grad():
                with lock:
                    compute_theta_grad(args,
                            equi_prop_grads_slow, process_id, 
                            grad_params_beta_zero, 
                            grad_params_beta_plus, 
                            grad_params_third_phase, 
                            L_out_partial_theta,
                            wait_flag, processed_tasks)

def accuracy(logits, targets):
    with torch.no_grad():
        _,predictions = torch.max(logits,dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()


def update_init_params(model, params_init_data):
    for (name, param) in model.named_parameters():
        params_init_data[name] *= 0.
        params_init_data[name] += param.data.detach().clone()

def update_model_params(model, params_init_data):
    for (name, param) in model.named_parameters():
        param.copy_(params_init_data[name])

def write_some_stats(writer, training_stats, n_it):
    #writer.add_scalar("Loss_Free_phase" , training_stats[1], n_it)
    #writer.add_scalar("Loss_Sec_phase" , training_stats[3], n_it)
    #writer.add_scalar("Loss_Third_phase", training_stats[5], n_it)
    
    if training_stats[1] != 0:
        ratio_1 = training_stats[0]/training_stats[1]
    else:
        ratio_1 = 1.
    if training_stats[3] != 0:  
        ratio_2 = training_stats[2]/training_stats[3]
    else:
        ratio_2 = 1.
    if training_stats[4] != 0: 
        ratio_3 = training_stats[4]/training_stats[5]
    else:
        ratio_3 = 1.
    #writer.add_scalar("Rate_Free_phase" , ratio_1, n_it)
    #writer.add_scalar("Rate_Sec_phase" , ratio_2, n_it)
    #writer.add_scalar("Rate_Third_phase", ratio_3, n_it)

def meta_train(args, model, loss_function, writer, epoch, lock, 
                                        sema, params_polyak_data, polyak_n):
    
    num_batches = 0
    if args.debug_iter > 0:
        best_acc_test = 0.
    
    # shared variables over processes 
    with torch.no_grad():
        params_init_data = OrderedDict()
        equi_prop_grads_slow = OrderedDict()
        for (name, param) in model.named_parameters():
            params_init_data[name] = param.detach().clone()
            params_init_data[name].share_memory_()

            equi_prop_grads_slow[name] = param.detach().clone()*0
            equi_prop_grads_slow[name].share_memory_()

    batch_acc = torch.tensor(0.0).share_memory_()
    processed_tasks = torch.tensor(0.0).share_memory_()
    train_flag = torch.ones([args.max_batches_process]).int().share_memory_()
    training_stats = torch.ones([6]).share_memory_()

    # Logic:
    #  0: currently training
    #  1: done training and wainting 
    wait_flag = torch.ones([args.max_batches_process]).int().share_memory_()

    task_list_shared = Queue()
    processes = []
    models = [copy.deepcopy(model) for _ in range(args.max_batches_process)]

    # start the processes
    for process_id in range(args.max_batches_process):
        p = mp.Process(target=step, args=(args, models[process_id], 
                                    loss_function, task_list_shared, process_id, 
                                    batch_acc, train_flag, wait_flag, 
                                    processed_tasks, equi_prop_grads_slow, 
                                    params_init_data, training_stats, lock))
        p.start()
        processes.append(p)

    # helper variables
    waiting = True
    batches_not_finished = True
    first_init_done = False
    deiter = 0

    # TODO this loop is very bulky - clean up 
    for batch in meta_dataloader["train"]:
        if num_batches >= args.batches_train or deiter > 0:
            break
        for deiter in range(args.debug_iter):
            waiting = True
            # wait until the batch of training data is done
            while waiting:  
                if first_init_done and processed_tasks < args.batch_size:
                    # if finished process
                    if 1 in wait_flag and not 0 in wait_flag:
                        pro_index = 0
                        with torch.no_grad():
                            for cur_process in wait_flag:
                                if cur_process == 1:
                                    # if finished process, reset parameters
                                    update_model_params(models[pro_index], 
                                                        params_init_data)
                                    # start process
                                    wait_flag[pro_index] *= 0
                                pro_index += 1
                    else:
                        sleep(.1)
                else:
                    if first_init_done: # and num_batches % 10 == 0:
                        if num_batches % 25 == 0 or args.debug_iter > 1:
                            # some printing
                            b_val_acc = batch_acc/(args.batch_size)*100
                            print("\tBatch val acc {:.2f} % after {} batches"\
                                                 .format(b_val_acc,num_batches))
                            if num_batches == 0 or args.debug_iter > 1:
                                ratio_1 = training_stats[0]/training_stats[1]
                                ratio_2 = training_stats[2]/training_stats[3]
                                print("\tLoss after 1. phase {:.5f},\
                                        ratio {:.5f}".format(training_stats[1],
                                        ratio_1))
                                print("\tLoss after 2. phase {:.5f},\
                                        ratio {:.5f}".format(training_stats[3], 
                                        ratio_2))
                                if args.symmetric_ep or args.forward_fd:
                                    ratio_3 =training_stats[4]/training_stats[5]
                                    print("\tLoss aftrt 3. phase {:.5f},\
                                        ratio {:.5f}".format(training_stats[5],
                                        ratio_3))
                        # hijack some hpsearching for debugging 
                        num_batches += 1
                        if args.debug_iter > 1:
                            if best_acc_test < batch_acc.item()/args.batch_size:
                                args.best_acc_epoch = num_batches
                                args.best_acc = batch_acc.item()/args.batch_size
                                best_acc_test = args.best_acc
                            args.end_acc = batch_acc.item()/args.batch_size
                        
                        if num_batches >= args.batches_train and \
                                                        not args.debug_iter > 1:
                            waiting = False
                            break

                    batch_acc *= 0.
                    with torch.no_grad():
                        # Batch gradients will be used to update
                        if first_init_done:

                            n_it = epoch*args.batches_train + num_batches
                            if num_batches == 1 or args.debug_iter > 1:
                                write_some_stats(writer, training_stats, n_it)
                            
                            # asign theta gradients to model and take a step
                            model.zero_grad()              
                            for (name, param) in model.named_parameters():
                                #if num_batches == 1:        
                                #    writer.add_histogram("slow_weights_param_"+\
                                #                name, param.data.view(-1), n_it)
                                #if num_batches == 1 and args.verbose:
                                #    writer.add_histogram("slow_weights_grads_"+\
                                #        name, equi_prop_grads_slow[name].data.\
                                #                                 view(-1), n_it)
                                param.grad =equi_prop_grads_slow[name].clone()/\
                                                               (args.batch_size)
                                if args.clamp_grads > 0:
                                    param.grad.data.clamp_(-args.clamp_grads, 
                                                      args.clamp_grads)

                            optimizer_slow.step()
        
                        # Update new theta in models in the processes
                        update_init_params(model, params_init_data)
                        
                        if epoch >= args.polyak_start:
                            polyak_n += 1
                            #print("Averging model for polyak n %i" % (polyak_n))
                            for (name, param) in model.named_parameters():
                                params_polyak_data[name] = ((polyak_n-1)*\
                                                    params_polyak_data[name] + \
                                                param.detach().clone())/polyak_n
                        
                        for m in models:
                            for (name, param)  in m.named_parameters():
                                param.copy_(params_init_data[name])

                        # resert gradients to zero
                        for key in equi_prop_grads_slow:
                            equi_prop_grads_slow[key] *= 0.

                        # resert gradients to zero
                        for task_id, task in enumerate(zip(*batch["train"], 
                                                        *batch["test"])):
                            task_list_shared.put(task)
                        
                        tp_count = 0
                        for cur_process in wait_flag:
                            if tp_count < args.max_batches_process:
                                cur_process *= 0
                                tp_count += 1
                        processed_tasks *= 0
                        first_init_done = True
                        waiting = False        
        
    task_list_shared.close()
    train_flag *= 0
    for p in processes:
        p.join()

def step_evaluate(args, model, loss_function, task_list_shared, process_id, 
                        params_init_data, outer_accuracy_all_sh, 
                        outer_loss_all_sh, eval_flag, 
                        wait_flag_eval, processed_tasks, lock):


    while eval_flag[process_id] == 1:
        if wait_flag_eval[process_id] == 1:
            sleep(0.1)
        else:
            # get data
            with lock:
                task = task_list_shared.get()

            # get optimiser for inner loop
            optimizer_fast, fast_weights_names=create_fast_optimizer(model,args)
            
            # create train and test data
            train_inputs, train_targets = task[0].to(args.device),\
                                        task[1].to(args.device) #support set
            test_inputs, test_targets = task[2].to(args.device),\
                                        task[3].to(args.device) #querry set
                                                    
            # --------- Meta-test train loop ---------
            for kk in range(args.steps_first_phase_test):
                
                train_logits = model(train_inputs)
                relax_loss = loss_function(train_logits, train_targets)
                # Add weight decay term
                if args.wd_fast > 0:
                    wd_loss = 0 
                    for (name, param) in model.named_parameters():
                        if name in fast_weights_names:
                            wd_loss += 0.5*torch.sum((param - \
                                                     params_init_data[name])**2)
                    relax_loss += args.wd_fast*wd_loss

                model.zero_grad()
                relax_loss.backward()
                optimizer_fast.step()

            with torch.no_grad():
                # Outer Loop Loss & Acc
                test_logit = model(test_inputs)
                outer_loss = loss_function(test_logit, test_targets)
                outer_accuracy = accuracy(test_logit, test_targets)

            with torch.no_grad():
                with lock:
                    outer_accuracy_all_sh += float(outer_accuracy)
                    outer_loss_all_sh += float(outer_loss)
                    wait_flag_eval[process_id] += 1
                    processed_tasks += 1

def meta_test(args, model, loss_function, lock, sema, split="test"):

    # copy params_init_data - used to compute the inner loop weight decay 
    with torch.no_grad():
        params_init_data = OrderedDict()
        for (name, param) in model.named_parameters():    
            params_init_data[name] = param.detach().clone()
            params_init_data[name].share_memory_()

    # 
    outer_accuracy_all_sh = torch.tensor(0.0).share_memory_()
    outer_loss_all_sh = torch.tensor(0.0).share_memory_()
    processed_tasks = torch.tensor(0.0).share_memory_()
    
    # Logic in eval_flag:
    #  0: done - process can be terminated
    #  1: meta-test training still running=
    eval_flag =torch.ones([args.max_batches_process_test]).int().share_memory_()

    # Logic in wait_flag_eval:
    #  0: currently training
    #  1: done training and waiting 
    wait_flag_eval = \
               torch.ones([args.max_batches_process_test]).int().share_memory_()

    task_list_shared =  Queue()
    processes_test = []
    models =[copy.deepcopy(model) for _ in range(args.max_batches_process_test)]

    # start the processes
    for process_id in range(args.max_batches_process_test):
        p = mp.Process(target=step_evaluate, args=(args, models[process_id], 
                                            loss_function, task_list_shared, 
                                            process_id, params_init_data,
                                            outer_accuracy_all_sh, 
                                            outer_loss_all_sh, eval_flag, 
                                            wait_flag_eval,processed_tasks,
                                            lock))
        p.start()
        processes_test.append(p)

    count = 0.
    for batch in meta_dataloader[split]:
        if count >= args.batches_test*args.test_batch_size:
            break

        for task_id, task in enumerate(zip(*batch["train"], *batch["test"])):
            task_list_shared.put(task)
            count += 1
        
        # wait until all tasks in task_list_shared are solved
        while processed_tasks < count:
            if 1 in wait_flag_eval and not 0 in wait_flag_eval:
                pro_index = 0
                with torch.no_grad():
                    for cur_process in wait_flag_eval:
                        if cur_process == 1:
                            # if finished process, reset parameters
                            update_model_params(models[pro_index], 
                                                params_init_data)
                            # start process
                            wait_flag_eval[pro_index] *= 0
                        pro_index += 1
            else:
                sleep(0.1)
    
    # end processes
    task_list_shared.close()
    eval_flag *= 0
    for p in processes_test:
        p.terminate()
    
    return outer_accuracy_all_sh/count, outer_loss_all_sh/count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EquiMeta')


    ## General configs
    parser.add_argument('--out_dir', type=str, default="out_dir_default",
                        help='Path to the output folder of the experiment.')
    parser.add_argument('--dont_use_cuda', action='store_true',
                        help='Dont use CUDA if available.')
    parser.add_argument('--seed', type=int, default=1, 
                        help='Random seed.')
    parser.add_argument('--set_seed', action='store_true',
                        help='Set random seed.')
    parser.add_argument('--save_model', action='store_true',
                        help='save the model')
    parser.add_argument('--load_model', action='store_true',
                        help='load the model')
    parser.add_argument('--save_load_path_model', type=str, 
                        default="/ckpt_model.pt",
                        help='Path to save/load a model')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='Use tensorboard')
    parser.add_argument('--verbose', action='store_true',
                        help='Print gradient norms and stats during training.')

    # Few-shot setup
    parser.add_argument('--num_shots_train',type=int,default=1, metavar='ktr',
                        help='K shots training phase')
    parser.add_argument('--num_shots_test',type=int,default=15, metavar='kts',
                    help='K shots test phase')
    parser.add_argument('--num_ways',type=int,default=20, metavar='nw',
                        help='Number of labels per task')
    parser.add_argument('--dataset', type=str, default="Omniglot",
                        help='Datsets supported: MiniImagenet, Omniglot')
    parser.add_argument('--num_workers',type=int,default=0, metavar='wo',
                        help='number of workers')
    parser.add_argument('--use_val_set', action='store_true',
                        help='Use val set to checkpoint (For MiniImagenet).')

    # Training & test setup
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='Number of inner loop tasks during training.')
    parser.add_argument('--test_batch_size', type=int, default=20, metavar='N',
                        help='Number of tasks in a test batch.')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='Training epochs. (1 epoch = batches_train')
    parser.add_argument('--batches_train', type=int,default=125, metavar='bt',
                        help='Number of batches per training epoch.')
    parser.add_argument('--batches_test',type=int,default=15, metavar='btt',
                        help='Number of batches of test data.')
    parser.add_argument('--test_after',type=int, default=5,
                        help='Test after every X epochs.')
    parser.add_argument('--test_start',type=int, default=0,
                        help='Test after processing X epochs.')
    parser.add_argument('--val_start',type=int, default=0,
                        help='Test after processing X epochs.')

    # Network specifications
    parser.add_argument('--mlp', action='store_true',
                        help='Use MLP for Few-Shot with Omniglot.')
    parser.add_argument('--hidden_size',type=int,default=64,metavar='h',
                        help='Classifier hidden size')
    parser.add_argument('--param_init', type=str, default="xavier",
                        help='Initialisation strategy parameters (or kaiming).')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for conv operation.')

    #EqProp general hps
    parser.add_argument('--hnet_model', action='store_true', 
                        help='Use HyperNetwork model.')
    parser.add_argument('--num_templates', type=int, default=100, 
                        help='Number of templates used ofr HyperNetwork.')
    parser.add_argument('--only_head', action='store_true', 
                        help='Use only head parameters in inner loop.')
    parser.add_argument('--clamp_grads', type=int, default=0, 
                        help='Clamp theta grads.')
    parser.add_argument('--beta_annealing_rate', type=int,default=25, 
                        help='Beta annealing step size.')
    parser.add_argument('--beta_decay_rate', type=int,default=0.1, 
                        help='Beta annealing rate.')

    parser.add_argument('--symmetric_ep', action='store_true', 
                        help='Symmetric equi prop.')
    parser.add_argument('--forward_fd', action='store_true', 
                        help='Forward differences equi prop.')
    parser.add_argument('--beta_rescaling', action='store_true', 
                        help='Beta rescaling equi prop.')
    
    parser.add_argument('--outer_loop_step_scheduler', type=int, default=100,
                        help='Outer loop lr scheduler step size.')
    parser.add_argument('--outer_loop_decay', type=float, default=0.1,
                        help='Outer loop decay,')

    #EqProp inner loop hps
    parser.add_argument('--reset_optimiser',action='store_true', 
                        help='Reset optimizer during phases.')
    parser.add_argument('--wd_fast', type=float, default=1., 
                        help='Weight decay for fast weights (default: 0.0)')
    parser.add_argument('--optimizer_fast',type=str,default="SGD",
                        help='Optimizer fast weights (SGD, ADAM or RMSprop).')
    parser.add_argument('--lr_fast', type=float, default=0.01, 
                        help='Learning rate fast weights (default: 0.01)')
    parser.add_argument('--momentum_fast',type=float,default=0.9, 
                        help='Momentum for fast weights')
    parser.add_argument('--steps_first_phase',type=int,default=200, 
                        help='Number of meta-train train - first phase')
    parser.add_argument('--steps_first_phase_test',type=int,default=50, 
                        help='Number of meta-train test steps - first phase.')
    parser.add_argument('--steps_sec_phase', type=int,default=100, 
                        help='Number of meta-train train - second phase')
    parser.add_argument('--steps_third_phase', type=int,default=100, 
                        help='Number of meta-train train - third phase')
    parser.add_argument('--beta', type=float, default=0.1, 
                        help='EqProp nudging strength.')
    parser.add_argument('--undo_learning', action='store_true', 
                        help='Restart training between 1. and 2. phase.')
    parser.add_argument('--undo_learning_third_first', action='store_true', 
                        help='Restart training between 2. and 3. phase.')
    
    #EqProp outer loop hps
    parser.add_argument('--optimizer_slow',type=str,default="ADAM",
                        help='Optimizer slow weights (SGD, ADAM or RMSprop).')
    parser.add_argument('--lr_slow', type=float, default=0.001, 
                        help='Learning rate slow weights  (default: 0.001)')
    parser.add_argument('--momentum_slow',type=float,default=0.9, 
                        help='Momentum for slow weights')    
    parser.add_argument('--polyak_avg', action='store_true',
                        help='Use polyak averging in the outer loop.')
    parser.add_argument('--polyak_start', type=int, default=10,
                        help='Starting point of polyak averging.')

    # Multi processing hps
    parser.add_argument('--max_batches_process', type=int,default=1, 
                        help='Accumulate over multiple baches.')
    parser.add_argument('--max_batches_process_test', type=int,default=1, 
                        help='Accumulate over multiple test baches.')

    #Debug inner loop
    parser.add_argument('--debug_iter', type=int, default=1,
                        help='Debug loop.')    

    args = parser.parse_args()
    
    # SETUP
    args.out_dir_w = args.out_dir
    if args.out_dir == "out_dir_default":

        hpsearch_dt = datetime.now()
        args.out_dir = os.path.join(args.out_dir,
                'search_' + hpsearch_dt.strftime('%Y-%m-%d_%H-%M-%S'))

    if not path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print("Created output folder %s." % (args.out_dir))
    # Save user configs to ensure reproducibility of this experiment.
    with open(os.path.join(args.out_dir, 'config.pickle'), 'wb') as f:
        pickle.dump(args, f)

    args.device = torch.device('cuda' if not args.dont_use_cuda
                                and torch.cuda.is_available() else 'cpu')

    # WRITER
    if args.tensorboard:
        writer=SummaryWriter()
        if args.out_dir == "out_dir_default":
            writer=SummaryWriter()
            args.out_dir = writer.log_dir
            with open(os.path.join(args.out_dir, 'config.pickle'), 'wb') as f:
                pickle.dump(args, f)
        else:
            writer=SummaryWriter(args.out_dir)

    # SEED
    if args.set_seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
    
    # DATA
    meta_dataloader, feature_size, input_channels = utils.load_data(args)

    # MODEL
    if args.mlp:
        classifier = model.LinearModel(out_features=args.num_ways
                                    ).to(args.device)
    else:
        classifier = model.ConvModel(input_channels, 
                                    out_features=args.num_ways,
                                    stride=args.stride,
                                    hidden_size=args.hidden_size,
                                    feature_size=feature_size,
                                    hnet_model=args.hnet_model,
                                    num_templates=args.num_templates,
                                    Omniglot = True if args.dataset ==\
                                               "Omniglot" else False,
                                    ).to(args.device)
    

    # MODEL INIT
    for name, params in classifier.named_parameters():
        #this is from: https://github.com/aravindr93/imaml_dev
        if args.param_init == "xavier":
            if "Conv2d.weight" in name or "Linear.weight" in name: 
                torch.nn.init.xavier_uniform_(params, gain=1.7)
            if "classifier.weight" in name:
                torch.nn.init.xavier_uniform_(params, gain=1.7)
            if "bias" in name: 
                torch.nn.init.uniform_(params, a=0.0, b=0.05)

        # this is the classic CAVIA init
        if args.param_init == "kaiming":
            if "Conv2d.weight" in name or "Linear.weight" in name:
                torch.nn.init.kaiming_uniform_(params, nonlinearity='relu')
            if "classifier.weight" in name:
                torch.nn.init.kaiming_uniform_(params, nonlinearity='linear')
            if "bias" in name: 
                params.data.fill_(0)

    loss_function = torch.nn.CrossEntropyLoss().to(args.device)

    # OPTIMIZER slow weights
    optimizer_slow = create_slow_optimizer(classifier, args)
    scheduler = StepLR(optimizer_slow, 
                      step_size=args.outer_loop_step_scheduler, 
                      gamma=args.outer_loop_decay)
    
    # MULTIPROCESS config
    mp.set_start_method('forkserver')
    lock = mp.Lock()
    sema = Semaphore(args.max_batches_process)

    # TRAINING
    best_test_acc = 0
    if args.polyak_avg:
        params_polyak_data = OrderedDict()
        polyak_n = 0
    else:
        params_polyak_data = None
        polyak_n = None

    for epoch in range(args.epochs):
        print("Start training epoch %i." % (epoch + 1))
        
        if args.polyak_avg:
            if epoch == args.polyak_start:
                polyak_n = 1
                params_polyak_data = OrderedDict()
                for (name, param) in classifier.named_parameters():
                    params_polyak_data[name] = param.detach().clone()

        meta_train(args, classifier, loss_function, writer, epoch, lock, sema, 
                                                 params_polyak_data, polyak_n)

        # if debugging ... 
        if args.debug_iter > 1:
            break 
        
        # beta decay 
        if args.beta_annealing_rate > 0:
            if epoch >= args.epochs/2. and epoch% args.beta_annealing_rate == 0:
                args.beta = args.beta*args.beta_decay_rate
        
        # TESTING 
        if epoch >= args.test_start and epoch % args.test_after == 0 or \
                                                       epoch == args.epochs - 1:
            if args.use_val_set:
                split = "val"
            else:
                split = "test"

            print("Start testing on %s set - Epoch %i." % (split, epoch + 1))

            outer_accuracy_all, outer_loss_all = meta_test(args, classifier,
                                                        loss_function,
                                                        lock, sema, split=split)                

            print("{:} Loss: {:.5f}".format(split, outer_loss_all.item()))
            print("{:} Accuracy: {:.2f} %\n".format(split, 
                                                  outer_accuracy_all*100))
            #writer.add_scalar('%s Loss' % (split), outer_loss_all, epoch)
            #writer.add_scalar('%s Accuracy' %(split), outer_accuracy_all, epoch)

            if args.polyak_avg and epoch > args.polyak_start:
                print("Start polyak testing epoch %i." % (epoch + 1))
                with torch.no_grad():
                    polyak_model = copy.deepcopy(classifier)
                    update_model_params(polyak_model, params_polyak_data)
                outer_accuracy_all, _ = meta_test(args, 
                                                     polyak_model,
                                                     loss_function,
                                                     lock, sema, split=split)
                print("Polyak {:} Accuracy: {:.2f} % \n".format(split, 
                                                     outer_accuracy_all*100))
                #writer.add_scalar('%s Polyak Accuracy' % (split), 
                #                                      outer_accuracy_all, epoch)
                

            if best_test_acc < outer_accuracy_all and  epoch >= args.val_start:
                # overwrite best val accuracy
                best_test_acc = outer_accuracy_all
                if args.use_val_set:
                    print("Better Val Accuracy: {:.2f} % \n".format(
                                                        best_test_acc*100))
                    print("Start testing on val set - Epoch %i." % (epoch + 1))
                    if args.polyak_avg and epoch > args.polyak_start:
                        outer_accuracy_all, outer_loss_all = meta_test(args,
                                                            polyak_model,
                                                            loss_function,
                                                            lock, sema,
                                                            split="test")
                    else:
                        outer_accuracy_all, outer_loss_all = meta_test(args,
                                                            classifier,
                                                            loss_function,
                                                            lock, sema,
                                                            split="test")
                print("Test set Accuracy: {:.2f} % \n".format(
                                                        outer_accuracy_all*100))
                args.best_acc_epoch = epoch
                args.best_acc = outer_accuracy_all
            
            if args.polyak_avg and epoch > args.polyak_start:
                del polyak_model

        scheduler.step()

    if args.debug_iter == 1:
        args.end_acc = outer_accuracy_all
    writer.close()
