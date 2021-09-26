import torch
import numpy as np

import model
import svhn
import utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import test
import mnist
import mnistm
from utils import save_model
from utils import visualize
from utils import set_model_mode
import params
import os
import usps
import argparse

# Source : 0, Target :1

mnist_test_loader = mnist.mnist_test_loader
mnistm_test_loader = mnistm.mnistm_test_loader

usps_test_loader = usps.usps_test_loader

svhn_test_loader = svhn.svhn_test_loader

result_list = []


def source_only(source, target, encoder, classifier, source_train_loader, target_train_loader, save_dir, save_name):
    print("Source-only training")
    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)

    best_acc = 0
    for epoch in range(params.epochs):
        print('Epoch : {}'.format(epoch))
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps

            source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            source_feature, _ = encoder(source_image)

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            class_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 50 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(source_image),
                                                                     len(source_train_loader.dataset),
                                                                     100. * batch_idx / len(source_train_loader),
                                                                     class_loss.item()))

            if (epoch + 1) % 10 == 0:
                # if epoch < 1:
                global result_list

                if source == "mnist":
                    if target == "mnistm":
                        result_list, current_acc = test.tester(source, target, encoder, classifier,
                                                               mnist_test_loader,
                                                               mnistm_test_loader, epoch,
                                                               training_mode='dann')
                    elif target == "usps":
                        result_list, current_acc = test.tester(source, target, encoder, classifier,
                                                               mnist_test_loader,
                                                               usps_test_loader, epoch,
                                                               training_mode='dann')
                elif source == "usps":
                    if target == "mnist":
                        result_list, current_acc = test.tester(source, target, encoder, classifier,
                                                               usps_test_loader,
                                                               mnist_test_loader, epoch,
                                                               training_mode='dann')
                if current_acc > best_acc:
                    best_acc = current_acc
                    save_model(epoch, encoder, classifier, 'dann', save_dir)
                # visualize(epoch, encoder, 'source', save_name)

    print(max(result_list, key=lambda x: x['target_acc']))
    best_dir = 'best_result/'

    if not os.path.exists(best_dir):
        os.mkdir(best_dir)
    target_acc = max(result_list, key=lambda x: x['target_acc'])
    target_acc = str(target_acc.get('target_acc'))
    result = best_dir + save_name + "_" + target_acc + ".txt"
    with open(result, 'a') as best_result:
        best_result.write(str(max(result_list, key=lambda x: x['target_acc'])))

    print("Best result is saved!!")


def dann(source, target, encoder, classifier, discriminator, source_train_loader, target_train_loader, sum_pooling_mode,
         sumdiscriminator, consistency, save_dir, save_name,vis):
    print("DANN training")
    print(source, " ", len(source_train_loader.dataset))
    print(target, " ", len(target_train_loader.dataset))
    # print(len(source_train_loader.dataset))
    # print(len(target_train_loader.dataset))
    # exit()
    if vis:
        visualize(source, target, -1, encoder, 'source', save_name)
    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()
    sum_discriminator_criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = nn.MSELoss().cuda()

    if sum_pooling_mode is not 0:
        optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(classifier.parameters()) +
            list(discriminator.parameters()) +
            list(sumdiscriminator.parameters()),
            lr=0.01,
            momentum=0.9)
    else:
        optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(classifier.parameters()) +
            list(discriminator.parameters()),
            lr=0.01,
            momentum=0.9)

    best_acc = 0
    for epoch in range(params.epochs):

        print('Epoch : {}'.format(epoch))
        if sum_pooling_mode is not 0:
            set_model_mode('train', [encoder, classifier, discriminator, sumdiscriminator])
        else:
            set_model_mode('train', [encoder, classifier, discriminator])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            # print(source_image.shape)  # torch.Size([32, 1, 28, 28])

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if source == "mnist":
                source_image = torch.cat((source_image, source_image, source_image), 1)

            if target == "mnist":
                target_image = torch.cat((target_image, target_image, target_image), 1)

            # print(source_image.shape)  # torch.Size([32, 1, 28, 28])
            # print(target_image.shape)  # torch.Size([32, 1, 28, 28])
            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            combined_image = torch.cat((source_image, target_image), 0)

            # print(source_image.shape)  # torch.Size([32, 3, 28, 28])
            # print(combined_image.shape)  # torch.Size([64, 3, 28, 28]) || usps torch.Size([64, 1, 28, 28])

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            # print(combined_image.shape) # s->m torch.Size([64, 3, 28, 28])

            combined_feature = encoder(combined_image)  # combined_feature --> 64 * 2352(3*28*28)
            # print(combined_feature.shape) # s->m torch.Size([64, 128, 3, 3])

            source_feature = encoder(source_image)
            # print(source_feature.shape) # s->m : torch.Size([32, 128, 3, 3])    m->mm : torch.Size([32, 48, 7, 7])

            # 1.Classification loss
            class_pred, class_feature = classifier(source_feature)  # class_pred: torch.Size([32, 10])
            # print(class_pred.shape)

            class_loss = classifier_criterion(class_pred, source_label)

            # 2. Domain loss
            domain_pred, domain_feature = discriminator(combined_feature, alpha)
            sum_pooling_pred = 0
            sum_pooling_pred2 = 0
            sum_pooling = 0
            sum_loss = 0
            sum_loss2 = 0

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()

            if sum_pooling_mode == 1:  # height sum pooling
                # print(feat.shape) #torch.Size([64, 48, 7, 7])
                sum_combined_feature = torch.sum(combined_feature, dim=2).unsqueeze(dim=2)
                # print(sum_combined_feature.shape) #torch.Size([64, 48, 1, 7])
                sum_pooling_pred, sum_feature = sumdiscriminator(sum_combined_feature, alpha)
                # sum_loss = sum_discriminator_criterion(sum_pooling_pred, domain_combined_label)

            elif sum_pooling_mode == 2:  # width sum pooling
                sum_combined_feature = torch.sum(combined_feature, dim=3).unsqueeze(dim=3)
                # print(sum_combined_feature.shape) #torch.Size([64, 48, 7, 1])
                sum_pooling_pred, sum_feature = sumdiscriminator(sum_combined_feature, alpha)
                # sum_loss = sum_discriminator_criterion(sum_pooling_pred, domain_combined_label)

            elif sum_pooling_mode == 3:  # both sum pooling
                # combined_feature_c1 = encoder(combined_image,cst=True)

                sum_combined_feature = torch.sum(combined_feature, dim=2).unsqueeze(dim=2)
                # print(sum_combined_feature.shape) #torch.Size([64, 48, 1, 7])
                # sum_pooling_pred, sum_feature = sumdiscriminator(sum_combined_feature, alpha)

                sum_combined_feature2 = torch.sum(combined_feature, dim=3).unsqueeze(dim=3)
                sum_pooling = torch.matmul(sum_combined_feature2, sum_combined_feature)

                sum_pooling_pred, sum_feature = sumdiscriminator(sum_pooling, alpha)
                # sum_pooling_pred2, sum_feature2 = sumdiscriminator(sum_combined_feature2, alpha)
                sum_loss = sum_discriminator_criterion(sum_pooling_pred, domain_combined_label)
                # sum_loss2 = sum_discriminator_criterion(sum_pooling_pred2, domain_combined_label)

            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            total_loss = class_loss + domain_loss

            if sum_pooling_mode == 1 or sum_pooling_mode == 2:  # 1 = height 2 = width 3 = both
                # print("sum mode2", sum_pooling_mode)
                total_loss += sum_loss
            elif sum_pooling_mode == 3:
                # print("both sum mode", sum_pooling_mode)
                total_loss += (sum_loss + sum_loss2)
                # print(sum_loss,"\n",sum_loss2)

            # consistency loss
            if consistency:
                if sum_pooling_mode == 1 or sum_pooling_mode == 2:
                    consistency_prob = combined_feature  # [:, 0]
                    consistency_prob = torch.mean(consistency_prob)
                    consistency_prob = consistency_prob.repeat(sum_combined_feature.size())
                    cst_loss = consistency_criterion(sum_combined_feature, consistency_prob.detach())

                    consistency_loss = 0.1 * alpha * (cst_loss)
                    total_loss += consistency_loss

                elif sum_pooling_mode == 3:
                    consistency_prob = combined_feature  # [:, 1]
                    consistency_prob = torch.mean(consistency_prob)
                    consistency_prob = consistency_prob.repeat(sum_pooling.size())
                    cst_loss = consistency_criterion(sum_pooling, consistency_prob.detach())

                    consistency_loss = 0.1 *alpha* (cst_loss)  # 0.1 = lambda
                    total_loss += consistency_loss

                else:
                    consistency_prob = combined_feature  # [:, 1]
                    consistency_prob = torch.mean(consistency_prob)
                    consistency_prob = consistency_prob.repeat(sum_feature.size())
                    cst_loss = consistency_criterion(sum_feature, consistency_prob.detach())

                    # domain_consistency_prob = domain_pred  # [:, 0]
                    # domain_consistency_prob = torch.mean(domain_consistency_prob)
                    # domain_consistency_prob = domain_consistency_prob.repeat(class_feature.size())
                    # domain_DA_cst_loss = consistency_criterion(class_feature, domain_consistency_prob.detach())

                    consistency_loss = (cst_loss)  # 0.1 = lambda
                    # consistency_loss = 0.1 * (domain_DA_cst_loss) # 0.1 = lambda
                    # consistency_loss = 0.1 * (class_DA_cst_loss + domain_DA_cst_loss)  # 0.1 = lambda
                    total_loss += consistency_loss

            total_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 50 == 0:
                if sum_pooling_mode == 1 or sum_pooling_mode == 2:
                    if consistency:
                        print(
                            '[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}\tSum Loss: {:.6f}\tConsistency Loss: {:.6f}'.format(
                                batch_idx * len(target_image), len(target_train_loader.dataset),
                                100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                                domain_loss.item(), sum_loss, (consistency_loss.item())))
                    else:
                        print(
                            '[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}\tSum Loss: {:.6f}'.format(
                                batch_idx * len(target_image), len(target_train_loader.dataset),
                                100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                                domain_loss.item(), sum_loss))
                elif sum_pooling_mode == 3:
                    if consistency:
                        print(
                            '[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}\tSum Loss: {:.6f}\tConsistency Loss: {:.6f}'.format(
                                batch_idx * len(target_image), len(target_train_loader.dataset),
                                100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                                domain_loss.item(), sum_loss, (consistency_loss.item())))
                    else:
                        print(
                            '[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}\tSum Loss: {:.6f}'.format(
                                batch_idx * len(target_image), len(target_train_loader.dataset),
                                100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                                domain_loss, sum_loss.item()))

                else:
                    if consistency:
                        print(
                            '[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}\tConsistency Loss: {:.6f}'.format(
                                batch_idx * len(target_image), len(target_train_loader.dataset),
                                100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                                domain_loss.item(), (consistency_loss.item())))
                    else:
                        print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                            batch_idx * len(target_image), len(target_train_loader.dataset),
                            100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                            domain_loss.item()))

        # if epoch == 0:
        #     visualize(epoch, encoder, 'source', save_name)

        if (epoch + 1) % 10 == 0:
            # if epoch < 1:
            # global result_list

            if source == "mnist":
                if target == "mnistm":
                    result_list, current_acc = test.tester(source, target, encoder, classifier, discriminator,
                                                           mnist_test_loader,
                                                           mnistm_test_loader, epoch,
                                                           training_mode='dann')
                elif target == "usps":
                    result_list, current_acc = test.tester(source, target, encoder, classifier, discriminator,
                                                           mnist_test_loader,
                                                           usps_test_loader, epoch,
                                                           training_mode='dann')
            elif source == "usps":
                if target == "mnist":
                    result_list, current_acc = test.tester(source, target, encoder, classifier, discriminator,
                                                           usps_test_loader,
                                                           mnist_test_loader, epoch,
                                                           training_mode='dann')
            elif source == "svhn":
                if target == "mnist":
                    result_list, current_acc = test.tester(source, target, encoder, classifier, discriminator,
                                                           svhn_test_loader,
                                                           mnist_test_loader, epoch,
                                                           training_mode='dann')
            if current_acc > best_acc:
                best_acc = current_acc
                save_model(epoch, encoder, classifier, discriminator, 'dann', save_dir)
            if vis:
                visualize(source, target, epoch, encoder, 'source', save_name)

    print(max(result_list, key=lambda x: x['target_acc']))
    best_dir = 'best_result/'

    if not os.path.exists(best_dir):
        os.mkdir(best_dir)
    target_acc = max(result_list, key=lambda x: x['target_acc'])
    target_acc = str(target_acc.get('target_acc'))
    result = best_dir + save_name + "_" + target_acc + ".txt"
    with open(result, 'a') as best_result:
        best_result.write(str(max(result_list, key=lambda x: x['target_acc'])))

    print("Best result is saved!!")

    # save_model(encoder, classifier, discriminator, 'dann', save_name)
    # visualize(encoder, 'source', save_name)
