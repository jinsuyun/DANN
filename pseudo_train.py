import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
from datasets import mnist, mnistm
from utils import save_model
from utils import visualize
from utils import set_model_mode
import torch.nn.functional as F
import params
import os

# Source : 0, Target :1
source_test_loader = mnist.mnist_test_loader
target_test_loader = mnistm.mnistm_test_loader

result_list = []


def source_only(encoder, classifier, source_train_loader, target_train_loader):
    print("Source-only training")
    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)

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
            list = test.tester(encoder, classifier, None, source_test_loader, target_test_loader, epoch,
                               training_mode='source_only')
    save_model(encoder, classifier, None, 'source')
    visualize(encoder, 'source')


def dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, sum_pooling_mode, sumdiscriminator, save_dir, save_name):
    print("DANN training")

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()

    # logit_criterion = F.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(discriminator.parameters()),
        lr=0.01,
        momentum=0.9)

    for epoch in range(params.epochs):
        print('Epoch : {}'.format(epoch))
        set_model_mode('train', [encoder, classifier, discriminator, sumdiscriminator])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            # print(source_image.shape)  # torch.Size([32, 1, 28, 28])

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = torch.cat((source_image, source_image, source_image), 1)

            # print(source_image.shape)  # torch.Size([32, 1, 28, 28])

            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            combined_image = torch.cat((source_image, target_image), 0)

            # print(source_image.shape)  # torch.Size([32, 3, 28, 28])
            # print(combined_image.shape)  # torch.Size([64, 3, 28, 28])

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            # combined_feature, feat = encoder(combined_image)  # combined_feature --> 64 * 2352(3*28*28)
            combined_feature = encoder(combined_image)  # combined_feature --> 64 * 2352(3*28*28)
            # print(feat.shape) # torch.Size([64, 48, 7, 7])

            source_feature = encoder(source_image)
            target_feature = encoder(target_image)

            # 1.Classification loss
            class_pred = classifier(source_feature)  # torch.Size([32, 10])

            logit_s = classifier(source_feature, pseudo=True)
            logit_t = classifier(target_feature, pseudo=True)
            logit_t.cuda()
            logit_s.cuda()
            # print(class_pred)


            total_logit= (logit_s + logit_t)/2
            total_logit.cuda()
            # print(logit_s)
            # print(source_label)
            # print(logit_t)
            # exit()

            # prit
            a= []
            pseudo_loss= F.cross_entropy(total_logit, source_label)

            a.append(pseudo_loss)
            logit_t = logit_t.type(torch.cuda.LongTensor)
            pseudo_loss2= F.cross_entropy(logit_t, total_logit.detach())
            a.append(pseudo_loss2)
            # pseudo_loss = logit_criterion(total_logit, source_label)
            # logit_t = logit_t.type(torch.cuda.LongTensor)
            # pseudo_loss2 = logit_criterion(logit_t, total_logit.detach())

            total_pseudo_loss = sum(a) /2


            class_loss = classifier_criterion(class_pred, source_label)

            # 2. Domain loss

            domain_pred = discriminator(combined_feature, alpha)
            # print(combined_feature.shape)

            if sum_pooling_mode == 1:  # height sum pooling
                # print(feat.shape) #torch.Size([64, 48, 7, 7])
                sum_combined_feature = torch.sum(combined_feature, dim=2).unsqueeze(dim=2)
                # print(sum_combined_feature.shape) #torch.Size([64, 48, 1, 7])
                sum_pooling_pred = sumdiscriminator(sum_combined_feature, alpha)

            elif sum_pooling_mode == 2:  # width sum pooling
                sum_combined_feature = torch.sum(combined_feature, dim=3).unsqueeze(dim=3)
                # print(sum_combined_feature.shape) #torch.Size([64, 48, 7, 1])
                sum_pooling_pred = sumdiscriminator(sum_combined_feature, alpha)

            elif sum_pooling_mode == 3:  # both sum pooling
                sum_combined_feature = torch.sum(combined_feature, dim=2).unsqueeze(dim=2)
                # print(sum_combined_feature.shape) #torch.Size([64, 48, 1, 7])
                sum_pooling_pred = sumdiscriminator(sum_combined_feature, alpha)

                sum_combined_feature2 = torch.sum(combined_feature, dim=3).unsqueeze(dim=3)
                sum_pooling_pred2 = sumdiscriminator(sum_combined_feature2, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            if sum_pooling_mode == 1 or sum_pooling_mode == 2:
                print("sum mode",sum_pooling_mode)
                sum_loss = discriminator_criterion(sum_pooling_pred, domain_combined_label)
            elif sum_pooling_mode == 3:
                print("both sum mode", sum_pooling_mode)
                sum_loss = discriminator_criterion(sum_pooling_pred, domain_combined_label)
                sum_loss2 = discriminator_criterion(sum_pooling_pred2, domain_combined_label)

            total_loss = class_loss + domain_loss + total_pseudo_loss

            if sum_pooling_mode == 1 or sum_pooling_mode == 2:
                print("sum mode2", sum_pooling_mode)
                total_loss += sum_loss
            elif sum_pooling_mode == 3:
                print("both sum mode", sum_pooling_mode)
                total_loss += sum_loss
                total_loss += sum_loss2

            total_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 50 == 0:
                if sum_pooling_mode == 1 or sum_pooling_mode == 2:
                    print(
                        '[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}\tSum Loss: {:.6f}'.format(
                            batch_idx * len(target_image), len(target_train_loader.dataset),
                            100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                            domain_loss.item(), sum_loss.item()))
                elif sum_pooling_mode == 3:
                    print(
                        '[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}\tSum Loss width: {:.6f}\tSum Loss height: {:.6f}'.format(
                            batch_idx * len(target_image), len(target_train_loader.dataset),
                            100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                            domain_loss.item(), sum_loss.item(), sum_loss2.item()))

                else:
                    # print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                    #     batch_idx * len(target_image), len(target_train_loader.dataset),
                    #     100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                    #     domain_loss.item()))
                    print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}\tPseudo Loss: {:.6f}'.format(
                        batch_idx * len(target_image), len(target_train_loader.dataset),
                        100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                        domain_loss.item()),total_pseudo_loss.item())
        # if epoch == 0:
        #     visualize(epoch, encoder, 'source', save_name)

        if (epoch + 1) % 10 == 0:
            # if epoch < 1:
            global result_list
            result_list = test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, epoch,
                                      training_mode='dann')
            save_model(epoch, encoder, classifier, discriminator, 'dann', save_dir)
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

    # save_model(encoder, classifier, discriminator, 'dann', save_name)
    # visualize(encoder, 'source', save_name)
