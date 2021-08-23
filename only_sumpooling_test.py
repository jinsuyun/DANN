import torch
import numpy as np
from utils import set_model_mode

result_list = []
best_acc = dict()


def tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, epoch, sum_pooling_mode,
           training_mode):
    print("Model test ...")

    encoder.cuda()
    classifier.cuda()
    set_model_mode('eval', [encoder, classifier])

    if training_mode == 'dann':
        discriminator.cuda()
        set_model_mode('eval', [discriminator])
        domain_correct = 0

    source_correct = 0
    target_correct = 0

    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        p = float(batch_idx) / len(source_test_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 1. Source input -> Source Classification
        source_image, source_label = source_data
        source_image, source_label = source_image.cuda(), source_label.cuda()
        source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
        source_feature = encoder(source_image)
        source_output = classifier(source_feature)
        source_pred = source_output.data.max(1, keepdim=True)[1]
        source_correct += source_pred.eq(source_label.data.view_as(source_pred)).cpu().sum()

        # 2. Target input -> Target Classification
        target_image, target_label = target_data
        target_image, target_label = target_image.cuda(), target_label.cuda()
        target_feature = encoder(target_image)
        target_output = classifier(target_feature)
        target_pred = target_output.data.max(1, keepdim=True)[1]
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).cpu().sum()

        if training_mode == 'dann':
            # 3. Combined input -> Domain Classificaion
            combined_image = torch.cat((source_image, target_image), 0)  # 64 = (S:32 + T:32)
            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_feature = encoder(combined_image)

            if sum_pooling_mode == 1:  # height sum pooling
                sum_combined_feature = torch.sum(domain_feature, dim=2).unsqueeze(dim=2)
                sum_pooling_pred = discriminator(sum_combined_feature, alpha)

            elif sum_pooling_mode == 2:  # width sum pooling
                sum_combined_feature = torch.sum(domain_feature, dim=3).unsqueeze(dim=3)
                sum_pooling_pred = discriminator(sum_combined_feature, alpha)

            elif sum_pooling_mode == 3:  # both sum pooling
                sum_combined_feature = torch.sum(domain_feature, dim=2).unsqueeze(dim=2)
                sum_pooling_pred = discriminator(sum_combined_feature, alpha)

                sum_combined_feature2 = torch.sum(domain_feature, dim=3).unsqueeze(dim=3)
                sum_pooling_pred2 = discriminator(sum_combined_feature2, alpha)

            # sum_combined_feature = discriminator(domain_feature, alpha)
            domain_pred = sum_pooling_pred.data.max(1, keepdim=True)[1]
            domain_correct += domain_pred.eq(domain_combined_label.data.view_as(domain_pred)).cpu().sum()

            # Sum pooling Discriminator TODO: sumpooling accuracy
            # domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            # domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            # domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            # domain_feature = encoder(combined_image)
            # domain_output = discriminator(domain_feature, alpha)
            # domain_pred = domain_output.data.max(1, keepdim=True)[1]
            # domain_correct += domain_pred.eq(domain_combined_label.data.view_as(domain_pred)).cpu().sum()

    if training_mode == 'dann':
        print("Test Results on DANN :")
        source_acc = 100. * source_correct.item() / len(source_test_loader.dataset)
        target_acc = 100. * target_correct.item() / len(target_test_loader.dataset)
        domain_acc = 100. * domain_correct.item() / (len(source_test_loader.dataset) + len(target_test_loader.dataset))

        best_acc = {'epoch': epoch + 1, 'source_acc': source_acc, 'target_acc': target_acc, 'domain_acc': domain_acc}
        result_list.append(best_acc)

        print('\nSource Accuracy: {}/{} ({:.2f}%)\n'
              'Target Accuracy: {}/{} ({:.2f}%)\n'
              'Domain Accuracy: {}/{} ({:.2f}%)\n'.
            format(
            source_correct, len(source_test_loader.dataset),
            100. * source_correct.item() / len(source_test_loader.dataset),
            target_correct, len(target_test_loader.dataset),
            100. * target_correct.item() / len(target_test_loader.dataset),
            domain_correct, len(source_test_loader.dataset) + len(target_test_loader.dataset),
            100. * domain_correct.item() / (len(source_test_loader.dataset) + len(target_test_loader.dataset)),
            # domain_correct, len(source_test_loader.dataset) + len(target_test_loader.dataset),#TODO:coding sum pooling accuracy
            # 100. * domain_correct.item() / (len(source_test_loader.dataset) + len(target_test_loader.dataset))
        ))
    else:
        print("Test results on source_only :")
        source_acc = 100. * source_correct.item() / len(source_test_loader.dataset)
        target_acc = 100. * target_correct.item() / len(target_test_loader.dataset)
        print('\nSource Accuracy: {}/{} ({:.2f}%)\n'
              'Target Accuracy: {}/{} ({:.2f}%)\n'.format(
            source_correct, len(source_test_loader.dataset),
            100. * source_correct.item() / len(source_test_loader.dataset),
            target_correct, len(target_test_loader.dataset),
            100. * target_correct.item() / len(target_test_loader.dataset)))

        best_acc = {'epoch': epoch + 1, 'source_acc': source_acc, 'target_acc': target_acc}
        result_list.append(best_acc)

    return result_list, target_acc
