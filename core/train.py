import torch
import torch.nn.functional as f
from torch.autograd import Variable
from core.utils import AverageMeter
import numpy as np


def train(net, criterion, optimizer, train_loader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()

    loss_all = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        # if options['loss'] == 'AMPFLoss' and epoch < 5:
        #     options['R_recording'].append(criterion.radius.cpu().detach().numpy())
        #     if options['cs'] or options['cs++']:
        #         r0 = criterion.radius.detach()
        #         o_center = criterion.points.mean(0, keepdim=True)
        #         d0 = (criterion.points - o_center).pow(2).mean(1).sum(0).detach()
        #         kappa = np.log(epoch + 3) * (options['gamma'] + d0 / r0)
        #         options['kR_recording'].append((kappa * criterion.radius).cpu().detach().numpy())

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y = net(data, True)
            logic, loss = criterion(x, y, labels)
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss classifier(encoder): {:.6f} ({:.6f})"
                  .format(batch_idx + 1, len(train_loader), losses.val, losses.avg))
        loss_all += losses.avg
    return loss_all


def train_cs(net, net_discriminator, net_generator, criterion, criterion_discriminator, optimizer,
             optimizer_discriminator, optimizer_generator, train_loader, **options):
    print('train with confusing samples')
    losses, losses_generator, losses_discriminator = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    net_discriminator.train()
    net_generator.train()

    torch.cuda.empty_cache()

    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            gan_target = gan_target.cuda()

        data, labels = Variable(data), Variable(labels)

        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = net_generator(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        target_variable = Variable(gan_target)
        optimizer_discriminator.zero_grad()
        output = net_discriminator(data)
        error_discriminator_real = criterion_discriminator(output, target_variable)
        error_discriminator_real.backward()

        # train with fake
        target_variable = Variable(gan_target.fill_(fake_label))
        output = net_discriminator(fake.detach())
        error_discriminator_fake = criterion_discriminator(output, target_variable)
        error_discriminator_fake.backward()
        error_discriminator = error_discriminator_real + error_discriminator_fake
        optimizer_discriminator.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizer_generator.zero_grad()
        # Original GAN loss
        target_variable = Variable(gan_target.fill_(real_label))
        output = net_discriminator(fake)
        error_generator = criterion_discriminator(output, target_variable)  # formula (18): log D(G(z_i))

        # minimize the true distribution
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        error_generator_fake = criterion.fake_loss(x).mean()  # formula (18): H(z_i, P)
        generator_loss = error_generator + options['beta'] * error_generator_fake
        generator_loss.backward()
        optimizer_generator.step()

        losses_generator.update(generator_loss.item(), labels.size(0))
        losses_discriminator.update(error_discriminator.item(), labels.size(0))

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        _, loss = criterion(x, y, labels)

        # KL divergence
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = net_generator(noise)
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        f_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + options['beta'] * f_loss_fake
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})"
                  .format(batch_idx + 1, len(train_loader), losses.val, losses.avg, losses_generator.val,
                          losses_generator.avg, losses_discriminator.val, losses_discriminator.avg))
        loss_all += losses.avg
    return loss_all


def train_cs_ampf(net, net_discriminator, net_generator, criterion, criterion_discriminator, optimizer,
                  optimizer_discriminator, optimizer_generator, train_loader, epoch=None, **options):
    print('train with confusing samples')
    losses, losses_generator, losses_discriminator = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    net_discriminator.train()
    net_generator.train()

    torch.cuda.empty_cache()
    loss_all, real_label, fake_label = 0, 1, 0

    r0 = torch.tensor(criterion.radius.item()).reshape(1).cuda()
    for batch_idx, (data, labels) in enumerate(train_loader):
        o_center = criterion.points.mean(0, keepdim=True)
        d0 = (criterion.points - o_center).pow(2).mean(1).sum(0).detach().item()
        kappa = np.log(epoch + 3) * (options['gamma'] + d0 / r0)

        if options['loss'] not in options['other_loss'] and epoch < 5:
            options['R_recording'].append(criterion.radius.cpu().detach().numpy())
            options['kR_recording'].append((kappa * criterion.radius).cpu().detach().numpy())

        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            gan_target = gan_target.cuda()

        data, labels = Variable(data), Variable(labels)

        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = net_generator(noise)

        # (1) Update D network    #
        gan_target.fill_(real_label)                # train with real
        target_variable = Variable(gan_target)
        optimizer_discriminator.zero_grad()
        output = net_discriminator(data)
        error_discriminator_real = criterion_discriminator(output, target_variable)
        error_discriminator_real.backward()

        # train with fake
        target_variable = Variable(gan_target.fill_(fake_label))
        output = net_discriminator(fake.detach())
        error_discriminator_fake = criterion_discriminator(output, target_variable)
        error_discriminator_fake.backward()
        error_discriminator = error_discriminator_real + error_discriminator_fake
        optimizer_discriminator.step()

        # (2) Update G network    #
        optimizer_generator.zero_grad()
        # Original GAN loss
        target_variable = Variable(gan_target.fill_(real_label))
        output = net_discriminator(fake)
        error_generator = criterion_discriminator(output, target_variable)  # formula (18): log D(G(z_i))

        # my H_1
        fake_features, _ = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        _dis_known = (fake_features - o_center).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        j_zi = criterion.margin_loss(_dis_known, kappa * r0, target)

        generator_loss = error_generator + options['alpha'] * j_zi
        generator_loss.backward()
        optimizer_generator.step()

        losses_generator.update(generator_loss.item(), labels.size(0))
        losses_discriminator.update(error_discriminator.item(), labels.size(0))

        # (3) Update classifier   #
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        _, loss, center_loss = criterion(x, y, labels)        # cross entropy loss

        # KL divergence
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = net_generator(noise)
        fake_features, _ = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())

        _dis_known = (fake_features - o_center).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        j0_zi = criterion.margin_loss(_dis_known, kappa * criterion.radius, target)

        total_loss = loss + options['beta'] * j0_zi
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) Net_center_loss {:.3f}\t G {:.3f} ({:.3f}) J0 {:.3f}\t"
                  " D {:.3f} ({:.3f})"
                  .format(batch_idx + 1, len(train_loader), losses.val, losses.avg, center_loss.item(),
                          losses_generator.val, losses_generator.avg, j0_zi.item(),
                          losses_discriminator.val, losses_discriminator.avg))
            print("R:{:.3f}".format(criterion.radius.item()))
        loss_all += losses.avg
    print()
    return loss_all


def train_cs_ampf_plus(net, net_generator_2, criterion, optimizer, optimizer_generator_2, train_loader, epoch=None,
                       **options):
    print('train with confusing samples plus')
    losses = AverageMeter()
    losses_generator_2 = AverageMeter()

    net.train()
    net_generator_2.train()
    torch.cuda.empty_cache()

    loss_all, real_label, fake_label = 0, 1, 0

    r0 = torch.tensor(criterion.radius.item()).reshape(1).cuda()
    for batch_idx, (data, labels) in enumerate(train_loader):
        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        o_center = criterion.points.mean(0)
        d0 = (criterion.points - o_center).pow(2).mean(1).sum(0).detach().item()
        kappa = np.log(epoch + 3) * (options['gamma'] + d0 / r0)

        if epoch < 5:
            options['R_recording'].append(criterion.radius.cpu().detach().numpy())
            options['kR_recording'].append((kappa * criterion.radius).cpu().detach().numpy())

        data, labels = Variable(data), Variable(labels)

        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake_2 = net_generator_2(noise)

        # (1) Update G_2 network  #
        delta_x = torch.randn((data.shape[0], len(o_center))).cuda()
        delta_x = delta_x * (criterion.points-o_center).pow(2).mean(1).mean() / (1 + 3 * np.sqrt(2/len(o_center)))
        optimizer_generator_2.zero_grad()

        # Original GAN loss
        fake_features, _ = net(fake_2, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        error_generator_2 = f.mse_loss(o_center.repeat(fake_features.shape[0], 1) + delta_x, fake_features)

        generator_loss_2 = error_generator_2
        generator_loss_2.backward()
        optimizer_generator_2.step()

        losses_generator_2.update(generator_loss_2.item(), labels.size(0))

        # (2) Update classifier   #
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        _, loss, center_loss = criterion(x, y, labels)        # cross entropy loss

        # KL divergence
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = net_generator_2(noise)
        fake_features, _ = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())

        # my J_2
        _dis_known = (fake_features - o_center).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        j2_zi = criterion.margin_loss(_dis_known, kappa * criterion.radius, target)

        total_loss = loss + options['beta'] * j2_zi
        total_loss.backward()
        optimizer.step()
        # print('J2', j2_zi)
        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) Net_center_loss {:.3f}\t"
                  "G_2 {:.3f} ({:.3f}) J2 {:.4f}"
                  .format(batch_idx + 1, len(train_loader), losses.val, losses.avg, center_loss.item(),
                          losses_generator_2.val, losses_generator_2.avg, j2_zi.item()))
            print("R:{:.4f}".format(criterion.radius.item()))
        loss_all += losses.avg
    print()
    return loss_all
