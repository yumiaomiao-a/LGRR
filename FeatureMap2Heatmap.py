import matplotlib.pyplot as plt
import torch

# FeatureMap2Heatmap(images, x_res,x_org_f,x_Block1, x_Block2, x_Block3, x_Block4, x_construct_240,x_Block11, x_Block22, x_Block33, x_Block44)
# feature  -->   [ batch, channel, height, width ]


def featuremap2heatmap(x, x_res,x_org_f,feature1, feature2, feature3,feature4,feature11, feature22, feature33,feature44):
    ## initial images
    feature_first_frame = x[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    # plt.savefig(args.log+'/'+args.log + '_x_visual.jpg')
    plt.savefig('./visualmap/input_visual.jpg')
    plt.close()


    feature_first_frame = x_res[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    # plt.savefig(args.log+'/'+args.log + '_x_visual.jpg')
    plt.savefig('./visualmap/x_res_visual.jpg')
    plt.close()


    feature_first_frame = x_org_f[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    # plt.savefig(args.log+'/'+args.log + '_x_visual.jpg')
    plt.savefig('./visualmap/x_org_f_visual.jpg')
    plt.close()



    ## first feature
    feature_first_frame = feature1[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig('./visualmap/x_Block1_visual.jpg')
    plt.close()

    feature_first_frame = feature11[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig('./visualmap/y_Block1_visual.jpg')
    plt.close()

    ## second feature
    feature_first_frame = feature2[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig('./visualmap/x_Block2_visual.jpg')
    plt.close()

    feature_first_frame = feature22[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig('./visualmap/y_Block2_visual.jpg')
    plt.close()


    ## third feature
    feature_first_frame = feature3[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig('./visualmap/x_Block3_visual.jpg')
    plt.close()



    feature_first_frame = feature33[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig('./visualmap/y_Block3_visual.jpg')
    plt.close()


    ## fourth feature
    feature_first_frame = feature4[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig('./visualmap/x_Block4_visual.jpg')
    plt.close()


    feature_first_frame = feature44[0,:,:,:].cpu()    ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))
    heatmap = heatmap.data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig('./visualmap/y_Block4_visual.jpg')
    plt.close()




    ## third feature
    # heatmap2 = torch.pow(map_x[0,:,:],2)    ## the middle frame
    #
    # heatmap2 = heatmap2.data.cpu().numpy()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.imshow(heatmap2)
    # plt.colorbar()
    # plt.savefig('./_x_DepthMap_visual.jpg')
    # plt.close()