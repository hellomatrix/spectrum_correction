import numpy as np
import matplotlib.pyplot as plt
import math

from l1regls import l1regls
from cvxopt import matrix

def transform(x):

    y = math.log(1+x)*50
    if y<0:
        y=0
    elif y>255:
        y=255

    return y

def poly_fit(x,y):

    # x_test = [[1,6,2],[1,8,1],[1,10,0],[1,14,2],[1,18,0]]
    # y_test = [[7],[9],[13],[17.5],[18]]
    # x=x_test
    # y=y_test
    x = list(x)
    y = list(y)

    paras = np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.dot(np.transpose(x),y))
    # print(paras)
    return paras

def trangle_poly_fit(x,y,para):

    alpha_0 = para[-1][-1]
    y = np.array(y)
    alpha_x= np.array(alpha_0*x[:,0])
    y=y-alpha_x
    pa = poly_fit(x[:,1:],y)
    return np.hstack((alpha_0,pa))

def trangle_fit():

    L = np.array([np.random.rand(3),np.random.rand(3)])

    #substance reflectance
    X_r = np.array([np.random.rand(3),np.random.rand(3),np.random.rand(3),
                    np.random.rand(3), np.random.rand(3), np.random.rand(3),
                    np.random.rand(3), np.random.rand(3), np.random.rand(3),
                    np.random.rand(3), np.random.rand(3), np.random.rand(3),
                    np.random.rand(3),np.random.rand(3),np.random.rand(3),
                    np.random.rand(3),np.random.rand(3),np.random.rand(3)])

    # quantum dots transmition
    # Q = [np.random.rand(3),np.random.rand(3),np.random.rand(3)]
    Q_t = np.array([np.random.rand(3),np.random.rand(3),np.random.rand(3)])

    # substance under different lights
    X_r_l1 = X_r*L[0]
    X_r_l2 = X_r*L[1]

    # quantum dots spectrum of substance
    Q_sxr_l1 = np.dot(X_r_l1,np.transpose(Q_t))
    Q_sxr_l2 = np.dot(X_r_l2,np.transpose(Q_t))

    # quantum dots spectrum of lights
    Q_L = np.dot(L,np.transpose(Q_t))
    Q_sxr_l1 = Q_sxr_l1/Q_L[0]
    Q_sxr_l2 = Q_sxr_l2/Q_L[1]

    board = 10
    x_samples = Q_sxr_l2[:board,:]
    paras = np.zeros([Q_sxr_l1.shape[1], Q_sxr_l1.shape[1]])
    para = []

    for j in range(Q_sxr_l1.shape[1]):

        y=Q_sxr_l1[:board,j]

        if j == 0:
            x = x_samples[:, :2]
            para.append(poly_fit(x, y))
            paras[j, 0:2] = para[j]
        elif j == Q_sxr_l1.shape[1]:
            x = x_samples[:, -2:]
            para.append(trangle_poly_fit(x,y, para))
            paras[j, -2:]= para[j]
        else:
            x = x_samples[:, j - 1:j + 2]
            para.append(trangle_poly_fit(x,y, para))
            paras[j, j - 1:j + 2]=para[j]

    print(paras)

    fits=np.dot(Q_sxr_l2,np.transpose(paras))

    spectrum_range = range(Q_sxr_l1.shape[1])

    for j in range(Q_sxr_l1.shape[0]):

        fig = plt.figure()

        ax1 = fig.add_subplot(211)
        ax1.plot(spectrum_range, Q_sxr_l1[j], label='gold_class_{}'.format(j))
        ax1.plot(spectrum_range, Q_sxr_l2[j], label='origin_class_{}'.format(j))


        ax2 = fig.add_subplot(212)
        ax2.plot(spectrum_range, Q_sxr_l1[j], label='gold_class_{}'.format(j))
        ax2.plot(spectrum_range, fits[j], label='fitted_class_{}'.format(j))

    plt.legend()
    plt.show()

    return L,Q_t,paras

def get_paras(L,X_r,Q_t,b1,b2):
    # L = np.array([np.random.rand(3),np.random.rand(3)])
    #
    # #substance reflectance
    # X_r = np.array([np.random.rand(3),np.random.rand(3),np.random.rand(3),
    #                 np.random.rand(3), np.random.rand(3), np.random.rand(3),
    #                 np.random.rand(3), np.random.rand(3), np.random.rand(3),
    #                 np.random.rand(3), np.random.rand(3), np.random.rand(3),
    #                 np.random.rand(3),np.random.rand(3),np.random.rand(3),
    #                 np.random.rand(3),np.random.rand(3),np.random.rand(3)])
    #
    # # quantum dots transmition
    # # Q = [np.random.rand(3),np.random.rand(3),np.random.rand(3)]
    # Q_t = np.array([np.random.rand(3),np.random.rand(3),np.random.rand(3)])

    # substance under different lights
    X_r_l1 = X_r*L[0]
    X_r_l2 = X_r*L[1]

    # quantum dots spectrum of substance
    Q_sxr_l1 = np.dot(X_r_l1,np.transpose(Q_t))
    Q_sxr_l2 = np.dot(X_r_l2,np.transpose(Q_t))

    # quantum dots spectrum of lights
    Q_L = np.dot(L,np.transpose(Q_t))
    Q_sxr_l1 = Q_sxr_l1/Q_L[0]
    Q_sxr_l2 = Q_sxr_l2/Q_L[1]

    # board = 4

    paras=[]
    x = Q_sxr_l2[b1:b2,:]

    for i in range(Q_sxr_l1.shape[1]):
        y=np.reshape(Q_sxr_l1[b1:b2,i],[b2-b1,1])

        temp_matrix= np.hstack((x, y))
        print('correction boards=%d:%d,rank=%d.'%(b1,b2,np.linalg.matrix_rank(temp_matrix)))

        paras.append(np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.dot(np.transpose(x),y)))

    print('transform matrix:',paras)

    fits=np.dot(Q_sxr_l2,np.transpose(paras))
    fits = np.reshape(fits,[X_r.shape[0],-1])
    spectrum_range = range(Q_sxr_l1.shape[1])

    for j in range(Q_sxr_l1.shape[0]):

        fig = plt.figure()

        ax1 = fig.add_subplot(211)
        ax1.plot(spectrum_range, Q_sxr_l1[j], label='gold_class_{}'.format(j))
        ax1.plot(spectrum_range, Q_sxr_l2[j], label='origin_class_{}'.format(j))

        ax2 = fig.add_subplot(212)
        ax2.plot(spectrum_range, Q_sxr_l1[j], label='gold_class_{}'.format(j))
        ax2.plot(spectrum_range, fits[j], label='fitted_class_{}'.format(j))

    plt.legend()
    plt.show()

    return L,Q_t,paras

def one_board(L,X_r,Q_t,k,bands):

    spectrum_range = range(len(L[0]))

    # substance under different lights
    X_r_l1 = X_r*L[0]
    X_r_l2 = X_r*L[1]

    # quantum dots spectrum of substance
    Q_sxr_l1 = np.dot(X_r_l1,np.transpose(Q_t))
    Q_sxr_l2 = np.dot(X_r_l2,np.transpose(Q_t))

    # quantum dots spectrum of lights
    Q_L = np.dot(L,np.transpose(Q_t))

    ## the average of origin lights
    block_L=[]
    # block_Q_L = []
    for i in range(L.shape[0]):
        # block_Q_L.append(np.mean(np.reshape(Q_L[i],(-1,k)),1))
        block_L.append(np.mean(np.reshape(L[i],(-1,k)),1)*k)
    # block_Q_L = np.array(block_Q_L)
    block_L = np.array(block_L)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('blocks lights1')
    ax1.plot(range(block_L.shape[1]), block_L[0], label='column block light')
    # ax1.plot(range(L.shape[1]), reconstruct_L[i]/np.max(reconstruct_L[i]), label='reconstruct light')
    ax1.legend(loc='upper right', shadow=True)

    block_Q_t = []
    for i in range(Q_t.shape[0]):
        block_Q_t.append(np.mean(np.reshape(Q_t[i],(-1,k)),1)*k)
    block_Q_t = np.array(block_Q_t)

    reconstruct_block_L = np.dot(np.dot(Q_L,block_Q_t),np.linalg.inv(np.dot(np.transpose(block_Q_t), block_Q_t)))

    reconstruct_block_L1 =[]
    A = matrix(block_Q_t)
    for i in range(Q_L.shape[0]):
        b = matrix(np.transpose(Q_L[i]))
        temp = l1regls(A,b)
        reconstruct_block_L1.append(np.transpose(temp))


    ## the reconstruct and origin
    for i in range(L.shape[0]):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_title('origin lights_{}'.format(i))
        ax1.plot(range(L.shape[1]), L[i], label='full origin light')
        # ax1.plot(range(L.shape[1]), reconstruct_L[i]/np.max(reconstruct_L[i]), label='reconstruct light')
        ax1.legend(loc='upper right', shadow=True)

        ax2 = fig.add_subplot(212)
        ax2.set_title('column block lights_{}'.format(i))
        ax2.plot(range(block_L.shape[1]), block_L[i], label=' column block origin light')
        ax2.plot(range(block_L.shape[1]), reconstruct_block_L[i], label='reconstruct column block light fangcheng')
        ax2.plot(range(block_L.shape[1]), reconstruct_block_L1[i][0], label='reconstruct column block light youhua')
        # ax1.plot(range(L.shape[1]), reconstruct_L[i]/np.max(reconstruct_L[i]), label='reconstruct light')
        ax2.legend(loc='upper right', shadow=True)


    # average lights


    # Q_sxr_l1 = Q_sxr_l1/Q_L[0]
    # Q_sxr_l2 = Q_sxr_l2/Q_L[1]

    # cal_L1 = np.dot(Q_L,np.linalg.inv(np.transpose(Q_t)))


    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.plot(range(len(L[0])),L[0],label='light 1')
    # ax1.plot(range(len(L[0])),cal_L[0],label='cal light 1')
    # ax1.set_title('')
    # ax1.legend(loc='upper right', shadow=True)
    #
    # ax2 = fig.add_subplot(212)
    # ax2.plot(range(len(L[0])),L[1],label='light 2')
    # ax2.plot(range(len(L[0])),cal_L[1],label='cal light 2')
    # ax2.set_title('')
    # ax2.legend(loc='upper right', shadow=True)

    # print(L)
    # print(cal_L)
    #
    # # cal_X_r_l1 = np.dot(Q_sxr_l1,np.linalg.inv(np.transpose(Q_t)))
    # # cal_X_r_l2 = np.dot(Q_sxr_l2,np.linalg.inv(np.transpose(Q_t)))
    #
    # cal_X_r_l1 = np.dot(np.dot(Q_sxr_l1,Q_t),np.linalg.inv(np.dot(np.transpose(Q_t), Q_t)))
    # cal_X_r_l2 = np.dot(np.dot(Q_sxr_l2,Q_t),np.linalg.inv(np.dot(np.transpose(Q_t), Q_t)))
    #
    # cal_X_rl1 = cal_X_r_l1/cal_L[0]
    # cal_X_rl2 = cal_X_r_l2/cal_L[1]

    # for j in range(X_r.shape[0]):
    # for j in range(3):
    #
    #     fig = plt.figure()
    #
    #     ax1 = fig.add_subplot(111)
    #     ax1.plot(spectrum_range, X_r[j], label='gold_spectrum_{}'.format(j))
    #     ax1.plot(spectrum_range, cal_X_rl1[j], label='light1_spectrum_{}'.format(j))
    #     ax1.plot(spectrum_range, cal_X_rl2[j], label='light2_spectrum_{}'.format(j))
    #     # title = 'class_'+str(j)
    #     ax1.set_title('class_{}'.format(j))
    #     ax1.legend(loc='upper right', shadow=True)


    # return cal_L,cal_X_rl1,cal_X_rl2


def spectrum_reconstruction():

    bands = 500
    k = 10

    L = np.array([np.random.rand(bands),np.random.rand(bands)])

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title('Origin lights')
    # ax1.plot(range(L[0].shape[0]), L[0], label='light1')
    # ax1.plot(range(L[0].shape[0]), L[1], label='light2')
    # ax1.legend(loc='upper right', shadow=True)

    #substance reflectance
    X_r = np.array([np.random.rand(bands)])

    # quantum dots transmittance
    Q_t = []
    for j in range(int(bands/k)):
        Q_t.append(np.random.rand(bands))
    Q_t = np.array(Q_t)

    ## reconstruct Light and substance reflectance
    one_board(L, X_r, Q_t,k,bands)







    #
    #
    # for i in range(L.shape[0]):
    #
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #     ax1.set_title('lights_{}'.format(i))
    #     ax1.plot(range(new_L.shape[1]), new_L[i], label='origin light')
    #     ax1.plot(range(new_L.shape[1]), new_reconstruct_L[i]/np.max(new_reconstruct_L[i]), label='reconstruct light')
    #     ax1.legend(loc='upper right', shadow=True)
    #
    # new_X_r = []
    # new_reconstruct_X_r = []
    # for i in range(X_r.shape[0]):
    #     new_X_r.append(np.mean(np.reshape(X_r[i],(-1,k)),1))
    #     new_reconstruct_X_r.append(np.mean(np.reshape(reconstruct_X_r[i],(-1,k)),1))
    # new_X_r=np.array(new_X_r)
    # new_reconstruct_X_r=np.array(new_reconstruct_X_r)
    #
    # for i in range(X_r.shape[0]):
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #     ax1.set_title('substance reflectance of class_{}'.format(i))
    #     ax1.plot(range(new_X_r.shape[1]), new_X_r[i], label='origin')
    #     ax1.legend(loc='upper right', shadow=True)
    #
    #
    #
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title('Quanta dots')
    # for i in range(5):
    #     ax1.plot(range(new_Q_t.shape[1]), new_Q_t[i], label='gold_spectrum_{}'.format(i))
    #
    # ax1.legend(loc='upper right', shadow=True)

    # one_board(new_L, new_X_r, new_Q_t)

    plt.legend()
    plt.show()

if __name__== '__main__':


    ### Test 3#######################################################################################

    spectrum_reconstruction()

    ### Test 3#######################################################################################


    # tf = transform(10)
    #
    # bands = 5
    # #l1 = np.random.rand(bands)
    # #l2 = l1 + 0.1*np.random.rand(bands)
    #
    # #L = np.array([l1,l2])
    # L = np.array([np.random.rand(bands)+np.random.rand(bands),np.random.rand(bands)])
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title('Origin lights')
    # ax1.plot(range(L[0].shape[0]), L[0], label='light1')
    # ax1.plot(range(L[0].shape[0]), L[1], label='light2')
    # ax1.legend(loc='upper right', shadow=True)
    #
    #
    # #substance reflectance
    # X_r = np.array([np.random.rand(bands),np.random.rand(bands),np.random.rand(bands),
    #                 np.random.rand(bands), np.random.rand(bands), np.random.rand(bands),
    #                 np.random.rand(bands), np.random.rand(bands), np.random.rand(bands)])
    #
    # # quantum dots transmition
    # # Q = [np.random.rand(3),np.random.rand(3),np.random.rand(3)]
    # Q_t = np.array([np.random.rand(bands),np.random.rand(bands),np.random.rand(bands),
    #                 np.random.rand(bands),np.random.rand(bands)])
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title('Quanta dots')
    # for i in range(Q_t.shape[0]):
    #     ax1.plot(range(Q_t[0].shape[0]), Q_t[i], label='gold_spectrum_{}'.format(i))
    #
    # ax1.legend(loc='upper right', shadow=True)
    #
    # b1=0
    # b2=6

    ### Test 2#######################################################################################

    # one_board(L, X_r, Q_t)

    ### Test 2#######################################################################################


    ### Test 1#######################################################################################

    # for i in range(3):
    #     L,Q_t,paras = get_paras(L,X_r,Q_t,b1+i,b2+i*2)

    ### Test 1#######################################################################################



    ### Test 0#######################################################################################

    # L,Q_t,paras = trangle_fit()
    #
    # L = np.array([[0.07491273,0.86788988,0.76165599],
    #               [0.38628058,0.6714299,0.87738926]])
    #
    # Q_t = np.array([[0.63939081 , 0.89087805 , 0.03213354],
    #                 [0.23244589 , 0.9223747 ,  0.9356657],
    #                 [0.85570454,  0.87826115,  0.0188989]])
    #
    # paras = np.array([[ 4.69222401, -0.11020381, -3.5820202 ],
    #                   [ 1.41708465,  0.82344584, -1.24053049],
    #                   [ 4.48342545, -0.12712379, -3.35630166]])
    #
    # X_r =  np.array([np.random.rand(3),np.random.rand(3),np.random.rand(3),np.random.rand(3)])
    #
    # # substance under different lights
    # X_r_l1 = X_r*L[0]
    # X_r_l2 = X_r*L[1]
    #
    # # quantum dots spectrum of substance
    # Q_sxr_l1 = np.dot(X_r_l1,np.transpose(Q_t))
    # Q_sxr_l2 = np.dot(X_r_l2,np.transpose(Q_t))
    #
    # # quantum dots spectrum of lights
    # Q_L = np.dot(L,np.transpose(Q_t))
    # Q_sxr_l1 = Q_sxr_l1/Q_L[0]
    # Q_sxr_l2 = Q_sxr_l2/Q_L[1]
    #
    # fits=np.dot(Q_sxr_l2,np.transpose(paras))
    #
    # spectrum_range = range(Q_sxr_l1.shape[1])
    # for j in range(Q_sxr_l1.shape[0]):
    #
    #     fig = plt.figure()
    #
    #     ax1 = fig.add_subplot(211)
    #     ax1.plot(spectrum_range, Q_sxr_l1[j], label='gold_class_{}'.format(j))
    #     ax1.plot(spectrum_range, Q_sxr_l2[j], label='origin_class_{}'.format(j))
    #
    #
    #     ax2 = fig.add_subplot(212)
    #     ax2.plot(spectrum_range, Q_sxr_l1[j], label='gold_class_{}'.format(j))
    #     ax2.plot(spectrum_range, fits[j], label='fitted_class_{}'.format(j))
    #
    # plt.legend()
    # plt.show()



    ### Test 0#######################################################################################