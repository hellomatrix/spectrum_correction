import numpy as np
import matplotlib.pyplot as plt

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

if __name__== '__main__':

    bands = 5
    #l1 = np.random.rand(bands)
    #l2 = l1 + 0.1*np.random.rand(bands)

    #L = np.array([l1,l2])
    L = np.array([np.random.rand(bands)+np.random.rand(bands),np.random.rand(bands)])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(L[0].shape[0]), L[0], label='gold_light')
    ax1.plot(range(L[0].shape[0]), L[1], label='light2')

    #substance reflectance
    X_r = np.array([np.random.rand(bands),np.random.rand(bands),np.random.rand(bands),
                    np.random.rand(bands), np.random.rand(bands), np.random.rand(bands),
                    np.random.rand(bands), np.random.rand(bands), np.random.rand(bands)])

    # quantum dots transmition
    # Q = [np.random.rand(3),np.random.rand(3),np.random.rand(3)]
    Q_t = np.array([np.random.rand(bands),np.random.rand(bands),np.random.rand(bands),np.random.rand(bands),np.random.rand(bands)])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(Q_t[0].shape[0]), Q_t[0])
    ax1.plot(range(Q_t[0].shape[0]), Q_t[1])
    ax1.plot(range(Q_t[0].shape[0]), Q_t[2])
    ax1.plot(range(Q_t[0].shape[0]), Q_t[3])
    ax1.plot(range(Q_t[0].shape[0]), Q_t[4])



    b1=0
    b2=6

    for i in range(3):
        L,Q_t,paras = get_paras(L,X_r,Q_t,b1+i,b2+i*2)

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