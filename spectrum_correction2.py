import matplotlib
matplotlib.use('TkAgg')
import scipy.io as sio
import cv2
import numpy as np
import matplotlib.pyplot as plt

def trangle_poly_fit(x,y,para):

    alpha_0 = para[-1][-1]
    y = np.array(y)
    alpha_x= np.array(alpha_0*x[:,0])
    y=y-alpha_x
    pa = poly_fit(x[:,1:],y)
    return np.hstack((alpha_0,pa))

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

def trangle_fit(light_gold,x_data_gold,spectrum_range,k=1):

    # light_gold = r_spectrum_l1  # set samples
    light_gold_samples = light_gold[0:-k]  # set samples
    # x_data_gold = r_spectrum_l2  # the x data for fitting

    # # add constant good or not
    #     for j in range(len(x_data_gold)):  # for each band
    #         x_data_gold[j] = list(x_data_gold[j])
    #         x_data_gold[j].insert(0,1)
    #         x_data_gold[j] = np.array(x_data_gold[j])
    # # add constant good or not

    x_data_samples = np.array(x_data_gold[0:(-k)])  # the x data for fitting

    # print(x_data)

    y_data = []  # the y data for fitting
    para = []
    y_fit = []

    # i = 0
    for j in spectrum_range:

        y_data.append([])
        y_fit.append([])

        for i in range(len(light_gold_samples)): # class range
            y_data[j].append(light_gold_samples[i][j]) # j means band

        # y_data[j] = np.reshape(np.array(y_data[j]),(-1,1))
        # parameters
        if j == 0:
            x = x_data_samples[:,:2]
            para.append(poly_fit(x, y_data[j]))  # paramters for band j
        elif j==len(spectrum_range):
            x = x_data_samples[:,-2:]
            para.append(trangle_poly_fit(x, y_data[j], para))
        else:
            x = x_data_samples[:,j-1:j+2]
            para.append(trangle_poly_fit(x, y_data[j], para))
        # k=k+1
        # fit curve

    # put para to angle line
    paras = np.zeros([len(spectrum_range),len(spectrum_range)])
    for j in spectrum_range:
        if j == 0:
            paras[j,0:2]= para[j]

        elif j==len(spectrum_range):
            paras[j, -2:] = para[j]

        else:
            paras[j,j-1:j+2] = para[j]

    return paras

def show_results(light_gold,x_data_gold,paras,spectrum_range,figs):

    # y_fit = np.transpose(np.reshape(y_fit, [3, -1]))
    y_fit=[]
    for j in range(len(light_gold)):  # class range

        # y_fit[j].append(np.dot(paras,np.transpose(x_data_gold[j])))
        temp = list(np.dot(paras,np.transpose(x_data_gold[j])))
        y_fit.append(temp)

        fig = figs[j]
        ax2 = fig.add_subplot(212)
        ax2.plot(spectrum_range, light_gold[j], label='gold_class_{}'.format(j))
        ax2.plot(spectrum_range, y_fit[j][:], label='fitted_class_{}'.format(j))
        # ax2.text()
        plt.legend()

        # fig.savefig('result_{}'.format(i),dpi=fig.dpi)
        # plt.figure(i + 1000)
        # plt.plot(spectrum_range, r_spectrum_l1[i], label='reflectance_light1')
        # plt.plot(spectrum_range, [fit_y[0][0][j],fit_y[1][0][j],fit_y[2][0][j]], label='reflectance_light2')
    #
    # fig.canvas.draw()
    # plt.legend(loc='upper right')
    cv2.imshow('input', img2)
    plt.show()
    # cv2.imshow('input',img2)
    # cv2.waitKey(0)

def get_reflecSpectrum(data,blocks):

    r_spectrum_l1=[]
    r_spectrum_l2=[]
    r_spectrum_l3=[]
    figs = []
    spectrum_range = range(data[0].shape[2])

    for i in range(len(blocks)):

        h1 = blocks[i][1]
        h2 = blocks[i][1]+blocks[i][3]

        r1 = blocks[i][0]
        r2 = blocks[i][0]+blocks[i][2]
        #
        # img2 = cv2.rectangle(img2, (blocks[i][0], blocks[i][1]),
        #                      ( blocks[i][0]+blocks[i][2], blocks[i][1]+blocks[i][3]),WHITE, 2)

        spectrum_l1 = np.mean(np.mean(data[0][h1:h2, r1:r2, :], 0), 0)
        spectrum_l2 = np.mean(np.mean(data[1][h1:h2, r1:r2, :], 0), 0)
        spectrum_l3 = np.mean(np.mean(data[2][h1:h2, r1:r2, :], 0), 0)

        # # point spectrum
        # spectrum_l1 = data[0][h1, r1, :]
        # spectrum_l2 = data[1][h1, r1, :]
        # spectrum_l3 = data[2][h1, r1, :]

        # plt.figure(i+10)
        # plt.plot(spectrum_range, spectrum_l1, label='light1')
        # plt.plot(spectrum_range, spectrum_l2, label='light2')
        # plt.plot(spectrum_range, spectrum_l3, label='light3')
        # plt.legend(loc='upper right')

        # save light spectrum
        if i == 0:
            l1 = spectrum_l1
            l2 = spectrum_l2
            l3 = spectrum_l3
        # save reflectance of all class
        else:
            r_spectrum_l1.append(spectrum_l1/l1)
            r_spectrum_l2.append(spectrum_l2/l2)
            r_spectrum_l3.append(spectrum_l3/l3)

            # draw the reflectance of all class
            fig = plt.figure()
            figs.append(fig)

            ax1 = fig.add_subplot(211)
            ax1.plot(spectrum_range,r_spectrum_l1[i-1],label='gold_class_{}'.format(i-1))
            ax1.plot(spectrum_range,r_spectrum_l2[i-1],label='origin_class_{}'.format(i-1))

            # plt.figure(i+100)
            # plt.plot(spectrum_range,r_spectrum_l1[i-1] ,label='reflectance_light1')
            # plt.plot(spectrum_range,r_spectrum_l2[i-1] ,label='reflectance_light2')
            # # plt.plot(spectrum_range,r_spectrum_l3[i-1] ,label='reflectance_light3')
            # plt.legend(loc='upper right')

    return r_spectrum_l1,r_spectrum_l2,r_spectrum_l3,spectrum_range,figs

def fit_normal(r_spectrum_l1,r_spectrum_l2,spectrum_range,figs,k=1):

    light_gold = r_spectrum_l1 # set samples
    light_gold_samples = light_gold[0:-k] # set samples
    x_data_gold = r_spectrum_l2 # the x data for fitting

# # add constant good or not
#     for j in range(len(x_data_gold)):  # for each band
#         x_data_gold[j] = list(x_data_gold[j])
#         x_data_gold[j].insert(0,1)
#         x_data_gold[j] = np.array(x_data_gold[j])
# # add constant good or not

    x_data_samples = x_data_gold[0:(-k)] # the x data for fitting

    # print(x_data)

    y_data = [] # the y data for fitting
    para=[]
    y_fit = []

    for j in spectrum_range: # for each band

        y_data.append([])
        y_fit.append([])

        for i in range(len(light_gold_samples)): # class range
            y_data[j].append([light_gold_samples[i][j]]) # j means band

        # parameters
        para.append(poly_fit(x_data_samples,y_data[j])) # paramters for band j

        # fit curve
        y_fit[j].append(np.dot(x_data_gold, para[j]))

    y_fit = np.transpose(np.reshape(y_fit, [3, -1]))

    for j in range(len(light_gold)):# class range

        fig=figs[j]
        ax2 = fig.add_subplot(212)
        ax2.plot(spectrum_range, light_gold[j], label='gold_class_{}'.format(j))
        ax2.plot(spectrum_range, y_fit[j][:], label='fitted_class_{}'.format(j))
        # ax2.text()
        plt.legend()

        # fig.savefig('result_{}'.format(i),dpi=fig.dpi)
        # plt.figure(i + 1000)
        # plt.plot(spectrum_range, r_spectrum_l1[i], label='reflectance_light1')
        # plt.plot(spectrum_range, [fit_y[0][0][j],fit_y[1][0][j],fit_y[2][0][j]], label='reflectance_light2')
    #
    # fig.canvas.draw()
    # plt.legend(loc='upper right')
    cv2.imshow('input',img2)
    plt.show()
    # cv2.imshow('input',img2)
    # cv2.waitKey(0)

if __name__=='__main__':

    light1 = sio.loadmat('light1/light1.mat')['light1']
    light2 = sio.loadmat('light2/light2.mat')['light2']
    light3 = sio.loadmat('light3/light3.mat')['light3']
    # k = 1
    data = [light1[:, :, [1, 51, 101]], light2[:, :, [1, 51, 101]], light3[:, :, [1, 51, 101]]]
    # data = [light1, light2, light3]

    blocks1 = [[30, 195, 20, 20], [140, 195, 20, 20], [280, 195, 10, 20],
               [390, 195, 20, 20], [485, 195, 20, 20], [640, 195, 20, 20], [804, 195, 20, 20], [1000, 195, 20, 20]]

    blocks2 = [[20, 350, 20, 20], [140, 350, 20, 20], [259, 350, 10, 20],
               [470, 350, 20, 20], [640, 350, 20, 20], [770, 350, 20, 20], [875, 350, 20, 20], [980, 350, 20, 20]]

    blocks = blocks1

    WHITE = (255, 255, 255)  # sure FG
    cv2.namedWindow('input', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('input', 900, 700)
    img = np.array(light1[:, :, 1])
    img2 = img.copy()

    # draw class
    for i in range(len(blocks)):

        x = blocks[i][0] + blocks[i][2]
        y = blocks[i][1] + blocks[i][3]

        img2 = cv2.rectangle(img2, (blocks[i][0], blocks[i][1]),
                             (blocks[i][0] + blocks[i][2], blocks[i][1] + blocks[i][3]), WHITE, 2)

        if i == 0:
            cv2.putText(img2, 'light', (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE)
        else:
            cv2.putText(img2, 'class_' + str(i-1), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE)

    # cv2.imshow('input', img2)
    # cv2.waitKey(0)

    r_spectrum_l1, r_spectrum_l2, r_spectrum_l3, spectrum_range, figs = get_reflecSpectrum(data,blocks)

    # fit_normal(r_spectrum_l1,r_spectrum_l2,spectrum_range,figs,k=1)

    light_gold = r_spectrum_l1  # set samples
    x_data_gold = r_spectrum_l2  # the x data for fitting

    para = trangle_fit(light_gold,x_data_gold,spectrum_range,k=1)
    show_results(light_gold, x_data_gold, para, spectrum_range,figs)