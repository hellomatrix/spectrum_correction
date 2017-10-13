import numpy as np




#w= 20   # 量子点宽度,像素个数
#v = 200 # 速度 m/s
#h = []  # 高度 m
# fps = 25 # 帧率 fps
# f = 15  # 焦距 mm
# delta = 3.3  # 像素大小 um

class UAV:

    ## 量子点条带像素宽度w，飞行速度v， 飞行高度h，帧速fps，焦距f,
    def __init__(self,fps=24,f=15,delta=3.3):
        self.fps=fps
        self.f=f
        self.delta=delta

    def v2h(self,w,v):
        h = v*self.f*2/(w*self.delta*(self.fps-1))*1000
        return h

    def h2v(self,w,h):
        v = (h*w*self.delta*(self.fps-1))/(self.f*2*1000)
        return v

    def groundR2h(self,ground_r): #根据地面分辨率设置高度

        h = ground_r*self.f*1000/self.delta
        return h

    def h2groundR(self,h):

        ground_r = h/(self.f*1000/self.delta)
        return ground_r

    def range_v(self,h): #不同高度下，允许的速度误差

        return (0.5*(self.fps-1)*h*self.delta)/(self.f*1000)

if __name__=='__main__':

    uav = UAV()
    w = 200/3.3
    v = 24
    h = uav.v2h(w,v)
    print('量子点宽度为%f个像素，速度为%f（m/s）时，无人机飞行高度设置为%f米'%(w,v,h))
    #
    # w = 18
    # v = 10.56
    # h = uav.v2h(w,v)
    # print('量子点宽度为%f个像素，速度为%f（m/s）时，无人机飞行高度设置为%f米'%(w,v,h))

    w = 200/3.3
    h = 50
    v = uav.h2v(w,h)
    r = uav.h2groundR(h)
    print('量子点宽度为%f个像素，无人机飞行高度设置为%f米时, 速度为%f（m/s）'%(w,h,v))
    print('量子点宽度为%f个像素，无人机飞行高度设置为%f米时, 地面分辨率为%f（m）'%(w,h,r))

    w = 200/3.3
    h = 160
    v = uav.h2v(w,h)
    print('量子点宽度为%f个像素，无人机飞行高度设置为%f米时, 速度为%f（m/s）'%(w,h,v))

    w = (200*0.9)*2/3.3  #200um宽度，重合10%的
    h = 50
    v = uav.h2v(w,h)
    print('量子点宽度为%f个像素，无人机飞行高度设置为%f米时, 速度为%f（m/s）'%(w,h,v))

    # w = 18.8
    # v = 15
    # h = uav.v2h(w,v)
    # print('量子点宽度为%f个像素，速度为%f（m/s）时，无人机飞行高度设置为%f米'%(w,v,h))
    #
    # ground_r = 0.05 #5米长度的物体
    # h = uav.ground_r2h(ground_r)
    # print('当地面分辨率设置为%f米大小时,无人机高度设置为%f'%(ground_r,h))
    #
    # h = 200
    # temp = uav.range_v(h)
    # print('当无人机高度为%f米时,允许的速度变化范围为：%f(m/s)' % (h, temp))
    #
    # h = 400
    # temp = uav.range_v(h)
    # print('当无人机高度为%f米时,允许的速度变化范围为：%f(m/s)' % (h, temp))